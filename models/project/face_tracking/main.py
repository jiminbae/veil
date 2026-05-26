# main.py
import cv2
import logging
from time import perf_counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from face_swapper import FaceSwapper, blend_face

from config import (
    VIDEO_PATH,
    FACE_MODEL_PATH,
    TARGET_DIR,
    TARGET_PATTERN,
    TARGET_IMAGE_PATH,
    OUTPUT_PATH,
    LOG_PATH,
    CROP_ROOT,
    METADATA_PATH,
    TRACKING_METADATA_PATH,
    SIM_THRESHOLD,
    TARGET_THRESHOLD,
    SMOOTH_ALPHA,
    TARGET_HOLD_FRAMES,
    EMBEDDING_REFRESH_INTERVAL,
    LOG_EVERY_N_FRAMES,
    CROP_WRITER_WORKERS,
    ENABLE_FACE_SWAP,
    FACE_SWAP_BATCH_SIZE,
    SWAP_HOLD_FRAMES,
    MAX_SWAP_FACES_PER_FRAME,
    device,
)

from detector_tracker import init_detector, init_tracker, detect_faces_multiscale
from face_utils import crop_with_padding, clip_bbox
from face_identifier import (
    get_target_image_paths,
    get_target_embedding,
    get_track_embedding,
    check_target_match,
    cleanup_old_face_ids,
    assign_stable_face_id,
    smooth_bbox,
    target_face_ids,
    target_track_ids,
)
from crop_manager import assess_face_quality, apply_fallback_blur
from metadata_manager import make_face_data, save_metadata

def setup_dirs_and_logging():
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(CROP_ROOT).mkdir(parents=True, exist_ok=True)
    Path(METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(TRACKING_METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s | %(message)s"
    )

    logging.info("===== Experiment Started =====")
    logging.info(f"VIDEO_PATH={VIDEO_PATH}")
    logging.info(f"FACE_MODEL_PATH={FACE_MODEL_PATH}")
    logging.info(f"TARGET_DIR={TARGET_DIR}")
    logging.info(f"TARGET_PATTERN={TARGET_PATTERN}")
    logging.info(f"OUTPUT_PATH={OUTPUT_PATH}")
    logging.info(f"CROP_ROOT={CROP_ROOT}")
    logging.info(f"METADATA_PATH={METADATA_PATH}")
    logging.info(f"TRACKING_METADATA_PATH={TRACKING_METADATA_PATH}")
    logging.info(f"SIM_THRESHOLD={SIM_THRESHOLD}")
    logging.info(f"TARGET_THRESHOLD={TARGET_THRESHOLD}")
    logging.info(f"SMOOTH_ALPHA={SMOOTH_ALPHA}")
    logging.info(f"TARGET_HOLD_FRAMES={TARGET_HOLD_FRAMES}")
    logging.info(f"EMBEDDING_REFRESH_INTERVAL={EMBEDDING_REFRESH_INTERVAL}")
    logging.info(f"LOG_EVERY_N_FRAMES={LOG_EVERY_N_FRAMES}")
    logging.info(f"FACE_SWAP_BATCH_SIZE={FACE_SWAP_BATCH_SIZE}")
    logging.info(f"SWAP_HOLD_FRAMES={SWAP_HOLD_FRAMES}")
    logging.info(f"MAX_SWAP_FACES_PER_FRAME={MAX_SWAP_FACES_PER_FRAME}")

target_last_seen = {}

def mark_target_seen(stable_face_id, raw_track_id, current_frame_idx):
    if stable_face_id is not None:
        target_face_ids.add(stable_face_id)
        target_last_seen[f"face_{stable_face_id}"] = current_frame_idx

    target_track_ids.add(raw_track_id)
    target_last_seen[f"track_{raw_track_id}"] = current_frame_idx


def is_known_target_face(stable_face_id):
    return stable_face_id is not None and stable_face_id in target_face_ids


def is_recent_target_track(raw_track_id, current_frame_idx):
    last_seen = target_last_seen.get(f"track_{raw_track_id}", -999999)

    return (
        raw_track_id in target_track_ids
        and current_frame_idx - last_seen <= TARGET_HOLD_FRAMES
    )

swap_patch_cache = {}

def cleanup_target_last_seen(current_frame_idx):
    stale_keys = [
        key
        for key, last_seen in target_last_seen.items()
        if current_frame_idx - last_seen > TARGET_HOLD_FRAMES
    ]

    for key in stale_keys:
        target_last_seen.pop(key, None)

        if key.startswith("track_"):
            try:
                stale_track_id = int(key.replace("track_", ""))
                target_track_ids.discard(stale_track_id)
            except ValueError:
                pass

    if stale_keys and current_frame_idx % LOG_EVERY_N_FRAMES == 0:
        logging.info(
            f"Frame={current_frame_idx} "
            f"CleanedTargetLastSeen={len(stale_keys)} "
            f"RemainingTargetLastSeen={len(target_last_seen)}"
        )

def get_swap_key(track_ctx):
    stable_face_id = track_ctx.get("stable_face_id")

    if stable_face_id is not None:
        return f"face_{stable_face_id}"

    return f"track_{track_ctx['raw_track_id']}"

def compute_bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 1e-6:
        return 0.0

    return inter / union

def overlaps_with_target(ctx, track_contexts, iou_threshold=0.15):
    if not ctx["is_background"]:
        return False

    bg_bbox = ctx["smoothed_bbox"]

    for other in track_contexts:
        if not other["is_target_final"]:
            continue

        target_bbox = other["smoothed_bbox"]

        if compute_bbox_iou(bg_bbox, target_bbox) >= iou_threshold:
            return True

    return False

def is_cached_swap_compatible(cached_bbox, current_bbox):
    iou = compute_bbox_iou(cached_bbox, current_bbox)

    if iou < 0.65:
        return False

    cw = max(1, cached_bbox[2] - cached_bbox[0])
    ch = max(1, cached_bbox[3] - cached_bbox[1])
    nw = max(1, current_bbox[2] - current_bbox[0])
    nh = max(1, current_bbox[3] - current_bbox[1])

    width_ratio = nw / cw
    height_ratio = nh / ch

    return 0.80 <= width_ratio <= 1.25 and 0.80 <= height_ratio <= 1.25

def paste_cached_swap(render_frame, track_ctx, current_frame_idx):
    swap_key = get_swap_key(track_ctx)
    cached = swap_patch_cache.get(swap_key)

    if cached is None:
        return render_frame, False

    if current_frame_idx - cached["frame_idx"] > SWAP_HOLD_FRAMES:
        return render_frame, False

    clipped = clip_bbox(render_frame, track_ctx["smoothed_bbox"])

    if clipped is None:
        return render_frame, False

    x1, y1, x2, y2 = clipped
    target_w = x2 - x1
    target_h = y2 - y1

    if target_w <= 0 or target_h <= 0:
        return render_frame, False

    current_bbox = [x1, y1, x2, y2]
    cached_bbox = cached.get("bbox")

    if cached_bbox is None or not is_cached_swap_compatible(cached_bbox, current_bbox):
        return render_frame, False

    cached_patch = cached["patch"]

    if cached_patch is None or cached_patch.size == 0:
        return render_frame, False

    render_frame = blend_face(render_frame, cached_patch, [x1, y1, x2, y2])

    return render_frame, True

def load_target_embeddings():
    target_paths = get_target_image_paths(TARGET_DIR, TARGET_PATTERN)
    target_embeddings = []

    for target_path in target_paths:
        try:
            emb = get_target_embedding(str(target_path))
            target_embeddings.append(emb)
            logging.info(f"Loaded target embedding: {target_path}")
        except RuntimeError as e:
            logging.warning(f"Skipped target image: {target_path} | {e}")

    if len(target_embeddings) == 0:
        raise RuntimeError("No valid target embeddings loaded")

    logging.info(f"Total target embeddings loaded: {len(target_embeddings)}")
    return target_embeddings

def prepare_track(
    original_frame,
    track,
    target_embeddings,
    current_frame_idx,
):
    x1, y1, x2, y2, raw_track_id = map(int, track[:5])

    emb, track_kps, embedding_refreshed = get_track_embedding(
        original_frame,
        [x1, y1, x2, y2],
        raw_track_id,
        current_frame_idx
    )

    stable_face_id = assign_stable_face_id(
        raw_track_id,
        emb,
        current_frame_idx,
        [x1, y1, x2, y2]
    )
    is_target = False
    target_sim = -1.0

    if emb is not None:
        is_target, target_sim = check_target_match(emb, target_embeddings)

        if is_target:
            mark_target_seen(stable_face_id, raw_track_id, current_frame_idx)

    if is_known_target_face(stable_face_id):
        mark_target_seen(stable_face_id, raw_track_id, current_frame_idx)

    elif is_recent_target_track(raw_track_id, current_frame_idx):
        mark_target_seen(stable_face_id, raw_track_id, current_frame_idx)

    if stable_face_id is not None:
        sx1, sy1, sx2, sy2 = smooth_bbox(stable_face_id, [x1, y1, x2, y2])
    else:
        sx1, sy1, sx2, sy2 = x1, y1, x2, y2

    raw_bbox = [x1, y1, x2, y2]
    smoothed_bbox = [sx1, sy1, sx2, sy2]
    is_target_final = (
        is_known_target_face(stable_face_id)
        or is_recent_target_track(raw_track_id, current_frame_idx)
    )

    is_background = not is_target_final

    raw_crop = crop_with_padding(original_frame, raw_bbox)
    quality, fallback_reasons = assess_face_quality(original_frame, raw_bbox, emb, raw_crop)

    return {
        "raw_track_id": raw_track_id,
        "stable_face_id": stable_face_id,
        "raw_bbox": raw_bbox,
        "smoothed_bbox": smoothed_bbox,
        "emb": emb,
        "track_kps": track_kps,
        "embedding_refreshed": embedding_refreshed,
        "is_target": is_target,
        "target_sim": target_sim,
        "is_target_final": is_target_final,
        "is_background": is_background,
        "quality": quality,
        "fallback_reasons": fallback_reasons,
    }

def make_face_metadata(
    current_frame_idx,
    raw_track_id,
    stable_face_id,
    raw_bbox,
    smoothed_bbox,
    is_target_final,
    is_background,
    target_sim,
    emb,
    quality,
    fallback_reasons,
    crop_path,
):
    return make_face_data(
        frame_idx=current_frame_idx,
        raw_track_id=raw_track_id,
        stable_face_id=stable_face_id,
        bbox=raw_bbox,
        smoothed_bbox=smoothed_bbox,
        is_target=is_target_final,
        is_background=is_background,
        target_sim=target_sim,
        embedding_ok=emb is not None,
        quality=quality,
        fallback_reasons=fallback_reasons,
        crop_path=crop_path,
    )

def finalize_track(
    render_frame,
    track_ctx,
    current_frame_idx,
    crop_executor,
    crop_write_futures,
    swap_success=False,
):
    raw_track_id = track_ctx["raw_track_id"]
    stable_face_id = track_ctx["stable_face_id"]
    raw_bbox = track_ctx["raw_bbox"]
    smoothed_bbox = track_ctx["smoothed_bbox"]
    emb = track_ctx["emb"]
    embedding_refreshed = track_ctx["embedding_refreshed"]
    is_target = track_ctx["is_target"]
    target_sim = track_ctx["target_sim"]
    is_target_final = track_ctx["is_target_final"]
    is_background = track_ctx["is_background"]
    quality = track_ctx["quality"]
    fallback_reasons = track_ctx["fallback_reasons"]
    x1, y1, x2, y2 = raw_bbox
    sx1, sy1, sx2, sy2 = smoothed_bbox
    crop_path = None

    if is_background:
        if quality == "GOOD":
            # if smooth_crop is not None:
            #     crop_path = save_background_crop(
            #         smooth_crop,
            #         stable_face_id,
            #         raw_track_id,
            #         current_frame_idx,
            #         crop_executor,
            #         crop_write_futures
            #     )
            # else:
            #     logging.warning(
            #         f"Frame={current_frame_idx} "
            #         f"TrackID={raw_track_id} "
            #         "Quality=GOOD but smooth_crop is None"
            #     )

            if not swap_success:
                render_frame = apply_fallback_blur(render_frame, smoothed_bbox)
        else:
            if not swap_success:
                render_frame = apply_fallback_blur(render_frame, smoothed_bbox)

    face_data = make_face_metadata(
        current_frame_idx=current_frame_idx,
        raw_track_id=raw_track_id,
        stable_face_id=stable_face_id,
        raw_bbox=raw_bbox,
        smoothed_bbox=smoothed_bbox,
        is_target_final=is_target_final,
        is_background=is_background,
        target_sim=target_sim,
        emb=emb,
        quality=quality,
        fallback_reasons=fallback_reasons,
        crop_path=crop_path,
    )

    if current_frame_idx % LOG_EVERY_N_FRAMES == 0:
        target_track_flag = raw_track_id in target_track_ids
        target_face_flag = stable_face_id in target_face_ids if stable_face_id is not None else False
        logging.info(
            f"Frame={current_frame_idx} "
            f"TrackID={raw_track_id} "
            f"FaceID={stable_face_id} "
            f"BBox=({x1},{y1},{x2},{y2}) "
            f"SmoothedBBox=({sx1},{sy1},{sx2},{sy2}) "
            f"Embedding={'OK' if emb is not None else 'None'} "
            f"EmbeddingRefreshed={embedding_refreshed} "
            f"TargetSim={target_sim:.4f} "
            f"TargetDirect={is_target} "
            f"TargetFinal={is_target_final} "
            f"TargetTrack={target_track_flag} "
            f"TargetFace={target_face_flag} "
            f"Background={is_background} "
            f"SwapSuccess={swap_success} "
            f"Quality={quality} "
            f"FallbackReasons={fallback_reasons} "
            f"CropPath={crop_path}"
        )

    if stable_face_id is not None:
        if is_target_final:
            label = f"TARGET ID {stable_face_id}"
            color = (0, 0, 255)
        else:
            if swap_success:
                label = f"BG ID {stable_face_id} SWAP"
                color = (255, 0, 255)
            elif quality == "GOOD":
                label = f"BG ID {stable_face_id} CROP"
                color = (0, 255, 0)
            else:
                label = f"BG ID {stable_face_id} BLUR"
                color = (0, 165, 255)
    else:
        if is_target_final:
            label = f"TARGET Track {raw_track_id}"
            color = (0, 0, 255)
        else:
            if swap_success:
                label = f"BG Track {raw_track_id} SWAP"
                color = (255, 0, 255)
            elif quality == "GOOD":
                label = f"BG Track {raw_track_id} CROP"
                color = (0, 255, 0)
            else:
                label = f"BG Track {raw_track_id} BLUR"
                color = (0, 165, 255)

    cv2.rectangle(render_frame, (sx1, sy1), (sx2, sy2), color, 2)
    cv2.putText(render_frame, label, (sx1, max(20, sy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return face_data, render_frame

def main():
    setup_dirs_and_logging()

    detector = init_detector()
    tracker = init_tracker(device)
    target_embeddings = load_target_embeddings()

    swapper = None

    if ENABLE_FACE_SWAP:
        swapper = FaceSwapper(TARGET_IMAGE_PATH, device=device)
        logging.info(f"Face swap enabled: {TARGET_IMAGE_PATH}")
    else:
        logging.info("Face swap disabled")

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))

    if not out.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {OUTPUT_PATH}")

    current_frame_idx = 0
    all_face_metadata = []
    all_tracking_metadata = []
    crop_write_futures = []

    with ThreadPoolExecutor(max_workers=CROP_WRITER_WORKERS) as crop_executor:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            original_frame = frame
            render_frame = frame.copy()

            current_frame_idx += 1

            if current_frame_idx % 30 == 0:
                cleanup_old_face_ids(current_frame_idx)
                cleanup_target_last_seen(current_frame_idx)

            frame_started = perf_counter()

            detect_started = perf_counter()
            dets = detect_faces_multiscale(
                original_frame,
                detector,
                device,
                current_frame_idx
            )
            detect_elapsed = perf_counter() - detect_started

            if current_frame_idx % LOG_EVERY_N_FRAMES == 0:
                logging.info(
                    f"Frame={current_frame_idx} Detections={len(dets)}"
                )

            track_started = perf_counter()
            tracks = tracker.update(dets, original_frame)
            track_elapsed = perf_counter() - track_started
            
            for track in tracks:
                x1, y1, x2, y2 = map(int, track[:4])
                raw_track_id = int(track[4])

                all_tracking_metadata.append({
                    "frame_idx": int(current_frame_idx),
                    "raw_track_id": raw_track_id,
                    "bbox": [x1, y1, x2, y2]
                })

            if current_frame_idx % LOG_EVERY_N_FRAMES == 0:
                logging.info(
                    f"Frame={current_frame_idx} Tracks={len(tracks)}"
                )

            prepare_tracks_started = perf_counter()
            track_contexts = [
                prepare_track(
                    original_frame,
                    track,
                    target_embeddings,
                    current_frame_idx,
                )
                for track in tracks
            ]
            prepare_tracks_elapsed = perf_counter() - prepare_tracks_started

            swap_indices = [
                idx
                for idx, ctx in enumerate(track_contexts)
                if (
                    ctx["is_background"]
                    and ctx["quality"] == "GOOD"
                    and not overlaps_with_target(ctx, track_contexts)
                )
            ]

            swap_indices = swap_indices[:MAX_SWAP_FACES_PER_FRAME]
            swap_success_by_index = {}
            swap_elapsed = 0.0
            swap_timings = {
                "prepare_sec": 0.0,
                "inference_sec": 0.0,
                "paste_sec": 0.0,
                "total_sec": 0.0,
            }

            if swapper is not None and swap_indices:
                swap_bboxes = [
                    track_contexts[idx]["smoothed_bbox"]
                    for idx in swap_indices
                ]

                track_kps_list = [
                    track_contexts[idx].get("track_kps")
                    for idx in swap_indices
                ]

                swap_started = perf_counter()

                render_frame, swap_success_flags, _, swap_timings = swapper.swap_many_into_frame(
                    render_frame,
                    swap_bboxes,
                    landmarks_list=track_kps_list,
                    target_kps_list=track_kps_list,
                    batch_size=FACE_SWAP_BATCH_SIZE,
                )
                swap_elapsed = perf_counter() - swap_started
                swap_success_by_index = dict(zip(swap_indices, swap_success_flags))

                for idx, success in swap_success_by_index.items():
                    if not success:
                        continue

                    track_ctx = track_contexts[idx]
                    swap_key = get_swap_key(track_ctx)

                    clipped = clip_bbox(render_frame, track_ctx["smoothed_bbox"])

                    if clipped is None:
                        continue

                    x1, y1, x2, y2 = clipped
                    patch = render_frame[y1:y2, x1:x2].copy()

                    if patch.size == 0:
                        continue

                    swap_patch_cache[swap_key] = {
                        "patch": patch,
                        "bbox": [x1, y1, x2, y2],
                        "frame_idx": current_frame_idx,
                    }

            if current_frame_idx % LOG_EVERY_N_FRAMES == 0:
                logging.info(
                    f"Frame={current_frame_idx} "
                    f"SwapCandidates={len(swap_indices)} "
                    f"SwapSuccessCount={sum(1 for v in swap_success_by_index.values() if v)} "
                    f"SwapBatchSize={FACE_SWAP_BATCH_SIZE} "
                    f"SwapElapsedSec={swap_elapsed:.4f} "
                    f"SwapPrepareSec={swap_timings['prepare_sec']:.4f} "
                    f"SwapInferenceSec={swap_timings['inference_sec']:.4f} "
                    f"SwapPasteSec={swap_timings['paste_sec']:.4f}"
                )

            finalize_started = perf_counter()
            for idx, track_ctx in enumerate(track_contexts):
                swap_success = swap_success_by_index.get(idx, False)

                if (
                    not swap_success
                    and track_ctx["is_background"]
                    and not overlaps_with_target(track_ctx, track_contexts)
                ):
                    render_frame, cache_success = paste_cached_swap(
                        render_frame,
                        track_ctx,
                        current_frame_idx,
                    )

                    if cache_success:
                        swap_success = True

                face_data, render_frame = finalize_track(
                    render_frame,
                    track_ctx,
                    current_frame_idx,
                    crop_executor,
                    crop_write_futures,
                    swap_success=swap_success,
                )
                all_face_metadata.append(face_data)
            finalize_elapsed = perf_counter() - finalize_started

            write_started = perf_counter()
            out.write(render_frame)
            write_elapsed = perf_counter() - write_started

            if current_frame_idx % LOG_EVERY_N_FRAMES == 0:
                logging.info(
                    f"Frame={current_frame_idx} "
                    f"DetectSec={detect_elapsed:.4f} "
                    f"TrackSec={track_elapsed:.4f} "
                    f"PrepareTracksSec={prepare_tracks_elapsed:.4f} "
                    f"FinalizeSec={finalize_elapsed:.4f} "
                    f"WriteSec={write_elapsed:.4f} "
                    f"FrameTotalSec={perf_counter() - frame_started:.4f}"
                )

            if current_frame_idx % 100 == 0:
                failed = []
                still_running = []

                for save_path, future in crop_write_futures:
                    if future.done():
                        try:
                            if not future.result():
                                failed.append(str(save_path))
                        except Exception as e:
                            failed.append(f"{save_path}: {e}")
                    else:
                        still_running.append((save_path, future))

                if failed:
                    logging.warning(f"Failed crop writes (frame {current_frame_idx}): {failed}")

                crop_write_futures = still_running

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 비동기 crop 저장 결과 확인
    failed_crop_writes = []

    for save_path, future in crop_write_futures:
        try:
            if not future.result():
                failed_crop_writes.append(str(save_path))
        except Exception as e:
            failed_crop_writes.append(f"{save_path}: {e}")

    if failed_crop_writes:
        logging.warning(f"Failed crop writes: {failed_crop_writes[:20]}")

    tracking_metadata_path = Path(TRACKING_METADATA_PATH)

    save_metadata(tracking_metadata_path, all_tracking_metadata)
    save_metadata(METADATA_PATH, all_face_metadata)

    logging.info("===== Experiment Finished =====")
    logging.info(f"Saved result to: {OUTPUT_PATH}")
    logging.info(f"Saved log to: {LOG_PATH}")
    logging.info(f"Saved face metadata to: {METADATA_PATH}")
    logging.info(f"Saved tracking metadata to: {tracking_metadata_path}")
    #logging.info(f"Saved background crops to: {CROP_ROOT}")

    print(f"Saved result to: {OUTPUT_PATH}")
    print(f"Saved log to: {LOG_PATH}")
    print(f"Saved face metadata to: {METADATA_PATH}")
    print(f"Saved tracking metadata to: {tracking_metadata_path}")
    #print(f"Saved background crops to: {CROP_ROOT}")

if __name__ == "__main__":
    main()
import cv2
import logging
from time import perf_counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from face_swapper import FaceSwapper

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
    SIM_THRESHOLD,
    TARGET_THRESHOLD,
    SMOOTH_ALPHA,
    TARGET_HOLD_FRAMES,
    EMBEDDING_REFRESH_INTERVAL,
    LOG_EVERY_N_FRAMES,
    CROP_WRITER_WORKERS,
    ENABLE_FACE_SWAP,
    FACE_SWAP_BATCH_SIZE,
    device,
)

from detector_tracker import init_detector, init_tracker, detect_faces_multiscale
from face_utils import crop_with_padding
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
from crop_manager import assess_face_quality, apply_fallback_blur, save_background_crop
from metadata_manager import make_face_data, save_metadata


def setup_dirs_and_logging():
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(CROP_ROOT).mkdir(parents=True, exist_ok=True)
    Path(METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)

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
    logging.info(f"SIM_THRESHOLD={SIM_THRESHOLD}")
    logging.info(f"TARGET_THRESHOLD={TARGET_THRESHOLD}")
    logging.info(f"SMOOTH_ALPHA={SMOOTH_ALPHA}")
    logging.info(f"TARGET_HOLD_FRAMES={TARGET_HOLD_FRAMES}")
    logging.info(f"EMBEDDING_REFRESH_INTERVAL={EMBEDDING_REFRESH_INTERVAL}")
    logging.info(f"LOG_EVERY_N_FRAMES={LOG_EVERY_N_FRAMES}")
    logging.info(f"FACE_SWAP_BATCH_SIZE={FACE_SWAP_BATCH_SIZE}")


target_last_seen = {}


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

    emb, target_kps, embedding_refreshed = get_track_embedding(
        original_frame,
        [x1, y1, x2, y2],
        raw_track_id,
        current_frame_idx
    )

    stable_face_id = assign_stable_face_id(raw_track_id, emb, current_frame_idx)
    is_target = False
    target_sim = -1.0

    if emb is not None:
        is_target, target_sim = check_target_match(emb, target_embeddings)
        if is_target:
            target_track_ids.add(raw_track_id)
            target_last_seen[raw_track_id] = current_frame_idx

            if stable_face_id is not None:
                target_face_ids.add(stable_face_id)
                target_last_seen[stable_face_id] = current_frame_idx

    if stable_face_id in target_face_ids:
        target_track_ids.add(raw_track_id)

    if raw_track_id in target_track_ids and stable_face_id is not None:
        target_face_ids.add(stable_face_id)

    if stable_face_id is not None:
        sx1, sy1, sx2, sy2 = smooth_bbox(stable_face_id, [x1, y1, x2, y2])
    else:
        sx1, sy1, sx2, sy2 = x1, y1, x2, y2

    raw_bbox = [x1, y1, x2, y2]
    smoothed_bbox = [sx1, sy1, sx2, sy2]
    is_target_final = False

    if stable_face_id is not None and stable_face_id in target_face_ids:
        last_seen = target_last_seen.get(stable_face_id, -999999)

        if current_frame_idx - last_seen <= TARGET_HOLD_FRAMES:
            is_target_final = True

    if not is_target_final and raw_track_id in target_track_ids:
        last_seen = target_last_seen.get(raw_track_id, -999999)

        if current_frame_idx - last_seen <= TARGET_HOLD_FRAMES:
            is_target_final = True

    is_background = not is_target_final

    raw_crop = crop_with_padding(original_frame, raw_bbox)
    smooth_crop = crop_with_padding(original_frame, smoothed_bbox)
    quality, fallback_reasons = assess_face_quality(original_frame, raw_bbox, emb, raw_crop)

    return {
        "raw_track_id": raw_track_id,
        "stable_face_id": stable_face_id,
        "raw_bbox": raw_bbox,
        "smoothed_bbox": smoothed_bbox,
        "emb": emb,
        "target_kps": target_kps,
        "embedding_refreshed": embedding_refreshed,
        "is_target": is_target,
        "target_sim": target_sim,
        "is_target_final": is_target_final,
        "is_background": is_background,
        "quality": quality,
        "fallback_reasons": fallback_reasons,
        "smooth_crop": smooth_crop,
    }


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
    smooth_crop = track_ctx["smooth_crop"]
    x1, y1, x2, y2 = raw_bbox
    sx1, sy1, sx2, sy2 = smoothed_bbox
    crop_path = None

    if is_background:
        if quality == "GOOD":
            crop_path = save_background_crop(
                smooth_crop,
                stable_face_id,
                raw_track_id,
                current_frame_idx,
                crop_executor,
                crop_write_futures
            )
            if not swap_success:
                render_frame = apply_fallback_blur(render_frame, smoothed_bbox)
        else:
            render_frame = apply_fallback_blur(render_frame, smoothed_bbox)

    face_data = make_face_data(
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
        crop_path=crop_path
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
        if raw_track_id in target_track_ids:
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

    current_frame_idx = 0
    all_face_metadata = []
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
                if ctx["is_background"] and ctx["quality"] == "GOOD"
            ]
            swap_success_by_index = {}
            swap_elapsed = 0.0
            swap_timings = {
                "prepare_sec": 0.0,
                "inference_sec": 0.0,
                "paste_sec": 0.0,
                "total_sec": 0.0,
            }

            if swapper is not None and swap_indices:
                swap_bboxes = [track_contexts[idx]["smoothed_bbox"] for idx in swap_indices]
                swap_started = perf_counter()
                swap_landmarks = [None] * len(swap_bboxes)
                swap_target_kps = [track_contexts[idx].get("target_kps") for idx in swap_indices]

                render_frame, swap_success_flags, _, swap_timings = swapper.swap_many_into_frame(
                    render_frame,
                    swap_bboxes,
                    landmarks_list=swap_landmarks,
                    target_kps_list=swap_target_kps,
                    batch_size=FACE_SWAP_BATCH_SIZE,
                )
                swap_elapsed = perf_counter() - swap_started
                swap_success_by_index = dict(zip(swap_indices, swap_success_flags))

            if current_frame_idx % LOG_EVERY_N_FRAMES == 0:
                logging.info(
                    f"Frame={current_frame_idx} "
                    f"SwapCandidates={len(swap_indices)} "
                    f"SwapSuccessCount={sum(swap_success_by_index.values())} "
                    f"SwapBatchSize={FACE_SWAP_BATCH_SIZE} "
                    f"SwapElapsedSec={swap_elapsed:.4f} "
                    f"SwapPrepareSec={swap_timings['prepare_sec']:.4f} "
                    f"SwapInferenceSec={swap_timings['inference_sec']:.4f} "
                    f"SwapPasteSec={swap_timings['paste_sec']:.4f}"
                )

            finalize_started = perf_counter()
            for idx, track_ctx in enumerate(track_contexts):
                face_data, render_frame = finalize_track(
                    render_frame,
                    track_ctx,
                    current_frame_idx,
                    crop_executor,
                    crop_write_futures,
                    swap_success=swap_success_by_index.get(idx, False),
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

    save_metadata(METADATA_PATH, all_face_metadata)

    logging.info("===== Experiment Finished =====")
    logging.info(f"Saved result to: {OUTPUT_PATH}")
    logging.info(f"Saved log to: {LOG_PATH}")
    logging.info(f"Saved metadata to: {METADATA_PATH}")
    logging.info(f"Saved background crops to: {CROP_ROOT}")

    print(f"Saved result to: {OUTPUT_PATH}")
    print(f"Saved log to: {LOG_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")
    print(f"Saved background crops to: {CROP_ROOT}")


if __name__ == "__main__":
    main()
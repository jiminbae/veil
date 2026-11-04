import cv2
import json
import numpy as np
import torch
import logging
import queue
import threading
from time import perf_counter
from pathlib import Path

from face_inswapper import FaceSwapper

from config import (
    VIDEO_PATH,
    FACE_MODEL_PATH,
    TARGET_DIR,
    TARGET_PATTERN,
    TARGET_IMAGE_PATH,
    OUTPUT_PATH,
    LOG_PATH,
    METADATA_PATH,
    TRACKING_METADATA_PATH,
    SIM_THRESHOLD,
    SMOOTH_ALPHA,
    TARGET_HOLD_FRAMES,
    SWAP_HOLD_FRAMES,
    MAX_SWAP_FACES_PER_FRAME,
    EMBEDDING_REFRESH_INTERVAL,
    LOG_EVERY_N_FRAMES,
    ENABLE_FACE_SWAP,
    ENABLE_FALLBACK_BLUR,
    IDENTITY_LOCK_ENABLED,
    ENABLE_SWAP_CACHE,
    DRAW_ANNOTATIONS,
    FRAME_METADATA_PATH,
    RUNTIME_METADATA_PATH,
    FACE_SWAP_BATCH_SIZE,
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
    FACE_PROVIDERS,
    face_app,
)
from crop_manager import assess_face_quality, apply_fallback_blur
from metadata_manager import make_face_data, save_metadata
from render_safety import restore_protected_regions, changed_pixels


target_last_seen = {}
target_track_faces = {}
swap_patch_cache = {}


class AsyncVideoWriter:
    def __init__(self, writer, max_queue_size=16):
        self._writer = writer
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            frame = self._queue.get()
            try:
                if frame is None:
                    return

                self._writer.write(frame)
            except Exception as exc:
                self._error = exc
            finally:
                self._queue.task_done()

    def write(self, frame):
        if self._error is not None:
            raise RuntimeError("Async video writer failed") from self._error

        self._queue.put(frame)

    def release(self):
        self._queue.put(None)
        self._queue.join()
        self._thread.join()
        self._writer.release()

        if self._error is not None:
            raise RuntimeError("Async video writer failed") from self._error


def setup_dirs_and_logging():
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(TRACKING_METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        force=True,
    )

    logging.info("===== InSwapper Pipeline Started =====")
    logging.info(f"VIDEO_PATH={VIDEO_PATH}")
    logging.info(f"FACE_MODEL_PATH={FACE_MODEL_PATH}")
    logging.info(f"TARGET_DIR={TARGET_DIR}")
    logging.info(f"TARGET_PATTERN={TARGET_PATTERN}")
    logging.info(f"OUTPUT_PATH={OUTPUT_PATH}")
    logging.info(f"METADATA_PATH={METADATA_PATH}")
    logging.info(f"TRACKING_METADATA_PATH={TRACKING_METADATA_PATH}")
    logging.info(f"SIM_THRESHOLD={SIM_THRESHOLD}")
    logging.info(f"SMOOTH_ALPHA={SMOOTH_ALPHA}")
    logging.info(f"TARGET_HOLD_FRAMES={TARGET_HOLD_FRAMES}")
    logging.info(f"SWAP_HOLD_FRAMES={SWAP_HOLD_FRAMES}")
    logging.info(f"EMBEDDING_REFRESH_INTERVAL={EMBEDDING_REFRESH_INTERVAL}")
    logging.info(f"LOG_EVERY_N_FRAMES={LOG_EVERY_N_FRAMES}")
    logging.info(f"FACE_SWAP_BATCH_SIZE={FACE_SWAP_BATCH_SIZE}")
    logging.info(f"ENABLE_FALLBACK_BLUR={ENABLE_FALLBACK_BLUR}")
    logging.info(f"ONNXRuntime providers={FACE_PROVIDERS}")
    logging.info("Swap engine=inswapper_128.onnx")


def synchronize_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mark_target_seen(stable_face_id, raw_track_id, current_frame_idx):
    if stable_face_id is not None:
        target_face_ids.add(stable_face_id)
        target_last_seen[f"face_{stable_face_id}"] = current_frame_idx

    target_track_ids.add(raw_track_id)
    target_track_faces[raw_track_id] = stable_face_id
    target_last_seen[f"track_{raw_track_id}"] = current_frame_idx


def is_known_target_face(stable_face_id, current_frame_idx):
    return (stable_face_id is not None and stable_face_id in target_face_ids
            and current_frame_idx - target_last_seen.get(f"face_{stable_face_id}", -999999)
            <= TARGET_HOLD_FRAMES)


def is_recent_target_track(raw_track_id, current_frame_idx, stable_face_id=None):
    last_seen = target_last_seen.get(f"track_{raw_track_id}", -999999)
    return (
        raw_track_id in target_track_ids
        and target_track_faces.get(raw_track_id) == stable_face_id
        and current_frame_idx - last_seen <= TARGET_HOLD_FRAMES
    )


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
                target_track_faces.pop(stale_track_id, None)
            except ValueError:
                pass
        elif key.startswith("face_"):
            target_face_ids.discard(int(key[5:]))


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

        if compute_bbox_iou(bg_bbox, other["smoothed_bbox"]) >= iou_threshold:
            return True

    return False


def is_cached_swap_compatible(cached_bbox, current_bbox):
    if compute_bbox_iou(cached_bbox, current_bbox) < 0.65:
        return False

    cw = max(1, cached_bbox[2] - cached_bbox[0])
    ch = max(1, cached_bbox[3] - cached_bbox[1])
    nw = max(1, current_bbox[2] - current_bbox[0])
    nh = max(1, current_bbox[3] - current_bbox[1])

    return (0.80 <= nw / cw <= 1.25) and (0.80 <= nh / ch <= 1.25)


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
    if x2 <= x1 or y2 <= y1:
        return render_frame, False

    cached_bbox = cached.get("bbox")
    if cached_bbox is None:
        return render_frame, False

    if not is_cached_swap_compatible(cached_bbox, [x1, y1, x2, y2]):
        return render_frame, False

    cached_patch = cached.get("patch")
    if cached_patch is None or cached_patch.size == 0:
        return render_frame, False

    patch_resized = cv2.resize(cached_patch, (x2 - x1, y2 - y1))
    render_frame[y1:y2, x1:x2] = patch_resized

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

    if not target_embeddings:
        raise RuntimeError("No valid target embeddings loaded")

    logging.info(f"Total target embeddings loaded: {len(target_embeddings)}")
    return target_embeddings


def prepare_track(original_frame, track, target_embeddings, current_frame_idx):
    x1, y1, x2, y2, raw_track_id = map(int, track[:5])
    raw_bbox = [x1, y1, x2, y2]

    emb, track_kps, embedding_refreshed = get_track_embedding(
        original_frame,
        raw_bbox,
        raw_track_id,
        current_frame_idx,
    )

    stable_face_id = assign_stable_face_id(
        raw_track_id,
        emb,
        current_frame_idx,
        raw_bbox,
    )

    is_target = False
    target_sim = -1.0

    if emb is not None:
        is_target, target_sim = check_target_match(emb, target_embeddings)

        if is_target and embedding_refreshed:
            mark_target_seen(stable_face_id, raw_track_id, current_frame_idx)

    if stable_face_id is not None:
        sx1, sy1, sx2, sy2 = smooth_bbox(stable_face_id, raw_bbox)
    else:
        sx1, sy1, sx2, sy2 = x1, y1, x2, y2

    smoothed_bbox = [sx1, sy1, sx2, sy2]

    is_target_final = (
        is_target or (IDENTITY_LOCK_ENABLED and (
            is_known_target_face(stable_face_id, current_frame_idx)
            or is_recent_target_track(raw_track_id, current_frame_idx, stable_face_id)
        ))
    )
    is_background = not is_target_final

    raw_crop = crop_with_padding(original_frame, raw_bbox)
    quality, fallback_reasons = assess_face_quality(
        original_frame,
        raw_bbox,
        emb,
        raw_crop,
    )

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


def finalize_track(render_frame, track_ctx, current_frame_idx, swap_success=False):
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

    blur_applied = False
    if is_background and not swap_success and ENABLE_FALLBACK_BLUR:
        render_frame, blur_applied = apply_fallback_blur(
            render_frame, smoothed_bbox, return_status=True)
        if not blur_applied:
            fallback_reasons = [*fallback_reasons, "blur_no_pixel_change"]

    if is_target_final:
        final_action = "PRESERVE"
    elif swap_success:
        final_action = "SWAP"
    elif blur_applied:
        final_action = "BLUR"
    elif is_background:
        final_action = "UNPROCESSED"
    else:
        final_action = "UNKNOWN"

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
        crop_path=None,
        is_target_direct=is_target,
        is_target_final=is_target_final,
        final_action=final_action,
        swap_success=swap_success,
        blur_applied=blur_applied,
    )

    if current_frame_idx % LOG_EVERY_N_FRAMES == 0:
        logging.info(
            f"Frame={current_frame_idx} "
            f"TrackID={raw_track_id} FaceID={stable_face_id} "
            f"BBox=({x1},{y1},{x2},{y2}) "
            f"SmoothedBBox=({sx1},{sy1},{sx2},{sy2}) "
            f"Emb={'OK' if emb is not None else 'None'} "
            f"EmbRefreshed={embedding_refreshed} "
            f"TargetSim={target_sim:.4f} "
            f"TargetDirect={is_target} TargetFinal={is_target_final} "
            f"Background={is_background} SwapSuccess={swap_success} "
            f"Quality={quality} FallbackReasons={fallback_reasons}"
        )

    if not DRAW_ANNOTATIONS:
        return face_data, render_frame

    if stable_face_id is not None:
        if final_action == "PRESERVE":
            label, color = f"TARGET ID {stable_face_id}", (0, 0, 255)
        elif final_action == "SWAP":
            label, color = f"BG ID {stable_face_id} SWAP", (255, 0, 255)
        elif final_action == "BLUR":
            label, color = f"BG ID {stable_face_id} BLUR", (0, 165, 255)
        elif final_action == "UNPROCESSED":
            label, color = f"BG ID {stable_face_id} RAW", (0, 255, 255)
        else:
            label, color = f"BG ID {stable_face_id} UNKNOWN", (0, 255, 0)
    else:
        if final_action == "PRESERVE":
            label, color = f"TARGET Track {raw_track_id}", (0, 0, 255)
        elif final_action == "SWAP":
            label, color = f"BG Track {raw_track_id} SWAP", (255, 0, 255)
        elif final_action == "BLUR":
            label, color = f"BG Track {raw_track_id} BLUR", (0, 165, 255)
        elif final_action == "UNPROCESSED":
            label, color = f"BG Track {raw_track_id} RAW", (0, 255, 255)
        else:
            label, color = f"BG Track {raw_track_id} UNKNOWN", (0, 255, 0)

    cv2.rectangle(render_frame, (sx1, sy1), (sx2, sy2), color, 2)
    cv2.putText(
        render_frame,
        label,
        (sx1, max(20, sy1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    return face_data, render_frame


def main():
    process_started = perf_counter()
    setup_dirs_and_logging()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"Output: {OUTPUT_PATH}")

    detector = init_detector()
    tracker = init_tracker(device)
    target_embeddings = load_target_embeddings()

    swapper = None
    if ENABLE_FACE_SWAP:
        swapper = FaceSwapper(TARGET_IMAGE_PATH, device=device)
        logging.info(f"Face swap enabled: {TARGET_IMAGE_PATH}")
    else:
        logging.info("Face swap disabled")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_sec = perf_counter() - process_started

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    raw_out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not raw_out.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {OUTPUT_PATH}")

    out = AsyncVideoWriter(raw_out)

    current_frame_idx = 0
    all_face_metadata = []
    all_tracking_metadata = []
    frame_metadata = []
    loop_started = perf_counter()

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
            current_frame_idx,
        )
        synchronize_gpu()
        detect_elapsed = perf_counter() - detect_started

        num_dets = 0 if dets is None else len(dets)

        if current_frame_idx % LOG_EVERY_N_FRAMES == 0:
            logging.info(f"Frame={current_frame_idx} Detections={num_dets}")

        track_started = perf_counter()

        # Empty frames must age the tracker as well.
        if dets is None:
            dets = np.empty((0, 6), dtype=np.float32)
        tracks = tracker.update(dets, original_frame)
        synchronize_gpu()

        track_elapsed = perf_counter() - track_started

        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            raw_track_id = int(track[4])
            all_tracking_metadata.append(
                {
                    "frame_idx": int(current_frame_idx),
                    "raw_track_id": raw_track_id,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        if current_frame_idx % LOG_EVERY_N_FRAMES == 0:
            logging.info(f"Frame={current_frame_idx} Tracks={len(tracks)}")

        prepare_tracks_started = perf_counter()
        track_contexts = [
            prepare_track(original_frame, track, target_embeddings, current_frame_idx)
            for track in tracks
        ]
        synchronize_gpu()
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
            synchronize_gpu()
            swap_elapsed = perf_counter() - swap_started
            swap_success_by_index = dict(zip(swap_indices, swap_success_flags))

            for idx, success in swap_success_by_index.items():
                if not ENABLE_SWAP_CACHE:
                    continue
                if not success:
                    continue

                track_ctx = track_contexts[idx]
                swap_key = get_swap_key(track_ctx)
                clipped = clip_bbox(render_frame, track_ctx["smoothed_bbox"])

                if clipped is None:
                    continue

                cx1, cy1, cx2, cy2 = clipped
                patch = render_frame[cy1:cy2, cx1:cx2].copy()

                if patch.size == 0:
                    continue

                swap_patch_cache[swap_key] = {
                    "patch": patch,
                    "bbox": [cx1, cy1, cx2, cy2],
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
                and ENABLE_FACE_SWAP
                and ENABLE_SWAP_CACHE
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
                    track_ctx['swap_cache_used'] = True

            face_data, render_frame = finalize_track(
                render_frame,
                track_ctx,
                current_frame_idx,
                swap_success=swap_success,
            )
            face_data['swap_cache_used'] = bool(track_ctx.get('swap_cache_used', False))
            all_face_metadata.append(face_data)

        render_frame = restore_protected_regions(render_frame, original_frame, track_contexts)
        for ctx, row in zip(track_contexts, all_face_metadata[-len(track_contexts):] if track_contexts else []):
            row['render_changed_pixels'] = changed_pixels(original_frame, render_frame, ctx['smoothed_bbox'])
            if row['final_action'] in {'BLUR', 'SWAP'} and row['render_changed_pixels'] == 0:
                row['final_action'] = 'UNPROCESSED'
                row['blur_applied'] = False
                row['swap_success'] = False
                row['fallback_reasons'].append('no_final_pixel_change_after_target_protection')

        finalize_elapsed = perf_counter() - finalize_started

        write_started = perf_counter()
        out.write(render_frame)
        write_elapsed = perf_counter() - write_started

        frame_total_sec = perf_counter() - frame_started
        frame_metadata.append({
            'frame_idx': current_frame_idx, 'detections': num_dets, 'tracks': len(tracks),
            'detect_sec': detect_elapsed, 'track_sec': track_elapsed,
            'identify_sec': prepare_tracks_elapsed, 'swap_sec': swap_elapsed,
            'finalize_sec': finalize_elapsed, 'writer_enqueue_sec': write_elapsed,
            'frame_wall_sec': frame_total_sec,
        })

        print(
            f"[Frame {current_frame_idx}] "
            f"tracks={len(tracks)} "
            f"swap_try={len(swap_indices)} "
            f"swap_ok={sum(1 for v in swap_success_by_index.values() if v)} "
            f"frame_time={frame_total_sec:.2f}s",
            flush=True,
        )

        if current_frame_idx % LOG_EVERY_N_FRAMES == 0:
            logging.info(
                f"Frame={current_frame_idx} "
                f"DetectSec={detect_elapsed:.4f} "
                f"TrackSec={track_elapsed:.4f} "
                f"PrepareTracksSec={prepare_tracks_elapsed:.4f} "
                f"FinalizeSec={finalize_elapsed:.4f} "
                f"WriteSec={write_elapsed:.4f} "
                f"FrameTotalSec={frame_total_sec:.4f}"
            )

    cap.release()
    out.release()
    # Headless execution must not require OpenCV GUI support.

    save_metadata(TRACKING_METADATA_PATH, all_tracking_metadata)
    save_metadata(METADATA_PATH, all_face_metadata)
    save_metadata(FRAME_METADATA_PATH or str(Path(METADATA_PATH).with_name('frames.json')), frame_metadata)
    runtime = {
        'frames': current_frame_idx, 'width': width, 'height': height, 'fps': fps,
        'model_load_sec': model_load_sec, 'warm_wall_sec': perf_counter() - loop_started,
        'process_wall_sec': perf_counter() - process_started,
        'peak_torch_gpu_memory_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        'clean_output': not DRAW_ANNOTATIONS,
        'recognizer_actual_providers': {k: v.session.get_providers() for k, v in face_app.models.items()},
        'gpu_stage_timing_note': 'host wall times with CUDA synchronization after detection/tracking/identification/swap; writer enqueue excludes final drain',
        'gpu_memory_note': 'PyTorch allocations only; excludes ONNX Runtime; NVML measured externally',
    }
    save_metadata(RUNTIME_METADATA_PATH or str(Path(METADATA_PATH).with_name('runtime.json')), runtime)

    logging.info("===== InSwapper Pipeline Finished =====")
    logging.info(f"Saved result to: {OUTPUT_PATH}")
    logging.info(f"Saved log to: {LOG_PATH}")
    logging.info(f"Saved metadata to: {METADATA_PATH}")
    logging.info(f"Saved tracking metadata to: {TRACKING_METADATA_PATH}")

    print()
    print(f"Saved result to: {OUTPUT_PATH}")
    print(f"Saved log to: {LOG_PATH}")
    print(f"Saved metadata to: {METADATA_PATH}")
    print(f"Saved tracking metadata to: {TRACKING_METADATA_PATH}")


if __name__ == "__main__":
    main()

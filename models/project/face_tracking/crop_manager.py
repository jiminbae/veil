import cv2

from config import (
    SWAP_MIN_FACE_AREA,
    SWAP_MIN_CROP_SIZE,
    SWAP_MAX_ASPECT_RATIO,
    EDGE_MARGIN,
    SWAP_MIN_FACE_SIZE,
    SWAP_MAX_FACE_AREA_RATIO,
    ENABLE_POSE_FALLBACK,
    SIDE_FACE_ASPECT_RATIO_THRESHOLD,
)
from face_utils import clip_bbox


def assess_face_quality(frame, bbox, embedding, crop):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    box_w = x2 - x1
    box_h = y2 - y1

    reasons = []

    if box_w <= 0 or box_h <= 0:
        reasons.append("invalid_bbox")
        return "BAD", reasons

    if embedding is None:
        reasons.append("embedding_failed")

    area = box_w * box_h
    aspect_ratio = max(box_w, box_h) / (min(box_w, box_h) + 1e-6)
    frame_area = w * h

    if ENABLE_POSE_FALLBACK:
        if aspect_ratio > SIDE_FACE_ASPECT_RATIO_THRESHOLD:
            reasons.append("side_face_or_unstable_pose")

    if min(box_w, box_h) < SWAP_MIN_FACE_SIZE:
        reasons.append("small_face_size")

    if area / frame_area > SWAP_MAX_FACE_AREA_RATIO:
        reasons.append("too_large_face")

    if area < SWAP_MIN_FACE_AREA:
        reasons.append("small_face")

    if aspect_ratio > SWAP_MAX_ASPECT_RATIO:
        reasons.append("bad_aspect_ratio")

    if crop is None or crop.size == 0:
        reasons.append("crop_failed")
    else:
        crop_h, crop_w = crop.shape[:2]
        if crop_w < SWAP_MIN_CROP_SIZE or crop_h < SWAP_MIN_CROP_SIZE:
            reasons.append("small_crop")

    if (
        x1 <= EDGE_MARGIN
        or y1 <= EDGE_MARGIN
        or x2 >= w - EDGE_MARGIN
        or y2 >= h - EDGE_MARGIN
    ):
        reasons.append("near_frame_edge")

    if reasons:
        return "BAD", reasons

    return "GOOD", reasons


def apply_fallback_blur(frame, bbox):
    clipped = clip_bbox(frame, bbox)

    if clipped is None:
        return frame

    x1, y1, x2, y2 = clipped
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return frame

    blur = cv2.GaussianBlur(roi, (45, 45), 0)
    frame[y1:y2, x1:x2] = blur

    return frame


def save_background_crop(
    crop,
    stable_face_id,
    raw_track_id,
    current_frame_idx,
    crop_executor,
    crop_write_futures,
):
    if crop is None or crop.size == 0:
        return None

    from pathlib import Path
    from config import CROP_ROOT

    face_id = stable_face_id if stable_face_id is not None else f"track_{raw_track_id}"

    save_dir = Path(CROP_ROOT) / str(face_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"frame_{current_frame_idx:06d}.jpg"

    future = crop_executor.submit(cv2.imwrite, str(save_path), crop)
    crop_write_futures.append((save_path, future))

    return str(save_path)
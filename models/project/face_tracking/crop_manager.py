import cv2
from pathlib import Path

from config import (
    CROP_ROOT,
    LIVEPORTRAIT_MIN_FACE_AREA,
    LIVEPORTRAIT_MIN_CROP_SIZE,
    LIVEPORTRAIT_MAX_ASPECT_RATIO,
    EDGE_MARGIN,
)
from face_utils import clip_bbox


# 품질 판단
def assess_face_quality(frame, bbox, embedding, crop):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    box_w = x2 - x1
    box_h = y2 - y1

    reasons = []

    if box_w <= 0 or box_h <= 0:
        reasons.append("invalid_bbox")
        return "BAD", reasons

    area = box_w * box_h
    aspect_ratio = max(box_w, box_h) / (min(box_w, box_h) + 1e-6)

    # 얼굴 크기 부족
    if area < LIVEPORTRAIT_MIN_FACE_AREA:
        reasons.append("small_face")

    # 비정상 비율 얼굴
    if aspect_ratio > LIVEPORTRAIT_MAX_ASPECT_RATIO:
        reasons.append("bad_aspect_ratio")

    # crop 실패
    if crop is None or crop.size == 0:
        reasons.append("crop_failed")
    else:
        crop_h, crop_w = crop.shape[:2]

        # crop 크기 부족
        if (
            crop_w < LIVEPORTRAIT_MIN_CROP_SIZE
            or crop_h < LIVEPORTRAIT_MIN_CROP_SIZE
        ):
            reasons.append("small_crop")

    # 프레임 가장자리 얼굴
    if (
        x1 <= EDGE_MARGIN
        or y1 <= EDGE_MARGIN
        or x2 >= w - EDGE_MARGIN
        or y2 >= h - EDGE_MARGIN
    ):
        reasons.append("near_frame_edge")

    if len(reasons) > 0:
        return "BAD", reasons

    return "GOOD", reasons


# Fallback blur
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


# 비동기 crop 저장
def save_background_crop(
    crop,
    stable_face_id,
    raw_track_id,
    frame_idx,
    executor,
    crop_write_futures
):
    if crop is None or crop.size == 0:
        return None

    if stable_face_id is not None:
        person_key = f"face_{stable_face_id}"
    else:
        person_key = f"track_{raw_track_id}"

    save_dir = Path(CROP_ROOT) / person_key
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"frame_{frame_idx:06d}.png"

    future = executor.submit(
        cv2.imwrite,
        str(save_path),
        crop.copy()
    )

    crop_write_futures.append((save_path, future))

    return str(save_path)
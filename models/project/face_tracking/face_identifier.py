import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

from config import (
    TARGET_THRESHOLD,
    SIM_THRESHOLD,
    SMOOTH_ALPHA,
    MAX_FACE_AGE,
    EMBEDDING_REFRESH_INTERVAL,
)
from face_utils import l2_normalize, crop_with_padding


# InsightFace 초기화 
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

face_app.prepare(ctx_id=0, det_size=(640, 640))


# 전역 상태
next_face_id = 1
face_gallery = {}       # face_id → embedding
face_last_seen = {}     # face_id → last frame idx
track_to_face = {}      # track_id → face_id
track_last_emb = {}     # track_id → last embedding
bbox_smoother = {}      # face_id → smoothed bbox

target_face_ids = set()
target_track_ids = set()


# ── 임베딩 추출 ──────────────────────────────────────────────────────────────
def get_arcface_embedding(frame, box):
    crop = crop_with_padding(frame, box)

    if crop is None or crop.size == 0:
        return None

    faces = face_app.get(crop)

    if len(faces) == 0:
        return None

    best_face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    if hasattr(best_face, "normed_embedding"):
        emb = best_face.normed_embedding
    else:
        emb = best_face.embedding

    return l2_normalize(emb)


def get_track_embedding(frame, box, track_id, current_frame):
    """
    같은 track_id는 EMBEDDING_REFRESH_INTERVAL 프레임마다만 새로 계산.
    중간 프레임에서는 캐시된 embedding 재사용.
    반환: (embedding, refreshed: bool)
    """
    cached_emb = track_last_emb.get(track_id)

    should_refresh = (
        cached_emb is None
        or current_frame % EMBEDDING_REFRESH_INTERVAL == 0
    )

    if not should_refresh:
        return cached_emb, False

    emb = get_arcface_embedding(frame, box)

    if emb is None:
        return cached_emb, False
    
    track_last_emb[track_id] = emb

    return emb, True


# 타겟 이미지 로드
def get_target_image_paths(target_dir, pattern="target*"):
    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    paths = [
        p for p in sorted(Path(target_dir).glob(pattern))
        if p.suffix.lower() in image_exts
    ]

    if len(paths) == 0:
        raise RuntimeError(
            f"No target images found in {target_dir} with pattern {pattern}"
        )

    return paths


def get_target_embedding(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise RuntimeError(f"Cannot read target image: {image_path}")

    faces = face_app.get(image)

    if len(faces) == 0:
        raise RuntimeError(f"No face detected in target image: {image_path}")

    best_face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    if hasattr(best_face, "normed_embedding"):
        emb = best_face.normed_embedding
    else:
        emb = best_face.embedding

    emb = l2_normalize(emb)

    if emb is None:
        raise RuntimeError("Failed to normalize target embedding")

    return emb


def check_target_match(embedding, target_embeddings):
    if embedding is None or len(target_embeddings) == 0:
        return False, -1.0

    sims = [
        float(np.dot(embedding, target_emb))
        for target_emb in target_embeddings
    ]

    best_sim = max(sims)

    return best_sim >= TARGET_THRESHOLD, best_sim


# 안정 Face ID 관리
def cleanup_old_face_ids(current_frame):
    expired_face_ids = [
        face_id
        for face_id, last_seen in face_last_seen.items()
        if current_frame - last_seen > MAX_FACE_AGE
    ]

    for face_id in expired_face_ids:
        face_gallery.pop(face_id, None)
        face_last_seen.pop(face_id, None)
        bbox_smoother.pop(face_id, None)
        target_face_ids.discard(face_id)

    for track_id, face_id in list(track_to_face.items()):
        if face_id in expired_face_ids:
            track_to_face.pop(track_id, None)
            track_last_emb.pop(track_id, None)
            target_track_ids.discard(track_id)


def assign_stable_face_id(track_id, embedding, current_frame):
    global next_face_id

    if embedding is None:
        embedding = track_last_emb.get(track_id, None)
    else:
        track_last_emb[track_id] = embedding

    if track_id in track_to_face:
        stable_id = track_to_face[track_id]

        if stable_id in face_gallery:
            if embedding is None:
                face_last_seen[stable_id] = current_frame
                return stable_id

            sim = float(np.dot(embedding, face_gallery[stable_id]))

            if sim >= SIM_THRESHOLD * 0.85:
                updated = (
                    SMOOTH_ALPHA * face_gallery[stable_id]
                    + (1 - SMOOTH_ALPHA) * embedding
                )
                face_gallery[stable_id] = l2_normalize(updated)
                face_last_seen[stable_id] = current_frame
                return stable_id

            track_to_face.pop(track_id, None)

        else:
            track_to_face.pop(track_id, None)

    if embedding is None:
        return None

    best_id = None
    best_sim = -1.0

    for face_id, gallery_emb in face_gallery.items():
        sim = float(np.dot(embedding, gallery_emb))

        if sim > best_sim:
            best_sim = sim
            best_id = face_id

    if best_id is not None and best_sim >= SIM_THRESHOLD:
        stable_id = best_id
        updated = (
            SMOOTH_ALPHA * face_gallery[stable_id]
            + (1 - SMOOTH_ALPHA) * embedding
        )
        face_gallery[stable_id] = l2_normalize(updated)
    else:
        stable_id = next_face_id
        face_gallery[stable_id] = embedding
        next_face_id += 1

    face_last_seen[stable_id] = current_frame
    track_to_face[track_id] = stable_id

    return stable_id


def smooth_bbox(face_id, bbox, alpha=0.65):
    bbox = np.array(bbox, dtype=np.float32)

    if face_id not in bbox_smoother:
        bbox_smoother[face_id] = bbox
        return bbox.astype(int)

    smoothed = alpha * bbox_smoother[face_id] + (1 - alpha) * bbox
    bbox_smoother[face_id] = smoothed

    return smoothed.astype(int)

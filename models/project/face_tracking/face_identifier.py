import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
import onnxruntime as ort

from config import (
    TARGET_THRESHOLD,
    SIM_THRESHOLD,
    SMOOTH_ALPHA,
    MAX_FACE_AGE,
    EMBEDDING_REFRESH_INTERVAL,
)
from face_utils import l2_normalize, crop_with_padding

EMBEDDING_POOL_SIZE = 5
SMOOTH_RESET_IOU_THRESHOLD = 0.10
SMOOTH_RESET_CENTER_DISTANCE_RATIO = 1.5


def get_face_analysis_runtime():
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    available_providers = ort.get_available_providers()

    if "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0

    return ["CPUExecutionProvider"], -1


FACE_PROVIDERS, FACE_CTX_ID = get_face_analysis_runtime()

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=FACE_PROVIDERS
)

face_app.prepare(ctx_id=FACE_CTX_ID, det_size=(640, 640))


next_face_id = 1
face_gallery = {}
face_last_seen = {}
track_to_face = {}
track_last_emb = {}
bbox_smoother = {}

target_face_ids = set()
target_track_ids = set()


def cosine_similarity(emb1, emb2):
    if emb1 is None or emb2 is None:
        return -1.0

    return float(np.dot(emb1, emb2))


def get_gallery_best_similarity(embedding, gallery_embs):
    if embedding is None or len(gallery_embs) == 0:
        return -1.0

    sims = [
        cosine_similarity(embedding, gallery_emb)
        for gallery_emb in gallery_embs
    ]

    return max(sims)


def update_gallery_embedding(face_id, embedding):
    if embedding is None:
        return

    embedding = l2_normalize(embedding)

    if embedding is None:
        return

    if face_id not in face_gallery:
        face_gallery[face_id] = [embedding]
        return

    gallery_embs = face_gallery[face_id]

    if len(gallery_embs) > 0:
        base_emb = gallery_embs[-1]
        updated = (
            SMOOTH_ALPHA * base_emb
            + (1 - SMOOTH_ALPHA) * embedding
        )
        updated = l2_normalize(updated)

        if updated is not None:
            gallery_embs.append(updated)
        else:
            gallery_embs.append(embedding)
    else:
        gallery_embs.append(embedding)

    if len(gallery_embs) > EMBEDDING_POOL_SIZE:
        face_gallery[face_id] = gallery_embs[-EMBEDDING_POOL_SIZE:]


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

            best_sim = get_gallery_best_similarity(
                embedding,
                face_gallery[stable_id]
            )

            if best_sim >= SIM_THRESHOLD * 0.85:
                update_gallery_embedding(stable_id, embedding)
                face_last_seen[stable_id] = current_frame
                return stable_id

            track_to_face.pop(track_id, None)

        else:
            track_to_face.pop(track_id, None)

    if embedding is None:
        return None

    best_id = None
    best_sim = -1.0

    for face_id, gallery_embs in face_gallery.items():
        sim = get_gallery_best_similarity(embedding, gallery_embs)

        if sim > best_sim:
            best_sim = sim
            best_id = face_id

    if best_id is not None and best_sim >= SIM_THRESHOLD:
        stable_id = best_id
        update_gallery_embedding(stable_id, embedding)
    else:
        stable_id = next_face_id
        face_gallery[stable_id] = [embedding]
        next_face_id += 1

    face_last_seen[stable_id] = current_frame
    track_to_face[track_id] = stable_id

    return stable_id


def compute_single_iou(box1, box2):
    box1 = np.array(box1, dtype=np.float32)
    box2 = np.array(box2, dtype=np.float32)

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter

    if union <= 1e-6:
        return 0.0

    return float(inter / union)


def compute_center_distance(box1, box2):
    box1 = np.array(box1, dtype=np.float32)
    box2 = np.array(box2, dtype=np.float32)

    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2

    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))


def compute_box_diag(box):
    box = np.array(box, dtype=np.float32)

    w = max(1.0, box[2] - box[0])
    h = max(1.0, box[3] - box[1])

    return float(np.sqrt(w ** 2 + h ** 2))


def smooth_bbox(face_id, bbox, alpha=0.65):
    bbox = np.array(bbox, dtype=np.float32)

    if face_id is None:
        return bbox.astype(int)

    if face_id not in bbox_smoother:
        bbox_smoother[face_id] = bbox
        return bbox.astype(int)

    prev_bbox = bbox_smoother[face_id]

    iou = compute_single_iou(prev_bbox, bbox)
    center_dist = compute_center_distance(prev_bbox, bbox)
    diag = compute_box_diag(prev_bbox)

    should_reset = (
        iou < SMOOTH_RESET_IOU_THRESHOLD
        and center_dist > SMOOTH_RESET_CENTER_DISTANCE_RATIO * diag
    )

    if should_reset:
        bbox_smoother[face_id] = bbox
        return bbox.astype(int)

    smoothed = alpha * prev_bbox + (1 - alpha) * bbox
    bbox_smoother[face_id] = smoothed

    return smoothed.astype(int)

import ctypes
import sysconfig
import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
import onnxruntime as ort

from config import (
    TARGET_THRESHOLD,
    SIM_THRESHOLD,
    SMOOTH_ALPHA,
    SMOOTH_RESET_IOU_THRESHOLD,
    SMOOTH_RESET_CENTER_DISTANCE_RATIO,
    MAX_FACE_AGE,
    EMBEDDING_REFRESH_INTERVAL,
    ID_MIN_FACE_SIZE,
)
from face_utils import l2_normalize


# Gallery / smoothing 설정
EMBEDDING_POOL_SIZE = 5
BBOX_FALLBACK_MAX_AGE = 30
BBOX_FALLBACK_IOU_THRESHOLD = 0.35
BBOX_FALLBACK_CENTER_RATIO = 0.75
TRACK_REUSE_THRESHOLD_RATIO = 0.85
IDENTITY_MIN_ASPECT_RATIO = 0.55
IDENTITY_MAX_ASPECT_RATIO = 1.8
IDENTITY_MIN_AREA_RATIO = 0.35

def can_load_tensorrt_runtime():
    search_dirs = [Path(sysconfig.get_paths()["purelib"]) / "tensorrt_libs"]
    library_names = (
        "libnvinfer.so.10",
        "libnvinfer_plugin.so.10",
        "libnvonnxparser.so.10",
    )

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        try:
            for library_name in library_names:
                ctypes.CDLL(str(search_dir / library_name), mode=ctypes.RTLD_GLOBAL)
            return True
        except OSError:
            continue

    try:
        ctypes.CDLL("libnvinfer.so.10", mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


def get_face_analysis_runtime():
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    available_providers = ort.get_available_providers()

    if (
        "TensorrtExecutionProvider" in available_providers
        and can_load_tensorrt_runtime()
    ):
        providers = ["TensorrtExecutionProvider"]

        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")

        providers.append("CPUExecutionProvider")
        return providers, 0

    if "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0

    return ["CPUExecutionProvider"], -1


# InsightFace 초기화
FACE_PROVIDERS, FACE_CTX_ID = get_face_analysis_runtime()

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=FACE_PROVIDERS
)

face_app.prepare(ctx_id=FACE_CTX_ID, det_size=(640, 640))


# 전역 상태
next_face_id = 1

face_gallery = {}   
face_last_seen = {}    
face_last_bbox = {}
track_to_face = {}    
track_last_emb = {}  
bbox_smoother = {}
track_last_kps = {}    

target_face_ids = set()
target_track_ids = set()


def is_too_small_for_identity(box):
    x1, y1, x2, y2 = map(int, box)
    bw = x2 - x1
    bh = y2 - y1

    return bw <= 0 or bh <= 0 or min(bw, bh) < ID_MIN_FACE_SIZE

def is_invalid_identity_bbox(box, prev_box=None):
    x1, y1, x2, y2 = map(int, box)

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return True

    if min(bw, bh) < ID_MIN_FACE_SIZE:
        return True

    aspect_ratio = bw / max(bh, 1)

    if (
        aspect_ratio < IDENTITY_MIN_ASPECT_RATIO
        or aspect_ratio > IDENTITY_MAX_ASPECT_RATIO
    ):
        return True

    if prev_box is not None:
        px1, py1, px2, py2 = map(int, prev_box)

        pw = px2 - px1
        ph = py2 - py1

        if pw > 0 and ph > 0:
            cur_area = bw * bh
            prev_area = pw * ph

            if cur_area < prev_area * IDENTITY_MIN_AREA_RATIO:
                return True

    return False

# 임베딩 추출
def get_arcface_embedding(frame, box):
    x1, y1, x2, y2 = map(int, box)
    H, W = frame.shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return None, None
    if min(bw, bh) < ID_MIN_FACE_SIZE:
        return None, None

    pad_w = int(bw * 0.25)
    pad_h = int(bh * 0.25)

    px1 = max(0, x1 - pad_w)
    py1 = max(0, y1 - pad_h)
    px2 = min(W, x2 + pad_w)
    py2 = min(H, y2 + pad_h)

    crop = frame[py1:py2, px1:px2]

    if crop is None or crop.size == 0:
        return None, None

    faces = face_app.get(crop)

    if len(faces) == 0:
        return None, None

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
        return None, None

    kps = None

    if hasattr(best_face, "kps") and best_face.kps is not None:
        kps = np.asarray(best_face.kps, dtype=np.float32).copy()
        kps[:, 0] += px1
        kps[:, 1] += py1

    return emb, kps


# Track embedding 캐시
def get_track_embedding(frame, box, track_id, current_frame):
    cached_emb = track_last_emb.get(track_id)
    cached_kps = track_last_kps.get(track_id)

    prev_box = None

    if track_id in track_to_face:
        stable_id = track_to_face[track_id]
        prev_box = face_last_bbox.get(stable_id)

    if is_invalid_identity_bbox(box, prev_box):
        if cached_emb is not None:
            return cached_emb, cached_kps, False

        return None, None, False

    should_refresh = (
        cached_emb is None
        or current_frame % EMBEDDING_REFRESH_INTERVAL == 0
    )

    if not should_refresh:
        return cached_emb, cached_kps, False

    emb, kps = get_arcface_embedding(frame, box)

    if emb is None:
        if cached_emb is not None:
            return cached_emb, cached_kps, False

        return None, None, False

    track_last_emb[track_id] = emb
    track_last_kps[track_id] = kps

    return emb, kps, True


# 타겟 이미지 경로 로드
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


# 타겟 이미지 embedding 추출
def prepare_target_image(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255.0)
        image = (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)

    height, width = image.shape[:2]
    min_side = min(height, width)

    if min_side > 0 and min_side < 320:
        scale = 320 / min_side
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    return image


def get_target_embedding(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise RuntimeError(f"Cannot read target image: {image_path}")

    image = prepare_target_image(image)
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

# 타겟 얼굴 판별
def check_target_match(embedding, target_embeddings):
    if embedding is None or len(target_embeddings) == 0:
        return False, -1.0

    sims = [
        float(np.dot(embedding, target_emb))
        for target_emb in target_embeddings
    ]

    best_sim = max(sims)

    return best_sim >= TARGET_THRESHOLD, best_sim


# Gallery similarity 계산
def get_gallery_best_similarity(embedding, gallery_embeddings):
    if embedding is None or len(gallery_embeddings) == 0:
        return -1.0

    sims = [
        float(np.dot(embedding, gallery_emb))
        for gallery_emb in gallery_embeddings
    ]

    return max(sims)


# Gallery embedding 업데이트
def update_gallery_embedding(face_id, embedding):
    if embedding is None:
        return

    if face_id not in face_gallery:
        face_gallery[face_id] = [embedding]
        return

    face_gallery[face_id].append(embedding)

    if len(face_gallery[face_id]) > EMBEDDING_POOL_SIZE:
        face_gallery[face_id] = face_gallery[face_id][-EMBEDDING_POOL_SIZE:]


# 오래된 Face ID 정리
def cleanup_old_face_ids(current_frame):
    expired_face_ids = [
        face_id
        for face_id, last_seen in face_last_seen.items()
        if current_frame - last_seen > MAX_FACE_AGE
    ]

    for face_id in expired_face_ids:
        face_gallery.pop(face_id, None)
        face_last_seen.pop(face_id, None)
        face_last_bbox.pop(face_id, None)
        bbox_smoother.pop(face_id, None)

    for track_id, face_id in list(track_to_face.items()):
        if face_id in expired_face_ids:
            track_to_face.pop(track_id, None)
            track_last_emb.pop(track_id, None)
            track_last_kps.pop(track_id, None)
            target_track_ids.discard(track_id)


# Stable Face ID 부여
def assign_stable_face_id(track_id, embedding, current_frame, bbox=None):
    global next_face_id

    if embedding is not None:
        track_last_emb[track_id] = embedding

    # 기존 track_id가 이미 stable face_id에 연결된 경우
    if track_id in track_to_face:
        stable_id = track_to_face[track_id]

        if stable_id in face_gallery:
            if embedding is None:
                fallback_id = assign_bbox_fallback(
                    track_id,
                    bbox,
                    current_frame
                )

                if fallback_id is not None:
                    return fallback_id

                return None

            best_sim = get_gallery_best_similarity(
                embedding,
                face_gallery[stable_id]
            )

            if best_sim >= SIM_THRESHOLD * TRACK_REUSE_THRESHOLD_RATIO:
                update_gallery_embedding(stable_id, embedding)
                face_last_seen[stable_id] = current_frame

                if (
                    bbox is not None
                    and not is_invalid_identity_bbox(
                        bbox,
                        face_last_bbox.get(stable_id),
                    )
                ):
                    face_last_bbox[stable_id] = bbox

                return stable_id

            track_to_face.pop(track_id, None)

        else:
            track_to_face.pop(track_id, None)

    if embedding is None:
        fallback_id = assign_bbox_fallback(
            track_id,
            bbox,
            current_frame
        )

        if fallback_id is not None:
            return fallback_id

        return None

    # 전체 gallery에서 가장 비슷한 face_id 검색
    best_id = None
    best_sim = -1.0

    for face_id, gallery_embeddings in face_gallery.items():
        sim = get_gallery_best_similarity(embedding, gallery_embeddings)

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

    if (
        bbox is not None
        and not is_invalid_identity_bbox(
            bbox,
            face_last_bbox.get(stable_id),
        )
    ):
        face_last_bbox[stable_id] = bbox

    track_to_face[track_id] = stable_id

    return stable_id


# bbox IoU 계산
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


# bbox 중심 거리 계산
def compute_center_distance(box1, box2):
    box1 = np.array(box1, dtype=np.float32)
    box2 = np.array(box2, dtype=np.float32)

    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2

    return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))


# bbox 대각선 길이 계산
def compute_box_diag(box):
    box = np.array(box, dtype=np.float32)

    w = max(1.0, box[2] - box[0])
    h = max(1.0, box[3] - box[1])

    return float(np.sqrt(w ** 2 + h ** 2))


def find_bbox_fallback_face_id(bbox, current_frame):
    if bbox is None:
        return None

    best_id = None
    best_score = -1.0

    cur_diag = compute_box_diag(bbox)

    for face_id, last_bbox in face_last_bbox.items():
        last_seen = face_last_seen.get(face_id, -999999)

        if current_frame - last_seen > BBOX_FALLBACK_MAX_AGE:
            continue

        iou = compute_single_iou(bbox, last_bbox)
        center_dist = compute_center_distance(bbox, last_bbox)
        center_score = max(0.0, 1.0 - center_dist / max(cur_diag, 1.0))

        if iou < BBOX_FALLBACK_IOU_THRESHOLD and center_score < BBOX_FALLBACK_CENTER_RATIO:
            continue

        score = 0.7 * iou + 0.3 * center_score

        if score > best_score:
            best_score = score
            best_id = face_id

    return best_id


def assign_bbox_fallback(track_id, bbox, current_frame):
    if bbox is None or is_invalid_identity_bbox(bbox):
        return None

    fallback_id = find_bbox_fallback_face_id(bbox, current_frame)

    if fallback_id is None:
        return None

    face_last_seen[fallback_id] = current_frame
    face_last_bbox[fallback_id] = bbox
    track_to_face[track_id] = fallback_id

    return fallback_id


# bbox smoothing
def smooth_bbox(face_id, bbox, alpha=SMOOTH_ALPHA):
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
        or center_dist > SMOOTH_RESET_CENTER_DISTANCE_RATIO * diag
    )

    if should_reset:
        if is_invalid_identity_bbox(bbox, prev_bbox):
            return prev_bbox.astype(int)

        bbox_smoother[face_id] = bbox
        return bbox.astype(int)

    smoothed = alpha * bbox + (1 - alpha) * prev_bbox
    bbox_smoother[face_id] = smoothed

    return smoothed.astype(int)

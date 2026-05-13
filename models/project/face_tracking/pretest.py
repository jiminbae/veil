import cv2
import numpy as np
import logging
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from ultralytics import YOLO
from boxmot.trackers.botsort.botsort import BotSort
from insightface.app import FaceAnalysis


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

VIDEO_PATH = str(BASE_DIR / "videos/test.mp4")
FACE_MODEL_PATH = str(BASE_DIR / "weights/yolo26x-face.pt")
REID_MODEL_PATH = str(PROJECT_DIR / "boxmot/models/osnet_x0_25_msmt17.pt")

TARGET_DIR = str(BASE_DIR / "target")
TARGET_PATTERN = "target*"


# results 저장 경로 설정
OUTPUT_PATH = str(BASE_DIR / "outputs/result/output_target1.mp4")
LOG_PATH = str(BASE_DIR / "outputs/log/tracking_target1_log.txt")
CROP_ROOT = str(BASE_DIR / "outputs/crop/crop1")
METADATA_PATH = str(BASE_DIR / "outputs/metadata/face_metadata1.json")


# 맥북 GPU 사용시 설정
#device = "mps"
# GPU 사용시 설정
device = "cuda"

detector = YOLO(FACE_MODEL_PATH)

tracker = BotSort(
    reid_weights=Path(REID_MODEL_PATH),
    device=device,
    half=False,
    with_reid=False,
    track_buffer=150,
    match_thresh=0.6,
    proximity_thresh=0.7,
    appearance_thresh=0.5
)

face_app = FaceAnalysis(
    name="buffalo_l",
    # CPU 사용시 설정
    # providers=["CPUExecutionProvider"]
    # GPU 사용시 설정
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)


# CPU 사용시 설정
#face_app.prepare(ctx_id=-1, det_size=(640, 640))
# GPU 사용시 설정
face_app.prepare(ctx_id=0, det_size=(640, 640))


SIM_THRESHOLD = 0.38
TARGET_THRESHOLD = 0.50
SMOOTH_ALPHA = 0.8
MAX_FACE_AGE = 120
MIN_FACE_AREA = 400
MAX_ASPECT_RATIO = 2.2

LIVEPORTRAIT_MIN_FACE_AREA = 2500
LIVEPORTRAIT_MIN_CROP_SIZE = 64
LIVEPORTRAIT_MAX_ASPECT_RATIO = 2.5
EDGE_MARGIN = 2
EMBEDDING_REFRESH_INTERVAL = 5
LOG_EVERY_N_FRAMES = 30
CROP_WRITER_WORKERS = 2

next_face_id = 1
face_gallery = {}
face_last_seen = {}
track_to_face = {}
track_last_emb = {}
bbox_smoother = {}

target_embeddings = []
target_face_ids = set()
target_track_ids = set()

current_frame_idx = 0
all_face_metadata = []
crop_write_futures = []

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
logging.info(f"EMBEDDING_REFRESH_INTERVAL={EMBEDDING_REFRESH_INTERVAL}")
logging.info(f"LOG_EVERY_N_FRAMES={LOG_EVERY_N_FRAMES}")


def l2_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x)
    if norm < 1e-6:
        return None
    return x / norm


def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area1 + area2 - inter
    return inter / np.maximum(union, 1e-6)


def apply_nms(dets, iou_thresh=0.40):
    if len(dets) == 0:
        return dets

    boxes = dets[:, :4]
    scores = dets[:, 4]
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        cur = order[0]
        keep.append(cur)

        if len(order) == 1:
            break

        ious = compute_iou(boxes[cur], boxes[order[1:]])
        order = order[1:][ious < iou_thresh]

    return dets[keep].astype(np.float32)


def filter_detections(dets):
    if len(dets) == 0:
        return dets

    filtered = []

    for det in dets:
        x1, y1, x2, y2 = det[:4]
        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 0 or box_h <= 0:
            continue

        area = box_w * box_h
        aspect_ratio = max(box_w, box_h) / (min(box_w, box_h) + 1e-6)

        if area >= MIN_FACE_AREA and aspect_ratio <= MAX_ASPECT_RATIO:
            filtered.append(det)

    if len(filtered) == 0:
        return np.empty((0, 6), dtype=np.float32)

    return np.array(filtered, dtype=np.float32)


def detect_faces_multiscale(frame, detector, device):
    h, w = frame.shape[:2]
    all_dets = []

    full_result = detector(
        frame,
        conf=0.40,
        imgsz=768,
        verbose=False,
        device=device
    )[0]

    if full_result.boxes is not None:
        boxes = full_result.boxes.xyxy.cpu().numpy()
        confs = full_result.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box
            all_dets.append([x1, y1, x2, y2, conf, 0])

    tiles = [
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w, h // 2),
        (0, h // 2, w // 2, h),
        (w // 2, h // 2, w, h),
    ]

    tile_images = []
    tile_offsets = []

    for tx1, ty1, tx2, ty2 in tiles:
        tile = frame[ty1:ty2, tx1:tx2]

        if tile.size == 0:
            continue

        tile_images.append(tile)
        tile_offsets.append((tx1, ty1))

    if tile_images:
        tile_results = detector(
            tile_images,
            conf=0.40,
            imgsz=640,
            verbose=False,
            device=device
        )

        for result, (tx1, ty1) in zip(tile_results, tile_offsets):
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box
                all_dets.append([
                    x1 + tx1,
                    y1 + ty1,
                    x2 + tx1,
                    y2 + ty1,
                    conf,
                    0
                ])

    if len(all_dets) == 0:
        return np.empty((0, 6), dtype=np.float32)

    dets = np.array(all_dets, dtype=np.float32)
    dets = apply_nms(dets, iou_thresh=0.40)
    dets = filter_detections(dets)

    return dets


def clip_bbox(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def crop_with_padding(frame, box, pad_ratio=0.4):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * pad_ratio)
    pad_y = int(box_h * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


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

    if area < LIVEPORTRAIT_MIN_FACE_AREA:
        reasons.append("small_face")

    if aspect_ratio > LIVEPORTRAIT_MAX_ASPECT_RATIO:
        reasons.append("bad_aspect_ratio")

    if embedding is None:
        reasons.append("embedding_failed")

    if crop is None or crop.size == 0:
        reasons.append("crop_failed")
    else:
        crop_h, crop_w = crop.shape[:2]

        if crop_w < LIVEPORTRAIT_MIN_CROP_SIZE or crop_h < LIVEPORTRAIT_MIN_CROP_SIZE:
            reasons.append("small_crop")

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


def save_background_crop(crop, stable_face_id, raw_track_id, frame_idx, executor):
    if crop is None or crop.size == 0:
        return None

    if stable_face_id is not None:
        person_key = f"face_{stable_face_id}"
    else:
        person_key = f"track_{raw_track_id}"

    save_dir = Path(CROP_ROOT) / person_key
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"frame_{frame_idx:06d}.png"
    future = executor.submit(cv2.imwrite, str(save_path), crop.copy())
    crop_write_futures.append((save_path, future))

    return str(save_path)


def make_face_data(
    frame_idx,
    raw_track_id,
    stable_face_id,
    bbox,
    smoothed_bbox,
    is_target,
    is_background,
    target_sim,
    embedding_ok,
    quality,
    fallback_reasons,
    crop_path
):
    return {
        "frame_idx": int(frame_idx),
        "raw_track_id": int(raw_track_id),
        "stable_face_id": int(stable_face_id) if stable_face_id is not None else None,
        "bbox": [int(v) for v in bbox],
        "smoothed_bbox": [int(v) for v in smoothed_bbox],
        "is_target": bool(is_target),
        "is_background": bool(is_background),
        "target_similarity": float(target_sim),
        "embedding_ok": bool(embedding_ok),
        "quality": str(quality),
        "fallback_reasons": list(fallback_reasons),
        "crop_path": crop_path
    }


target_paths = get_target_image_paths(
    TARGET_DIR,
    TARGET_PATTERN
)

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

logging.info(
    f"Total target embeddings loaded: {len(target_embeddings)}"
)

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

with ThreadPoolExecutor(max_workers=CROP_WRITER_WORKERS) as crop_executor:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        current_frame_idx += 1

        if current_frame_idx % 30 == 0:
            cleanup_old_face_ids(current_frame_idx)

        dets = detect_faces_multiscale(frame, detector, device)

        if current_frame_idx % LOG_EVERY_N_FRAMES == 0:
            logging.info(
                f"Frame={current_frame_idx} "
                f"Detections={len(dets)}"
            )

        tracks = tracker.update(dets, frame)

        if current_frame_idx % LOG_EVERY_N_FRAMES == 0:
            logging.info(
                f"Frame={current_frame_idx} "
                f"Tracks={len(tracks)}"
            )

        for track in tracks:
            x1, y1, x2, y2, raw_track_id = map(int, track[:5])

            emb, embedding_refreshed = get_track_embedding(
                frame,
                [x1, y1, x2, y2],
                raw_track_id,
                current_frame_idx
            )

            stable_face_id = assign_stable_face_id(
                raw_track_id,
                emb,
                current_frame_idx
            )

            is_target = False
            target_sim = -1.0

            if emb is not None:
                is_target, target_sim = check_target_match(
                    emb,
                    target_embeddings
                )

                if is_target:
                    target_track_ids.add(raw_track_id)

                    if stable_face_id is not None:
                        target_face_ids.add(stable_face_id)

            if stable_face_id in target_face_ids:
                target_track_ids.add(raw_track_id)

            if raw_track_id in target_track_ids and stable_face_id is not None:
                target_face_ids.add(stable_face_id)

            if stable_face_id is not None:
                sx1, sy1, sx2, sy2 = smooth_bbox(
                    stable_face_id,
                    [x1, y1, x2, y2]
                )
            else:
                sx1, sy1, sx2, sy2 = x1, y1, x2, y2

            smoothed_bbox = [sx1, sy1, sx2, sy2]

            is_target_final = (
                stable_face_id in target_face_ids
                if stable_face_id is not None
                else raw_track_id in target_track_ids
            )

            is_background = not is_target_final

            raw_bbox = [x1, y1, x2, y2]

            raw_crop = crop_with_padding(frame, raw_bbox)
            smooth_crop = crop_with_padding(frame, smoothed_bbox)

            quality, fallback_reasons = assess_face_quality(
                frame,
                raw_bbox,
                emb,
                raw_crop
            )

            crop_path = None

            if is_background:
                if quality == "GOOD":
                    crop_path = save_background_crop(
                        smooth_crop,
                        stable_face_id,
                        raw_track_id,
                        current_frame_idx,
                        crop_executor
                    )
                else:
                    frame = apply_fallback_blur(frame, smoothed_bbox)

            face_data = make_face_data(
                frame_idx=current_frame_idx,
                raw_track_id=raw_track_id,
                stable_face_id=stable_face_id,
                bbox=[x1, y1, x2, y2],
                smoothed_bbox=smoothed_bbox,
                is_target=is_target_final,
                is_background=is_background,
                target_sim=target_sim,
                embedding_ok=emb is not None,
                quality=quality,
                fallback_reasons=fallback_reasons,
                crop_path=crop_path
            )

            all_face_metadata.append(face_data)

            target_track_flag = raw_track_id in target_track_ids
            target_face_flag = (
                stable_face_id in target_face_ids
                if stable_face_id is not None
                else False
            )

            if current_frame_idx % LOG_EVERY_N_FRAMES == 0:
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
                    f"Quality={quality} "
                    f"FallbackReasons={fallback_reasons} "
                    f"CropPath={crop_path}"
                )

            if stable_face_id is not None:
                if is_target_final:
                    label = f"TARGET ID {stable_face_id}"
                    color = (0, 0, 255)
                else:
                    if quality == "GOOD":
                        label = f"BG ID {stable_face_id} CROP"
                        color = (0, 255, 0)
                    else:
                        label = f"BG ID {stable_face_id} BLUR"
                        color = (0, 165, 255)
            else:
                if raw_track_id in target_track_ids:
                    label = f"TARGET Track ID {raw_track_id}"
                    color = (0, 0, 255)
                else:
                    if quality == "GOOD":
                        label = f"BG Track {raw_track_id} CROP"
                        color = (0, 255, 0)
                    else:
                        label = f"BG Track {raw_track_id} BLUR"
                        color = (0, 165, 255)

            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
            cv2.putText(
                frame,
                label,
                (sx1, max(20, sy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

failed_crop_writes = []

for save_path, future in crop_write_futures:
    try:
        if not future.result():
            failed_crop_writes.append(str(save_path))
    except Exception as e:
        failed_crop_writes.append(f"{save_path}: {e}")

if failed_crop_writes:
    logging.warning(f"Failed crop writes: {failed_crop_writes[:20]}")

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(all_face_metadata, f, ensure_ascii=False, indent=2)

logging.info("===== Experiment Finished =====")
logging.info(f"Saved result to: {OUTPUT_PATH}")
logging.info(f"Saved log to: {LOG_PATH}")
logging.info(f"Saved metadata to: {METADATA_PATH}")
logging.info(f"Saved background crops to: {CROP_ROOT}")

print(f"Saved result to: {OUTPUT_PATH}")
print(f"Saved log to: {LOG_PATH}")
print(f"Saved metadata to: {METADATA_PATH}")
print(f"Saved background crops to: {CROP_ROOT}")

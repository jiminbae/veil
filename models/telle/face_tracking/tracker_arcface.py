import argparse
import cv2
import numpy as np
import logging
from pathlib import Path
from ultralytics import YOLO
from boxmot.trackers.botsort.botsort import BotSort
from insightface.app import FaceAnalysis


DEFAULT_VIDEO_PATH = "../people_crossing.mp4"
FACE_MODEL_PATH = "runs/face/yolo26x_widerface/weights/best.pt"
REID_MODEL_PATH = "../boxmot/models/osnet_x0_25_msmt17.pt"

DEFAULT_OUTPUT_PATH = "tracker_arcface.mp4"
DEFAULT_LOG_PATH = "tracker_arcface.log"

# device = "mps"

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
    # providers=["CPUExecutionProvider"]
    # GPU 사용시 설정
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

SIM_THRESHOLD = 0.38
SMOOTH_ALPHA = 0.8
MAX_FACE_AGE = 120
MIN_FACE_AREA = 400
MAX_ASPECT_RATIO = 2.2

next_face_id = 1
face_gallery = {}
face_last_seen = {}
track_to_face = {}
track_last_emb = {}
bbox_smoother = {}
current_frame_idx = 0

def build_logger(log_path):
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        force=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="ArcFace 기반 얼굴 추적")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH, help="입력 동영상 경로")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="출력 동영상 경로")
    parser.add_argument("--log", type=str, default=DEFAULT_LOG_PATH, help="로그 파일 경로")
    return parser.parse_args()


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

    results = detector(
        frame,
        conf=0.40,
        imgsz=1280,
        verbose=False,
        device=device
    )[0]

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box
            all_dets.append([x1, y1, x2, y2, conf, 0])

    tiles = [
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w, h // 2),
        (0, h // 2, w // 2, h),
        (w // 2, h // 2, w, h),
    ]

    for tx1, ty1, tx2, ty2 in tiles:
        tile = frame[ty1:ty2, tx1:tx2]

        if tile.size == 0:
            continue

        results = detector(
            tile,
            conf=0.40,
            imgsz=1280,
            verbose=False,
            device=device
        )[0]

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

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

    for track_id, face_id in list(track_to_face.items()):
        if face_id in expired_face_ids:
            track_to_face.pop(track_id, None)


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


args = parse_args()
VIDEO_PATH = args.video
OUTPUT_PATH = args.output
LOG_PATH = args.log

build_logger(LOG_PATH)

logging.info("===== Experiment Started =====")
logging.info(f"VIDEO_PATH={VIDEO_PATH}")
logging.info(f"FACE_MODEL_PATH={FACE_MODEL_PATH}")
logging.info(f"OUTPUT_PATH={OUTPUT_PATH}")
logging.info(f"SIM_THRESHOLD={SIM_THRESHOLD}")
logging.info(f"SMOOTH_ALPHA={SMOOTH_ALPHA}")

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

while True:
    ret, frame = cap.read()

    if not ret:
        break

    current_frame_idx += 1

    if current_frame_idx % 30 == 0:
        cleanup_old_face_ids(current_frame_idx)

    dets = detect_faces_multiscale(frame, detector, device)

    logging.info(
        f"Frame={current_frame_idx} "
        f"Detections={len(dets)}"
    )

    tracks = tracker.update(dets, frame)

    logging.info(
        f"Frame={current_frame_idx} "
        f"Tracks={len(tracks)}"
    )

    for track in tracks:
        x1, y1, x2, y2, raw_track_id = map(int, track[:5])

        emb = get_arcface_embedding(frame, [x1, y1, x2, y2])

        stable_face_id = assign_stable_face_id(
            raw_track_id,
            emb,
            current_frame_idx
        )

        logging.info(
            f"Frame={current_frame_idx} "
            f"TrackID={raw_track_id} "
            f"FaceID={stable_face_id} "
            f"BBox=({x1},{y1},{x2},{y2}) "
            f"Embedding={'OK' if emb is not None else 'None'}"
        )

        if stable_face_id is not None:
            sx1, sy1, sx2, sy2 = smooth_bbox(
                stable_face_id,
                [x1, y1, x2, y2]
            )
            label = f"Face ID {stable_face_id}"
            color = (0, 255, 0)
        else:
            sx1, sy1, sx2, sy2 = x1, y1, x2, y2
            label = f"Track ID {raw_track_id}"
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

logging.info("===== Experiment Finished =====")
logging.info(f"Saved result to: {OUTPUT_PATH}")
print(f"Saved result to: {OUTPUT_PATH}")
print(f"Saved log to: {LOG_PATH}")
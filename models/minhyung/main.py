import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from boxmot.trackers.botsort.botsort import BotSort
from insightface.app import FaceAnalysis
from ultralytics import YOLO

# GPU 설정
if torch.cuda.is_available():
    DEVICE = 0
    CTX_ID = 0
    FACE_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    HALF_PRECISION = True
    print("[INFO] CUDA 사용")
else:
    DEVICE = "cpu"
    CTX_ID = -1
    FACE_PROVIDERS = ["CPUExecutionProvider"]
    HALF_PRECISION = False
    print("[INFO] CPU 사용")

# ReID 모델 경로
REID_MODEL_PATH = "models/weights/osnet_x0_25_msmt17.pt"

# Stable Identity 설정
SIM_THRESHOLD = 0.45
EMBED_SMOOTH_ALPHA = 0.90
MAX_MISSING_FRAMES = 120

# 제외할 Stable ID
EXCLUDE_STABLE_IDS = [1]

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True, help="입력 비디오 경로")
parser.add_argument("--output", type=str, required=True, help="출력 비디오 경로")
parser.add_argument("--log", type=str, required=True, help="로그 파일 저장 경로")
args = parser.parse_args()

input_path = args.video
output_path = args.output
log_path = args.log

log_file = open(log_path, "w", encoding="utf-8")


def log_print(msg: str) -> None:
    print(msg)
    log_file.write(msg + "\n")


# ArcFace 초기화
face_app = FaceAnalysis(name="buffalo_l", providers=FACE_PROVIDERS)
face_app.prepare(ctx_id=CTX_ID, det_size=(640, 640))

# YOLO 모델
person_model = YOLO("models/yolov8x.pt")
face_model = YOLO("/home/jmbae/DL-project/model/telle/face_tracking/weights/yolov8x-face-lindevs.pt")

# BoTSORT + ReID Tracker
tracker = BotSort(
    reid_weights=Path(REID_MODEL_PATH),
    device=DEVICE,
    half=HALF_PRECISION,
    with_reid=True,
    track_buffer=150,
    match_thresh=0.75,
    proximity_thresh=0.5,
    appearance_thresh=0.25,
)

# 비디오 로드
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("[ERROR] 영상 열기 실패")
    raise SystemExit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
imgsz = 960 if DEVICE == "cuda" else 640

if fps <= 0:
    fps = 25

print(f"[INFO] width={width}, height={height}, fps={fps}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Stable Identity 저장소
next_stable_id = 1
face_gallery = {}
face_last_seen = {}
track_to_stable = {}
track_last_embedding = {}
frame_count = 0


def l2_normalize(x):
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x)
    if norm < 1e-6:
        return None
    return x / norm


def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def get_face_embedding(frame, face_box):
    fx1, fy1, fx2, fy2 = face_box
    h, w = frame.shape[:2]

    fx1 = max(0, fx1)
    fy1 = max(0, fy1)
    fx2 = min(w, fx2)
    fy2 = min(h, fy2)

    pad_x = int((fx2 - fx1) * 0.25)
    pad_y = int((fy2 - fy1) * 0.25)

    x1 = max(0, fx1 - pad_x)
    y1 = max(0, fy1 - pad_y)
    x2 = min(w, fx2 + pad_x)
    y2 = min(h, fy2 + pad_y)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    faces = face_app.get(crop)
    if len(faces) == 0:
        return None

    best_face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )

    if hasattr(best_face, "normed_embedding"):
        emb = best_face.normed_embedding
    else:
        emb = best_face.embedding

    return l2_normalize(emb)


def assign_stable_id(track_id, embedding, current_frame):
    global next_stable_id

    if track_id is None:
        return None

    if embedding is None:
        embedding = track_last_embedding.get(track_id)
    else:
        track_last_embedding[track_id] = embedding

    if track_id in track_to_stable:
        stable_id = track_to_stable[track_id]

        if stable_id in face_gallery:
            if embedding is None:
                face_last_seen[stable_id] = current_frame
                return stable_id

            similarity = float(np.dot(embedding, face_gallery[stable_id]))
            if similarity >= SIM_THRESHOLD * 0.85:
                updated_embedding = (
                    EMBED_SMOOTH_ALPHA * face_gallery[stable_id]
                    + (1 - EMBED_SMOOTH_ALPHA) * embedding
                )
                face_gallery[stable_id] = l2_normalize(updated_embedding)
                face_last_seen[stable_id] = current_frame
                return stable_id

            track_to_stable.pop(track_id, None)

    if embedding is None:
        return None

    best_id = None
    best_similarity = -1.0

    for stable_id, gallery_embedding in face_gallery.items():
        similarity = float(np.dot(embedding, gallery_embedding))
        if similarity > best_similarity:
            best_similarity = similarity
            best_id = stable_id

    if best_id is not None and best_similarity >= SIM_THRESHOLD:
        stable_id = best_id
        updated_embedding = (
            EMBED_SMOOTH_ALPHA * face_gallery[stable_id]
            + (1 - EMBED_SMOOTH_ALPHA) * embedding
        )
        face_gallery[stable_id] = l2_normalize(updated_embedding)
    else:
        stable_id = next_stable_id
        face_gallery[stable_id] = embedding
        next_stable_id += 1

    face_last_seen[stable_id] = current_frame
    track_to_stable[track_id] = stable_id
    return stable_id


def cleanup_old_identities(current_frame):
    expired_ids = []

    for stable_id, last_seen in face_last_seen.items():
        if current_frame - last_seen > MAX_MISSING_FRAMES:
            expired_ids.append(stable_id)

    for stable_id in expired_ids:
        face_gallery.pop(stable_id, None)
        face_last_seen.pop(stable_id, None)

    for track_id, stable_id in list(track_to_stable.items()):
        if stable_id in expired_ids:
            track_to_stable.pop(track_id, None)


while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] 영상 끝")
        break

    frame_count += 1
    print(f"\n[INFO] processing frame {frame_count}")

    if frame_count % 30 == 0:
        cleanup_old_identities(frame_count)

    det_results = person_model(
        frame,
        conf=0.25,
        iou=0.5,
        imgsz=imgsz,
        device=DEVICE,
        verbose=False,
    )[0]

    detections = []
    if det_results.boxes is not None:
        boxes = det_results.boxes.xyxy.cpu().numpy()
        confs = det_results.boxes.conf.cpu().numpy()
        classes = det_results.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            if int(cls) != 0:
                continue

            x1, y1, x2, y2 = box
            detections.append([x1, y1, x2, y2, conf, cls])

    if len(detections) == 0:
        out.write(frame)
        continue

    detections = np.array(detections, dtype=np.float32)
    log_print(f"Frame={frame_count} Detections={len(detections)}")

    tracks = tracker.update(detections, frame)

    person_list = []
    for track in tracks:
        if len(track) < 5:
            continue

        x1, y1, x2, y2, track_id = map(int, track[:5])
        person_list.append({"track_id": track_id, "bbox": (x1, y1, x2, y2)})
        print(f"[INFO] Track ID {track_id} BBox ({x1}, {y1}, {x2}, {y2})")

    print(f"[INFO] 사람 수 {len(person_list)}")
    log_print(f"Frame={frame_count} Tracks={len(person_list)}")

    face_results = face_model(frame, conf=0.4, device=DEVICE, verbose=False)

    face_list = []
    for r in face_results:
        if r.boxes is None:
            continue

        face_boxes = r.boxes.xyxy.cpu().numpy()
        for box in face_boxes:
            fx1, fy1, fx2, fy2 = map(int, box)
            face_list.append((fx1, fy1, fx2, fy2))

    print(f"[INFO] 얼굴 개수 {len(face_list)}")

    matched = []
    iou_threshold = 0.01

    for face_box in face_list:
        best_iou = 0.0
        matched_track_id = None

        for person in person_list:
            iou = compute_iou(face_box, person["bbox"])
            if iou > best_iou:
                best_iou = iou
                matched_track_id = person["track_id"]

        if best_iou < iou_threshold:
            matched_track_id = None

        embedding = get_face_embedding(frame, face_box)
        stable_id = assign_stable_id(matched_track_id, embedding, frame_count)

        matched.append(
            {
                "face_box": face_box,
                "track_id": matched_track_id,
                "stable_id": stable_id,
                "iou": best_iou,
            }
        )

    for m in matched:
        fx1, fy1, fx2, fy2 = m["face_box"]
        track_id = m["track_id"]
        stable_id = m["stable_id"]
        iou = m["iou"]

        if track_id is not None and stable_id is not None:
            log_print(f"Frame={frame_count} TrackID={track_id} FaceID={stable_id}")

        print(f"[INFO] Track={track_id} | Stable={stable_id} | IoU={iou:.4f}")

    for m in matched:
        fx1, fy1, fx2, fy2 = m["face_box"]
        stable_id = m["stable_id"]

        if stable_id is not None and stable_id not in EXCLUDE_STABLE_IDS:
            face_roi = frame[fy1:fy2, fx1:fx2]
            if face_roi.size != 0:
                blurred_face = cv2.GaussianBlur(face_roi, (51, 51), 30)
                frame[fy1:fy2, fx1:fx2] = blurred_face

        color = (0, 0, 255) if stable_id in EXCLUDE_STABLE_IDS else (255, 0, 0)
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 2)
        cv2.putText(
            frame,
            f"S{stable_id}",
            (fx1, fy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    out.write(frame)

cap.release()
out.release()

log_print(f"Saved result to: {output_path}")
log_print(f"Saved log to: {log_path}")
log_file.close()

print("[INFO] 저장 완료")
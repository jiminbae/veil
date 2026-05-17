import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from boxmot.trackers.botsort.botsort import BotSort

from config import (
    FACE_MODEL_PATH,
    REID_MODEL_PATH,
    MIN_FACE_AREA,
    MAX_ASPECT_RATIO,
)


def init_detector():
    return YOLO(FACE_MODEL_PATH)


def init_tracker(device):
    tracker_device = normalize_tracker_device(device)

    return BotSort(
        reid_weights=Path(REID_MODEL_PATH),
        device=tracker_device,
        half=False,
        with_reid=True,
        track_buffer=250,
        match_thresh=0.5,
        proximity_thresh=0.7,
        appearance_thresh=0.4
    )


def normalize_tracker_device(device):
    if isinstance(device, str) and device.lower() == "cuda":
        return "0" if torch.cuda.device_count() > 0 else "cpu"

    return device


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

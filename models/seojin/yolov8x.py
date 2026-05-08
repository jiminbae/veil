# -*- coding: utf-8 -*-
"""
YOLOv8x 기반 얼굴 추적 모델
BotSort 트래커 + 모자이크 + 텍스트 로그 출력
"""

import argparse
import cv2
import logging
from pathlib import Path
from ultralytics import YOLO
import torch


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 기본 경로 설정
DEFAULT_VIDEO_PATH = "../people_crossing.mp4"
DEFAULT_OUTPUT_PATH = "yolov8x.mp4"
DEFAULT_LOG_PATH = "yolov8x.log"

# 모델 경로
FACE_MODEL_PATH = "../telle/face_tracking/weights/yolov8x-face-lindevs.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8x 기반 얼굴 추적")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH, help="입력 동영상 경로")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="출력 동영상 경로")
    parser.add_argument("--log", type=str, default=DEFAULT_LOG_PATH, help="로그 파일 경로")
    parser.add_argument("--device", type=int, default=0, help="CUDA 디바이스 ID (음수면 CPU)")
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)
    log_path = Path(args.log)
    device = args.device

    logger.info(f"입력 동영상: {video_path}")
    logger.info(f"출력 동영상: {output_path}")
    logger.info(f"로그 파일: {log_path}")
    logger.info(f"CUDA 디바이스: {device}")

    # 모델 로드
    logger.info("YOLOv8x 모델 로딩...")
    model = YOLO(FACE_MODEL_PATH)

    # 입력 동영상 열기
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"동영상을 열 수 없습니다: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"동영상 정보: {width}x{height}, {fps} FPS, {total_frames} 프레임")

    # 출력 동영상 설정
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # 로그 파일 열기
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, 'w', encoding='utf-8')

    # 프레임 처리
    frame_num = 0
    frame_detections = {}
    frame_tracks = {}
    all_track_ids = set()
    all_face_ids = set()

    logger.info("동영상 처리 시작...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        timestamp = round(frame_num / fps, 3)

        # YOLOv8x로 트래킹
        results = model.track(
            frame,
            persist=True,
            tracker='botsort.yaml',
            conf=0.5,
            verbose=False,
            device=device
        )

        detection_count = 0
        track_count = 0

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            detection_count = len(boxes)
            track_count = len(set(ids))

            for box_idx, (box, track_id) in enumerate(zip(boxes, ids)):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                face_id = track_id  # 같은 track_id를 face_id로 사용

                all_track_ids.add(track_id)
                all_face_ids.add(face_id)

                # 모자이크
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    blurred = cv2.GaussianBlur(roi, (51, 51), 0)
                    frame[y1:y2, x1:x2] = blurred

                # ID 표시
                cv2.putText(
                    frame,
                    f'ID:{track_id}',
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # 텍스트 로그 (evaluate.py 호환 형식)
                log_file.write(f"Frame={frame_num} TrackID={track_id} FaceID={face_id}\n")

        # 프레임별 통계 로그
        log_file.write(f"Frame={frame_num} Detections={detection_count}\n")
        log_file.write(f"Frame={frame_num} Tracks={track_count}\n")

        frame_detections[frame_num] = detection_count
        frame_tracks[frame_num] = track_count

        out.write(frame)

        if frame_num % 100 == 0:
            logger.info(f"처리 완료: {frame_num}/{total_frames} 프레임")

    cap.release()
    out.release()
    log_file.close()

    # 최종 통계
    total_detections = sum(frame_detections.values())
    total_tracks = sum(frame_tracks.values())
    unique_track_ids = len(all_track_ids)
    unique_face_ids = len(all_face_ids)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"처리 완료!")
    logger.info(f"  - 총 프레임: {frame_num}")
    logger.info(f"  - 총 감지: {total_detections}")
    logger.info(f"  - 총 트래킹: {total_tracks}")
    logger.info(f"  - 고유 트랙 ID: {unique_track_ids}")
    logger.info(f"  - 고유 얼굴 ID: {unique_face_ids}")
    logger.info(f"  - 출력 동영상: {output_path}")
    logger.info(f"  - 로그 파일: {log_path}")
    logger.info(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
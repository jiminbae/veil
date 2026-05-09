# 얼굴 탐지 및 블러 처리 모델 평가

이 폴더는 동영상에서 얼굴을 탐지하고 블러 처리하는 여러 파이프라인과,
얼굴 탐지 모델 weight를 같은 조건에서 비교하기 위한 평가 스크립트를 담고 있습니다.

## 폴더 구성

- `evaluate.py`: 여러 모델 스크립트를 실행하고 로그 기반 진단 리포트를 출력합니다.
- `seojin/yolov8x.py`: YOLO face tracking 기반 얼굴 블러 파이프라인입니다. 단순 얼굴 블러 목적에는 가장 빠르고 실용적입니다.
- `telle/face_tracking/tracker_arcface.py`: YOLO face detector + BoTSORT + ArcFace 기반 stable identity 파이프라인입니다.
- `minhyung/main.py`: person detector + face detector + ArcFace 매칭 기반 파이프라인입니다.
- `people_crossing.mp4`: 로컬 평가용 동영상입니다. git에는 올라가지 않습니다.
- `evaluation_results/`: 평가 결과 영상과 로그가 저장되는 폴더입니다. git에는 올라가지 않습니다.

## 최종 얼굴 블러 목적 추천

최종 목표가 "동영상 속 얼굴을 탐지해서 블러 처리"라면 아래 조합을 추천합니다.

```bash
cd /home/jmbae/DL-project/models

/home/jmbae/DL-project/venv/bin/python seojin/yolov8x.py \
  --video /home/jmbae/DL-project/models/people_crossing.mp4 \
  --output /home/jmbae/DL-project/models/output_blur_yolo26.mp4 \
  --log /home/jmbae/DL-project/models/output_blur_yolo26.log \
  --face-model /home/jmbae/yolo26x-face/runs/detect/runs/face/yolo26x_widerface/weights/best.pt
```

`tracker_arcface.py`와 `main.py`는 ArcFace 기반 identity 추적이 들어가서 훨씬 느립니다.
같은 사람을 계속 같은 ID로 유지하는 것이 중요하다면 유용하지만, 단순 얼굴 블러 목적에는 과한 편입니다.

## YOLOv8l Face와 YOLO26 Face 비교 실행

`evaluate.py`는 여러 face detector weight를 같은 동영상과 같은 파이프라인에서 반복 실행합니다.
예를 들어 `yolov8l-face-lindevs.pt`와 학습한 `yolo26` face 모델을 전체 3개 파이프라인에서 비교하려면:

### 비교 대상 face detector weight

- `yolov8l-face-lindevs.pt`
  - YOLOv8l 기반 face pretrained weight입니다.
  - 현재 프로젝트에서는 외부 pretrained face detector baseline으로 사용합니다.
  - 권장 위치:

```text
/home/jmbae/DL-project/models/telle/face_tracking/weights/yolov8l-face-lindevs.pt
```

- `yolo26x_widerface/weights/best.pt`
  - [`yolo26x-face`](https://github.com/jiminbae/yolo26x-face) repo에서 WIDERFace 데이터셋으로 학습한 YOLO26x face detector weight입니다.
  - 현재 비교에서 직접 학습한 face detector로 사용합니다.
  - 권장 위치:

```text
/home/jmbae/yolo26x-face/runs/detect/runs/face/yolo26x_widerface/weights/best.pt
```

```bash
cd /home/jmbae/DL-project/models

/home/jmbae/DL-project/venv/bin/python evaluate.py \
  --mode compare \
  --face-models \
  yolov8l=/home/jmbae/DL-project/models/telle/face_tracking/weights/yolov8l-face-lindevs.pt \
  yolo26=/home/jmbae/yolo26x-face/runs/detect/runs/face/yolo26x_widerface/weights/best.pt
```

특정 파이프라인만 비교하려면:

```bash
/home/jmbae/DL-project/venv/bin/python evaluate.py \
  --mode single \
  --model telle \
  --face-models \
  yolov8l=/home/jmbae/DL-project/models/telle/face_tracking/weights/yolov8l-face-lindevs.pt \
  yolo26=/home/jmbae/yolo26x-face/runs/detect/runs/face/yolo26x_widerface/weights/best.pt
```

사용 가능한 모델 키:

- `seojin`: 빠른 YOLO face tracking 및 블러 파이프라인
- `telle`: YOLO face detector + BoTSORT + ArcFace identity 파이프라인
- `minhyung`: person detector + face matching + ArcFace identity 파이프라인

## 평가 리포트 컬럼 의미

`evaluate.py`는 아래와 같은 표를 출력합니다.

- `model`: 실행된 파이프라인 이름
- `face`: 사용한 face detector weight 이름
- `status`: 실행 성공 여부
- `sec`: 실행 시간
- `frames`: 로그에 기록된 처리 프레임 수
- `miss`: 입력 동영상 메타데이터 프레임 수에서 로그 프레임 수를 뺀 값
- `det`: 로그에 기록된 detection 총합
- `trk`: 로그에 기록된 tracking box 총합
- `trk_id`: 고유 track ID 개수
- `face_id`: 고유 Face ID 개수
- `face%`: track line 중 `None`이 아닌 FaceID가 붙은 비율

주의: 이 리포트는 진짜 정확도 평가가 아닙니다. ground truth annotation이 없기 때문에
precision, recall, IDF1, MOTA, ID switch, mAP 같은 지표는 계산하지 못합니다.
현재 리포트는 실행 일관성, 로그 상태, 파이프라인 동작 양상을 확인하기 위한 진단표입니다.

## 로그 저장

터미널 출력까지 파일로 남기려면 `tee`를 사용합니다.

```bash
/home/jmbae/DL-project/venv/bin/python evaluate.py --mode compare 2>&1 | \
  tee evaluation_results/evaluate_compare_$(date +%Y%m%d_%H%M%S).log
```

## Git에 올리지 않는 파일

아래 파일들은 `.gitignore`로 제외됩니다.

- `*.mp4`
- `*.log`
- `*.pt`
- `__pycache__/`
- `evaluation_results/`

대용량 모델 weight와 평가 결과 영상은 로컬에만 두고, git에는 코드만 올립니다.

---

# Face Detection and Blur Model Evaluation

This directory contains several video processing pipelines for face detection,
face blurring, and detector-weight comparison.

## Directory Overview

- `evaluate.py`: runs model scripts and prints a log-based diagnostic report.
- `seojin/yolov8x.py`: YOLO face tracking pipeline. This is the fastest and most practical option for direct face blurring.
- `telle/face_tracking/tracker_arcface.py`: YOLO face detector + BoTSORT + ArcFace stable identity pipeline.
- `minhyung/main.py`: person detector + face detector + ArcFace matching pipeline.
- `people_crossing.mp4`: local evaluation video. This file is not tracked by git.
- `evaluation_results/`: generated videos and logs. This directory is not tracked by git.

## Recommended Pipeline for Face Blur

If the final goal is to detect faces in a video and blur them, use:

```bash
cd /home/jmbae/DL-project/models

/home/jmbae/DL-project/venv/bin/python seojin/yolov8x.py \
  --video /home/jmbae/DL-project/models/people_crossing.mp4 \
  --output /home/jmbae/DL-project/models/output_blur_yolo26.mp4 \
  --log /home/jmbae/DL-project/models/output_blur_yolo26.log \
  --face-model /home/jmbae/yolo26x-face/runs/detect/runs/face/yolo26x_widerface/weights/best.pt
```

`tracker_arcface.py` and `main.py` are much slower because they include
ArcFace-based identity tracking. They are useful when stable person identity is
important, but they are heavier than necessary for simple face blurring.

## Compare YOLOv8l Face and YOLO26 Face

`evaluate.py` runs multiple face detector weights under the same video and model
pipeline. To compare `yolov8l-face-lindevs.pt` with the trained `yolo26` face
model across all three pipelines:

### Face Detector Weights

- `yolov8l-face-lindevs.pt`
  - A YOLOv8l-based pretrained face detector weight.
  - It is used as the external pretrained face detector baseline in this project.
  - Recommended location:

```text
/home/jmbae/DL-project/models/telle/face_tracking/weights/yolov8l-face-lindevs.pt
```

- `yolo26x_widerface/weights/best.pt`
  - A YOLO26x face detector trained on WIDERFace in the [`yolo26x-face`](https://github.com/jiminbae/yolo26x-face) repository.
  - It is used as the custom trained face detector in this comparison.
  - Recommended location:

```text
/home/jmbae/yolo26x-face/runs/detect/runs/face/yolo26x_widerface/weights/best.pt
```

```bash
cd /home/jmbae/DL-project/models

/home/jmbae/DL-project/venv/bin/python evaluate.py \
  --mode compare \
  --face-models \
  yolov8l=/home/jmbae/DL-project/models/telle/face_tracking/weights/yolov8l-face-lindevs.pt \
  yolo26=/home/jmbae/yolo26x-face/runs/detect/runs/face/yolo26x_widerface/weights/best.pt
```

To compare only one pipeline:

```bash
/home/jmbae/DL-project/venv/bin/python evaluate.py \
  --mode single \
  --model telle \
  --face-models \
  yolov8l=/home/jmbae/DL-project/models/telle/face_tracking/weights/yolov8l-face-lindevs.pt \
  yolo26=/home/jmbae/yolo26x-face/runs/detect/runs/face/yolo26x_widerface/weights/best.pt
```

Available model keys:

- `seojin`: fast YOLO face tracking and blur pipeline
- `telle`: YOLO face detector + BoTSORT + ArcFace identity pipeline
- `minhyung`: person detector + face matching + ArcFace identity pipeline

## Report Columns

`evaluate.py` prints a table with these columns:

- `model`: pipeline name
- `face`: face detector weight name
- `status`: script exit status
- `sec`: runtime in seconds
- `frames`: processed frames found in the log
- `miss`: input video metadata frame count minus logged frames
- `det`: total detections recorded in the log
- `trk`: total tracked boxes recorded in the log
- `trk_id`: number of unique track IDs
- `face_id`: number of unique Face IDs
- `face%`: percentage of track lines that received a non-`None` FaceID

Important: this report is not a true accuracy benchmark. There is no ground
truth annotation, so it cannot compute precision, recall, IDF1, MOTA, ID
switches, or mAP. Treat it as a diagnostic report for run consistency, log
health, and pipeline behavior.

## Save Terminal Logs

Use `tee` if terminal output should be saved:

```bash
/home/jmbae/DL-project/venv/bin/python evaluate.py --mode compare 2>&1 | \
  tee evaluation_results/evaluate_compare_$(date +%Y%m%d_%H%M%S).log
```

## Files Ignored by Git

The following generated or large files are ignored:

- `*.mp4`
- `*.log`
- `*.pt`
- `__pycache__/`
- `evaluation_results/`

Keep large model weights and generated result videos local. Commit code only.

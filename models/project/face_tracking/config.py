from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# 입출력 경로
VIDEO_PATH = str(BASE_DIR / "videos/test.mp4")
FACE_MODEL_PATH = str(BASE_DIR / "weights" / "yolo26x-face.pt")
REID_MODEL_PATH = str(BASE_DIR.parent / "boxmot" / "models" / "osnet_x0_25_msmt17.pt")

TARGET_DIR = str(BASE_DIR / "target")
TARGET_PATTERN = "target*"
TARGET_IMAGE_PATH = str(BASE_DIR / "virtual_face" / "fake_face.jpg")

OUTPUT_PATH = str(BASE_DIR / "outputs/result/output_target1.mp4")
LOG_PATH = str(BASE_DIR / "outputs/log/tracking_target1_log.txt")
CROP_ROOT = str(BASE_DIR / "outputs/crop/crop1")
METADATA_PATH = str(BASE_DIR / "outputs/metadata/face_metadata1.json")

LIVEPORTRAIT_DIR = str(PROJECT_DIR / "LivePortrait")

ENABLE_FACE_SWAP = True
STITCH_BLUR_KERNEL = 21

device = "cuda"

# 얼굴 인식 임계값
SIM_THRESHOLD = 0.38
TARGET_THRESHOLD = 0.50
SMOOTH_ALPHA = 0.8
MAX_FACE_AGE = 300
MIN_FACE_AREA = 400
MAX_ASPECT_RATIO = 2.2

# LivePortrait 품질 기준
LIVEPORTRAIT_MIN_FACE_AREA = 2500
LIVEPORTRAIT_MIN_CROP_SIZE = 64
LIVEPORTRAIT_MAX_ASPECT_RATIO = 2.5
EDGE_MARGIN = 2

# 성능 최적화 설정
EMBEDDING_REFRESH_INTERVAL = 5   # N프레임마다 embedding 재계산
LOG_EVERY_N_FRAMES = 30          # N프레임마다 상세 로그
CROP_WRITER_WORKERS = 2          # 비동기 crop 저장 worker 수

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

OUTPUT_PATH = str(BASE_DIR / "outputs/result/output_target5.mp4")
LOG_PATH = str(BASE_DIR / "outputs/log/tracking_target5_log.txt")
METADATA_PATH = str(BASE_DIR / "outputs/metadata/face_metadata5.json")
CROP_ROOT = str(BASE_DIR / "outputs/crop/crop5")
LIVEPORTRAIT_DIR = str(PROJECT_DIR / "LivePortrait")

ENABLE_FACE_SWAP = True
STITCH_BLUR_KERNEL = 21
FACE_SWAP_BATCH_SIZE = 32

device = "cuda"

# 얼굴 인식 / Target 판별 기준
SIM_THRESHOLD = 0.38
TARGET_THRESHOLD = 0.50
SMOOTH_ALPHA = 0.55
SMOOTH_RESET_IOU_THRESHOLD = 0.15
SMOOTH_RESET_CENTER_DISTANCE_RATIO = 0.45
MAX_FACE_AGE = 300

# Target 유지 프레임 수
TARGET_HOLD_FRAMES = 60

# Detection 필터 기준
MIN_FACE_AREA = 500
MAX_ASPECT_RATIO = 2.2

# Detection 설정
DETECTION_CONF = 0.40
FULL_DETECT_IMGSZ = 768
TILE_DETECT_IMGSZ = 640
TILE_DETECT_INTERVAL = 3
NMS_IOU_THRESHOLD = 0.35
MAX_DETECTIONS_PER_FRAME = 60

# BoT-SORT Tracking 설정
TRACK_BUFFER = 150
MATCH_THRESH = 0.55
PROXIMITY_THRESH = 0.65
APPEARANCE_THRESH = 0.45
USE_BOTSORT_REID = False

# LivePortrait 품질 기준
LIVEPORTRAIT_MIN_FACE_AREA = 2500
LIVEPORTRAIT_MIN_CROP_SIZE = 64
LIVEPORTRAIT_MAX_ASPECT_RATIO = 2.5

# 너무 작은 얼굴 swap 방지
LIVEPORTRAIT_MIN_FACE_SIZE = 90

# 화면 전체 가까운 큰 얼굴 swap 방지
LIVEPORTRAIT_MAX_FACE_AREA_RATIO = 0.12

# 프레임 가장자리 얼굴 swap 방지
EDGE_MARGIN = 40

# 성능 최적화 설정
EMBEDDING_REFRESH_INTERVAL = 5
LOG_EVERY_N_FRAMES = 10

# crop 저장 worker 수
CROP_WRITER_WORKERS = 2

# swap flicker 방지
SWAP_HOLD_FRAMES = 5
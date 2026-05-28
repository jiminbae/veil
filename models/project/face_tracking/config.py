import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


def _next_output_index():
    outputs_dir = BASE_DIR / "outputs"
    patterns = (
        ("result", r"output_target(\d+)\.mp4$"),
        ("log", r"tracking_target(\d+)_log\.txt$"),
        ("metadata", r"face_metadata(\d+)\.json$"),
        ("metadata", r"tracking_metadata(\d+)\.json$"),
        ("crop", r"crop(\d+)$"),
    )
    max_index = 0

    for subdir, pattern in patterns:
        search_dir = outputs_dir / subdir

        if not search_dir.exists():
            continue

        regex = re.compile(pattern)

        for path in search_dir.iterdir():
            match = regex.match(path.name)

            if match:
                max_index = max(max_index, int(match.group(1)))

    return max_index + 1


RUN_INDEX = _next_output_index()


VIDEO_PATH = str(BASE_DIR / "videos/test.mp4")
FACE_MODEL_PATH = str(BASE_DIR / "weights" / "yolo26x-face.pt")
REID_MODEL_PATH = str(BASE_DIR.parent / "boxmot" / "models" / "osnet_x0_25_msmt17.pt")

TARGET_DIR = str(BASE_DIR / "target")
TARGET_PATTERN = "target*"
TARGET_IMAGE_PATH = str(BASE_DIR / "virtual_face" / "fake_face.jpg")

OUTPUT_PATH = str(BASE_DIR / f"outputs/result/output_target{RUN_INDEX}.mp4")
LOG_PATH = str(BASE_DIR / f"outputs/log/tracking_target{RUN_INDEX}_log.txt")
METADATA_PATH = str(BASE_DIR / f"outputs/metadata/face_metadata{RUN_INDEX}.json")
TRACKING_METADATA_PATH = str(BASE_DIR / f"outputs/metadata/tracking_metadata{RUN_INDEX}.json")
CROP_ROOT = str(BASE_DIR / f"outputs/crop/crop{RUN_INDEX}")

INSWAPPER_MODEL_PATH = str(BASE_DIR / "weights" / "inswapper_128.onnx")
INSWAPPER_DET_SIZE = (640, 640)

ENABLE_MOUTH_PASTE = True
MOUTH_WIDTH_PAD = 1.3
MOUTH_HEIGHT_RATIO = 0.75
MOUTH_BLUR_SIZE = 31

ENABLE_FACE_SWAP = True
STITCH_BLUR_KERNEL = 21
FACE_SWAP_BATCH_SIZE = 8
MAX_SWAP_FACES_PER_FRAME = 2

device = "cuda"


SIM_THRESHOLD = 0.38
TARGET_THRESHOLD = 0.50
SMOOTH_ALPHA = 0.65
ID_MIN_FACE_SIZE = 80
SMOOTH_RESET_IOU_THRESHOLD = 0.15
SMOOTH_RESET_CENTER_DISTANCE_RATIO = 0.45
MAX_FACE_AGE = 300


TARGET_HOLD_FRAMES = 60


MIN_FACE_AREA = 500
MAX_ASPECT_RATIO = 2.2


DETECTION_CONF = 0.40
FULL_DETECT_IMGSZ = 768
TILE_DETECT_IMGSZ = 640
TILE_DETECT_INTERVAL = 3
NMS_IOU_THRESHOLD = 0.35
MAX_DETECTIONS_PER_FRAME = 60


TRACK_BUFFER = 150
MATCH_THRESH = 0.55
PROXIMITY_THRESH = 0.65
APPEARANCE_THRESH = 0.45
USE_BOTSORT_REID = False

SWAP_MIN_FACE_AREA = 2500
SWAP_MIN_CROP_SIZE = 64
SWAP_MAX_ASPECT_RATIO = 2.5
SWAP_MIN_FACE_SIZE = 110
SWAP_MAX_FACE_AREA_RATIO = 0.08


EDGE_MARGIN = 40


EMBEDDING_REFRESH_INTERVAL = 5
LOG_EVERY_N_FRAMES = 10


SWAP_HOLD_FRAMES = 5


ENABLE_POSE_FALLBACK = True

SIDE_FACE_CENTER_RATIO_THRESHOLD = 0.38
SIDE_FACE_ASPECT_RATIO_THRESHOLD = 1.75


ENABLE_OCCLUSION_FALLBACK = True


ENABLE_MASK_BLEND = True
SWAP_FEATHER_RATIO = 0.08
SWAP_MASK_BLUR_KERNEL = 21

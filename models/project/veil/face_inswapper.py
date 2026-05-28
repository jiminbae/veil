import logging
import cv2
import numpy as np
from time import perf_counter
from pathlib import Path

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from face_identifier import face_app, FACE_PROVIDERS

try:
    from config import INSWAPPER_MODEL_PATH
except ImportError:
    INSWAPPER_MODEL_PATH = str(
        Path(__file__).resolve().parent / "weights" / "inswapper_128.onnx"
    )

try:
    from config import INSWAPPER_DET_SIZE
except ImportError:
    INSWAPPER_DET_SIZE = (640, 640)

try:
    from config import ENABLE_MOUTH_PASTE
except ImportError:
    ENABLE_MOUTH_PASTE = True

try:
    from config import MOUTH_WIDTH_PAD
except ImportError:
    MOUTH_WIDTH_PAD = 1.3

try:
    from config import MOUTH_HEIGHT_RATIO
except ImportError:
    MOUTH_HEIGHT_RATIO = 0.75

try:
    from config import MOUTH_BLUR_SIZE
except ImportError:
    MOUTH_BLUR_SIZE = 31


def _iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    areaA = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    areaB = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    return inter / (areaA + areaB - inter + 1e-9)


def _center_score(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    acx = (ax1 + ax2) * 0.5
    acy = (ay1 + ay2) * 0.5

    bcx = (bx1 + bx2) * 0.5
    bcy = (by1 + by2) * 0.5

    aw = max(1.0, ax2 - ax1)
    ah = max(1.0, ay2 - ay1)

    dx = abs(acx - bcx) / aw
    dy = abs(acy - bcy) / ah

    return 1.0 - min(1.0, (dx + dy) * 0.5)


def _match_faces_to_bboxes(detected_faces, target_bboxes, iou_thresh=0.25):
    result = [None] * len(target_bboxes)
    used = [False] * len(detected_faces)

    for ti, tbbox in enumerate(target_bboxes):
        best_score = iou_thresh
        best_fi = -1

        for fi, face in enumerate(detected_faces):
            if used[fi]:
                continue

            fb = face.bbox
            iou_score = _iou(
                [float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3])],
                [float(tbbox[0]), float(tbbox[1]), float(tbbox[2]), float(tbbox[3])],
            )

            center_score = _center_score(
                [float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3])],
                [float(tbbox[0]), float(tbbox[1]), float(tbbox[2]), float(tbbox[3])],
            )

            score = 0.7 * iou_score + 0.3 * center_score

            if score > best_score:
                best_score = score
                best_fi = fi

        if best_fi >= 0:
            result[ti] = detected_faces[best_fi]
            used[best_fi] = True

    return result


def _build_ellipse_mask(frame_shape, bbox):
    H, W = frame_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return np.zeros((H, W, 3), dtype=np.float32)

    mask = np.zeros((H, W), dtype=np.uint8)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    rx = max(1, int(bw * 0.46))
    ry = max(1, int(bh * 0.52))

    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    blur_size = max(21, int(max(bw, bh) * 0.08))
    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1

    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    return np.stack([mask] * 3, axis=-1).astype(np.float32) / 255.0


def _build_mouth_mask_roi(
    frame_shape,
    target_face,
    width_pad=MOUTH_WIDTH_PAD,
    height_ratio=MOUTH_HEIGHT_RATIO,
    blur_size=MOUTH_BLUR_SIZE,
):
    H, W = frame_shape[:2]

    kps = getattr(target_face, "kps", None)
    if kps is None or len(kps) < 5:
        return None

    kps = np.asarray(kps, dtype=np.float32)

    mouth_l = kps[3]
    mouth_r = kps[4]

    cx = float((mouth_l[0] + mouth_r[0]) * 0.5)
    cy = float((mouth_l[1] + mouth_r[1]) * 0.5)

    width = float(np.linalg.norm(mouth_r - mouth_l))
    if width < 4:
        return None

    ellipse_w = width * width_pad
    ellipse_h = width * height_ratio

    rx = max(1, int(round(ellipse_w * 0.5)))
    ry = max(1, int(round(ellipse_h * 0.5)))

    b = blur_size if blur_size % 2 == 1 else blur_size + 1
    pad = b + 2

    x1 = max(0, int(np.floor(cx - rx - pad)))
    y1 = max(0, int(np.floor(cy - ry - pad)))
    x2 = min(W, int(np.ceil(cx + rx + pad)))
    y2 = min(H, int(np.ceil(cy + ry + pad)))

    if x2 <= x1 or y2 <= y1:
        return None

    angle = float(
        np.degrees(
            np.arctan2(
                mouth_r[1] - mouth_l[1],
                mouth_r[0] - mouth_l[0],
            )
        )
    )

    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

    cv2.ellipse(
        mask,
        (int(round(cx - x1)), int(round(cy - y1))),
        (rx, ry),
        angle,
        0,
        360,
        255,
        -1,
    )

    mask = cv2.GaussianBlur(mask, (b, b), 0).astype(np.float32) / 255.0

    return x1, y1, x2, y2, mask


class FaceSwapper:
    def __init__(self, target_image_path: str, device: str = "cpu"):
        self._ctx_id = 0 if "CUDAExecutionProvider" in FACE_PROVIDERS else -1
        self._app = face_app

        model_path = Path(INSWAPPER_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"inswapper 모델을 찾을 수 없습니다: {model_path}\n"
                "weights/inswapper_128.onnx 파일을 확인하세요."
            )

        print(f"[FaceSwapper] Loading inswapper: {model_path}")
        self._swapper = get_model(str(model_path), providers=FACE_PROVIDERS)

        target_img = cv2.imread(str(target_image_path))
        if target_img is None:
            raise RuntimeError(f"Cannot read target image: {target_image_path}")

        self._source_face = self._detect_source_face(target_img)

        print(f"[FaceSwapper] Source face loaded: {target_image_path}")


    def _detect_with_det_size(self, img_bgr, det_size):
        try:
            if not hasattr(self, "_source_apps"):
                self._source_apps = {}

            if det_size not in self._source_apps:
                source_app = FaceAnalysis(name="buffalo_l", providers=FACE_PROVIDERS)
                source_app.prepare(ctx_id=self._ctx_id, det_size=det_size)
                self._source_apps[det_size] = source_app

            return self._source_apps[det_size].get(img_bgr)

        except Exception:
            logging.exception(f"[FaceSwapper] source detection failed at det_size={det_size}")
            return []

    def _detect_source_face(self, img_bgr):
        h, w = img_bgr.shape[:2]

        candidates = [("original", img_bgr)]

        scale = 640 / max(h, w)
        if abs(scale - 1.0) > 0.05:
            resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
            candidates.append(("resize640", resized))

        pad = max(h, w) // 2
        padded = cv2.copyMakeBorder(
            img_bgr,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_REPLICATE,
        )
        candidates.append(("padded", padded))

        det_sizes = [
            INSWAPPER_DET_SIZE,
            (1024, 1024),
        ]

        for det_size in det_sizes:
            for tag, candidate in candidates:
                faces = self._detect_with_det_size(candidate, det_size)
                if faces:
                    best = max(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    )
                    print(f"[FaceSwapper] Source face detected: {tag}, det_size={det_size}")
                    return best

        raise RuntimeError(
            "소스 이미지에서 얼굴을 검출하지 못했습니다. "
            "정면 얼굴이고 배경이 단순한 이미지를 사용하세요."
        )

    def swap_into_frame(
        self,
        frame_bgr,
        bbox,
        landmarks=None,
        target_kps=None,
    ):
        result_bgr, success_flags, masks, _ = self.swap_many_into_frame(
            frame_bgr,
            [bbox],
            landmarks_list=[landmarks],
            target_kps_list=[target_kps],
            batch_size=1,
        )

        if not success_flags[0]:
            return None, None

        return result_bgr, masks[0]

    def swap_many_into_frame(
        self,
        frame_bgr,
        bboxes,
        landmarks_list=None,
        target_kps_list=None,
        batch_size=16,
    ):
        total_started = perf_counter()

        success_flags = [False] * len(bboxes)
        masks = [None] * len(bboxes)

        if not bboxes:
            return frame_bgr, success_flags, masks, {
                "prepare_sec": 0.0,
                "inference_sec": 0.0,
                "paste_sec": 0.0,
                "total_sec": perf_counter() - total_started,
            }

        prepare_started = perf_counter()

        try:
            detected_faces = self._app.get(frame_bgr)
        except Exception:
            logging.exception("[FaceSwapper] face detection failed")
            detected_faces = []

        matched_faces = _match_faces_to_bboxes(detected_faces, bboxes)
        prepare_elapsed = perf_counter() - prepare_started

        result_bgr = frame_bgr.copy()
        inference_elapsed = 0.0
        paste_elapsed = 0.0

        for i, (bbox, face) in enumerate(zip(bboxes, matched_faces)):
            if face is None:
                continue

            try:
                inf_started = perf_counter()

                swapped_bgr = self._swapper.get(
                    result_bgr,
                    face,
                    self._source_face,
                    paste_back=True,
                )

                if ENABLE_MOUTH_PASTE:
                    mouth_roi = _build_mouth_mask_roi(frame_bgr.shape, face)

                    if mouth_roi is not None:
                        x1, y1, x2, y2, mouth_mask = mouth_roi
                        result_bgr = swapped_bgr
                        m3 = mouth_mask[:, :, None]
                        original_roi = frame_bgr[y1:y2, x1:x2].astype(np.float32)
                        swapped_roi = swapped_bgr[y1:y2, x1:x2].astype(np.float32)

                        result_bgr[y1:y2, x1:x2] = (
                            m3 * original_roi
                            + (1.0 - m3) * swapped_roi
                        ).astype(np.uint8)
                    else:
                        result_bgr = swapped_bgr
                else:
                    result_bgr = swapped_bgr

                inference_elapsed += perf_counter() - inf_started

                paste_started = perf_counter()
                masks[i] = _build_ellipse_mask(frame_bgr.shape, bbox)
                paste_elapsed += perf_counter() - paste_started

                success_flags[i] = True

            except Exception:
                logging.exception(f"[FaceSwapper] swap error at bbox index {i}")

        timings = {
            "prepare_sec": prepare_elapsed,
            "inference_sec": inference_elapsed,
            "paste_sec": paste_elapsed,
            "total_sec": perf_counter() - total_started,
        }

        return result_bgr, success_flags, masks, timings

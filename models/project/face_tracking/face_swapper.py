# face_swapper.py
# LivePortrait의 내장 stitching + paste_back을 사용하는 face swap
import cv2
import numpy as np
import torch
from pathlib import Path
import sys

# C:\Workspace\DL-project\LivePortrait 가 import path에 들어가야 함
# face_swapper.py 위치: DL-project/models/face_vision/face_swapper.py
# → .parent.parent.parent = DL-project/
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from LivePortrait.src.live_portrait_pipeline import LivePortraitPipeline
from LivePortrait.src.utils.cropper import Cropper
from LivePortrait.src.utils.crop import prepare_paste_back, paste_back
from LivePortrait.src.config.inference_config import InferenceConfig
from LivePortrait.src.config.crop_config import CropConfig


class LPProcessor:
    """
    set_source(fake_face): 한 번만 호출 → f_s, x_s 캐시
    swap_into_frame(frame, bbox): 매 프레임 호출 → swap된 전체 프레임 + 마스크 반환
    """

    def __init__(self, device_type="cpu"):
        is_cuda = (device_type == "cuda" and torch.cuda.is_available())

        # 🔑 핵심 플래그: stitching + pasteback 활성화
        self.inference_cfg = InferenceConfig(
            flag_force_cpu=not is_cuda,
            flag_use_half_precision=is_cuda,
            flag_stitching=True,         # 키포인트 자연스럽게 보정
            flag_pasteback=True,         # 원본 프레임 합성
            flag_do_crop=True,
            flag_relative_motion=True,   # source 외형 보존, 표정만 가져옴
            flag_normalize_lip=True,
            flag_lip_retargeting=False,
            flag_eye_retargeting=False,
            flag_do_rot=True,
        )
        self.crop_cfg = CropConfig()

        device_name = "GPU" if is_cuda else "CPU"
        print(f"[FaceSwapper] Initializing LivePortrait on: {device_name}")

        self.pipeline = LivePortraitPipeline(
            inference_cfg=self.inference_cfg,
            crop_cfg=self.crop_cfg,
        )
        self.wrapper = self.pipeline.live_portrait_wrapper
        self.cropper = Cropper(crop_cfg=self.crop_cfg)

        # source(fake_face) 캐시 — 한 번만 계산
        self.f_s = None            # 3D appearance feature
        self.x_s = None            # transformed source keypoints
        self.x_s_info = None       # raw kp info

    def set_source(self, image_bgr):
        """fake_face.jpg를 한 번 분석해 f_s와 x_s 캐시."""
        if image_bgr is None:
            raise RuntimeError("source image is None")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        crop_info = self.cropper.crop_source_image(image_rgb, self.crop_cfg)
        if crop_info is None or 'img_crop_256x256' not in crop_info:
            raise RuntimeError("No face detected in source image.")

        source_crop = crop_info['img_crop_256x256']  # RGB 256x256

        with torch.inference_mode():
            I_s = self.wrapper.prepare_source(source_crop)
            self.f_s = self.wrapper.extract_feature_3d(I_s)
            self.x_s_info = self.wrapper.get_kp_info(I_s)
            # 🔑 pose/expression 적용한 실제 워핑용 키포인트
            self.x_s = self.wrapper.transform_keypoint(self.x_s_info)

        print("[FaceSwapper] Source features cached (f_s, x_s).")

    def swap_into_frame(self, frame_bgr, bbox, landmarks=None):
        """
        프레임의 bbox 얼굴을 source(fake_face)로 swap한 전체 프레임 + 마스크 반환.
        실패 시 (None, None).

        Parameters
        ----------
        frame_bgr : np.ndarray HxWx3 BGR
        bbox : [x1, y1, x2, y2]
        landmarks : optional Nx2 array (insightface의 landmark_2d_106 등)
            제공되면 얼굴 모양 정밀 마스크 사용 (액자 효과 제거).
            None이면 기본 oval mask + erosion 사용.

        Returns
        -------
        result_bgr : np.ndarray HxWx3 uint8
        mask_ori   : np.ndarray HxWx3 float32 (0~1) — 합성 가중치
        """
        if self.f_s is None:
            raise RuntimeError("set_source()가 호출되지 않음.")

        x1, y1, x2, y2 = map(int, bbox)
        H, W = frame_bgr.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return None, None

        # LivePortrait cropper가 랜드마크 잡으려면 컨텍스트 여유 필요 (1.0x 패딩)
        pad_w, pad_h = bw, bh
        px1, py1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
        px2, py2 = min(W, x2 + pad_w), min(H, y2 + pad_h)
        face_region_bgr = frame_bgr[py1:py2, px1:px2]
        if face_region_bgr.size == 0:
            return None, None

        face_region_rgb = cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2RGB)

        try:
            # 1. driving 영역에 cropper 적용 → 256x256 crop + M_c2o(역변환 행렬)
            crop_info_d = self.cropper.crop_source_image(
                face_region_rgb, self.crop_cfg
            )
            if crop_info_d is None or 'img_crop_256x256' not in crop_info_d:
                return None, None

            driving_crop = crop_info_d['img_crop_256x256']
            M_c2o_local = crop_info_d['M_c2o']  # 3x3, crop -> face_region 좌표계

            with torch.inference_mode():
                # 2. driving keypoints (현재 프레임 인물의 표정/포즈)
                I_d = self.wrapper.prepare_source(driving_crop)
                x_d_info = self.wrapper.get_kp_info(I_d)
                x_d = self.wrapper.transform_keypoint(x_d_info)

                # 3. 🔑 stitching — driving 키포인트를 source 모양에 자연스럽게 맞춤
                x_d_new = self.wrapper.stitching(self.x_s, x_d)

                # 4. 워핑 + generator (source 외형 + driving 표정)
                out = self.wrapper.warp_decode(self.f_s, self.x_s, x_d_new)
                I_p = self.wrapper.parse_output(out['out'])[0]  # 256x256 RGB uint8

            # 5. M_c2o는 face_region 로컬 좌표계 → 전체 frame 좌표계로 평행이동 보정
            M_c2o_full = M_c2o_local.copy().astype(np.float32)
            M_c2o_full[0, 2] += px1
            M_c2o_full[1, 2] += py1

            # 6. paste_back 마스크 준비
            if landmarks is not None and len(landmarks) >= 33:
                # 🆕 방법 3: landmark 기반 정밀 마스크 (타이트하게)
                # 얼굴 landmark의 convex hull → 눈/코/입/턱선 영역에만 mask 적용.
                # 이마 확장 없음! source(fake_face)의 배경/머리카락이 안 들어가게 함.
                pts = np.asarray(landmarks, dtype=np.int32)
                hull = cv2.convexHull(pts)

                custom_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillConvexPoly(custom_mask, hull, 255)

                # 살짝 안쪽으로 깎아서 가장자리에 source 배경 안 보이게
                face_size = max(bw, bh)
                erode_size = max(7, face_size // 25)
                erode_size = erode_size if erode_size % 2 == 1 else erode_size + 1
                custom_mask = cv2.erode(
                    custom_mask, np.ones((erode_size, erode_size), np.uint8)
                )

                # 가장자리 부드럽게 (얼굴 안→밖 그라데이션)
                blur_size = max(31, int(face_size * 0.10))
                blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
                custom_mask = cv2.GaussianBlur(custom_mask, (blur_size, blur_size), 0)

                mask_ori = np.stack([custom_mask] * 3, axis=-1).astype(np.float32) / 255.0

            else:
                # 기본 fallback: 정적 oval mask + erosion
                mask_ori = prepare_paste_back(
                    self.inference_cfg.mask_crop,
                    M_c2o_full,
                    dsize=(W, H),
                )
                face_size = max(bw, bh)
                erode_size = max(15, int(face_size * 0.10))
                erode_size = erode_size if erode_size % 2 == 1 else erode_size + 1
                blur_size = max(31, int(face_size * 0.15))
                blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
                mask_uint8 = (mask_ori * 255).clip(0, 255).astype(np.uint8)
                kernel = np.ones((erode_size, erode_size), np.uint8)
                mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=2)
                mask_uint8 = cv2.GaussianBlur(mask_uint8, (blur_size, blur_size), 0)
                mask_ori = mask_uint8.astype(np.float32) / 255.0

            # 7. 🔑 paste_back — affine 역변환 + 마스크 알파블렌딩
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result_rgb = paste_back(I_p, M_c2o_full, frame_rgb, mask_ori)
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

            return result_bgr, mask_ori

        except Exception as e:
            print(f"[FaceSwapper] swap_into_frame error: {e}")
            return None, None


class FaceSwapper:
    def __init__(self, target_image_path, device="cpu"):
        self.processor = LPProcessor(device_type=device)
        target_img = cv2.imread(target_image_path)
        if target_img is None:
            raise RuntimeError(f"Cannot read target image: {target_image_path}")
        self.processor.set_source(target_img)

    def swap_into_frame(self, frame, bbox, landmarks=None):
        """
        프레임 전체와 마스크를 반환.
        landmarks 제공 시 얼굴 모양 정밀 마스크 사용 (액자 효과 제거).
        """
        return self.processor.swap_into_frame(frame, bbox, landmarks=landmarks)


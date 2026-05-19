import cv2
import numpy as np
import torch
from time import perf_counter
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from LivePortrait.src.live_portrait_pipeline import LivePortraitPipeline
from LivePortrait.src.utils.cropper import Cropper
from LivePortrait.src.utils.crop import paste_back
from LivePortrait.src.config.inference_config import InferenceConfig
from LivePortrait.src.config.crop_config import CropConfig


class LPProcessor:
    def __init__(self, device_type="cpu"):
        is_cuda = (device_type == "cuda" and torch.cuda.is_available())

        self.inference_cfg = InferenceConfig(
            flag_force_cpu=not is_cuda,
            flag_use_half_precision=is_cuda,
            flag_stitching=True,
            flag_pasteback=True,
            flag_do_crop=True,
            flag_relative_motion=True,
            flag_normalize_lip=True,
            flag_lip_retargeting=False,
            flag_eye_retargeting=False,
            flag_do_rot=True,
            flag_do_torch_compile=False,
        )

        self.crop_cfg = CropConfig()

        device_name = "GPU" if is_cuda else "CPU"
        print(f"[FaceSwapper] Initializing LivePortrait on: {device_name}")

        if is_cuda:
            print(f"[FaceSwapper] CUDA device: {torch.cuda.get_device_name(0)}")

        self.pipeline = LivePortraitPipeline(
            inference_cfg=self.inference_cfg,
            crop_cfg=self.crop_cfg,
        )

        self.wrapper = self.pipeline.live_portrait_wrapper
        self.cropper = Cropper(crop_cfg=self.crop_cfg)

        self.f_s = None
        self.x_s = None
        self.x_s_info = None

    def set_source(self, image_bgr):
        if image_bgr is None:
            raise RuntimeError("source image is None")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        crop_info = self.cropper.crop_source_image(image_rgb, self.crop_cfg)

        if crop_info is None or "img_crop_256x256" not in crop_info:
            raise RuntimeError("No face detected in source image.")

        source_crop = crop_info["img_crop_256x256"]

        with torch.inference_mode():
            I_s = self.wrapper.prepare_source(source_crop)
            self.f_s = self.wrapper.extract_feature_3d(I_s)
            self.x_s_info = self.wrapper.get_kp_info(I_s)
            self.x_s = self.wrapper.transform_keypoint(self.x_s_info)

        print("[FaceSwapper] Source features cached.")

    def _prepare_driving_item(self, frame_bgr, bbox, landmarks=None):
        x1, y1, x2, y2 = map(int, bbox)
        H, W = frame_bgr.shape[:2]

        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            return None

        pad_w = int(bw * 0.45)
        pad_h = int(bh * 0.45)

        px1 = max(0, x1 - pad_w)
        py1 = max(0, y1 - pad_h)
        px2 = min(W, x2 + pad_w)
        py2 = min(H, y2 + pad_h)

        face_region_bgr = frame_bgr[py1:py2, px1:px2]

        if face_region_bgr.size == 0:
            return None

        face_region_rgb = cv2.cvtColor(face_region_bgr, cv2.COLOR_BGR2RGB)

        crop_info_d = self.cropper.crop_source_image(
            face_region_rgb,
            self.crop_cfg
        )

        if crop_info_d is None or "img_crop_256x256" not in crop_info_d:
            return None

        M_c2o_full = crop_info_d["M_c2o"].copy().astype(np.float32)
        M_c2o_full[0, 2] += px1
        M_c2o_full[1, 2] += py1

        return {
            "bbox": [x1, y1, x2, y2],
            "landmarks": landmarks,
            "driving_crop": crop_info_d["img_crop_256x256"],
            "M_c2o_full": M_c2o_full,
        }

    def _prepare_batch_tensor(self, batch_items):
        crops = np.stack(
            [item["driving_crop"] for item in batch_items],
            axis=0
        )

        x = crops.astype(np.float32) / 255.0
        x = np.clip(x, 0, 1)
        x = torch.from_numpy(x).permute(0, 3, 1, 2)

        return x.to(self.wrapper.device)

    def _get_default_roi(self, frame_shape, bbox, M_c2o_full):
        h, w = frame_shape[:2]

        corners = np.array(
            [[[0, 0], [255, 0], [255, 255], [0, 255]]],
            dtype=np.float32,
        )

        transformed = cv2.transform(corners, M_c2o_full[:2, :])[0]

        x1, y1 = np.floor(transformed.min(axis=0)).astype(int)
        x2, y2 = np.ceil(transformed.max(axis=0)).astype(int)

        bx1, by1, bx2, by2 = bbox
        face_size = max(bx2 - bx1, by2 - by1)

        blur_size = max(21, int(face_size * 0.08))
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1

        pad = blur_size + 4

        return (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(w, x2 + pad + 1),
            min(h, y2 + pad + 1),
        )

    def _paste_back_default_mask_roi(self, img_crop, M_c2o_full, frame_rgb, bbox):
        rx1, ry1, rx2, ry2 = self._get_default_roi(
            frame_rgb.shape,
            bbox,
            M_c2o_full
        )

        if rx2 <= rx1 or ry2 <= ry1:
            return frame_rgb, None

        roi_w = rx2 - rx1
        roi_h = ry2 - ry1

        M_roi = M_c2o_full.copy()
        M_roi[0, 2] -= rx1
        M_roi[1, 2] -= ry1

        mask_roi = cv2.warpAffine(
            self.inference_cfg.mask_crop,
            M_roi[:2, :],
            dsize=(roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
        ).astype(np.float32) / 255.0

        bx1, by1, bx2, by2 = bbox
        face_size = max(bx2 - bx1, by2 - by1)

        erode_size = max(25, int(face_size * 0.18))
        erode_size = erode_size if erode_size % 2 == 1 else erode_size + 1

        blur_size = max(21, int(face_size * 0.08))
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1

        mask_uint8 = (mask_roi * 255).clip(0, 255).astype(np.uint8)

        kernel = np.ones((erode_size, erode_size), np.uint8)
        mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=2)
        mask_uint8 = cv2.GaussianBlur(mask_uint8, (blur_size, blur_size), 0)

        mask_roi = mask_uint8.astype(np.float32) / 255.0

        result_roi = cv2.warpAffine(
            img_crop,
            M_roi[:2, :],
            dsize=(roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
        )

        frame_roi = frame_rgb[ry1:ry2, rx1:rx2]

        if mask_roi.ndim == 2:
            mask_roi_3ch = np.stack([mask_roi] * 3, axis=-1)
        else:
            mask_roi_3ch = mask_roi

        blended_roi = np.clip(
            mask_roi_3ch * result_roi +
            (1 - mask_roi_3ch) * frame_roi,
            0,
            255
        ).astype(np.uint8)

        frame_rgb[ry1:ry2, rx1:rx2] = blended_roi

        return frame_rgb, mask_roi

    def _build_landmark_mask(self, frame_shape, bbox, landmarks):
        H, W = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1

        pts = np.asarray(landmarks, dtype=np.int32)
        hull = cv2.convexHull(pts)

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        face_size = max(bw, bh)

        erode_size = max(7, face_size // 25)
        erode_size = erode_size if erode_size % 2 == 1 else erode_size + 1

        mask = cv2.erode(
            mask,
            np.ones((erode_size, erode_size), np.uint8),
            iterations=1
        )

        blur_size = max(31, int(face_size * 0.10))
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1

        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        return np.stack([mask] * 3, axis=-1).astype(np.float32) / 255.0

    def swap_many_into_frame(self, frame_bgr, bboxes, landmarks_list=None, batch_size=16):
        if self.f_s is None:
            raise RuntimeError("set_source()가 호출되지 않음.")

        total_started = perf_counter()
        prepare_started = perf_counter()

        if landmarks_list is None:
            landmarks_list = [None] * len(bboxes)

        prepared = []
        success_flags = [False] * len(bboxes)
        masks = [None] * len(bboxes)

        for idx, (bbox, landmarks) in enumerate(zip(bboxes, landmarks_list)):
            try:
                item = self._prepare_driving_item(
                    frame_bgr,
                    bbox,
                    landmarks
                )
            except Exception as e:
                print(f"[FaceSwapper] prepare error: {e}")
                item = None

            if item is not None:
                item["index"] = idx
                prepared.append(item)

        prepare_elapsed = perf_counter() - prepare_started

        if not prepared:
            timings = {
                "prepare_sec": prepare_elapsed,
                "inference_sec": 0.0,
                "paste_sec": 0.0,
                "total_sec": perf_counter() - total_started,
            }
            return frame_bgr, success_flags, masks, timings

        result_bgr = frame_bgr.copy()
        frame_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        batch_size = max(1, int(batch_size))

        inference_elapsed = 0.0
        paste_elapsed = 0.0

        for start_idx in range(0, len(prepared), batch_size):
            batch_items = prepared[start_idx:start_idx + batch_size]

            try:
                inference_started = perf_counter()

                with torch.inference_mode():
                    I_d = self._prepare_batch_tensor(batch_items)
                    x_d_info = self.wrapper.get_kp_info(I_d)
                    x_d = self.wrapper.transform_keypoint(x_d_info)

                    batch_count = len(batch_items)

                    x_s_batch = self.x_s.expand(batch_count, -1, -1)
                    f_s_batch = self.f_s.expand(
                        batch_count,
                        -1,
                        -1,
                        -1,
                        -1
                    )

                    x_d_new = self.wrapper.stitching(x_s_batch, x_d)

                    out = self.wrapper.warp_decode(
                        f_s_batch,
                        x_s_batch,
                        x_d_new
                    )

                    outputs = self.wrapper.parse_output(out["out"])

                inference_elapsed += perf_counter() - inference_started

                paste_started = perf_counter()

                for item, I_p in zip(batch_items, outputs):
                    if item["landmarks"] is not None and len(item["landmarks"]) >= 33:
                        mask_ori = self._build_landmark_mask(
                            frame_rgb.shape,
                            item["bbox"],
                            item["landmarks"]
                        )

                        frame_rgb = paste_back(
                            I_p,
                            item["M_c2o_full"],
                            frame_rgb,
                            mask_ori
                        )

                    else:
                        frame_rgb, mask_ori = self._paste_back_default_mask_roi(
                            I_p,
                            item["M_c2o_full"],
                            frame_rgb,
                            item["bbox"]
                        )

                    idx = item["index"]
                    success_flags[idx] = True
                    masks[idx] = mask_ori

                paste_elapsed += perf_counter() - paste_started

            except Exception as e:
                print(f"[FaceSwapper] batch swap error: {e}")

        result_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        timings = {
            "prepare_sec": prepare_elapsed,
            "inference_sec": inference_elapsed,
            "paste_sec": paste_elapsed,
            "total_sec": perf_counter() - total_started,
        }

        return result_bgr, success_flags, masks, timings

    def swap_into_frame(self, frame_bgr, bbox, landmarks=None):
        result_bgr, success_flags, masks, _ = self.swap_many_into_frame(
            frame_bgr,
            [bbox],
            [landmarks],
            batch_size=1,
        )

        if not success_flags[0]:
            return None, None

        return result_bgr, masks[0]


class FaceSwapper:
    def __init__(self, target_image_path, device="cpu"):
        self.processor = LPProcessor(device_type=device)

        target_img = cv2.imread(target_image_path)

        if target_img is None:
            raise RuntimeError(f"Cannot read target image: {target_image_path}")

        self.processor.set_source(target_img)

    def swap_into_frame(self, frame, bbox, landmarks=None):
        return self.processor.swap_into_frame(frame, bbox, landmarks=landmarks)

    def swap_many_into_frame(self, frame, bboxes, landmarks_list=None, batch_size=16):
        return self.processor.swap_many_into_frame(
            frame,
            bboxes,
            landmarks_list=landmarks_list,
            batch_size=batch_size,
        )
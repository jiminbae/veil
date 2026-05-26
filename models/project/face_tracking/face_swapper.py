# face_swapper.py
import cv2
import numpy as np
import torch
from time import perf_counter
from pathlib import Path
import sys
import logging

from config import (
    ENABLE_MASK_BLEND,
    SWAP_FEATHER_RATIO,
    SWAP_MASK_BLUR_KERNEL,
)

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from LivePortrait.src.live_portrait_pipeline import LivePortraitPipeline
from LivePortrait.src.utils.cropper import Cropper
from LivePortrait.src.config.inference_config import InferenceConfig
from LivePortrait.src.config.crop_config import CropConfig

def match_color(source, target, code_to_lab, code_from_lab):
    source_lab = cv2.cvtColor(source, code_to_lab).astype(np.float32)
    target_lab = cv2.cvtColor(target, code_to_lab).astype(np.float32)

    s_mean, s_std = cv2.meanStdDev(source_lab)
    t_mean, t_std = cv2.meanStdDev(target_lab)

    s_mean = s_mean.reshape(1, 1, 3)
    s_std = s_std.reshape(1, 1, 3)
    t_mean = t_mean.reshape(1, 1, 3)
    t_std = t_std.reshape(1, 1, 3)

    result = (source_lab - s_mean) / (s_std + 1e-6) * t_std + t_mean
    result = np.clip(result, 0, 255).astype(np.uint8)

    return cv2.cvtColor(result, code_from_lab)

def blend_face(frame_bgr, swapped_face_bgr, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    H, W = frame_bgr.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)

    if x2 <= x1 or y2 <= y1:
        return frame_bgr

    w = x2 - x1
    h = y2 - y1

    roi = frame_bgr[y1:y2, x1:x2]
    swapped_resized = cv2.resize(swapped_face_bgr, (w, h))

    gray = cv2.cvtColor(swapped_resized, cv2.COLOR_BGR2GRAY)
    non_black_mask = (gray > 35).astype(np.float32)

    ellipse_mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    axes = (max(1, int(w * 0.42)), max(1, int(h * 0.48)))
    cv2.ellipse(ellipse_mask, center, axes, 0, 0, 360, 1.0, -1)

    mask = non_black_mask * ellipse_mask

    if ENABLE_MASK_BLEND:
        blur_size = max(SWAP_MASK_BLUR_KERNEL, int(min(w, h) * SWAP_FEATHER_RATIO))
    else:
        blur_size = 1

    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1

    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    mask = np.clip(mask, 0.0, 1.0)[..., None]

    swapped_matched = match_color(
        swapped_resized,
        roi,
        cv2.COLOR_BGR2LAB,
        cv2.COLOR_LAB2BGR,
    )

    blended = (
        swapped_matched.astype(np.float32) * mask
        + roi.astype(np.float32) * (1.0 - mask)
    )

    frame_bgr[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return frame_bgr

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

    def _prepare_driving_item(self, frame_bgr, bbox, landmarks=None, target_kps=None):
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
            self.crop_cfg,
        )

        if crop_info_d is None or "img_crop_256x256" not in crop_info_d:
            return None

        M_c2o_full = crop_info_d["M_c2o"].copy().astype(np.float32)
        M_c2o_full[0, 2] += px1
        M_c2o_full[1, 2] += py1

        return {
            "bbox": [x1, y1, x2, y2],
            "landmarks": landmarks,
            "target_kps": target_kps,
            "driving_crop": crop_info_d["img_crop_256x256"],
            "M_c2o_full": M_c2o_full,
        }

    def _prepare_batch_tensor(self, batch_items):
        crops = np.stack([item["driving_crop"] for item in batch_items], axis=0)
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

        if ENABLE_MASK_BLEND:
            blur_size = max(SWAP_MASK_BLUR_KERNEL, int(face_size * SWAP_FEATHER_RATIO))
        else:
            blur_size = 1

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
            M_c2o_full,
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

        erode_size = max(11, int(face_size * 0.08))
        erode_size = erode_size if erode_size % 2 == 1 else erode_size + 1

        if ENABLE_MASK_BLEND:
            blur_size = max(SWAP_MASK_BLUR_KERNEL, int(face_size * SWAP_FEATHER_RATIO))
        else:
            blur_size = 1

        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1

        mask_uint8 = (mask_roi * 255).clip(0, 255).astype(np.uint8)
        kernel = np.ones((erode_size, erode_size), np.uint8)
        mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=1)
        mask_uint8 = cv2.GaussianBlur(mask_uint8, (blur_size, blur_size), 0)

        mask_roi = mask_uint8.astype(np.float32) / 255.0

        result_roi = cv2.warpAffine(
            img_crop,
            M_roi[:2, :],
            dsize=(roi_w, roi_h),
            flags=cv2.INTER_LINEAR,
        )

        frame_roi = frame_rgb[ry1:ry2, rx1:rx2]

        if result_roi.shape == frame_roi.shape:
            result_roi = match_color(
                result_roi,
                frame_roi,
                cv2.COLOR_RGB2LAB,
                cv2.COLOR_LAB2RGB,
            )

        if mask_roi.ndim == 2:
            mask_roi_3ch = np.stack([mask_roi] * 3, axis=-1)
        else:
            mask_roi_3ch = mask_roi

        blended_roi = np.clip(
            mask_roi_3ch * result_roi + (1 - mask_roi_3ch) * frame_roi,
            0,
            255,
        ).astype(np.uint8)

        frame_rgb[ry1:ry2, rx1:rx2] = blended_roi

        return frame_rgb, mask_roi

    def _build_landmark_mask(self, frame_shape, bbox, landmarks):
        H, W = frame_shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            return np.zeros((H, W, 3), dtype=np.float32)

        face_size = max(bw, bh)

        mask = np.zeros((H, W), dtype=np.uint8)
        pts = np.asarray(landmarks, dtype=np.float32)

        if pts.ndim != 2 or pts.shape[0] < 5:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            rx = max(1, int(bw * 0.45))
            ry = max(1, int(bh * 0.52))

            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        else:
            pts = pts[:, :2]

            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))

            rx = max(1, int(bw * 0.45))
            ry = max(1, int(bh * 0.52))

            cv2.ellipse(
                mask,
                (cx, cy),
                (rx, ry),
                0,
                0,
                360,
                255,
                -1,
            )

            hull_pts = pts.astype(np.int32)
            hull = cv2.convexHull(hull_pts)

            landmark_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(landmark_mask, hull, 255)

            dilate_size = max(25, int(face_size * 0.18))
            dilate_size = dilate_size if dilate_size % 2 == 1 else dilate_size + 1

            landmark_mask = cv2.dilate(
                landmark_mask,
                np.ones((dilate_size, dilate_size), np.uint8),
                iterations=2,
            )

            mask = cv2.bitwise_or(mask, landmark_mask)

        bbox_limit = np.zeros((H, W), dtype=np.uint8)

        margin_x = int(bw * 0.12)
        margin_y = int(bh * 0.12)

        lx1 = max(0, x1 - margin_x)
        ly1 = max(0, y1 - margin_y)
        lx2 = min(W, x2 + margin_x)
        ly2 = min(H, y2 + margin_y)

        bbox_limit[ly1:ly2, lx1:lx2] = 255
        mask = cv2.bitwise_and(mask, bbox_limit)

        blur_size = max(21, int(face_size * 0.08))
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1

        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

        return np.stack([mask] * 3, axis=-1).astype(np.float32) / 255.0

    def _try_build_align_transform(self, I_p, target_kps):
        try:
            from face_identifier import face_app

            I_p_bgr = cv2.cvtColor(I_p, cv2.COLOR_RGB2BGR)
            out_faces = face_app.get(I_p_bgr)

            if len(out_faces) == 0:
                return None

            out_face = max(
                out_faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )

            if not hasattr(out_face, "kps") or out_face.kps is None:
                return None

            out_kps = np.asarray(out_face.kps, dtype=np.float32)
            tgt_kps = np.asarray(target_kps, dtype=np.float32)

            if out_kps.shape[0] != 5 or tgt_kps.shape[0] != 5:
                return None

            M_align, _ = cv2.estimateAffinePartial2D(
                out_kps,
                tgt_kps,
                method=cv2.LMEDS,
            )

            if M_align is None:
                return None

            return M_align.astype(np.float32)

        except Exception:
            logging.exception("[FaceSwapper] M_align calculation failed")
            return None

    def _paste_with_align(self, I_p, M_align, frame_rgb, bbox, landmarks):
        H, W = frame_rgb.shape[:2]

        warped_face = cv2.warpAffine(
            I_p,
            M_align,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        if landmarks is not None:
            mask_ori = self._build_landmark_mask(frame_rgb.shape, bbox, landmarks)
        else:
            mask_ori = self._build_output_face_mask(I_p, M_align, frame_rgb.shape, bbox)

        bx1, by1, bx2, by2 = map(int, bbox)
        bx1 = max(0, bx1)
        by1 = max(0, by1)
        bx2 = min(W, bx2)
        by2 = min(H, by2)

        if bx2 > bx1 and by2 > by1:
            warped_roi = warped_face[by1:by2, bx1:bx2]
            frame_roi = frame_rgb[by1:by2, bx1:bx2]

            if warped_roi.shape == frame_roi.shape:
                warped_face[by1:by2, bx1:bx2] = match_color(
                    warped_roi,
                    frame_roi,
                    cv2.COLOR_RGB2LAB,
                    cv2.COLOR_LAB2RGB,
                )

        m = mask_ori.astype(np.float32)

        if m.ndim == 2:
            m = m[..., None]

        frame_rgb = np.clip(
            m * warped_face.astype(np.float32)
            + (1.0 - m) * frame_rgb.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)

        return frame_rgb, mask_ori

    def _build_output_face_mask(self, I_p, M_align, frame_shape, bbox):
        H, W = frame_shape[:2]
        bx1, by1, bx2, by2 = map(int, bbox)
        face_size = max(bx2 - bx1, by2 - by1)

        try:
            from face_identifier import face_app

            I_p_bgr = cv2.cvtColor(I_p, cv2.COLOR_RGB2BGR)
            out_faces = face_app.get(I_p_bgr)

            if len(out_faces) > 0:
                out_face = max(
                    out_faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )

                out_lmk = None

                for attr in ["landmark_2d_106", "landmark_3d_68", "kps"]:
                    if hasattr(out_face, attr):
                        val = getattr(out_face, attr)

                        if val is not None:
                            val = np.asarray(val)
                            out_lmk = val[:, :2] if val.shape[-1] == 3 else val
                            break

                if out_lmk is not None and len(out_lmk) >= 5:
                    out_h, out_w = I_p.shape[:2]
                    pts = np.asarray(out_lmk, dtype=np.int32)
                    hull = cv2.convexHull(pts)

                    erode_size = max(5, face_size // 30)
                    erode_size = erode_size if erode_size % 2 == 1 else erode_size + 1

                    blur_size = max(21, face_size // 12)
                    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1

                    mask_out = np.zeros((out_h, out_w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask_out, hull, 255)
                    mask_out = cv2.erode(
                        mask_out,
                        np.ones((erode_size, erode_size), np.uint8),
                    )
                    mask_out = cv2.GaussianBlur(mask_out, (blur_size, blur_size), 0)

                    mask_out_3ch = np.stack([mask_out] * 3, axis=-1)

                    mask_warped = cv2.warpAffine(
                        mask_out_3ch,
                        M_align,
                        (W, H),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )

                    return mask_warped.astype(np.float32) / 255.0

        except Exception:
            logging.exception("[FaceSwapper] output face mask failed")

        mask_fb = np.zeros((H, W), dtype=np.uint8)
        cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
        rx, ry = max(1, (bx2 - bx1) // 2), max(1, (by2 - by1) // 2)

        cv2.ellipse(mask_fb, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        blur_size = max(21, face_size // 8)
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1

        mask_fb = cv2.GaussianBlur(mask_fb, (blur_size, blur_size), 0)

        return np.stack([mask_fb] * 3, axis=-1).astype(np.float32) / 255.0

    def swap_many_into_frame(
        self,
        frame_bgr,
        bboxes,
        landmarks_list=None,
        target_kps_list=None,
        batch_size=16,
    ):
        if self.f_s is None:
            raise RuntimeError("set_source()가 호출되지 않음.")

        total_started = perf_counter()
        prepare_started = perf_counter()

        if landmarks_list is None:
            landmarks_list = [None] * len(bboxes)

        if target_kps_list is None:
            target_kps_list = [None] * len(bboxes)

        prepared = []
        success_flags = [False] * len(bboxes)
        masks = [None] * len(bboxes)

        for idx, (bbox, landmarks, target_kps) in enumerate(
            zip(bboxes, landmarks_list, target_kps_list)
        ):
            try:
                item = self._prepare_driving_item(
                    frame_bgr,
                    bbox,
                    landmarks=landmarks,
                    target_kps=target_kps,
                )
            except Exception:
                logging.exception("[FaceSwapper] prepare error")
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
                        -1,
                    )

                    x_d_new = self.wrapper.stitching(x_s_batch, x_d)

                    out = self.wrapper.warp_decode(
                        f_s_batch,
                        x_s_batch,
                        x_d_new,
                    )

                    outputs = self.wrapper.parse_output(out["out"])

                inference_elapsed += perf_counter() - inference_started
                paste_started = perf_counter()

                for item, I_p in zip(batch_items, outputs):
                    target_kps = item.get("target_kps")
                    landmarks = item.get("landmarks")
                    bbox = item["bbox"]

                    frame_rgb, mask_ori = self._paste_back_default_mask_roi(
                        I_p,
                        item["M_c2o_full"],
                        frame_rgb,
                        bbox,
                    )

                    if mask_ori is None:
                        continue

                    cur_idx = item["index"]
                    success_flags[cur_idx] = True
                    masks[cur_idx] = mask_ori

                paste_elapsed += perf_counter() - paste_started

            except Exception:
                logging.exception("[FaceSwapper] batch swap error")

        result_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        timings = {
            "prepare_sec": prepare_elapsed,
            "inference_sec": inference_elapsed,
            "paste_sec": paste_elapsed,
            "total_sec": perf_counter() - total_started,
        }

        return result_bgr, success_flags, masks, timings

    def swap_into_frame(self, frame_bgr, bbox, landmarks=None, target_kps=None):
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

class FaceSwapper:
    def __init__(self, target_image_path, device="cpu"):
        self.processor = LPProcessor(device_type=device)

        target_img = cv2.imread(target_image_path)

        if target_img is None:
            raise RuntimeError(f"Cannot read target image: {target_image_path}")

        self.processor.set_source(target_img)

    def swap_into_frame(self, frame, bbox, landmarks=None, target_kps=None):
        return self.processor.swap_into_frame(
            frame,
            bbox,
            landmarks=landmarks,
            target_kps=target_kps,
        )

    def swap_many_into_frame(
        self,
        frame,
        bboxes,
        landmarks_list=None,
        target_kps_list=None,
        batch_size=16,
    ):
        return self.processor.swap_many_into_frame(
            frame,
            bboxes,
            landmarks_list=landmarks_list,
            target_kps_list=target_kps_list,
            batch_size=batch_size,
        )
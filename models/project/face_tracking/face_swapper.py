import cv2
import numpy as np
import torch
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from LivePortrait.src.live_portrait_pipeline import LivePortraitPipeline
from LivePortrait.src.utils.cropper import Cropper
from LivePortrait.src.config.inference_config import InferenceConfig
from LivePortrait.src.config.crop_config import CropConfig

class LPProcessor:
    def __init__(self, device_type="cpu"):
        is_cuda = (device_type == "cuda" and torch.cuda.is_available())
        
        self.inference_cfg = InferenceConfig(
            flag_force_cpu=not is_cuda,
            flag_use_half_precision=is_cuda,
            device_id=0 if is_cuda else -1
        )
        self.crop_cfg = CropConfig()

        print(f"Initializing LivePortrait on: {'GPU' if is_cuda else 'CPU'}")

        self.pipeline = LivePortraitPipeline(
            inference_cfg=self.inference_cfg,
            crop_cfg=self.crop_cfg
        )
        
        self.wrapper = self.pipeline.live_portrait_wrapper
        self.cropper = Cropper(crop_cfg=self.crop_cfg)
        
        self.source_image = None
        self.f_s = None
        self.x_s_info = None

    def set_source(self, image_bgr):
        if image_bgr is None: 
            return
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        crop_info = self.cropper.crop_source_image(image_rgb, self.crop_cfg)

        if crop_info is None or 'img_crop' not in crop_info:
            raise RuntimeError("No face detected in source image.")

        cropped = crop_info['img_crop']
        self.source_image = cropped
        
        with torch.no_grad():
            source_tensor = self.wrapper.prepare_source(cropped)
            # 1. 소스 특징량(f_s) 추출
            self.f_s = self.wrapper.appearance_feature_extractor(source_tensor)
            # 2. 소스 키포인트(x_s) 추출
            self.x_s_info = self.wrapper.get_kp_info(source_tensor)
            
        print("Success: Source face features extracted.")

    def inference(self, driving_bgr):
        if self.f_s is None or self.x_s_info is None:
            raise RuntimeError("Source features not extracted.")

        driving_rgb = cv2.cvtColor(driving_bgr, cv2.COLOR_BGR2RGB)

        if hasattr(self.cropper, "crop_driving_video"):
            crop_info = self.cropper.crop_driving_video([driving_rgb])

            if crop_info is None:
                return None

            if isinstance(crop_info, dict):
                if "frame_crop_lst" in crop_info:
                    if len(crop_info["frame_crop_lst"]) == 0:
                        return None
                    driving_crop = crop_info["frame_crop_lst"][0]
                elif "img_crop" in crop_info:
                    driving_crop = crop_info["img_crop"]
                else:
                    return None

            elif isinstance(crop_info, list):
                if len(crop_info) == 0:
                    return None

                first_crop = crop_info[0]

                if isinstance(first_crop, dict):
                    if "img_crop" in first_crop:
                        driving_crop = first_crop["img_crop"]
                    elif "frame_crop_lst" in first_crop and len(first_crop["frame_crop_lst"]) > 0:
                        driving_crop = first_crop["frame_crop_lst"][0]
                    else:
                        return None
                else:
                    driving_crop = first_crop
            else:
                return None

        else:
            crop_info = self.cropper.crop_source_image(driving_rgb, self.crop_cfg)

            if crop_info is None or 'img_crop' not in crop_info:
                return None

            driving_crop = crop_info['img_crop']

        driving_crop = cv2.resize(driving_crop, (256, 256))
        
        with torch.no_grad():
            # 1. Driving 키포인트 추출
            if hasattr(self.wrapper, "prepare_driving"):
                driving_tensor = self.wrapper.prepare_driving(driving_crop)
            elif hasattr(self.wrapper, "prepare_driving_video"):
                driving_tensor = self.wrapper.prepare_driving_video(driving_crop)
            else:
                driving_tensor = self.wrapper.prepare_source(driving_crop)

            x_d_info = self.wrapper.get_kp_info(driving_tensor)
            
            # 2. 딕셔너리에서 'kp' 텐서만 추출
            kp_s = self.x_s_info['kp']
            kp_d = x_d_info['kp']
            
            # 3. Warping 수행
            # warping_network.py 분석 결과: 반환값은 {'out': tensor, 'occlusion_map': tensor}
            warping_result = self.wrapper.warping_module(self.f_s, kp_d, kp_s)
            
            # [핵심 수정] 딕셔너리에서 'out' 키에 해당하는 실제 텐서만 추출
            if isinstance(warping_result, dict):
                warped_feature = warping_result['out']
            else:
                warped_feature = warping_result

            # 4. Generator 호출 (이제 순수 텐서가 전달됨)
            out_tensor = self.wrapper.spade_generator(warped_feature)
            
        # 5. 후처리
        if out_tensor.ndim == 4: 
            out_tensor = out_tensor[0]
            
        result = out_tensor.permute(1, 2, 0).cpu().numpy()
        result = (result * 255).clip(0, 255).astype(np.uint8)
        
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


class FaceSwapper:
    def __init__(self, target_image_path, device="cpu"):
        self.processor = LPProcessor(device_type=device)
        target_img = cv2.imread(target_image_path)
        if target_img is None:
            raise RuntimeError(f"Cannot read target image: {target_image_path}")
        self.processor.set_source(target_img)

    def swap(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]

        # LivePortrait의 내부 크로퍼가 랜드마크를 찾을 수 있는 '공간'을 제공함
        bw, bh = x2 - x1, y2 - y1
        pad_w, pad_h = int(bw * 0.5), int(bh * 0.5)

        px1 = max(0, x1 - pad_w)
        py1 = max(0, y1 - pad_h)
        px2 = min(w, x2 + pad_w)
        py2 = min(h, y2 + pad_h)

        # 확장된 영역으로 크롭
        face_crop_padded = frame[py1:py2, px1:px2]
        if face_crop_padded.size == 0: 
            return None

        try:
            # 확장된 이미지를 넘겨주어 내부 크로퍼가 정상 작동하게 함
            swapped_padded = self.processor.inference(face_crop_padded)
            if swapped_padded is None: 
                return None

            # 크롭된 이미지 내에서의 상대적 좌표 계산
            rel_x1, rel_y1 = x1 - px1, y1 - py1
            
            # swapped_padded는 LivePortrait 출력 크기(보통 256x256 등)이므로 리사이즈 필요
            swapped_resized = cv2.resize(swapped_padded, (px2 - px1, py2 - py1))
            final_face = swapped_resized[rel_y1:rel_y1 + bh, rel_x1:rel_x1 + bw]

            if final_face.size == 0:
                return None
            
            return final_face
            
        except Exception as e:
            print(f"Inference error during padded swap: {e}")
            return None
import cv2
import numpy as np

class FaceStitcher:

    def __init__(self, blur_kernel=21):
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        self.blur_kernel = max(3, blur_kernel)

    def _rotate_face(self, face, angle):
        if angle == 0:
            return face

        h, w = face.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

        return cv2.warpAffine(
            face,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

    def stitch(self, frame, fake_face, bbox, angle=0):
        """
        frame     : 원본 프레임
        fake_face : face_swapper.swap()이 반환한 가상 얼굴
        bbox      : [x1, y1, x2, y2]
        angle     : 고개 돌림 각도 (기본 0)
        """
        if frame is None:
            raise ValueError("frame이 None입니다.")

        if fake_face is None:
            return frame

        x1, y1, x2, y2 = map(int, bbox)
        h_frame, w_frame = frame.shape[:2]

        # bbox 클리핑
        x1 = max(0, min(x1, w_frame))
        x2 = max(0, min(x2, w_frame))
        y1 = max(0, min(y1, h_frame))
        y2 = max(0, min(y2, h_frame))

        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            return frame

        # fake_face 리사이즈 + 고개 각도 적용
        resized_face = cv2.resize(fake_face, (w, h))
        resized_face = self._rotate_face(resized_face, angle)

        # 타원 마스크
        seamless_mask = np.zeros((h, w), dtype=np.uint8)
        axes = (max(1, w // 2 - 2), max(1, h // 2 - 2))

        cv2.ellipse(
            seamless_mask,
            (w // 2, h // 2),
            axes,
            0,
            0,
            360,
            255,
            -1
        )

        # 가우시안 블러로 경계 부드럽게
        mask_blur = cv2.GaussianBlur(
            seamless_mask,
            (self.blur_kernel, self.blur_kernel),
            0
        )

        mask_3ch = cv2.merge([mask_blur] * 3).astype(np.float32) / 255.0

        center = (x1 + w // 2, y1 + h // 2)

        try:
            result = cv2.seamlessClone(
                resized_face,
                frame,
                seamless_mask,
                center,
                cv2.NORMAL_CLONE
            )
        except Exception:
            # fallback: 가우시안 블렌딩
            result = frame.copy()
            roi = result[y1:y2, x1:x2].astype(np.float32)

            blended = (
                roi * (1 - mask_3ch)
                + resized_face.astype(np.float32) * mask_3ch
            )

            result[y1:y2, x1:x2] = blended.astype(np.uint8)

        return result
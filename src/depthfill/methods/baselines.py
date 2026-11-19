"""기준선 방법들. 모델이 이보다 나아야 의미가 있다."""
from __future__ import annotations

import numpy as np


class NoFill:
    """보완하지 않음(구멍이 0 그대로). 아무것도 안 했을 때의 오차."""
    name = "nofill"

    def fill(self, depth, valid, rgb=None):
        return depth.copy()


class MeanFill:
    """구멍을 측정된 값의 평균으로 채운다."""
    name = "meanfill"

    def fill(self, depth, valid, rgb=None):
        out = depth.copy()
        out[~valid] = depth[valid].mean() if valid.any() else 0.0
        return out


class DiffusionFill:
    """구멍 가장자리의 값이 안쪽으로 번지게 한다(주변 4칸 평균을 반복)."""
    name = "diffusion"

    def __init__(self, iterations: int = 300):
        self.iterations = iterations

    def fill(self, depth, valid, rgb=None):
        hole = ~valid
        out = depth.copy()
        out[hole] = depth[valid].mean() if valid.any() else 0.0
        for _ in range(self.iterations):
            p = np.pad(out, 1, mode="edge")
            avg = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
            out[hole] = avg[hole]
        return out


class OpenCVInpaint:
    """OpenCV 채우기(Telea). opencv가 설치된 경우에만 쓸 수 있다."""
    name = "opencv_telea"

    def __init__(self, radius: int = 3):
        import cv2  # 없으면 여기서 ImportError
        self._cv2 = cv2
        self.radius = radius

    def fill(self, depth, valid, rgb=None):
        mask = (~valid).astype(np.uint8)
        # 실수(float32) 깊이를 그대로 넣으면 이 OpenCV(5.1)에서는 구멍 속 값이 픽셀마다 0.6~3m로 튀었다.
        # 밀리미터 단위 16비트 정수로 바꿔 넣으면 정상이다(판단 기록 2026-11-19, 시험 test_opencv_telea_is_sane).
        mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        out = self._cv2.inpaint(mm, mask, self.radius, self._cv2.INPAINT_TELEA).astype(np.float32) / 1000.0
        return np.where(valid, depth, out).astype(np.float32)

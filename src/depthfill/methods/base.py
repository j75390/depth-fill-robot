"""보완 방법의 공통 형태. 평가 도구는 이 형태만 안다."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Method(Protocol):
    name: str

    def fill(self, depth: np.ndarray, valid: np.ndarray, rgb: np.ndarray | None = None) -> np.ndarray:
        """depth: 구멍 자리가 0인 깊이(H, W), valid: 측정값이 있는 곳 True, rgb: 컬러(선택).
        구멍을 채운 깊이(H, W)를 돌려준다. 정답 깊이는 절대 받지 않는다."""
        ...

"""데이터셋. 지금은 시험·연습용 합성 데이터만 있고, 실제 공개 데이터 로더는 이슈 #2에서 만든다."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    name: str
    rgb: np.ndarray    # (H, W, 3) float32, 0~1
    depth: np.ndarray  # (H, W) float32, 미터. 정답 깊이(구멍 없음)


def make_scene(rng: np.random.Generator, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """바닥과 벽, 상자 몇 개로 이루어진 간단한 장면. 상자 하나는 반짝이는 색(반사 구멍 시험용)."""
    yy, _ = np.mgrid[:h, :w].astype(np.float32)
    depth = (4.0 - 2.0 * (yy / h)).astype(np.float32)           # 아래쪽일수록 가깝다
    rgb = np.stack([0.35 + 0.2 * (yy / h)] * 3, axis=-1).astype(np.float32)
    n_boxes = int(rng.integers(3, 6))
    for k in range(n_boxes):
        y0 = int(rng.integers(0, h * 2 // 3)); y1 = y0 + int(rng.integers(h // 8, h // 3))
        x0 = int(rng.integers(0, w * 2 // 3)); x1 = x0 + int(rng.integers(w // 8, w // 3))
        depth[y0:y1, x0:x1] = rng.uniform(1.0, 3.0)
        color = rng.uniform(0.1, 0.7, size=3)
        rgb[y0:y1, x0:x1] = 0.97 if k == 0 else color                 # 첫 상자는 반짝이는 흰색
    depth += rng.normal(0, 0.005, size=depth.shape).astype(np.float32)
    rgb += rng.normal(0, 0.005, size=rgb.shape).astype(np.float32)
    return np.clip(rgb, 0, 1), depth.astype(np.float32)


class SyntheticDataset:
    """시드가 같으면 항상 같은 장면을 돌려준다."""

    def __init__(self, n: int, size: tuple[int, int] = (120, 160), seed: int = 0):
        self.n, self.size, self.seed = n, size, seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> Sample:
        rng = np.random.default_rng([self.seed, index])
        rgb, depth = make_scene(rng, *self.size)
        return Sample(name=f"synthetic_{index:04d}", rgb=rgb, depth=depth)


class NYUDataset:
    """TODO(이슈 #2): NYU Depth v2 같은 공개 데이터를 읽는다. 사용 범위는 판단 기록에 남긴다."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("이슈 #2에서 구현한다")

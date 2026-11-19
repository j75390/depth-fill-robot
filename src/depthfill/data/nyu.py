"""NYU Depth v2 (labeled 1,449장) 준비와 읽기.

원본(.mat, HDF5)은 640x480이라 매번 읽기 무겁다. 한 번만 `prepare_nyu()`로 테두리를 자르고 줄여서
`data/nyu_<H>x<W>.npz`로 저장하고, 학습·평가는 그 파일만 읽는다.

정답 깊이에 대해 꼭 알아야 할 점(판단 기록 2026-11-19):
- `depths`는 센서가 못 잰 곳을 이미 알고리즘(색 기반 보간)으로 채운 값이다.
- `rawDepths`는 센서가 실제로 잰 값이고, 못 잰 곳은 0이다.
그래서 평가는 두 가지로 한다: 채운 정답 전체(`gt`)와, 실제로 잰 픽셀만(`raw_valid`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 640x480 원본 가장자리에는 값이 없는 흰 테두리가 있다. 흔히 쓰는 범위로 잘라 낸다.
CROP = (slice(45, 471), slice(41, 601))  # (행, 열) → 426 x 560


def _resize_area(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(a, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def _resize_raw(raw: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """측정값이 0(없음)인 칸을 섞어 평균내지 않도록, 있는 값만 평균낸다. 절반 넘게 비면 0으로 둔다."""
    valid = (raw > 0).astype(np.float32)
    s = _resize_area(raw * valid, size)
    c = _resize_area(valid, size)
    out = np.where(c > 0.5, s / np.maximum(c, 1e-6), 0.0)
    return out.astype(np.float32)


def prepare_nyu(mat_path: str | Path, splits_path: str | Path, out_path: str | Path, size: tuple[int, int]) -> Path:
    """원본 .mat을 잘라서 줄이고, 공식 학습/시험 분할과 함께 npz로 저장한다."""
    import h5py
    from scipy.io import loadmat

    split = loadmat(str(splits_path))
    train_idx = split["trainNdxs"].ravel() - 1  # MATLAB은 1부터 센다
    test_idx = split["testNdxs"].ravel() - 1

    with h5py.File(mat_path, "r") as f:
        n = f["depths"].shape[0]
        rgb = np.zeros((n, *size, 3), np.uint8)
        depth = np.zeros((n, *size), np.float16)
        raw = np.zeros((n, *size), np.float16)
        for i in range(n):
            # h5py로 읽으면 MATLAB 배열이 뒤집혀 나온다: (3, W, H) → (H, W, 3)
            img = np.transpose(f["images"][i], (2, 1, 0))[CROP]
            d = np.transpose(f["depths"][i], (1, 0))[CROP].astype(np.float32)
            r = np.transpose(f["rawDepths"][i], (1, 0))[CROP].astype(np.float32)
            rgb[i] = _resize_area(img, size)
            depth[i] = _resize_area(d, size)
            raw[i] = _resize_raw(r, size)
    out = Path(out_path)
    np.savez_compressed(out, rgb=rgb, depth=depth, raw=raw, train=train_idx, test=test_idx)
    return out


@dataclass
class NYUSample:
    name: str
    rgb: np.ndarray        # (H, W, 3) float32 0~1
    depth: np.ndarray      # (H, W) float32 미터, 채운 정답
    raw_valid: np.ndarray  # (H, W) bool, 센서가 실제로 잰 픽셀


class NYUDataset:
    """준비된 npz에서 학습(train) 또는 시험(test) 분할을 읽는다. 시험 분할은 학습에 쓰지 않는다."""

    def __init__(self, npz_path: str | Path, split: str = "test", limit: int | None = None):
        data = np.load(npz_path)
        idx = data[split]
        if limit:
            idx = idx[:limit]
        self.idx = idx
        self.split = split
        self.rgb = data["rgb"][idx]
        self.depth = data["depth"][idx].astype(np.float32)
        self.raw = data["raw"][idx].astype(np.float32)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> NYUSample:
        return NYUSample(
            name=f"nyu_{int(self.idx[i]):04d}",
            rgb=self.rgb[i].astype(np.float32) / 255.0,
            depth=self.depth[i],
            raw_valid=self.raw[i] > 0,
        )

    def sensor_masks(self) -> np.ndarray:
        """이 분할의 '실제 센서 누락' 마스크들(구멍 유형 sensor에서 다른 장면의 누락 모양을 빌려 쓴다)."""
        return self.raw <= 0

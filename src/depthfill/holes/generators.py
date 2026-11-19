"""구멍(측정값이 빠진 영역) 만들기. 반환값은 True가 구멍인 마스크(H, W)이다.

모든 함수는 rng(np.random.Generator)를 받아, 같은 시드면 같은 구멍이 나온다.
"""
from __future__ import annotations

import numpy as np

HOLE_KINDS = ("random", "edge", "reflect", "sensor")


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """마스크를 r칸만큼 부풀린다(가장자리는 늘어나지 않는다)."""
    h, w = mask.shape
    padded = np.pad(mask, r)
    out = np.zeros_like(mask)
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            out |= padded[dy:dy + h, dx:dx + w]
    return out


def box_mean(a: np.ndarray, r: int) -> np.ndarray:
    """(2r+1) 크기 창의 평균."""
    k = 2 * r + 1
    h, w = a.shape
    c = np.pad(np.pad(a, r, mode="edge").cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = c[k:k + h, k:k + w] - c[0:h, k:k + w] - c[k:k + h, 0:w] + c[0:h, 0:w]
    return s / (k * k)


def random_blobs(depth: np.ndarray, rng: np.random.Generator, ratio: float) -> np.ndarray:
    """무작위 타원 덩어리를 비율이 찰 때까지 뚫는다."""
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    mask = np.zeros((h, w), bool)
    for _ in range(300):
        if mask.mean() >= ratio:
            break
        cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
        ry = int(rng.integers(max(2, h // 20), max(3, h // 6)))
        rx = int(rng.integers(max(2, w // 20), max(3, w // 6)))
        mask |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
    return mask


def edge_holes(depth: np.ndarray, rng: np.random.Generator, ratio: float) -> np.ndarray:
    """깊이가 크게 변하는 물체 가장자리 주변을 뚫는다."""
    h, w = depth.shape
    gy, gx = np.gradient(depth)
    g = np.hypot(gx, gy)
    if g.max() < 1e-6:
        return np.zeros((h, w), bool)
    band = dilate(g >= max(np.quantile(g, 0.9), 1e-6), 2)
    frac = band.mean()
    if frac <= ratio:
        return band
    block = 3  # 6이면 실제 이미지에서 계단처럼 보여 3으로 줄였다(판단 기록 2026-11-19)
    coarse = (rng.random((h // block + 1, w // block + 1)) < ratio / frac).astype(np.uint8)
    keep = np.kron(coarse, np.ones((block, block), np.uint8)).astype(bool)[:h, :w]
    return band & keep


def reflective_holes(rgb: np.ndarray, rng: np.random.Generator, ratio: float) -> np.ndarray:
    """밝고 매끈한(질감이 적은) 면을 뚫는다. 반짝이는 면에서 깊이가 빠지는 상황을 흉내 낸다."""
    h, w = rgb.shape[:2]
    b = rgb.mean(-1)
    var = box_mean(b * b, 3) - box_mean(b, 3) ** 2
    smooth = var <= np.quantile(var, 0.5)
    n = int(smooth.sum())
    if n == 0:
        return np.zeros((h, w), bool)
    q = 1.0 - min(1.0, ratio * h * w / n)
    thr = np.quantile(b[smooth], q)
    return smooth & (b >= thr)


def sensor_holes(pool: np.ndarray, rng: np.random.Generator, exclude: int | None = None) -> np.ndarray:
    """실제 센서가 값을 못 잰 모양을 다른 장면에서 빌려 온다(사람이 만든 규칙이 아닌 실제 누락 패턴).

    pool: (N, H, W) bool, 실제 누락 마스크 모음. exclude: 지금 장면의 번호(자기 자신의 누락은 쓰지 않는다).
    """
    choices = [i for i in range(len(pool)) if i != exclude]
    return pool[int(rng.choice(choices))].copy()


def make_holes(kind: str, rgb: np.ndarray, depth: np.ndarray,
               rng: np.random.Generator, ratio: float,
               sensor_pool: np.ndarray | None = None, index: int | None = None) -> np.ndarray:
    if kind == "random":
        return random_blobs(depth, rng, ratio)
    if kind == "edge":
        return edge_holes(depth, rng, ratio)
    if kind == "reflect":
        return reflective_holes(rgb, rng, ratio)
    if kind == "sensor":
        if sensor_pool is None:
            raise ValueError("sensor 구멍은 실제 누락 마스크 모음(sensor_pool)이 있어야 만든다")
        return sensor_holes(sensor_pool, rng, exclude=index)
    raise ValueError(f"알 수 없는 구멍 유형: {kind} (가능: {HOLE_KINDS})")

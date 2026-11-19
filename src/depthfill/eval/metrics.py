"""지표. 평가는 항상 구멍 영역에서만 한다(값이 있던 곳은 제외)."""
from __future__ import annotations

import numpy as np


def masked_errors(pred: np.ndarray, gt: np.ndarray, hole: np.ndarray) -> dict:
    """구멍 영역의 MAE, RMSE와 픽셀 수."""
    # 집계 오차를 줄이려고 float64로 계산한다(float32끼리 빼면 지표가 미세하게 어긋난다)
    diff = (pred.astype(np.float64) - gt.astype(np.float64))[hole]
    if diff.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    return {
        "mae": float(np.abs(diff).mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "n": int(diff.size),
    }

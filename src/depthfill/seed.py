"""모든 무작위 값을 같은 시드로 고정한다(같은 설정이면 같은 결과가 나와야 한다)."""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:  # PyTorch가 설치된 경우에만
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

import numpy as np

from depthfill.data.datasets import SyntheticDataset
from depthfill.holes.generators import HOLE_KINDS, make_holes, sensor_holes

# 규칙으로 만드는 구멍(sensor는 실제 누락 마스크 모음이 있어야 해서 따로 시험한다)
RULE_KINDS = tuple(k for k in HOLE_KINDS if k != "sensor")


def _sample():
    return SyntheticDataset(1, (120, 160), seed=1)[0]


def test_sensor_holes_borrow_another_scene():
    pool = np.zeros((3, 4, 4), bool)
    pool[0, 0, 0] = True  # 장면 0의 누락
    pool[1, 1, 1] = True
    pool[2, 2, 2] = True
    for seed in range(10):
        m = sensor_holes(pool, np.random.default_rng(seed), exclude=0)
        assert not m[0, 0], "자기 장면의 누락 모양은 빌리지 않는다"
        assert m.sum() == 1


def test_sensor_kind_needs_pool():
    s = _sample()
    try:
        make_holes("sensor", s.rgb, s.depth, np.random.default_rng(0), 0.1)
    except ValueError:
        return
    raise AssertionError("ValueError가 나와야 한다")


def test_same_seed_same_holes():
    s = _sample()
    for kind in RULE_KINDS:
        a = make_holes(kind, s.rgb, s.depth, np.random.default_rng(7), 0.15)
        b = make_holes(kind, s.rgb, s.depth, np.random.default_rng(7), 0.15)
        assert (a == b).all(), kind


def test_hole_ratio_is_reasonable():
    s = _sample()
    for kind in RULE_KINDS:
        m = make_holes(kind, s.rgb, s.depth, np.random.default_rng(3), 0.15)
        assert 0.02 <= m.mean() <= 0.35, (kind, m.mean())


def test_unknown_kind_raises():
    s = _sample()
    try:
        make_holes("nope", s.rgb, s.depth, np.random.default_rng(0), 0.1)
    except ValueError:
        return
    raise AssertionError("ValueError가 나와야 한다")

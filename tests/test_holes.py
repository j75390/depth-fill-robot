import numpy as np

from depthfill.data.datasets import SyntheticDataset
from depthfill.holes.generators import HOLE_KINDS, make_holes


def _sample():
    return SyntheticDataset(1, (120, 160), seed=1)[0]


def test_same_seed_same_holes():
    s = _sample()
    for kind in HOLE_KINDS:
        a = make_holes(kind, s.rgb, s.depth, np.random.default_rng(7), 0.15)
        b = make_holes(kind, s.rgb, s.depth, np.random.default_rng(7), 0.15)
        assert (a == b).all(), kind


def test_hole_ratio_is_reasonable():
    s = _sample()
    for kind in HOLE_KINDS:
        m = make_holes(kind, s.rgb, s.depth, np.random.default_rng(3), 0.15)
        assert 0.02 <= m.mean() <= 0.35, (kind, m.mean())


def test_unknown_kind_raises():
    s = _sample()
    try:
        make_holes("nope", s.rgb, s.depth, np.random.default_rng(0), 0.1)
    except ValueError:
        return
    raise AssertionError("ValueError가 나와야 한다")

import math

import numpy as np

from depthfill.eval.metrics import masked_errors


def test_zero_error_when_equal():
    gt = np.ones((4, 4), np.float32)
    hole = np.zeros((4, 4), bool); hole[1:3, 1:3] = True
    e = masked_errors(gt.copy(), gt, hole)
    assert e["mae"] == 0 and e["rmse"] == 0 and e["n"] == 4


def test_known_values_only_inside_holes():
    gt = np.zeros((2, 2), np.float32)
    pred = np.array([[1, 100], [3, 100]], np.float32)  # 100은 구멍 밖이라 무시돼야 한다
    hole = np.array([[True, False], [True, False]])
    e = masked_errors(pred, gt, hole)
    assert math.isclose(e["mae"], 2.0) and math.isclose(e["rmse"], math.sqrt(5.0))


def test_empty_hole_gives_nan():
    z = np.zeros((2, 2), np.float32)
    e = masked_errors(z, z, np.zeros((2, 2), bool))
    assert e["n"] == 0 and math.isnan(e["mae"])

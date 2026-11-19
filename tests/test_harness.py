import numpy as np

from depthfill.data.datasets import SyntheticDataset
from depthfill.eval.harness import evaluate, summarize
from depthfill.methods.registry import get_methods


def _run():
    ds = SyntheticDataset(4, (60, 80), seed=5)
    return evaluate(ds, get_methods(["nofill", "meanfill", "diffusion"]), ["random", "edge", "reflect"], 0.15, 42)


def test_results_are_reproducible():
    assert summarize(_run()) == summarize(_run())


def test_fill_beats_nofill_on_random_holes():
    s = summarize(_run())
    assert s["random/diffusion"]["rmse"] < s["random/nofill"]["rmse"]
    assert s["random/diffusion"]["rmse"] < s["random/meanfill"]["rmse"]


def test_every_method_sees_same_holes():
    rows = _run()
    by_sample = {}
    for r in rows:
        by_sample.setdefault((r["sample"], r["kind"]), set()).add(round(r["hole_ratio"], 9))
    assert all(len(v) == 1 for v in by_sample.values())


def test_opencv_telea_is_sane():
    """OpenCV 채우기가 평균값 채우기보다 나빠지면 입력 형식 문제(float32 버그)를 의심한다."""
    from depthfill.data.datasets import SyntheticDataset
    from depthfill.eval.metrics import masked_errors
    from depthfill.holes.generators import random_blobs
    from depthfill.methods.baselines import MeanFill, OpenCVInpaint

    s = SyntheticDataset(1, (120, 160), seed=3)[0]
    hole = random_blobs(s.depth, np.random.default_rng(0), 0.15)
    valid = ~hole
    holed = np.where(valid, s.depth, 0.0).astype(np.float32)
    telea = masked_errors(OpenCVInpaint().fill(holed, valid), s.depth, hole)["mae"]
    mean = masked_errors(MeanFill().fill(holed, valid), s.depth, hole)["mae"]
    assert telea < mean, (telea, mean)

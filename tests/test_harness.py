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

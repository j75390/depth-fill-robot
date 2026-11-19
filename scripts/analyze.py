"""세밀한 분석: 평균 하나로는 안 보이는 것을 나눠서 잰다. 시험 654장, 센서가 실제로 잰 픽셀만.

  metrics   유형×방법별 MAE·RMSE·상대오차·δ 정확도·"5cm/10cm 넘게 틀린 픽셀 비율"(픽셀 단위로 모아서)
  distance  구멍 가장자리에서 몇 픽셀 안쪽인지에 따른 오차
  depth     실제 거리(가까움~멂)에 따른 오차
  ratio     구멍 크기를 5~35%로 바꿨을 때(모델은 15%로만 학습)
  seeds     학습 시드 3개(42·1·2)의 편차
  speed     한 장 처리 시간(CPU/GPU)

결과: results/analysis.json, 그림 docs/images/analysis_*.png
예) python scripts/analyze.py            # 전부(약 10분)
    python scripts/analyze.py --parts speed seeds
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from depthfill.config import ROOT, load_config
from depthfill.data.nyu import NYUDataset
from depthfill.eval.harness import make_case
from depthfill.methods.registry import get_methods

KIND_KO = {"random": "무작위", "edge": "물체 가장자리", "reflect": "반사 면", "sensor": "실제 센서 누락"}
NAME_KO = {"opencv_telea": "OpenCV Telea", "unet_depth": "U-Net(깊이)", "unet_rgbd": "U-Net(컬러+깊이)"}
COLOR = {"opencv_telea": "#7f8aa0", "unet_depth": "#ff8a3d", "unet_rgbd": "#3d9bff"}
DIST_BINS = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 999)]
DIST_LABEL = ["1~2", "2~4", "4~8", "8~16", "16+"]
DEPTH_BINS = [(0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 99)]
DEPTH_LABEL = ["~1.5m", "1.5~2.5m", "2.5~3.5m", "3.5~5m", "5m~"]
RATIOS = [0.05, 0.10, 0.15, 0.25, 0.35]
OUT_IMG = ROOT / "docs" / "images"


def korean_font() -> None:
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if any(name in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


class Acc:
    """픽셀 오차를 모아 두었다가 지표를 낸다(메모리를 아끼려고 합계만 쌓는다)."""

    def __init__(self):
        self.n = 0
        self.abs = self.sq = self.rel = 0.0
        self.d105 = self.d110 = self.d125 = self.b5 = self.b10 = 0

    def add(self, pred, gt):
        e = pred.astype(np.float64) - gt.astype(np.float64)
        a = np.abs(e)
        ratio = np.maximum(pred / gt, gt / np.maximum(pred, 1e-6))
        self.n += a.size
        self.abs += a.sum()
        self.sq += (e ** 2).sum()
        self.rel += (a / gt).sum()
        self.d105 += int((ratio < 1.05).sum())
        self.d110 += int((ratio < 1.10).sum())
        self.d125 += int((ratio < 1.25).sum())
        self.b5 += int((a > 0.05).sum())
        self.b10 += int((a > 0.10).sum())

    def out(self):
        n = max(self.n, 1)
        return {"n": self.n, "mae_m": self.abs / n, "rmse_m": (self.sq / n) ** 0.5, "absrel": self.rel / n,
                "delta_1.05": self.d105 / n, "delta_1.10": self.d110 / n, "delta_1.25": self.d125 / n,
                "over_5cm": self.b5 / n, "over_10cm": self.b10 / n}


def part_breakdowns(ds, pool, cfg, methods):
    """metrics·distance·depth를 한 번에(같은 예측을 재사용)."""
    kinds, ratio, seed = cfg["holes"]["kinds"], cfg["holes"]["ratio"], cfg["seed"]
    by_kind = defaultdict(Acc)
    by_dist = defaultdict(Acc)
    by_depth = defaultdict(Acc)
    for i in range(len(ds)):
        s = ds[i]
        for k, kind in enumerate(kinds):
            hole, valid, holed = make_case(s, kind, k, i, ratio, seed, pool)
            m = hole & s.raw_valid
            if not m.any():
                continue
            dist = cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3)
            gt = s.depth[m]
            dm = dist[m]
            for meth in methods:
                p = meth.fill(holed, valid, s.rgb)[m]
                by_kind[(kind, meth.name)].add(p, gt)
                by_kind[("all", meth.name)].add(p, gt)
                for b, (lo, hi) in enumerate(DIST_BINS):
                    sel = (dm >= lo) & (dm < hi)
                    if sel.any():
                        by_dist[(b, meth.name)].add(p[sel], gt[sel])
                for b, (lo, hi) in enumerate(DEPTH_BINS):
                    sel = (gt >= lo) & (gt < hi)
                    if sel.any():
                        by_depth[(b, meth.name)].add(p[sel], gt[sel])
        if i % 100 == 0:
            print(f"  분해 {i}/{len(ds)}", flush=True)
    names = [m.name for m in methods]
    metrics = {f"{k}/{n}": by_kind[(k, n)].out() for k in [*kinds, "all"] for n in names}
    distance = {n: [{"bin_px": DIST_LABEL[b], **by_dist[(b, n)].out()} for b in range(len(DIST_BINS))] for n in names}
    depth = {n: [{"bin": DEPTH_LABEL[b], **by_depth[(b, n)].out()} for b in range(len(DEPTH_BINS))] for n in names}
    return metrics, distance, depth


def scene_mae(ds, pool, cfg, method, kinds, ratio):
    """기존 결론과 같은 방식: 장면별 MAE(실측 픽셀)의 평균."""
    seed = cfg["seed"]
    all_kinds = cfg["holes"]["kinds"]
    out = {}
    for kind in kinds:
        vals = []
        for i in range(len(ds)):
            s = ds[i]
            hole, valid, holed = make_case(s, kind, all_kinds.index(kind), i, ratio, seed, pool)
            m = hole & s.raw_valid
            if m.any():
                vals.append(float(np.abs(method.fill(holed, valid, s.rgb)[m] - s.depth[m]).mean()))
        out[kind] = float(np.mean(vals))
    return out


def part_ratio(ds, pool, cfg):
    telea, unet = get_methods(["opencv_telea", "unet_depth"])
    kinds = ["random", "edge", "reflect"]  # 실제 센서 누락은 빌려 온 모양이라 크기를 정할 수 없다
    res = []
    for r in RATIOS:
        t, u = scene_mae(ds, pool, cfg, telea, kinds, r), scene_mae(ds, pool, cfg, unet, kinds, r)
        res.append({"ratio": r, "opencv_telea": t, "unet_depth": u})
        print(f"  구멍 {r:.0%}: " + ", ".join(f"{k} {t[k]:.3f}/{u[k]:.3f}" for k in kinds), flush=True)
    return res


def part_seeds(ds, pool, cfg):
    from depthfill.methods.unet import UNetFill

    kinds = cfg["holes"]["kinds"]
    telea = get_methods(["opencv_telea"])[0]
    base = scene_mae(ds, pool, cfg, telea, kinds, cfg["holes"]["ratio"])
    runs = {}
    for name in ("unet_depth", "unet_depth_s1", "unet_depth_s2"):
        if (ROOT / "checkpoints" / f"{name}.pt").exists():
            runs[name] = scene_mae(ds, pool, cfg, UNetFill(name, use_rgb=False), kinds, cfg["holes"]["ratio"])
            print(f"  {name}: " + ", ".join(f"{k} {runs[name][k]:.4f}" for k in kinds), flush=True)
    summary = {}
    for k in kinds:
        v = np.array([r[k] for r in runs.values()])
        summary[k] = {"telea": base[k], "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "min": float(v.min()), "max": float(v.max()), "all_beat_telea": bool((v < base[k]).all())}
    return {"runs": runs, "telea": base, "summary": summary}


def part_speed(ds, pool, cfg):
    import torch

    from depthfill.methods.unet import UNetFill

    s = ds[0]
    _, valid, holed = make_case(s, "random", 0, 0, cfg["holes"]["ratio"], cfg["seed"], pool)
    res = {}

    def bench(name, fn, n=200):
        for _ in range(10):
            fn()
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t) * 1000)
        res[name] = {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}
        print(f"  {name}: 중앙값 {res[name]['median_ms']:.2f}ms, 95% {res[name]['p95_ms']:.2f}ms", flush=True)

    telea = get_methods(["opencv_telea"])[0]
    bench("opencv_telea_cpu", lambda: telea.fill(holed, valid, s.rgb))
    u = UNetFill("unet_depth", use_rgb=False)
    if u.device == "cuda":
        def gpu():
            u.fill(holed, valid, s.rgb)
            torch.cuda.synchronize()
        bench("unet_depth_gpu", gpu)
    u.model = u.model.to("cpu")
    u.device = "cpu"
    bench("unet_depth_cpu", lambda: u.fill(holed, valid, s.rgb), n=100)
    res["note"] = "120x160 한 장, Telea 선채움 포함, 배치 1. " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    return res


def figures(a):
    korean_font()
    if "distance" in a:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for n, rows in a["distance"].items():
            ax.plot(DIST_LABEL, [r["mae_m"] * 100 for r in rows], "o-", color=COLOR[n], label=NAME_KO[n])
        ax.set_xlabel("구멍 가장자리에서 안쪽으로 떨어진 거리 (픽셀)")
        ax.set_ylabel("평균 오차 (cm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_IMG / "analysis_distance.png", dpi=110)
        plt.close(fig)
    if "depth" in a:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        for n, rows in a["depth"].items():
            ax.plot(DEPTH_LABEL, [r["mae_m"] * 100 for r in rows], "o-", color=COLOR[n], label=NAME_KO[n])
        ax.set_xlabel("실제 거리")
        ax.set_ylabel("평균 오차 (cm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_IMG / "analysis_depth.png", dpi=110)
        plt.close(fig)
    if "ratio" in a:
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
        for ax, kind in zip(axes, ["random", "edge", "reflect"], strict=True):
            xs = [r["ratio"] * 100 for r in a["ratio"]]
            for n in ("opencv_telea", "unet_depth"):
                ax.plot(xs, [r[n][kind] * 100 for r in a["ratio"]], "o-", color=COLOR[n], label=NAME_KO[n])
            ax.axvline(15, color="#999", ls=":", lw=1)
            ax.set_title(KIND_KO[kind], fontsize=10)
            ax.set_xlabel("가린 비율 (%)")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("평균 오차 (cm)")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_IMG / "analysis_ratio.png", dpi=110)
        plt.close(fig)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["breakdowns", "ratio", "seeds", "speed"])
    args = ap.parse_args()
    ds = NYUDataset(ROOT / cfg["data"]["nyu_npz"], split="test")
    pool = ds.sensor_masks()
    path = ROOT / "results" / "analysis.json"
    a = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if "breakdowns" in args.parts:
        print("[분해] 유형·가장자리 거리·실제 거리", flush=True)
        a["metrics"], a["distance"], a["depth"] = part_breakdowns(
            ds, pool, cfg, get_methods(["opencv_telea", "unet_depth", "unet_rgbd"]))
    if "ratio" in args.parts:
        print("[구멍 크기]", flush=True)
        a["ratio"] = part_ratio(ds, pool, cfg)
    if "seeds" in args.parts:
        print("[시드]", flush=True)
        a["seeds"] = part_seeds(ds, pool, cfg)
    if "speed" in args.parts:
        print("[속도]", flush=True)
        a["speed"] = part_speed(ds, pool, cfg)
    path.write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
    figures(a)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()

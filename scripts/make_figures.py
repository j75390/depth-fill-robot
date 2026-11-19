"""결과 그림과 뷰어 데이터 만들기 (이슈 #7).

- docs/images/examples.png     : 구멍 유형별 입력 / 방법별 결과 / 정답
- docs/images/failures.png     : 컬러 U-Net 오차가 가장 큰 시험 장면들(실패 사례)
- docs/images/training.png     : 학습 곡선(검증 MAE)
- docs/viewer/data.js          : 전/후 슬라이더·3D 점 보기용 데이터(장면 몇 개만, 작게)

예) python scripts/make_figures.py
"""
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from depthfill.config import ROOT, load_config
from depthfill.data.nyu import NYUDataset
from depthfill.eval.harness import make_case
from depthfill.eval.metrics import masked_errors
from depthfill.methods.registry import get_methods

KIND_KO = {"random": "무작위", "edge": "물체 가장자리", "reflect": "반사 면", "sensor": "실제 센서 누락"}
METHOD_KO = {"diffusion": "확산 채우기", "opencv_telea": "OpenCV", "unet_depth": "U-Net(깊이)", "unet_rgbd": "U-Net(컬러+깊이)"}
SHOW = ["diffusion", "opencv_telea", "unet_depth", "unet_rgbd"]


def korean_font() -> None:
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        if any(name in f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def depth_img(ax, d, vmax, title=None):
    ax.imshow(np.where(d > 0, d, np.nan), cmap="turbo", vmin=0, vmax=vmax)
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def png_b64(rgb01: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray((np.clip(rgb01, 0, 1) * 255).astype(np.uint8)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def colorize(d: np.ndarray, vmax: float) -> np.ndarray:
    rgb = plt.get_cmap("turbo")(np.clip(d / vmax, 0, 1))[..., :3]
    rgb[d <= 0] = 0
    return rgb


def main() -> None:
    korean_font()
    cfg = load_config()
    seed, ratio, kinds = cfg["seed"], cfg["holes"]["ratio"], cfg["holes"]["kinds"]
    ds = NYUDataset(ROOT / cfg["data"]["nyu_npz"], split="test")
    pool = ds.sensor_masks()
    methods = {m.name: m for m in get_methods(SHOW)}
    out_img = ROOT / "docs" / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    # 1) 예시: 한 장면 × 구멍 유형 4가지
    index = 12
    s = ds[index]
    vmax = float(np.quantile(s.depth, 0.99))
    fig, axes = plt.subplots(len(kinds), 3 + len(SHOW), figsize=(2.1 * (3 + len(SHOW)), 1.75 * len(kinds)))
    for r, kind in enumerate(kinds):
        k_index = cfg["holes"]["kinds"].index(kind)
        hole, valid, holed = make_case(s, kind, k_index, index, ratio, seed, pool)
        axes[r, 0].imshow(s.rgb)
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(KIND_KO[kind], fontsize=9)
        depth_img(axes[r, 1], holed, vmax, "입력(구멍)" if r == 0 else None)
        for c, name in enumerate(SHOW):
            pred = methods[name].fill(holed, valid, s.rgb)
            e = masked_errors(pred, s.depth, hole & s.raw_valid)
            depth_img(axes[r, 2 + c], pred, vmax, METHOD_KO[name] if r == 0 else None)
            axes[r, 2 + c].set_xlabel(f"MAE {e['mae']:.3f}m", fontsize=8)
        depth_img(axes[r, -1], np.where(s.raw_valid, s.depth, 0), vmax, "정답(실측)" if r == 0 else None)
    fig.tight_layout()
    fig.savefig(out_img / "examples.png", dpi=110)
    plt.close(fig)

    # 2) 실패 사례: 컬러 U-Net의 장면별 오차(실측 기준) 상위
    rows_path = ROOT / "results" / "rows.csv"
    import csv

    with open(rows_path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == "unet_rgbd" and r.get("mae_raw") not in (None, "", "nan")]
    rows.sort(key=lambda r: float(r["mae_raw"]), reverse=True)
    worst = rows[:4]
    fig, axes = plt.subplots(len(worst), 4, figsize=(9, 2.0 * len(worst)))
    names = [f"nyu_{int(i):04d}" for i in ds.idx]
    for r, row in enumerate(worst):
        i = names.index(row["sample"])
        sw = ds[i]
        kind = row["kind"]
        hole, valid, holed = make_case(sw, kind, kinds.index(kind), i, ratio, seed, pool)
        pred = methods["unet_rgbd"].fill(holed, valid, sw.rgb)
        vm = float(np.quantile(sw.depth, 0.99))
        axes[r, 0].imshow(sw.rgb)
        axes[r, 0].set_ylabel(f"{row['sample']}\n{KIND_KO[kind]}", fontsize=8)
        depth_img(axes[r, 1], holed, vm, "입력" if r == 0 else None)
        depth_img(axes[r, 2], pred, vm, "U-Net(컬러+깊이)" if r == 0 else None)
        err = np.where(hole & sw.raw_valid, np.abs(pred - sw.depth), np.nan)
        axes[r, 3].imshow(err, cmap="magma", vmin=0, vmax=1.0)
        axes[r, 3].set_facecolor("#222")
        axes[r, 3].set_title("구멍 속 오차(0~1m)" if r == 0 else None, fontsize=9)
        axes[r, 3].set_xlabel(f"MAE {float(row['mae_raw']):.3f}m", fontsize=8)
        for a in axes[r]:
            a.set_xticks([])
            a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_img / "failures.png", dpi=110)
    plt.close(fig)

    # 3) 학습 곡선
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for name in ("unet_depth", "unet_rgbd"):
        p = ROOT / "results" / f"train_{name}.json"
        if p.exists():
            h = json.loads(p.read_text(encoding="utf-8"))["history"]
            ax.plot([e["epoch"] for e in h], [e["val_mae_m"] for e in h], label=f"{METHOD_KO[name]} 검증")
            ax.plot([e["epoch"] for e in h], [e["train_mae_m"] for e in h], "--", alpha=0.5, label=f"{METHOD_KO[name]} 학습")
    ax.set_xlabel("에폭")
    ax.set_ylabel("가린 곳 MAE (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_img / "training.png", dpi=110)
    plt.close(fig)

    # 4) 뷰어 데이터: 장면 4개 × 구멍 유형, 그림은 PNG로, 3D 점은 절반 해상도로
    scenes = []
    for index in (12, 60, 150, 400):
        s = ds[index]
        vmax = float(np.quantile(s.depth, 0.99))
        entry = {"name": s.name, "rgb": png_b64(s.rgb), "gt": png_b64(colorize(np.where(s.raw_valid, s.depth, 0), vmax)), "cases": {}}
        for kind in kinds:
            hole, valid, holed = make_case(s, kind, kinds.index(kind), index, ratio, seed, pool)
            case = {"input": png_b64(colorize(holed, vmax)), "methods": {}}
            for name in SHOW:
                pred = methods[name].fill(holed, valid, s.rgb)
                e = masked_errors(pred, s.depth, hole & s.raw_valid)
                small = pred[::2, ::2]
                case["methods"][name] = {"img": png_b64(colorize(pred, vmax)), "mae": round(e["mae"], 4),
                                         "points": [round(float(v), 3) for v in small.ravel()]}
            case["hole"] = [int(v) for v in hole[::2, ::2].ravel()]
            entry["cases"][kind] = case
        entry["colors"] = [int(v) for v in (s.rgb[::2, ::2] * 255).astype(np.uint8).ravel()]
        entry["size"] = [s.depth.shape[0] // 2, s.depth.shape[1] // 2]
        scenes.append(entry)
    viewer = ROOT / "docs" / "viewer"
    viewer.mkdir(parents=True, exist_ok=True)
    (viewer / "data.js").write_text("window.VIEWER_DATA = " + json.dumps(
        {"scenes": scenes, "kinds": KIND_KO, "methods": METHOD_KO}, ensure_ascii=False) + ";", encoding="utf-8")
    print("저장:", out_img / "examples.png", out_img / "failures.png", out_img / "training.png", viewer / "data.js")


if __name__ == "__main__":
    main()

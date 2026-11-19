"""평가 도구: 여러 방법을 같은 장면·같은 구멍에서 비교하고, 유형별로 정리한다."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from depthfill.eval.metrics import masked_errors
from depthfill.holes.generators import make_holes


def make_case(sample, kind: str, k_index: int, index: int, ratio: float, seed: int, sensor_pool=None):
    """한 장면·한 구멍 유형의 입력을 만든다. 같은 인자면 항상 같은 구멍이 나온다.

    방법에 넘기는 입력은 '센서가 잰 값 - 우리가 뚫은 구멍'이다. 실제 데이터(NYU)에는 센서가 원래 못 잰 곳도
    있으므로 그곳도 값이 없는 채로 넘긴다. 정답 깊이는 방법에 넘기지 않는다.
    """
    rng = np.random.default_rng([seed, index, k_index])
    hole = make_holes(kind, sample.rgb, sample.depth, rng, ratio, sensor_pool=sensor_pool, index=index)
    raw_valid = getattr(sample, "raw_valid", None)
    measured = np.ones_like(hole) if raw_valid is None else raw_valid
    valid = measured & ~hole
    holed = np.where(valid, sample.depth, 0.0).astype(np.float32)
    return hole, valid, holed


def evaluate(dataset, methods, kinds, ratio: float, seed: int, sensor_pool=None) -> list[dict]:
    """모든 (장면 × 구멍 유형 × 방법) 조합의 오차를 한 줄씩 돌려준다.

    같은 장면·유형이면 방법과 상관없이 같은 구멍을 쓴다(공정한 비교).
    실제 데이터는 두 기준으로 잰다: 채운 정답 전체(mae, rmse)와 센서가 실제로 잰 픽셀만(mae_raw, rmse_raw).
    """
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        raw_valid = getattr(sample, "raw_valid", None)
        for k_index, kind in enumerate(kinds):
            hole, valid, holed = make_case(sample, kind, k_index, index, ratio, seed, sensor_pool)
            for method in methods:
                pred = method.fill(holed, valid, sample.rgb)
                err = masked_errors(pred, sample.depth, hole)
                row = {"sample": sample.name, "kind": kind, "method": method.name,
                       "hole_ratio": float(hole.mean()), **err}
                if raw_valid is not None:
                    raw_err = masked_errors(pred, sample.depth, hole & raw_valid)
                    row.update(mae_raw=raw_err["mae"], rmse_raw=raw_err["rmse"], n_raw=raw_err["n"])
                rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    """(구멍 유형, 방법)별 평균 오차."""
    groups = defaultdict(list)
    for row in rows:
        if row["n"] > 0:
            groups[(row["kind"], row["method"])].append(row)
    out = {}
    for (kind, method), rs in sorted(groups.items()):
        entry = {
            "mae": float(np.mean([r["mae"] for r in rs])),
            "rmse": float(np.mean([r["rmse"] for r in rs])),
            "samples": len(rs),
        }
        raw = [r for r in rs if r.get("n_raw", 0) > 0]
        if raw:
            entry["mae_raw"] = float(np.mean([r["mae_raw"] for r in raw]))
            entry["rmse_raw"] = float(np.mean([r["rmse_raw"] for r in raw]))
        out[f"{kind}/{method}"] = entry
    return out


def worst_cases(rows: list[dict], method: str, top: int = 5) -> list[dict]:
    """한 방법의 오차가 큰 사례(실패 사례 분석용)."""
    mine = [r for r in rows if r["method"] == method and r["n"] > 0]
    return sorted(mine, key=lambda r: r["rmse"], reverse=True)[:top]


def format_table(summary: dict) -> str:
    has_raw = any("mae_raw" in v for v in summary.values())
    head = f"{'유형/방법':<26}{'MAE(m)':>9}{'RMSE(m)':>9}"
    head += f"{'MAE실측':>9}{'RMSE실측':>9}" if has_raw else ""
    lines = [head + f"{'장면수':>7}"]
    for key, v in summary.items():
        line = f"{key:<26}{v['mae']:>9.4f}{v['rmse']:>9.4f}"
        if has_raw:
            line += f"{v.get('mae_raw', float('nan')):>9.4f}{v.get('rmse_raw', float('nan')):>9.4f}"
        lines.append(line + f"{v['samples']:>7d}")
    return "\n".join(lines)


def save_results(rows: list[dict], summary: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "rows.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

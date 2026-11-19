"""평가 도구: 여러 방법을 같은 장면·같은 구멍에서 비교하고, 유형별로 정리한다."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from depthfill.eval.metrics import masked_errors
from depthfill.holes.generators import make_holes


def evaluate(dataset, methods, kinds, ratio: float, seed: int) -> list[dict]:
    """모든 (장면 × 구멍 유형 × 방법) 조합의 오차를 한 줄씩 돌려준다.

    같은 장면·유형이면 방법과 상관없이 같은 구멍을 쓴다(공정한 비교).
    """
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        for k_index, kind in enumerate(kinds):
            rng = np.random.default_rng([seed, index, k_index])
            hole = make_holes(kind, sample.rgb, sample.depth, rng, ratio)
            valid = ~hole
            holed = np.where(valid, sample.depth, 0.0).astype(np.float32)
            for method in methods:
                pred = method.fill(holed, valid, sample.rgb)
                err = masked_errors(pred, sample.depth, hole)
                rows.append({
                    "sample": sample.name, "kind": kind, "method": method.name,
                    "hole_ratio": float(hole.mean()), **err,
                })
    return rows


def summarize(rows: list[dict]) -> dict:
    """(구멍 유형, 방법)별 평균 오차."""
    groups = defaultdict(list)
    for row in rows:
        if row["n"] > 0:
            groups[(row["kind"], row["method"])].append(row)
    return {
        f"{kind}/{method}": {
            "mae": float(np.mean([r["mae"] for r in rs])),
            "rmse": float(np.mean([r["rmse"] for r in rs])),
            "samples": len(rs),
        }
        for (kind, method), rs in sorted(groups.items())
    }


def worst_cases(rows: list[dict], method: str, top: int = 5) -> list[dict]:
    """한 방법의 오차가 큰 사례(실패 사례 분석용)."""
    mine = [r for r in rows if r["method"] == method and r["n"] > 0]
    return sorted(mine, key=lambda r: r["rmse"], reverse=True)[:top]


def format_table(summary: dict) -> str:
    lines = [f"{'유형/방법':<24}{'MAE(m)':>10}{'RMSE(m)':>10}{'장면수':>8}"]
    for key, v in summary.items():
        lines.append(f"{key:<24}{v['mae']:>10.4f}{v['rmse']:>10.4f}{v['samples']:>8d}")
    return "\n".join(lines)


def save_results(rows: list[dict], summary: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "rows.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

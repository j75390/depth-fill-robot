"""평가 실행: 방법 x 구멍 유형 비교표를 만들고 results/에 저장한다.

예) python scripts/run_eval.py --synthetic --n 20
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from depthfill.config import ROOT, load_config
from depthfill.data.datasets import SyntheticDataset
from depthfill.eval.harness import evaluate, format_table, save_results, summarize
from depthfill.methods.registry import get_methods
from depthfill.seed import set_seed


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="합성 장면으로 실행(실제 데이터 로더는 이슈 #2)")
    ap.add_argument("--n", type=int, default=cfg["eval"]["n_samples"])
    ap.add_argument("--methods", nargs="+", default=cfg["eval"]["methods"])
    ap.add_argument("--out", default=cfg["eval"]["results_dir"])
    args = ap.parse_args()
    if not args.synthetic:
        sys.exit("아직 실제 데이터 로더가 없습니다(이슈 #2). 지금은 --synthetic으로 실행하세요.")

    set_seed(cfg["seed"])
    dataset = SyntheticDataset(args.n, tuple(cfg["data"]["image_size"]), cfg["seed"])
    rows = evaluate(dataset, get_methods(args.methods), cfg["holes"]["kinds"], cfg["holes"]["ratio"], cfg["seed"])
    summary = summarize(rows)
    print(format_table(summary))
    out_dir = ROOT / args.out
    save_results(rows, summary, out_dir)
    print(f"\n저장: {out_dir / 'summary.json'}, {out_dir / 'rows.csv'}")


if __name__ == "__main__":
    main()

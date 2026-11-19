"""평가 실행: 방법 x 구멍 유형 비교표를 만들고 results/에 저장한다.

예) python scripts/run_eval.py                       # NYU 시험 분할(654장), 설정 파일의 방법 전부
    python scripts/run_eval.py --n 50 --methods nofill diffusion
    python scripts/run_eval.py --synthetic --n 20   # 합성 장면(틀 확인용, 숫자에 의미 없음)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from depthfill.config import ROOT, load_config
from depthfill.data.datasets import SyntheticDataset
from depthfill.eval.harness import evaluate, format_table, save_results, summarize, worst_cases
from depthfill.methods.registry import get_methods
from depthfill.seed import set_seed


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="합성 장면으로 실행(틀 확인용)")
    ap.add_argument("--n", type=int, default=cfg["eval"]["n_samples"])
    ap.add_argument("--methods", nargs="+", default=cfg["eval"]["methods"])
    ap.add_argument("--kinds", nargs="+", default=cfg["holes"]["kinds"])
    ap.add_argument("--out", default=cfg["eval"]["results_dir"])
    args = ap.parse_args()

    set_seed(cfg["seed"])
    kinds, pool = args.kinds, None
    if args.synthetic:
        dataset = SyntheticDataset(args.n, tuple(cfg["data"]["image_size"]), cfg["seed"])
        kinds = [k for k in kinds if k != "sensor"]  # 합성 장면에는 실제 누락 모양이 없다
    else:
        from depthfill.data.nyu import NYUDataset

        npz = ROOT / cfg["data"]["nyu_npz"]
        if not npz.exists():
            sys.exit(f"{npz}가 없습니다. 먼저 python scripts/prepare_nyu.py 를 실행하세요.")
        dataset = NYUDataset(npz, split="test", limit=args.n)
        pool = dataset.sensor_masks()  # 시험 장면끼리 누락 모양을 빌려 쓴다(자기 것은 제외)

    rows = evaluate(dataset, get_methods(args.methods), kinds, cfg["holes"]["ratio"], cfg["seed"], sensor_pool=pool)
    summary = summarize(rows)
    print(format_table(summary))
    out_dir = ROOT / args.out
    save_results(rows, summary, out_dir)
    worst = {m: worst_cases(rows, m) for m in args.methods}
    (out_dir / "worst_cases.json").write_text(__import__("json").dumps(worst, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out_dir / 'summary.json'}, {out_dir / 'rows.csv'}, {out_dir / 'worst_cases.json'}")


if __name__ == "__main__":
    main()

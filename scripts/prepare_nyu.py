"""NYU Depth v2 원본(.mat)을 잘라 줄여 data/nyu_<H>x<W>.npz로 저장한다(한 번만 실행).

예) python scripts/prepare_nyu.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from depthfill.config import ROOT, load_config
from depthfill.data.nyu import prepare_nyu


def main() -> None:
    cfg = load_config()["data"]
    t = time.time()
    out = prepare_nyu(ROOT / cfg["nyu_mat"], ROOT / cfg["nyu_splits"], ROOT / cfg["nyu_npz"], tuple(cfg["image_size"]))
    print(f"저장: {out} ({out.stat().st_size / 1e6:.1f}MB, {time.time() - t:.0f}초)")


if __name__ == "__main__":
    main()

"""품질 기준 시험(선택 과제 #9): 모델이 가장 강한 기준선보다 나빠지면 실패(종료 코드 1).

모델을 다시 학습하거나 코드를 바꾼 뒤 배포 전에 돌린다. 시험 분할 전체(654장, 결론과 같은 조건), 구멍 유형 4가지에서
실측 픽셀 기준 MAE를 비교하고, 같은 모델을 API로 불렀을 때 직접 부른 결과와 같은지도 확인한다.

앞부분 몇 장만 쓰면 결과가 뒤집힐 수 있다(실제 센서 누락 유형은 이긴 장면이 60%라, 앞 60장에서는
Telea가 0.0633m로 U-Net 0.0650m보다 낫게 나왔다). 그래서 기본값은 전체이고 약 35초 걸린다.

예) python scripts/quality_gate.py --model unet_rgbd
    python scripts/quality_gate.py --model unet_depth --n 100   # 빠른 확인용(판정은 전체로)
"""
import argparse
import base64
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from depthfill.config import ROOT, load_config
from depthfill.data.nyu import NYUDataset
from depthfill.eval.harness import evaluate, make_case, summarize
from depthfill.methods.registry import get_methods


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet_rgbd")
    ap.add_argument("--baseline", default="opencv_telea")
    ap.add_argument("--n", type=int, default=None, help="시험 장면 수(기본: 전체)")
    args = ap.parse_args()
    ds = NYUDataset(ROOT / cfg["data"]["nyu_npz"], split="test", limit=args.n)
    pool = ds.sensor_masks()
    kinds, ratio, seed = cfg["holes"]["kinds"], cfg["holes"]["ratio"], cfg["seed"]
    model, base = get_methods([args.model, args.baseline])
    summary = summarize(evaluate(ds, [model, base], kinds, ratio, seed, sensor_pool=pool))

    failures = []
    print("구멍 유형 " + f"{args.baseline:>14}{args.model:>14}")  # 한글 4자(8칸)+공백 2개 = 10칸, 아래 줄과 맞춤
    for kind in kinds:
        b, m = summary[f"{kind}/{args.baseline}"]["mae_raw"], summary[f"{kind}/{args.model}"]["mae_raw"]
        mark = "통과" if m <= b else "실패"
        print(f"{kind:<10}{b:>14.4f}{m:>14.4f}  {mark}")
        if m > b:
            failures.append(f"{kind}: {args.model} {m:.4f}m > {args.baseline} {b:.4f}m")

    # API로 부른 결과가 직접 부른 결과와 같은지(밀리미터 반올림 차이만 허용)
    from depthfill.api.app import create_app

    client = create_app([args.model]).test_client()
    s = ds[0]
    hole, valid, holed = make_case(s, "sensor", kinds.index("sensor"), 0, ratio, seed, pool)

    def png(a):
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    r = client.post("/v1/fill", json={"depth_png": png(np.round(holed * 1000).astype(np.uint16)),
                                      "rgb_png": png((s.rgb * 255).round().astype(np.uint8)), "method": args.model})
    if r.status_code != 200:
        failures.append(f"API 응답 {r.status_code}: {r.json}")
    else:
        api = np.array(Image.open(io.BytesIO(base64.b64decode(r.json["depth_png"])))).astype(np.float32) / 1000
        direct = model.fill(np.round(holed * 1000) / 1000, valid, (s.rgb * 255).round() / 255)
        gap = float(np.abs(api - direct)[hole].max())
        print(f"API와 직접 호출의 최대 차이: {gap * 1000:.1f}mm")
        if gap > 0.01:
            failures.append(f"API 결과가 직접 호출과 {gap:.3f}m 다르다")

    if failures:
        print("\n품질 기준 실패:\n- " + "\n- ".join(failures))
        sys.exit(1)
    print("\n품질 기준 통과")


if __name__ == "__main__":
    main()

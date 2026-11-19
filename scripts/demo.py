"""시연: 데이터 → 구멍 → 방법 비교 → API 호출 → 품질 기준(통과/실패) → 시험을 한 번에 돌린다.

화면 녹화용으로 단계마다 제목과 결과를 보기 좋게 찍는다. 학습된 체크포인트(checkpoints/*.pt)가 필요하다.

예) python scripts/demo.py               # 시험 654장 전체로 약 2분
    python scripts/demo.py --record results/demo_log.json   # 터미널 GIF용 기록도 남긴다
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from depthfill.config import ROOT, load_config
from depthfill.data.nyu import NYUDataset
from depthfill.eval.harness import evaluate, make_case, summarize
from depthfill.methods.registry import get_methods

if os.name == "nt":
    os.system("")  # Windows 콘솔에서 ANSI 색 켜기

STYLES = {"title": "\033[1;38;5;208m", "ok": "\033[1;32m", "bad": "\033[1;31m", "dim": "\033[90m",
          "hi": "\033[1;36m", "plain": ""}
KIND_KO = {"edge": "물체 가장자리", "random": "무작위", "reflect": "반사 면", "sensor": "실제 센서 누락"}
EVENTS: list[tuple[float, str, str]] = []
T0 = time.perf_counter()


def say(text: str = "", style: str = "plain") -> None:
    EVENTS.append((round(time.perf_counter() - T0, 2), style, text))
    print(f"{STYLES[style]}{text}\033[0m" if STYLES[style] else text, flush=True)


def step(n: int, title: str) -> None:
    say()
    say(f"[{n}/6] {title}", "title")
    say("─" * 60, "dim")


def pad(text: str, width: int) -> str:
    """한글은 터미널에서 두 칸을 차지하므로 칸 수로 맞춘다."""
    cells = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(0, width - cells)


def bar(value: float, vmax: float, width: int = 22) -> str:
    return "█" * max(1, round(value / vmax * width))


def png_b64(a: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run_gate(model: str, n: int | None) -> tuple[int, list[str]]:
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "quality_gate.py"), "--model", model, *(["--n", str(n)] if n else [])],
                       capture_output=True, text=True, encoding="utf-8", check=False, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return p.returncode, [ln for ln in p.stdout.splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="시험 장면 수(기본: 전체 654장)")
    ap.add_argument("--record", help="터미널 GIF용 기록(JSON) 경로")
    args = ap.parse_args()
    cfg = load_config()
    kinds, ratio, seed = cfg["holes"]["kinds"], cfg["holes"]["ratio"], cfg["seed"]

    say("depth-fill-robot 시연 — 로봇 카메라 깊이 영상의 빈 곳 채우기", "hi")
    say("모든 숫자는 지금 이 자리에서 다시 계산합니다.", "dim")

    step(1, "실제 센서 데이터 불러오기 (NYU Depth v2, 공식 시험 분할)")
    ds = NYUDataset(ROOT / cfg["data"]["nyu_npz"], split="test", limit=args.n)
    pool = ds.sensor_masks()
    s = ds[0]
    say(f"시험 장면 {len(ds)}장 · 해상도 {s.depth.shape[0]}x{s.depth.shape[1]} · 깊이 범위 "
        f"{s.depth[s.raw_valid].min():.2f}~{s.depth[s.raw_valid].max():.2f}m")
    say(f"센서가 원래 못 잰 픽셀: {100 * (~s.raw_valid).mean():.1f}%  → 정답은 실제로 잰 픽셀만 씁니다")

    step(2, "구멍 내기 (네 가지 유형, 시드 고정)")
    for k, kind in enumerate(kinds):
        hole, _, _ = make_case(s, kind, k, 0, ratio, seed, pool)
        say(f"  {pad(KIND_KO[kind], 16)}가린 픽셀 {100 * hole.mean():5.1f}%")

    step(3, f"같은 장면·같은 구멍으로 방법 비교 ({len(ds)}장 × {len(kinds)}유형)")
    names = ["meanfill", "opencv_telea", "unet_depth"]
    label = {"meanfill": "평균 채우기", "opencv_telea": "OpenCV Telea", "unet_depth": "U-Net(내 모델)"}
    t = time.perf_counter()
    summary = summarize(evaluate(ds, get_methods(names), kinds, ratio, seed, sensor_pool=pool))
    say(f"평가 {time.perf_counter() - t:.1f}초 · 오차(MAE, m) 낮을수록 좋음", "dim")
    for kind in kinds:
        vals = {m: summary[f"{kind}/{m}"]["mae_raw"] for m in names}
        best = min(vals, key=vals.get)
        say(f"  {KIND_KO[kind]}", "hi")
        for m in names[1:]:
            mark = "  ← 가장 좋음" if m == best else ""
            say(f"    {pad(label[m], 16)}{vals[m]:.3f}  {bar(vals[m], 0.25)}{mark}", "ok" if m == best else "plain")
        tel, me = vals["opencv_telea"], vals["unet_depth"]
        say(f"    Telea 대비 {100 * (me - tel) / tel:+.0f}%   (평균 채우기는 {vals['meanfill']:.3f}m)", "dim")

    step(4, "모델 API 호출 (POST /v1/fill, 16비트 PNG 밀리미터)")
    from depthfill.api.app import create_app

    client = create_app(["opencv_telea", "unet_depth"], default="unet_depth").test_client()
    h = client.get("/v1/health").json
    say(f"GET /v1/health → {h['status']} · 방법 {h['methods']}")
    _, _, holed = make_case(s, "sensor", kinds.index("sensor"), 0, ratio, seed, pool)
    r = client.post("/v1/fill", json={"depth_png": png_b64(np.round(holed * 1000).astype(np.uint16))})
    j = r.json
    say(f"POST /v1/fill → {r.status_code} · 채운 픽셀 {j['filled_pixels']}개 · {j['elapsed_ms']}ms", "ok")
    bad = client.post("/v1/fill", json={"depth_png": "이건 PNG가 아님"})
    say(f"잘못된 입력 → {bad.status_code} · {bad.json['error'][:40]}", "dim")

    step(5, "품질 기준 시험 — 모델이 기준선(Telea)보다 나쁘면 배포를 막는다")
    say("(1) 학습한 U-Net", "hi")
    code, lines = run_gate("unet_depth", args.n)
    for ln in lines:
        say("   " + ln, "ok" if "통과" in ln else ("bad" if "실패" in ln else "plain"))
    say(f"   종료 코드 {code}", "ok" if code == 0 else "bad")
    say("(2) 일부러 나쁜 모델(평균 채우기)을 넣으면", "hi")
    code, lines = run_gate("meanfill", args.n)
    for ln in lines[:6]:
        say("   " + ln, "bad" if "실패" in ln else "plain")
    say(f"   종료 코드 {code} → 배포 중단", "bad" if code else "ok")

    step(6, "자동 시험 (데이터 없이 도는 단위 시험)")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=ROOT,
                       capture_output=True, text=True, encoding="utf-8", check=False)
    last = [ln for ln in p.stdout.splitlines() if ln.strip()][-1]
    say("   " + last, "ok" if p.returncode == 0 else "bad")

    say()
    say(f"끝 — 총 {time.perf_counter() - T0:.0f}초", "hi")
    if args.record:
        Path(args.record).write_text(json.dumps(EVENTS, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

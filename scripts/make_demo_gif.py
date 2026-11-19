"""README 맨 위 시연 GIF: 컬러 | 깊이(구멍 → 슬라이더가 지나가며 U-Net으로 채움) | 돌아가는 3D 점.

장면은 뷰어와 같은 시험 장면(12, 60, 150)을 쓴다(잘 나온 장면을 고르지 않았다). 오차는 그 장면의 실측 픽셀 기준.

예) python scripts/make_demo_gif.py   → docs/images/demo.gif
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from depthfill.config import ROOT, load_config
from depthfill.data.nyu import NYUDataset
from depthfill.eval.harness import make_case
from depthfill.eval.metrics import masked_errors
from depthfill.methods.registry import get_methods

PW, PH, GAP, TOP, BOT = 320, 240, 14, 34, 46
W, H = 3 * PW + 4 * GAP, TOP + PH + BOT
BG, INK, MUTED, ORANGE = (17, 19, 24), (232, 234, 238), (150, 157, 170), (255, 138, 61)
SCENES = [(12, "reflect", "반사 면"), (60, "random", "무작위 구멍"), (150, "sensor", "실제 센서 누락")]
TURBO = matplotlib.colormaps["turbo"]


def font(size, bold=False):
    try:
        return ImageFont.truetype(f"C:/Windows/Fonts/{'malgunbd' if bold else 'malgun'}.ttf", size)
    except OSError:
        return ImageFont.load_default()


def colorize(d, vmax):
    rgb = (TURBO(np.clip(d / vmax, 0, 1))[..., :3] * 255).astype(np.uint8)
    rgb[d <= 0] = (40, 42, 48)
    return rgb


def up(a):
    return np.array(Image.fromarray(a).resize((PW, PH), Image.NEAREST))


def cloud(depth, rgb, hole, show_hole, yaw, pitch=0.28):
    """깊이를 3D 점으로 펴서 돌려 그린다. 채운 구멍 점은 주황색."""
    h, w = depth.shape
    sc = w / 560
    fx, cx, cy = 518.9 * sc, (325.6 - 41) * sc, (253.7 - 45) * sc
    ys, xs = np.mgrid[0:h, 0:w]
    keep = (depth > 0) & (show_hole | ~hole)
    z = depth[keep]
    X, Y = (xs[keep] - cx) * z / fx, (ys[keep] - cy) * z / fx
    Z = z - np.median(z)
    X1, Z1 = X * np.cos(yaw) - Z * np.sin(yaw), X * np.sin(yaw) + Z * np.cos(yaw)
    Y2, Z2 = Y * np.cos(pitch) - Z1 * np.sin(pitch), Y * np.sin(pitch) + Z1 * np.cos(pitch) + 4.2
    col = (rgb[keep] * 255).astype(np.uint8)
    col[hole[keep]] = ORANGE
    order = np.argsort(-Z2)
    f = PW * 0.95
    u = (PW / 2 + X1 / Z2 * f).astype(int)[order]
    v = (PH / 2 + Y2 / Z2 * f).astype(int)[order]
    col = col[order]
    img = np.zeros((PH, PW, 3), np.uint8) + np.array([6, 7, 11], np.uint8)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < PW) & (vv >= 0) & (vv < PH)
            img[vv[ok], uu[ok]] = col[ok]
    return img


def outline(panel, hole):
    """채운 구멍의 경계를 흰 선으로 그린다(깊이 색과 겹치지 않게)."""
    m = up(hole.astype(np.uint8) * 255) > 0
    edge = m ^ (np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    out = panel.copy()
    out[edge & m] = (255, 255, 255)
    return out


def compose(rgb, depth_panel, cloud_panel, titles, caption, sub, slider=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for i, panel in enumerate((rgb, depth_panel, cloud_panel)):
        x = GAP + i * (PW + GAP)
        img.paste(Image.fromarray(panel), (x, TOP))
        d.text((x, 8), titles[i], font=font(15, True), fill=INK)
    if slider is not None:
        x = GAP + PW + GAP + slider
        d.line([x, TOP, x, TOP + PH], fill=(255, 255, 255), width=2)
    d.text((GAP, TOP + PH + 10), caption, font=font(17, True), fill=ORANGE)
    d.text((GAP + 430, TOP + PH + 13), sub, font=font(14), fill=MUTED)
    return img


def main() -> None:
    cfg = load_config()
    kinds, ratio, seed = cfg["holes"]["kinds"], cfg["holes"]["ratio"], cfg["seed"]
    ds = NYUDataset(ROOT / cfg["data"]["nyu_npz"], split="test")
    pool = ds.sensor_masks()
    unet, telea = get_methods(["unet_depth", "opencv_telea"])
    frames, durations = [], []
    t = 0
    for index, kind, kind_ko in SCENES:
        s = ds[index]
        hole, valid, holed = make_case(s, kind, kinds.index(kind), index, ratio, seed, pool)
        pred = unet.fill(holed, valid, s.rgb)
        e_u = masked_errors(pred, s.depth, hole & s.raw_valid)["mae"]
        e_t = masked_errors(telea.fill(holed, valid, s.rgb), s.depth, hole & s.raw_valid)["mae"]
        vmax = float(np.quantile(s.depth, 0.99))
        rgb = up((s.rgb * 255).astype(np.uint8))
        before, after = up(colorize(holed, vmax)), outline(up(colorize(pred, vmax)), hole)
        titles = ("컬러 카메라", f"깊이 — {kind_ko}", "3D 점 (주황 = 채운 곳)")
        miss = f"측정값이 빠진 곳 {100 * hole.mean():.0f}%"
        # 1) 구멍 난 상태
        for _ in range(10):
            frames.append(compose(rgb, before, cloud(holed, s.rgb, hole, False, 0.45 * np.sin(t / 9)), titles,
                                  f"입력: {miss}", f"{s.name} · NYU Depth v2 시험 장면"))
            durations.append(90)
            t += 1
        # 2) 슬라이더가 지나가며 채움
        steps = 16
        for k in range(1, steps + 1):
            cut = int(PW * k / steps)
            panel = before.copy()
            panel[:, :cut] = after[:, :cut]
            frames.append(compose(rgb, panel, cloud(pred, s.rgb, hole, k == steps, 0.45 * np.sin(t / 9)), titles,
                                  "U-Net으로 채우는 중…", f"{s.name} · NYU Depth v2 시험 장면", slider=cut))
            durations.append(60)
            t += 1
        # 3) 결과 + 3D 회전
        print(f"{s.name} {kind}: U-Net {e_u:.4f}m, Telea {e_t:.4f}m")
        result = f"U-Net 오차 {e_u * 100:.1f}cm  (OpenCV Telea {e_t * 100:.1f}cm)"
        for _ in range(22):
            frames.append(compose(rgb, after, cloud(pred, s.rgb, hole, True, 0.45 * np.sin(t / 9)), titles, result,
                                  "센서가 실제로 잰 픽셀 기준 평균 오차"))
            durations.append(90)
            t += 1
        durations[-1] = 900
    out = ROOT / "docs" / "images" / "demo.gif"
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=96) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    print(f"저장: {out} ({len(frames)}프레임, {sum(durations) / 1000:.1f}초, {out.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()

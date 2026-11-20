"""demo.py가 남긴 기록(--record)을 터미널 화면 GIF로 만든다(README 맨 위 시연용).

기록에 있는 실제 출력만 그린다. 시간은 보기 좋게 줄인다(평가처럼 오래 걸린 단계는 짧게 표시).

예) python scripts/demo.py --record results/demo_log.json
    python scripts/make_terminal_gif.py results/demo_log.json docs/images/demo_terminal.gif
"""
import json
import sys
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, PAD, LINE, CELL = 980, 600, 18, 19, 8.4
ROWS = (H - 48 - PAD) // LINE
BG, FG = (17, 19, 24), (220, 223, 228)
COLORS = {"title": (255, 138, 61), "ok": (98, 214, 130), "bad": (255, 99, 99), "dim": (130, 137, 150),
          "hi": (102, 204, 230), "plain": FG}
FONTS = "C:/Windows/Fonts"


def load_fonts():
    try:
        return ImageFont.truetype(f"{FONTS}/consola.ttf", 14), ImageFont.truetype(f"{FONTS}/malgun.ttf", 15)
    except OSError:  # 다른 운영체제
        f = ImageFont.load_default()
        return f, f


def wide(c: str) -> bool:
    return unicodedata.east_asian_width(c) in "WF"


def draw_line(d: ImageDraw.ImageDraw, y: int, text: str, color, fonts) -> None:
    mono, kor = fonts
    x = PAD
    for c in text:
        if c == "█":  # 글꼴의 블록 문자는 틈이 생겨서 직접 칠한다
            d.rectangle([x, y + 2, x + CELL, y + LINE - 4], fill=color)
            x += CELL
        elif wide(c):
            d.text((x + 0.5, y - 3), c, font=kor, fill=color)
            x += CELL * 2
        else:
            d.text((x, y), c, font=mono, fill=color)
            x += CELL


def frame(lines, fonts, cursor: str | None = None) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 32], fill=(36, 39, 46))
    for i, col in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([14 + i * 20, 10, 26 + i * 20, 22], fill=col)
    draw_line(d, 8, " " * 10 + "Depth Patch  —  python scripts/demo.py", (160, 165, 175), fonts)
    shown = lines[-ROWS:]
    for i, (style, text) in enumerate(shown):
        draw_line(d, 44 + i * LINE, text, COLORS.get(style, FG), fonts)
    if cursor is not None:
        draw_line(d, 44 + len(shown) * LINE, cursor, FG, fonts)
    return img


def main() -> None:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/demo_log.json")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/images/demo_terminal.gif")
    events = json.loads(src.read_text(encoding="utf-8"))
    fonts = load_fonts()
    frames, durations, lines = [], [], []

    cmd = "> python scripts/demo.py"
    for k in range(2, len(cmd) + 1, 2):  # 명령 입력
        frames.append(frame(lines, fonts, cmd[:k] + "▌"))
        durations.append(45)
    frames.append(frame(lines, fonts, cmd))
    durations.append(500)
    lines.append(("dim", cmd))

    prev_t = 0.0
    for t, style, text in events:
        lines.append((style, text))
        gap = t - prev_t
        prev_t = t
        if style == "title":
            hold = 650
        elif gap > 3:  # 실제로 오래 걸린 계산(평가·품질 기준)
            hold = 900
        elif text.strip() == "":
            continue
        else:
            hold = 140
        frames.append(frame(lines, fonts))
        durations.append(hold)
    durations[-1] = 4000
    frames = [f.convert("P", palette=Image.ADAPTIVE, colors=32) for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"저장: {out} ({len(frames)}프레임, {sum(durations) / 1000:.0f}초, {out.stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()

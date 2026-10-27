#!/usr/bin/env python3
"""Record a real terminal run and render it as an animated GIF.

There is no ffmpeg or browser on this host, so the "screencast" is built from
actual captured output and its real timestamps: every line appears when the
process actually printed it. Nothing is typed ahead of time.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 880, 540
MARGIN = 18
LINE_H = 20
FONT_SIZE = 15
BG = (13, 17, 23)
FG = (201, 209, 217)
DIM = (110, 118, 129)
GREEN = (63, 185, 80)
ACCENT = (163, 113, 247)
NL = chr(10)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]


def load_font():
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, FONT_SIZE)
    return ImageFont.load_default()


def wrap(line: str, cols: int) -> list[str]:
    if not line:
        return [""]
    return [line[i:i + cols] for i in range(0, len(line), cols)]


def capture(cmd: list[str], cwd: str):
    t0 = time.monotonic()
    env = dict(os.environ, PYTHONPATH=cwd)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    out = []
    assert p.stdout is not None
    for raw in p.stdout:
        out.append((time.monotonic() - t0, raw.rstrip()))
    p.wait()
    return out, time.monotonic() - t0


def frame(lines, font, cursor: bool) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 30], fill=(22, 27, 34))
    d.text((MARGIN, 7), "ghostscore - terminal", font=font, fill=DIM)
    for i, text in enumerate(lines):
        y = 46 + i * LINE_H
        if y > H - 24:
            break
        color = FG
        if text.startswith("$"):
            color = GREEN
        elif text.startswith("synthetic holdout"):
            color = ACCENT
        d.text((MARGIN, y), text, font=font, fill=color)
    if cursor:
        y = 46 + min(len(lines), (H - 70) // LINE_H) * LINE_H
        d.rectangle([MARGIN, y + 2, MARGIN + 9, y + 15], fill=FG)
    return img


def main() -> None:
    cwd = str(Path(__file__).resolve().parents[1])
    cols = (W - 2 * MARGIN) // 9
    prompt = "$ python3 -m ghostscore.cli real --n 300 --seed 7"
    events, total = capture([sys.executable, "-m", "ghostscore.cli", "real",
                             "--n", "300", "--seed", "7"], cwd)

    transcript = [prompt] + [e[1] for e in events]
    Path(cwd, "results", "demo.txt").write_text(NL.join(transcript) + NL)

    font = load_font()
    frames = []
    durations = []
    visible = (H - 60) // LINE_H

    for _ in range(3):
        frames.append(frame(["$"], font, cursor=True))
        durations.append(120)

    shown = [prompt]
    last = 0.0
    for t, line in events:
        gap = max(0.05, t - last)
        last = t
        shown.extend(wrap(line, cols))
        frames.append(frame(shown[-visible:], font, cursor=True))
        durations.append(int(min(600, max(60, gap * 1000))))

    final = frame(shown[-visible:], font, cursor=False)
    for _ in range(12):
        frames.append(final)
        durations.append(120)

    out = Path(cwd, "results", "demo.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=2)
    print("captured %d lines in %.1fs -> %s (%.1f MB, %d frames)" % (
        len(events), total, out, out.stat().st_size / 1e6, len(frames)))


if __name__ == "__main__":
    main()

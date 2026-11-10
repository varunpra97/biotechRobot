"""Join the centrifuge clips (fable_centrifuge/record.py) into one labelled video: intro card with the scores, a title
card per clip, the episodes unedited (the renderer's black warm-up frame dropped, ~1 s episodes shown at half speed).

  python join_video.py <clips dir> <out.mp4> <ffmpeg> <cards.json>
cards.json: {"intro": [[text, scale, colour], ...], "clips": [[name, title, score line, note], ...], "outro": [...]}
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np

C, OUT, FFMPEG = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
cards = json.loads(Path(sys.argv[4]).read_text())
W, H = 1280, 720
COL = {"white": (255, 255, 255), "green": (90, 230, 120), "amber": (60, 190, 255), "grey": (190, 190, 190)}


def put(img, x, y, text, sc, col, th=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def card(lines, n, y0=90):
    img = np.full((H, W, 3), 24, np.uint8)
    y = y0
    for text, sc, col in lines:
        put(img, 60, y, text, sc, COL[col], 2 if sc >= 0.7 else 1)
        y += int(52 * sc / 0.72) if text else 20
    return [img] * n


def episodes(name):
    rows = [json.loads(x) for x in (C / f"{name}.jsonl").read_text().splitlines()]
    cap = cv2.VideoCapture(str(C / f"{name}.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    out, i = [], 0
    for r in rows:
        n = r["steps"] + 10
        ep = [f.copy() for f in frames[i:i + n]]
        i += n
        ep = [f for f in ep if f[100:-100].mean() > 15] or ep
        if r["steps"] < 40:
            ep = [f for f in ep for _ in (0, 1)]
            for f in ep[:-20]:
                put(f, 1110, 706, "0.5x speed", 0.6, COL["amber"], 1)
        out += ep
    return out, rows


frames = card(cards["intro"], 150)
for name, title, score, note in cards["clips"]:
    eps, rows = episodes(name)
    ok = sum(r["outcome"] == "SUCCESS" for r in rows)
    frames += card([(title, 1.0, "white"), (score, 0.75, "green"), *[(ln, 0.62, "grey") for ln in textwrap.wrap(note, 84)],
                    ("", 0.6, "grey"), (f"this clip: {ok} of {len(rows)} episodes succeed (unedited)", 0.62, "grey")], 75, 280)
    frames += eps
frames += card(cards["outro"], 120, 260)
raw = OUT.with_suffix(".raw.mp4")
w = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
for f in frames:
    w.write(cv2.resize(f, (W, H)))
w.release()
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24",
                "-preset", "medium", "-movflags", "+faststart", str(OUT)], check=True)
raw.unlink()
print(OUT, len(frames), "frames", round(len(frames) / 15, 1), "s", round(OUT.stat().st_size / 1e6, 1), "MB")

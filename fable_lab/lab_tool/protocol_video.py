r"""One labelled video of a protocol run (run_protocol.py): every step's recording in order, real time (15 frames per
second), three cameras side by side, with the high-level command and the primitive step written on each frame.

    .venv-isaac\Scripts\python.exe biotech-lab-robotics\fable_lab\lab_tool\protocol_video.py data\lab_tool\runs\demo
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CAMS = ["platform_cam1", "wrist_top", "wrist_side"]
FFMPEG = Path(r"C:\Users\varun\workspace\BiotechRobot\.venv-isaaclab\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe")
WHITE, GREEN, AMBER, RED, GREY = (255, 255, 255), (90, 230, 120), (60, 190, 255), (90, 90, 255), (190, 190, 190)
W, H, BAR = 1440, 360, 96


def put(img, x, y, text, sc, col, th=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def card(lines, n):
    img = np.full((H + BAR, W, 3), 24, np.uint8)
    y = 70
    for text, sc, col in lines:
        put(img, 50, y, text, sc, col, 2 if sc >= 0.75 else 1)
        y += int(46 * sc / 0.7) if text else 16
    return [img] * n


def step_clip(step_dir, cam="platform_cam1"):
    """The step's exterior-camera frames as one H.264 clip the browser can play (clip.mp4). True if written."""
    step_dir = Path(step_dir)
    if not FFMPEG.exists() or not (step_dir / "images" / f"{cam}_00000.jpg").exists():
        return False
    r = subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-framerate", "15", "-i",
                        str(step_dir / "images" / f"{cam}_%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "24", "-preset", "veryfast", "-movflags", "+faststart", str(step_dir / "clip.mp4")])
    return r.returncode == 0 and (step_dir / "clip.mp4").exists()


def make(out):
    out = Path(out)
    st = json.loads((out / "status.json").read_text())
    steps = [s for s in st["steps"] if s.get("dir")]
    raw, final = out / "protocol.raw.mp4", out / "protocol.mp4"
    w = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H + BAR))
    intro = [("Experiment run in the simulated lab", 0.95, WHITE), ("", 0.5, GREY)]
    words, line = st["text"].split(), ""
    for wd in words:                                   # the description as typed, wrapped
        if len(line) + len(wd) > 95:
            intro.append((line, 0.6, AMBER))
            line = ""
        line += wd + " "
    intro += [(line, 0.6, AMBER), ("", 0.5, GREY)]
    intro += [(f"{i + 1}. {c}", 0.62, WHITE) for i, c in enumerate(st["commands"])]
    intro += [("", 0.5, GREY), (f"{len(steps)} robot steps in one continuous simulation, real time, nothing cut.", 0.55, GREY),
              ("Each step is done by a scripted expert (motion planning + IK). Liquids are tracked volumes;", 0.55, GREY),
              ("the centrifuge does not spin; waits are shortened.", 0.55, GREY)]
    for f in card(intro, 15 * 7):
        w.write(f)
    total = 0
    for k, s in enumerate(steps):
        cmd = st["commands"][s["command"]]
        ep = out / s["dir"]
        t = 0
        while True:
            imgs = [cv2.imread(str(ep / "images" / f"{c}_{t:05d}.jpg")) for c in CAMS]
            if any(i is None for i in imgs):
                break
            frame = np.full((H + BAR, W, 3), 24, np.uint8)
            frame[BAR:] = cv2.resize(np.concatenate(imgs, axis=1), (W, H), interpolation=cv2.INTER_AREA)
            put(frame, 14, 30, f"Command {s['command'] + 1}/{len(st['commands'])}: {cmd}", 0.75, WHITE, 2)
            ok = s["state"] == "ok"
            put(frame, 14, 62, f"step {k + 1}/{len(steps)}: {s['label']}", 0.65, GREEN if ok else RED, 2)
            put(frame, 14, 86, "scripted expert (planning + IK)" + ("" if ok else "    STEP FAILED"), 0.5, GREY)
            w.write(frame)
            t += 1
            total += 1
    liq = st.get("liquids") or {}
    res = st.get("result", st["state"])
    end = [(f"Result: {res}", 0.95, GREEN if res == "SUCCESS" else RED), ("", 0.5, GREY),
           ("Liquid at the end (uL):", 0.65, WHITE),
           ("tubes: " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(liq.get("tubes", {}).items())), 0.6, WHITE),
           ("tray wells: " + (", ".join(f"{k} {v:.0f}" for k, v in sorted(liq.get("wells", {}).items())) or "none"), 0.6, WHITE),
           ("", 0.5, GREY), (f"steps done: {sum(s['state'] == 'ok' for s in steps)} of {len(st['steps'])}", 0.6, GREY)]
    for f in card(end, 15 * 6):
        w.write(f)
    w.release()
    if FFMPEG.exists():
        subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "26", "-preset", "medium", "-movflags", "+faststart", str(final)], check=True)
        raw.unlink()
    else:
        raw.replace(final)
    print("video", final, total, "frames", round(total / 15, 1), "s")
    return str(final)


if __name__ == "__main__":
    make(sys.argv[1])

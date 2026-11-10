"""Video of the version-2 model-in-the-loop episodes (several moves per model call).
  python loop_video_v2.py <loop dir> <out.mp4> <ffmpeg>
Every simulator step is one frame (15 per second, unedited). The picture holds for 0.6 s at each model call.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

L, OUT, FFMPEG = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
W, H = 1280, 720
WHITE, GREEN, AMBER, GREY, RED = (255, 255, 255), (90, 230, 120), (60, 190, 255), (190, 190, 190), (90, 90, 255)
# run, episode, title, what it shows, outcome override (for the episode the env never ruled on)
EPS = [("rem2", 0, "Remove, version 2", "whole removal in 1 model call (was 5)", None),
       ("rem2", 1, "Remove, version 2", "2 model calls: line up and check, then the rest", None),
       ("ins2", 0, "Insert, version 2", "tube already in the bore: 1 model call (was 3)", None),
       ("ins2", 1, "Insert, version 2", "2 model calls: line up and check, then the rest (was 4)", None),
       ("ins2", 2, "Insert, version 2", "all in 1 call, no check: the tube jams and is left cocked",
        "NOT RULED (server stopped early) - tube not seated, not a success"),
       ("ins2b", 2, "Insert, version 2", "same start with checks: jam seen, recovery tried", None)]


def put(img, x, y, text, sc, col, th=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def card(lines, n, y0=110):
    img = np.full((H, W, 3), 24, np.uint8)
    y = y0
    for text, sc, col in lines:
        put(img, 60, y, text, sc, col, 2 if sc >= 0.7 else 1)
        y += int(52 * sc / 0.72) if text else 20
    return [img] * n


frames = card([("Fable 5.2 in the loop, version 2: several moves per model call", 0.95, WHITE),
               ("Centrifuge remove and insert, same success rules, 2026-11-09", 0.62, GREY),
               ("", 0.6, GREY),
               ("Model calls per episode: 1 to 2 (version 1: 3 to 7). Env steps unchanged.", 0.72, GREEN),
               ("Remove: 2 of 2.   Insert: 4 successes, 1 failure, 1 not ruled.", 0.72, GREEN),
               ("", 0.6, GREY),
               ("The failed start: the tube sat 4 degrees off in the fingers and jammed on the bore mouth.", 0.62, AMBER),
               ("In one blind call nothing noticed it. With a picture after the push it was seen,", 0.62, AMBER),
               ("but the recovery lost the tube.", 0.62, AMBER),
               ("", 0.6, GREY),
               ("The picture holds at each model call (simulator paused while the model decides).", 0.62, GREY)], 120)
for run, ep, title, what, override in EPS:
    D = L / run
    log = [json.loads(x) for x in (D / "private.jsonl").read_text().splitlines()]
    fin = {r["finished"]["episode"]: r["finished"] for r in log if "finished" in r}
    files = sorted((D / "frames").glob(f"ep{ep:02d}_*.jpg"))
    last_call, ep_frames = None, []
    for i, p in enumerate(files):
        call = int(p.stem.split("_c")[1])
        img = cv2.resize(cv2.imread(str(p)), (W, H))
        put(img, 16, 40, title, 0.9, WHITE)
        put(img, 16, 78, what, 0.7, WHITE)
        put(img, 16, H - 20, f"step {i + 1}   model call {call}", 0.7, WHITE)
        if call != last_call and call > 0:
            hold = img.copy()
            put(hold, 16, H - 54, f"model call {call}: simulator paused while the model looks and decides", 0.7, AMBER)
            ep_frames += [hold] * 9
            last_call = call
        ep_frames.append(img)
    end = ep_frames[-1].copy()
    if override:
        put(end, 16, H - 54, f"result: {override}", 0.7, AMBER)
    else:
        f = fin[ep]
        put(end, 16, H - 54, f"result: {f['outcome']}  ({f['steps']} of 180 steps, {f['calls']} model calls with motion)",
            0.8, GREEN if f["outcome"] == "SUCCESS" else RED)
    frames += ep_frames + [end] * 30
    print(run, ep, len(files), "frames")
raw = OUT.with_suffix(".raw.mp4")
w = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
for f in frames:
    w.write(f)
w.release()
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "25",
                "-preset", "medium", "-movflags", "+faststart", str(OUT)], check=True)
raw.unlink()
print(OUT, len(frames), "frames", round(len(frames) / 15, 1), "s", round(OUT.stat().st_size / 1e6, 1), "MB")

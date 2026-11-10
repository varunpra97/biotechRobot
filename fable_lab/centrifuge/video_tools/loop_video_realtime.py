"""Model-in-the-loop episodes with NO speed-up: the arm moves at its real speed (15 control steps per second) and at
every model call the picture stands still for as long as the model actually took (time between the reply file of
the previous command and the next command file, look-only calls included).
  python loop_video_realtime.py <loop dir> <out.mp4> <ffmpeg> v1|v2
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

L, OUT, FFMPEG, VER = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
W, H, FPS = 1280, 720, 15
WHITE, GREEN, AMBER, GREY, RED = (255, 255, 255), (90, 230, 120), (60, 190, 255), (190, 190, 190), (90, 90, 255)
if VER == "v1":
    EPS = [("rem", e, "Remove: Fable 5.2 in the loop") for e in range(6)] + \
          [("ins", e, "Insert: Fable 5.2 in the loop") for e in range(6)] + \
          [("insc", e, "Insert (carry starts): Fable 5.2 in the loop") for e in range(3)]
    HEAD = "Fable 5.2 in the control loop, real time (version 1: one move per model call)"
else:
    EPS = [("rem2", 0, "Remove, version 2"), ("rem2", 1, "Remove, version 2"), ("ins2", 0, "Insert, version 2"),
           ("ins2", 1, "Insert, version 2"), ("ins2", 2, "Insert, version 2 (all in one call, no check)"),
           ("ins2b", 2, "Insert, version 2 (same start, with checks)")]
    HEAD = "Fable 5.2 in the control loop, real time (version 2: several moves per model call)"


def put(img, x, y, text, sc, col, th=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def clock(s):
    return f"{int(s) // 60}:{int(s) % 60:02d}"


raw = OUT.with_suffix(".raw.mp4")
w = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
total = {"frames": 0, "wait": 0.0, "move": 0.0}


def write(img):
    w.write(img)
    total["frames"] += 1


card = np.full((H, W, 3), 24, np.uint8)
y = 110
for text, sc, col in [(HEAD, 0.85, WHITE), ("Centrifuge remove and insert, 2026-11-09. Nothing is sped up or cut.", 0.62, GREY),
                      ("", 0.6, GREY),
                      ("The arm moves at its real speed: 15 control steps per second.", 0.68, WHITE),
                      ("At each model call the picture stands still for as long as the model really took", 0.68, WHITE),
                      ("to look at the camera images and send the next command.", 0.68, WHITE),
                      ("", 0.6, GREY),
                      ("The env allows 12 s per episode. In the scored runs the simulator was paused", 0.62, AMBER),
                      ("during these waits, so they did not count against it.", 0.62, AMBER)]:
    put(card, 60, y, text, sc, col, 2 if sc >= 0.7 else 1)
    y += int(52 * sc / 0.72) if text else 20
for _ in range(FPS * 8):
    write(card)
runs = {}
for k, (run, ep, title) in enumerate(EPS):
    D = L / run
    if run not in runs:
        log = [json.loads(x) for x in (D / "private.jsonl").read_text().splitlines()]
        cmds = {r["n"]: r["cmd"] for r in log if "cmd" in r}
        ends = [(r["n"], r["finished"]) for r in log if "finished" in r]
        ranges, first = {}, 0
        for end_n, fin in ends:
            ranges[fin["episode"]] = ([n for n in sorted(cmds) if first <= n <= end_n], fin)
            first = end_n + 1
        rest = [n for n in sorted(cmds) if n >= first and cmds[n].get("op", "move") != "quit"]
        if rest:
            ranges[len(ends)] = (rest, None)
        runs[run] = (cmds, ranges)
    cmds, ranges = runs[run]
    ns, fin = ranges[ep]
    waits, acc = [], 0.0                      # seconds the arm stood still before each call with motion
    for n in ns:
        if n > 0:
            acc += os.path.getmtime(D / f"cmd_{n:04d}.json") - os.path.getmtime(D / f"out_{n - 1:04d}.json")
        if cmds[n].get("op", "move") in ("move", "moves"):
            waits.append(acc)
            acc = 0.0
    files = sorted((D / "frames").glob(f"ep{ep:02d}_*.jpg"))
    t, last_call, img = 0.0, 0, None
    for i, p in enumerate(files):
        call = int(p.stem.split("_c")[1])
        img = cv2.resize(cv2.imread(str(p)), (W, H))
        put(img, 16, 40, title, 0.9, WHITE)
        if call != last_call:
            wait = waits[call - 1] if call - 1 < len(waits) else 0.0
            for j in range(int(round(wait * FPS))):
                hold = img.copy()
                put(hold, 16, 78, f"episode {k + 1} of {len(EPS)}    time in episode {clock(t)}", 0.7, WHITE)
                put(hold, 16, H - 54, f"waiting for the model (call {call}): {j / FPS:4.1f} s of {wait:.1f} s - arm stands still",
                    0.7, AMBER)
                put(hold, 16, H - 20, f"step {i}", 0.7, WHITE)
                write(hold)
                t += 1 / FPS
            total["wait"] += wait
            last_call = call
        put(img, 16, 78, f"episode {k + 1} of {len(EPS)}    time in episode {clock(t)}", 0.7, WHITE)
        put(img, 16, H - 20, f"step {i + 1}   moving (model call {call})", 0.7, WHITE)
        write(img)
        t += 1 / FPS
        total["move"] += 1 / FPS
    end = img.copy()
    if fin is None:
        put(end, 16, H - 54, "result: NOT RULED (server stopped early) - tube not seated, not a success", 0.7, AMBER)
    else:
        put(end, 16, H - 54, f"result: {fin['outcome']}   {fin['steps']} of 180 steps   real time {clock(t)} (env limit 0:12)",
            0.8, GREEN if fin["outcome"] == "SUCCESS" else RED)
    for _ in range(FPS * 3):
        write(end)
    print(run, ep, "moving", round(len(files) / FPS, 1), "s  waiting", [round(x, 1) for x in waits], flush=True)
w.release()
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27",
                "-preset", "medium", "-movflags", "+faststart", str(OUT)], check=True)
raw.unlink()
print(OUT, total["frames"], "frames", clock(total["frames"] / FPS), "min:s", round(OUT.stat().st_size / 1e6, 1), "MB",
      "moving", round(total["move"]), "s, waiting for the model", round(total["wait"]), "s")

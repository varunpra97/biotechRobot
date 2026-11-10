"""Video of a long model-in-the-loop run (every finished episode of one run folder, in order, nothing cut).
  python loop_video_100.py <loop dir> <run> <out.mp4> <ffmpeg> real|short "<title>"
real : the arm moves at its real speed (15 steps per second) and stands still at each model call for as long as the
       model really took (time between the previous reply file and the next command file).
short: the same motion, but each wait is shown as a 0.5 s hold.
A pause of more than 90 s before a look-only command is a hand-over between driving sessions and is not counted.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

L, RUN, OUT, FFMPEG, MODE, TITLE = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
W, H, FPS = 1280, 720, 15
WHITE, GREEN, AMBER, GREY, RED = (255, 255, 255), (90, 230, 120), (60, 190, 255), (190, 190, 190), (90, 90, 255)
D = L / RUN
log = [json.loads(x) for x in (D / "private.jsonl").read_text().splitlines()]
cmds = {r["n"]: r["cmd"] for r in log if "cmd" in r}
ends = [(r["n"], r["finished"]) for r in log if "finished" in r]
eps, first = [], 0
for end_n, fin in ends:
    ns = [n for n in sorted(cmds) if first <= n <= end_n]
    waits, acc, total_wait = [], 0.0, 0.0
    for n in ns:
        if n > 0:
            gap = os.path.getmtime(D / f"cmd_{n:04d}.json") - os.path.getmtime(D / f"out_{n - 1:04d}.json")
            if not (cmds[n].get("op", "move") == "obs" and gap > 90):
                acc += gap
        if cmds[n].get("op", "move") in ("move", "moves"):
            waits.append(acc)
            total_wait += acc
            acc = 0.0
    eps.append({"fin": fin, "waits": waits, "wait": total_wait, "calls": len(waits),
                "looks": sum(cmds[n].get("op", "move") == "obs" for n in ns)})
    first = end_n + 1
N = len(eps)
ok = sum(e["fin"]["outcome"] == "SUCCESS" for e in eps)
fail = sum(e["fin"]["outcome"] == "FAIL" for e in eps)
tout = N - ok - fail
steps = [e["fin"]["steps"] for e in eps]
wait_all = sum(e["wait"] for e in eps)


def put(img, x, y, text, sc, col, th=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def clock(s):
    return f"{int(s) // 60}:{int(s) % 60:02d}"


raw = OUT.with_suffix(".raw.mp4")
w = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
count = 0


def write(img, n=1):
    global count
    for _ in range(n):
        w.write(img)
    count += n


card = np.full((H, W, 3), 24, np.uint8)
y = 110
lines = [(TITLE, 0.9, WHITE),
         (f"{N} episodes in a row, none left out. Real-centrifuge scene, the env's own success test.", 0.62, GREY),
         ("", 0.6, GREY),
         (f"SUCCESS {ok}    FAIL {fail}    TIMEOUT {tout}", 0.9, GREEN),
         (f"Env steps per episode: median {int(np.median(steps))}, longest {max(steps)} (limit 180)", 0.62, WHITE),
         (f"Model calls with motion per episode: {np.mean([e['calls'] for e in eps]):.1f} on average", 0.62, WHITE),
         (f"Arm standing still waiting for the model: {wait_all / N:.0f} s per episode on average", 0.62, WHITE),
         ("", 0.6, GREY),
         ("Each model call: look at camera pictures, send a short batch of hand-pose moves.", 0.62, AMBER),
         ("Under the model: a Jacobian servo turns pose targets into joint commands (IK-like).", 0.62, AMBER),
         ("The simulator is paused while the model thinks; "
          + ("here every wait is shown in full." if MODE == "real" else "here each wait is shown as a 0.5 s hold."), 0.62, AMBER)]
for text, sc, col in lines:
    put(card, 60, y, text, sc, col, 2 if sc >= 0.7 else 1)
    y += int(52 * sc / 0.72) if text else 20
write(card, FPS * 8)
done_ok = 0
for k, e in enumerate(eps):
    fin = e["fin"]
    files = sorted((D / "frames").glob(f"ep{fin['episode']:02d}_*.jpg")) if fin["episode"] < 100 else \
        sorted((D / "frames").glob(f"ep{fin['episode']}_*.jpg"))
    t, last_call, img = 0.0, 0, None
    head = f"episode {k + 1} of {N}    successes so far {done_ok}"
    for i, p in enumerate(files):
        call = int(p.stem.split("_c")[1])
        img = cv2.resize(cv2.imread(str(p)), (W, H))
        put(img, 16, 40, TITLE, 0.8, WHITE)
        if call != last_call:
            wait = e["waits"][call - 1] if call - 1 < len(e["waits"]) else 0.0
            nh = int(round(wait * FPS)) if MODE == "real" else 8
            for j in range(nh):
                hold = img.copy()
                put(hold, 16, 78, head + (f"    time in episode {clock(t)}" if MODE == "real" else ""), 0.7, WHITE)
                txt = (f"waiting for the model (call {call}): {j / FPS:4.1f} s of {wait:.1f} s - arm stands still"
                       if MODE == "real" else f"model call {call}: the arm waited {wait:.1f} s here")
                put(hold, 16, H - 54, txt, 0.7, AMBER)
                put(hold, 16, H - 20, f"step {i}", 0.7, WHITE)
                write(hold)
                t += 1 / FPS
            last_call = call
        put(img, 16, 78, head + (f"    time in episode {clock(t)}" if MODE == "real" else ""), 0.7, WHITE)
        put(img, 16, H - 20, f"step {i + 1}   moving (model call {call})", 0.7, WHITE)
        write(img)
        t += 1 / FPS
    if img is None:
        continue
    done_ok += fin["outcome"] == "SUCCESS"
    end = img.copy()
    put(end, 16, H - 54, f"result: {fin['outcome']}   {fin['steps']} of 180 steps, {e['calls']} model calls, waited {e['wait']:.0f} s",
        0.8, GREEN if fin["outcome"] == "SUCCESS" else RED)
    write(end, FPS * (2 if MODE == "real" else 1) + (FPS if fin["outcome"] != "SUCCESS" else 0))
w.release()
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28",
                "-preset", "medium", "-movflags", "+faststart", str(OUT)], check=True)
raw.unlink()
print(json.dumps({"run": RUN, "episodes": N, "success": ok, "fail": fail, "timeout": tout,
                  "median_steps": int(np.median(steps)), "mean_calls": round(float(np.mean([e["calls"] for e in eps])), 2),
                  "mean_wait_s": round(wait_all / N, 1), "video": str(OUT), "length": clock(count / FPS),
                  "MB": round(OUT.stat().st_size / 1e6, 1)}))

"""Video + summary of the model-in-the-loop episodes (loop_server.py runs).
  python loop_video.py <loop dir> <out.mp4> <ffmpeg> <results.jsonl>
Every simulator step is one frame (15 per second, unedited). Where the model was called the picture holds for 0.6 s
with the call number: the simulator was paused while the model thought.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

L, OUT, FFMPEG, RES = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
W, H = 1280, 720
RUNS = [("rem", "remove_joint", "Remove: Fable 5.2 in the loop", "default start mix"),
        ("ins", "insert_joint", "Insert: Fable 5.2 in the loop", "default start mix"),
        ("insc", "insert_joint", "Insert: Fable 5.2 in the loop", "carry starts only")]


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


WHITE, GREEN, AMBER, GREY, RED = (255, 255, 255), (90, 230, 120), (60, 190, 255), (190, 190, 190), (90, 90, 255)
rows, frames = [], []
for name, task, title, mix in RUNS:
    D = L / name
    log = [json.loads(x) for x in (D / "private.jsonl").read_text().splitlines()]
    fin = {r["finished"]["episode"]: r["finished"] for r in log if "finished" in r}
    cmds = [r for r in log if "cmd" in r]
    for ep, f in sorted(fin.items()):
        # commands of this episode: the command that ended it is logged under the next episode number
        looks = sum(1 for r in cmds if r["episode"] == ep and r["cmd"].get("op", "move") != "move")
        first = next((r for r in cmds if r["episode"] == ep), None)
        rows.append({"task": task, "solver": "Fable 5.2 in the loop (pose targets from camera images)", "run": name,
                     "start_mix": mix, "episode": ep, "outcome": f["outcome"], "steps": f["steps"],
                     "move_calls": f["calls"], "look_only_calls": looks,
                     "start_truth": first["truth"] if first else None})
        files = sorted((D / "frames").glob(f"ep{ep:02d}_*.jpg"))
        last_call, ep_frames = None, []
        for i, p in enumerate(files):
            call = int(p.stem.split("_c")[1])
            img = cv2.resize(cv2.imread(str(p)), (W, H))
            put(img, 16, 40, title, 0.9, WHITE)
            put(img, 16, 78, f"episode {len([r for r in rows if r['run'] == name])}   ({mix})", 0.7, WHITE)
            put(img, 16, H - 20, f"step {i + 1}   model call {call}", 0.7, WHITE)
            if call != last_call and call > 0:
                hold = img.copy()
                put(hold, 16, H - 54, f"model call {call}: simulator paused while the model looks and decides", 0.7, AMBER)
                ep_frames += [hold] * 9
                last_call = call
            ep_frames.append(img)
        end = ep_frames[-1].copy()
        put(end, 16, H - 54, f"result: {f['outcome']}  ({f['steps']} of 180 steps, {f['calls']} move commands)", 0.8,
            GREEN if f["outcome"] == "SUCCESS" else RED)
        frames += ep_frames + [end] * 18
RES.parent.mkdir(parents=True, exist_ok=True)
with open(RES, "a") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
for r in rows:
    print(r["run"], r["episode"], r["outcome"], r["steps"], "moves", r["move_calls"], "looks", r["look_only_calls"],
          json.dumps(r["start_truth"]))
ok = {k: sum(r["outcome"] == "SUCCESS" for r in rows if r["task"] == k) for k in ("remove_joint", "insert_joint")}
n = {k: sum(1 for r in rows if r["task"] == k) for k in ok}
obs = cv2.imread(str(L / "rem" / "obs_0008.png"))
see = np.full((H, W, 3), 24, np.uint8)
o = cv2.resize(obs, (int(obs.shape[1] * 640 / obs.shape[0]), 640))
see[70:710, (W - o.shape[1]) // 2:(W - o.shape[1]) // 2 + o.shape[1]] = o
put(see, 60, 45, "What the model gets on each call: two fixed cameras (wide + crop), the hand's own pose, no object poses",
    0.62, WHITE, 1)
intro = card([("Claude Fable 5.2 in the control loop", 1.0, WHITE),
              ("Centrifuge remove and insert, real-centrifuge scene, unchanged success rules (2026-11-09)", 0.62, GREY),
              ("", 0.6, GREY),
              (f"Remove: {ok['remove_joint']} of {n['remove_joint']} episodes     Insert: {ok['insert_joint']} of {n['insert_joint']} episodes",
               0.8, GREEN),
              ("", 0.6, GREY),
              ("Each model call: look at the camera images, send one hand-pose target and gripper command.", 0.62, WHITE),
              ("2 to 5 move commands per episode. The simulator is paused while the model thinks.", 0.62, WHITE),
              ("", 0.6, GREY),
              ("Under the model: a Jacobian servo turns each pose target into joint commands (IK-like).", 0.62, AMBER),
              ("Small sample (15 episodes), not the 100-episode evaluation.", 0.62, AMBER)], 105)
video = intro + [see] * 75 + frames
raw = OUT.with_suffix(".raw.mp4")
w = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
for f in video:
    w.write(f)
w.release()
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "25",
                "-preset", "medium", "-movflags", "+faststart", str(OUT)], check=True)
raw.unlink()
print(OUT, len(video), "frames", round(len(video) / 15, 1), "s", round(OUT.stat().st_size / 1e6, 1), "MB")

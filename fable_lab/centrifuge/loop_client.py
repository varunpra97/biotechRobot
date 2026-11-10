"""One model call to loop_server.py:  python loop_client.py <dir> '<json command>'
Prints the reply (the robot's own state) and the path of the observation image.

  {"op":"obs"}                               look only (no steps)
  {"op":"move", ...}                         target for the TCP (or "point_tool": [x,y,z] m, a point fixed to the hand):
      "pos": [x,y,z]       absolute, robot base frame, metres      "dpos": base-frame offset     "dtool": tool-frame offset
      "z_axis": [..], "y_axis": [..]   hand approach / closing axis (base frame)   "rot_tool" / "rot_base": [deg x3]
      "grip": "open" | "close"     "max_steps": 40     "speed": 0.006 m/step     "wmax": 0.05 rad/step
  any command: "marks": [[x,y,z],..] points to draw, "focus": [x,y,z] crop centre (default the TCP), "crop_w": 400
  {"op":"moves", "moves": [ {move}, {move}, ... ]}   several moves in ONE model call, run in order; stops early when a
      move does not arrive, a "require_gap_mm": [lo, hi] on a move is not met, or the episode ends
  "obs": "full" (default: both views, wide + crop) | "crop" (both crops) | "one" + "view": "A"|"B" | "none" (no picture)
"""
import json
import sys
import time
from pathlib import Path

D = Path(sys.argv[1])
cmd = json.loads(sys.argv[2])
n = len(list(D.glob("cmd_*.json")))
tmp = D / f"cmd_{n:04d}.tmp"
tmp.write_text(json.dumps(cmd))
tmp.replace(D / f"cmd_{n:04d}.json")
out = D / f"out_{n:04d}.json"
t0 = time.time()
while not out.exists():
    if time.time() - t0 > 240:
        sys.exit(f"no reply to command {n} after 240 s")
    time.sleep(0.05)
time.sleep(0.05)
r = json.loads(out.read_text())
r.pop("trace", None) if "error" not in r else None
print(json.dumps(r))
img = D / f"obs_{n:04d}.png"
print("image:", img if img.exists() else "none")

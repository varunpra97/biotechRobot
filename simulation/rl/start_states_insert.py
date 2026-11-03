r"""Start-state library for the centrifuge-insert RL task (mini milestone m5):
the arm holding a tube near a rotor bore's entry, from recorded expert
inserts (insert_tube_into_bore, load_tube_into_centrifuge) with the
centrifuge at its default spot (rl/assets/centrifuge_bores.npz).

An episode counts if its tube ends seated in the named bore there (tip on the
axis within 1.5 mm, near the floor). Frames: gripper closed, the tube's tip
within NEAR_M of the bore's opening and not yet deeper than ENTERED_S along
the axis. Each entry: joint positions (7 + 2 fingers), the held tube's pose
([x, y, z, qw, qx, qy, qz]), the bore index (0..5 = label "1".."6").

Usage (any venv with numpy, from the BiotechRobot folder):
    python biotech-lab-robotics\simulation\rl\start_states_insert.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(r"C:\Users\varun\workspace\BiotechRobot\data")
OUT = HERE / "start_states_insert.npz"
BORES = np.load(HERE / "assets" / "centrifuge_bores.npz")
NEAR_M = 0.08
if "--near" in sys.argv:          # M5: 0.30 (the carry from the rack, so pi0.5 can hand over further out)
    NEAR_M = float(sys.argv[sys.argv.index("--near") + 1])
ENTERED_S = 0.0                   # tip deeper than the bore's axis point (s = 0; opening at s ~ +16 mm): already inside
SEATED_S = -0.010                 # an inserted tube's tip ends below this


def main():
    q, tube, bore, src = [], [], [], []
    used, skipped = 0, 0
    labels = list(BORES["label"])
    eps = []
    for run in ("primitives_insert_M5", "primitives_v2", "primitives_v3", "primitives_v4", "test_layouts", "test_free_tubes",
                "primitives"):
        for prim in ("insert_tube_into_bore", "load_tube_into_centrifuge"):
            eps += sorted(glob.glob(str(DATA / run / prim / "ep_*")))
    for ep in eps:
        ep = Path(ep)
        meta = json.loads((ep / "meta.json").read_text())
        if not meta.get("success"):
            continue
        b = labels.index(str(meta["args"]["bore"]))
        point, axis = BORES["point"][b], BORES["axis"][b]
        names = meta.get("object_names") or []
        d = np.load(ep / "steps.npz")
        P = d["object_poses"]
        tubes = [i for i, n in enumerate(names) if n.startswith("tube")]
        k = max(tubes, key=lambda i: np.linalg.norm(P[:, i, :3] - P[0, i, :3], axis=1).max())
        tp = P[:, k]
        rel = tp[:, :3] - point
        s = rel @ axis
        radial = np.linalg.norm(rel - s[:, None] * axis, axis=1)
        if not (radial[-1] < 0.0015 and s[-1] < SEATED_S):     # not seated in this bore here: centrifuge elsewhere
            skipped += 1
            continue
        used += 1
        qq = np.concatenate([d["joint_position"][:, :7], d["finger_position_m"].reshape(len(tp), -1)[:, :2]], axis=1)
        closed = d["action"][:, 7] > 0.9
        opening = point + BORES["opening_s"][b] * axis
        for t in range(len(tp)):
            if closed[t] and s[t] > ENTERED_S and np.linalg.norm(tp[t, :3] - opening) < NEAR_M:
                q.append(qq[t])
                tube.append(tp[t])
                bore.append(b)
                src.append(str(ep))
    np.savez(OUT, q=np.array(q, np.float32), tube=np.array(tube, np.float32), bore=np.array(bore, np.int64),
             source=np.array(src))
    print(f"{len(q)} start states from {used} episodes ({skipped} skipped: other centrifuge spot) -> {OUT}")
    if bore:
        print("bores:", {labels[i]: int((np.array(bore) == i).sum()) for i in sorted(set(bore))})


if __name__ == "__main__":
    main()

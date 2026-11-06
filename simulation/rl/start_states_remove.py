r"""Start-state library for the joint-space centrifuge-remove RL task (rl/tube_remove_joint): recorded expert
removals (remove_tube_from_bore, unload_tube_to_holder) with the centrifuge at its default spot
(rl/assets/centrifuge_bores.npz).

An episode counts if its tube starts seated in the named bore there (tip on the axis within 1.5 mm, below
SEATED_S) and the removal succeeded. Two kinds of frame:
  phase 0 (approach): gripper open, the tube still seated, the TCP within NEAR_M of the tube's grasp point;
  phase 1 (held):     gripper closed on the tube (3+ steps after the close), the tube's tip still below the clear
                      point (opening + 15 mm) -- the reverse curriculum, where the extraction is met early.
Each entry: joint positions (7 + 2 fingers), the tube's pose ([x, y, z, qw, qx, qy, qz]), the bore index (0..5 =
label "1".."6"), the phase, and the seated tube pose of that episode (for resets that start from an approach).

Usage (any venv with numpy, from the BiotechRobot folder):
    python biotech-lab-robotics\simulation\rl\start_states_remove.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(r"C:\Users\varun\workspace\BiotechRobot\data")
OUT = HERE / "start_states_remove.npz"
BORES = np.load(HERE / "assets" / "centrifuge_bores.npz")
NEAR_M = 0.35
SEATED_S = -0.010                 # a seated tube's tip is below this along the bore axis
TUBE_LEN, GRASP_DEPTH = 0.058, 0.010   # expert.REMOVE_GRASP_DEPTH: TCP 10 mm below the rim
CLEAR_S = 0.015                   # tip this far beyond the opening: out of the bore (expert's tip_clear_s)


def main():
    q, tube, bore, phase, seat, src = [], [], [], [], [], []
    used, skipped = 0, 0
    labels = list(BORES["label"])
    eps = []
    for run in ("primitives_insert_M5", "primitives_v2", "primitives_v3", "primitives_v4", "test_layouts", "test_free_tubes",
                "primitives"):
        for prim in ("remove_tube_from_bore", "unload_tube_to_holder"):
            eps += sorted(glob.glob(str(DATA / run / prim / "ep_*")))
    for ep in eps:
        ep = Path(ep)
        meta = json.loads((ep / "meta.json").read_text())
        if not meta.get("success") or "bore" not in (meta.get("args") or {}):
            continue
        b = labels.index(str(meta["args"]["bore"]))
        point, axis, opening = BORES["point"][b], BORES["axis"][b], BORES["opening_s"][b]
        names = meta.get("object_names") or []
        d = np.load(ep / "steps.npz")
        P = d["object_poses"]
        tubes = [i for i, n in enumerate(names) if n.startswith("tube")]
        k = max(tubes, key=lambda i: np.linalg.norm(P[:, i, :3] - P[0, i, :3], axis=1).max())
        tp = P[:, k]
        rel = tp[:, :3] - point
        s = rel @ axis
        radial = np.linalg.norm(rel - s[:, None] * axis, axis=1)
        if not (radial[0] < 0.0015 and s[0] < SEATED_S):          # not seated in this bore here: centrifuge elsewhere
            skipped += 1
            continue
        used += 1
        qq = np.concatenate([d["joint_position"][:, :7], d["finger_position_m"].reshape(len(tp), -1)[:, :2]], axis=1)
        closed = d["action"][:, 7] > 0.9
        t_close = int(np.argmax(closed)) if closed.any() else len(tp)
        w, x, y, z = tp[:, 3], tp[:, 4], tp[:, 5], tp[:, 6]
        tax = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)
        grasp = tp[:, :3] + tax * (TUBE_LEN - GRASP_DEPTH)
        tcp = d["tcp_pose"][:, :3]
        for t in range(len(tp)):
            if t < t_close and not closed[t] and s[t] < SEATED_S and np.linalg.norm(tcp[t] - grasp[t]) < NEAR_M:
                ph = 0
            elif t >= t_close + 3 and closed[t] and s[t] < opening + CLEAR_S:
                ph = 1
            else:
                continue
            q.append(qq[t])
            tube.append(tp[t])
            bore.append(b)
            phase.append(ph)
            seat.append(tp[0])
            src.append(str(ep))
    np.savez(OUT, q=np.array(q, np.float32), tube=np.array(tube, np.float32), bore=np.array(bore, np.int64),
             phase=np.array(phase, np.int64), seat=np.array(seat, np.float32), source=np.array(src))
    ph = np.array(phase)
    print(f"{len(q)} start states ({int((ph == 0).sum())} approach, {int((ph == 1).sum())} held) from {used} episodes "
          f"({skipped} skipped: other centrifuge spot) -> {OUT}")
    if bore:
        print("bores:", {labels[i]: int((np.array(bore) == i).sum()) for i in sorted(set(bore))})


if __name__ == "__main__":
    main()

r"""Held-tube start states for the joint-space tube-pick task (reverse
curriculum): the arm right after the expert's fingers closed on the tube,
from still in its bore up to part-way through the lift, from recorded expert
pick_tube_from_holder episodes with the rack at its default spot.

The joint-space teacher reached the grasp pose but never learned to lift: its
first closes knocked tubes over (the arm's moves after closing twisted the
tube out of the bore), so closing looked like a failure. Starting part of
the episodes from a hold lets it learn the lift on its own.

Frames: gripper closed and the pads on the tube (finger gap 4.5-7.5 mm), the
tube upright (< 5 deg) and its tip 0-5 cm above its rest height. Each entry:
joint positions (7 + 2 fingers), the tube's pose ([x, y, z, qw, qx, qy, qz]),
its slot (row, col).

Usage (any venv with numpy, from the BiotechRobot folder):
    python biotech-lab-robotics\simulation\rl\start_states_held.py
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(r"C:\Users\varun\workspace\BiotechRobot\data")
OUT = HERE / "start_states_held.npz"
RUNS = ("primitives_tube_picks", "primitives_v2", "primitives_place_M4", "primitives_insert_M5")
HOLDER_XY = (-0.05, 1.65)
TUBE_REST_Z = 0.952
GAP_M = (0.0045, 0.0075)
MAX_LIFT_M = 0.05


def tilt_deg(q):
    w, x, y, z = q
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


def main():
    q, tube, slot, src = [], [], [], []
    for run in RUNS:
        for ep in sorted(glob.glob(str(DATA / run / "pick_tube_from_holder" / "ep_*"))):
            ep = Path(ep)
            meta = json.loads((ep / "meta.json").read_text())
            lay = meta.get("layout") if isinstance(meta.get("layout"), dict) else {}
            holder = (lay.get("platform") or {}).get("holder")
            if not meta.get("success") or (holder is not None and np.hypot(holder[0] - HOLDER_XY[0],
                                                                            holder[1] - HOLDER_XY[1]) > 1e-3):
                continue
            label = meta["args"]["slot"]
            rc = ("ABCD".index(label[0]), int(label[1:]) - 1)
            names = meta.get("object_names") or []
            d = np.load(ep / "steps.npz")
            P = d["object_poses"]
            tubes = [i for i, n in enumerate(names) if n.startswith("tube")]
            k = max(tubes, key=lambda i: np.linalg.norm(P[:, i, :3] - P[0, i, :3], axis=1).max())
            fingers = d["finger_position_m"].reshape(len(P), -1)[:, :2]
            qq = np.concatenate([d["joint_position"][:, :7], fingers], axis=1)
            closed = d["action"][:, 7] > 0.9
            for t in range(len(P)):
                gap = fingers[t].sum()
                lift = P[t, k, 2] - TUBE_REST_Z
                if (closed[t] and GAP_M[0] < gap < GAP_M[1] and tilt_deg(P[t, k, 3:7]) < 5.0
                        and -0.002 < lift < MAX_LIFT_M):
                    q.append(qq[t])
                    tube.append(P[t, k])
                    slot.append(rc)
                    src.append(str(ep))
    np.savez(OUT, q=np.array(q, np.float32), tube=np.array(tube, np.float32), slot=np.array(slot, np.int64),
             source=np.array(src))
    t = np.array(tube)
    print(f"{len(q)} held start states from {len(set(src))} episodes -> {OUT}")
    if len(t):
        print(f"lift {np.percentile(t[:, 2] - TUBE_REST_Z, [0, 50, 100]).round(4).tolist()} m, "
              f"slots {len(set(map(tuple, np.array(slot).tolist())))}")


if __name__ == "__main__":
    main()

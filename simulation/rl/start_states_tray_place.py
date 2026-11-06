r"""Start-state library for the joint-space tray-place RL task (rl/tray_place_joint): the arm holding the sample
tray, from recorded expert place_tray episodes and the place part of move_tray chains (data/primitives*).

Frames: gripper closed, the tray lifted more than LIFTED_M off the bench (carried, not yet set down). Each entry:
joint positions (7 + 2 fingers), the tray's pose ([x, y, z, qw, qx, qy, qz]), the episode's target (x, y, yaw in
rad) -- both shifted to the default arm spot, as randomized layouts move the arm -- and the neighbours' poses
(neighbours.NAMES).

Usage (any venv with numpy, from the BiotechRobot folder):
    python biotech-lab-robotics\simulation\rl\start_states_tray_place.py
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np

import neighbours

HERE = Path(__file__).resolve().parent
DATA = Path(r"C:\Users\varun\workspace\BiotechRobot\data")
OUT = HERE / "start_states_tray_place.npz"
BENCH_Z = 0.94                    # lab_scene.BENCH_Z + 0.04: the tray's origin resting on the bench
LIFTED_M = 0.003


def main():
    q, tray, target, src, neigh = [], [], [], [], []
    eps = []
    for run in ("primitives", "primitives_v2", "primitives_v2b", "primitives_v3", "primitives_v4", "test_free_tray"):
        for prim in ("place_tray", "move_tray"):
            eps += sorted(glob.glob(str(DATA / run / prim / "ep_*")))
    used = 0
    for ep in eps:
        ep = Path(ep)
        meta = json.loads((ep / "meta.json").read_text())
        args = meta.get("args") or {}
        if not meta.get("success") or "xy" not in args:
            continue
        names = meta.get("object_names") or []
        if "tray" not in names:
            continue
        d = np.load(ep / "steps.npz")
        k = names.index("tray")
        tp = d["object_poses"][:, k]
        qq = np.concatenate([d["joint_position"][:, :7], d["finger_position_m"].reshape(len(tp), -1)[:, :2]], axis=1)
        closed = d["action"][:, 7] > 0.9
        shift = neighbours.arm_shift(meta)
        tgt = np.array([args["xy"][0] + shift[0], args["xy"][1] + shift[1], math.radians(args.get("yaw_deg", 0.0))])
        n0 = len(q)
        for t in range(len(tp)):
            if closed[t] and tp[t][2] > BENCH_Z + LIFTED_M:
                q.append(qq[t])
                tray.append(np.concatenate([tp[t][:3] + shift, tp[t][3:]]))
                target.append(tgt)
                src.append(str(ep))
                neigh.append(neighbours.frame_poses(meta, d["object_poses"][t]))
        used += len(q) > n0
    np.savez(OUT, q=np.array(q, np.float32), tray=np.array(tray, np.float32), target=np.array(target, np.float32),
             source=np.array(src), neigh=np.array(neigh, np.float32), neigh_names=np.array(neighbours.NAMES))
    t = np.array(target)
    print(f"{len(q)} held start states from {used} episodes -> {OUT}")
    if len(t):
        print(f"targets x {t[:, 0].min():.3f}..{t[:, 0].max():.3f}, y {t[:, 1].min():.3f}..{t[:, 1].max():.3f}, "
              f"yaw {np.degrees(t[:, 2]).min():.1f}..{np.degrees(t[:, 2]).max():.1f} deg; "
              f"{len(np.unique(t.round(3), axis=0))} distinct")


if __name__ == "__main__":
    main()

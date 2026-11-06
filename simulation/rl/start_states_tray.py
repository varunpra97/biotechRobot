"""Start-state library for the tray-grasp RL task (mini milestone m2): arm
states near the tray's grasp point from recorded default-layout episodes.

  - expert tray picks (data/primitives_v2, the pilot: default layout with
    small tray jitter): frames from where the TCP comes within NEAR_M of the
    grasp point until the gripper closes;
  - pi0.5's failed tray-pick trials on the fixed layout (data/eval/
    pilot100_2999, v2v4_3999_fixed): frames within NEAR_M.

Each entry: joint positions (7 + 2 fingers), the tray's pose at that frame
([x, y, z, qw, qx, qy, qz], env-local: shifted to the default arm spot), the TCP,
the neighbours' poses (neighbours.NAMES). Grasp point = tray
centre at bench top + TRAY_GRASP_Z (expert.Expert.TRAY_GRASP_Z, 5 mm).

Usage (Isaac venv, from the BiotechRobot folder):
    .venv-isaac\\Scripts\\python.exe biotech-lab-robotics\\simulation\\rl\\start_states_tray.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

import neighbours

HERE = Path(__file__).resolve().parent
DATA = Path(r"C:\Users\varun\workspace\BiotechRobot\data")
OUT = HERE / "start_states_tray.npz"
NEAR_M = 0.60                     # the finisher range (user: widen to ~60 cm, since pi0.5 ends 17-67 cm away)
TRAY_GRASP_Z = 0.005
TRAY_HALF = (0.026, 0.012)
# --held: the expert pick-ups' frames AFTER the close (3+ steps closed, the tray lifted < 4 cm) instead ->
# start_states_tray_held.npz, the reverse curriculum of the joint-space tray pick-up (tray_grasp_joint)
HELD = "--held" in sys.argv
if HELD:
    OUT = HERE / "start_states_tray_held.npz"


def grasp_point(pose):
    p = np.asarray(pose, float)
    return p[:3] + np.array([0.0, 0.0, TRAY_GRASP_Z])


def main():
    q, tray, tcps, src, neigh = [], [], [], [], []

    def add(ep, frames):
        ep = Path(ep)
        meta = json.loads((ep / "meta.json").read_text())
        names = meta.get("object_names") or []
        if "tray" not in names:
            return
        d = np.load(ep / "steps.npz")
        k = names.index("tray")
        tp = d["object_poses"][:, k]
        tcp = d["tcp_pose"][:, :3]
        qq = np.concatenate([d["joint_position"][:, :7], d["finger_position_m"].reshape(len(tcp), -1)[:, :2]], axis=1)
        shift = neighbours.arm_shift(meta)
        for t in frames(d, tcp, tp):
            q.append(qq[t])
            tray.append(np.concatenate([tp[t][:3] + shift, tp[t][3:]]))
            tcps.append(tcp[t] + shift)
            src.append(str(ep))
            neigh.append(neighbours.frame_poses(meta, d["object_poses"][t]))

    def near(d, tcp, tp, before_close=False):
        end = int(np.argmax(d["action"][:, 7] > 0.9)) if before_close and (d["action"][:, 7] > 0.9).any() else len(tcp)
        return [t for t in range(end) if np.linalg.norm(tcp[t] - grasp_point(tp[t])) < NEAR_M]

    def after_close(d, tcp, tp):
        closed = d["action"][:, 7] > 0.9
        closes = np.nonzero(closed)[0]
        if not len(closes):
            return []
        return [t for t in range(int(closes[0]) + 3, len(tcp)) if closed[t] and tp[t][2] < tp[0][2] + 0.04]

    if HELD:
        for run in ("primitives_v2", "primitives_v3", "primitives_v4"):
            for ep in sorted(glob.glob(str(DATA / run / "pick_up_tray" / "ep_*"))):
                if json.loads((Path(ep) / "meta.json").read_text()).get("success"):
                    add(ep, after_close)
        np.savez(OUT, q=np.array(q, np.float32), tray=np.array(tray, np.float32), tcp=np.array(tcps, np.float32),
                 source=np.array(src), n_expert=len(q), neigh=np.array(neigh, np.float32),
                 neigh_names=np.array(neighbours.NAMES))
        print(f"{len(q)} held start states -> {OUT}")
        return
    for ep in sorted(glob.glob(str(DATA / "primitives_v2" / "pick_up_tray" / "ep_*"))):
        add(ep, lambda d, tcp, tp: near(d, tcp, tp, before_close=True))
    n_expert = len(q)
    for run in ("pilot100_2999", "v2v4_3999_fixed"):
        for ep in sorted(glob.glob(str(DATA / "eval" / run / "pick_up_tray" / "ep_*"))):
            add(ep, near)
    np.savez(OUT, q=np.array(q, np.float32), tray=np.array(tray, np.float32), tcp=np.array(tcps, np.float32),
             source=np.array(src), n_expert=n_expert, neigh=np.array(neigh, np.float32),
             neigh_names=np.array(neighbours.NAMES))
    t = np.array(tray)
    print(f"{len(q)} start states ({n_expert} expert, {len(q) - n_expert} from failed pi0.5 trials) -> {OUT}")
    if len(t):
        print(f"tray x {t[:, 0].min():.3f}..{t[:, 0].max():.3f}, y {t[:, 1].min():.3f}..{t[:, 1].max():.3f}, "
              f"z {t[:, 2].min():.4f}..{t[:, 2].max():.4f}")


if __name__ == "__main__":
    main()

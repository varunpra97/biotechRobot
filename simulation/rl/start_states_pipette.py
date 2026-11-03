r"""Start-state library for the pipette-grasp RL task (mini milestone m3): arm
states within NEAR_M of the pipette's grasp point, from recorded episodes.

  - expert pick-ups (data/primitives_v2, v3, v4): frames from where the TCP
    comes within NEAR_M of the grasp point until the gripper closes;
  - recovery episodes (data/recovery_v1): pi0.5's part and the expert's
    finish up to its grasp (they start from pi0.5's failed end states);
  - pi0.5's failed pipette trials (data/eval/pilot100_2999, v2v4_3999,
    v2v4_3999_fixed): frames within NEAR_M.

Each entry: joint positions (7 + 2 fingers), the pipette's pose at that frame
([x, y, z, qw, qx, qy, qz], env-local: shifted to the default arm spot, as
randomized layouts move the arm), the TCP, the neighbours' poses
(neighbours.NAMES). Grasp point (as
pipette_expert.pick_up_pipette): PIPETTE_GRASP_Y along the pipette from its
nozzle end, the TCP PIPETTE_GRASP_BELOW_AXIS below the body axis.

Usage (any venv with numpy, from the BiotechRobot folder):
    python biotech-lab-robotics\simulation\rl\start_states_pipette.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

import neighbours

HERE = Path(__file__).resolve().parent
DATA = Path(r"C:\Users\varun\workspace\BiotechRobot\data")
OUT = HERE / "start_states_pipette.npz"
NEAR_M = 0.60                     # the finisher range (as the tray: pi0.5 ends far from the target)
GRASP_Y = 0.11                    # pipette_expert.PIPETTE_GRASP_Y
GRASP_BELOW_AXIS = 0.006          # pipette_expert.PIPETTE_GRASP_BELOW_AXIS


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def grasp_point(pose):
    p = np.asarray(pose, float)
    R = quat_to_matrix(p[3:7])
    return p[:3] + R[:, 1] * GRASP_Y - R[:, 2] * GRASP_BELOW_AXIS


def main():
    q, pip, tcps, src, kinds, neigh = [], [], [], [], [], []

    def add(ep, kind, frames):
        ep = Path(ep)
        meta = json.loads((ep / "meta.json").read_text())
        names = meta.get("object_names") or []
        if "pipette_0" not in names:
            return
        d = np.load(ep / "steps.npz")
        k = names.index("pipette_0")
        pp = d["object_poses"][:, k]
        tcp = d["tcp_pose"][:, :3]
        qq = np.concatenate([d["joint_position"][:, :7], d["finger_position_m"].reshape(len(tcp), -1)[:, :2]], axis=1)
        shift = neighbours.arm_shift(meta)                  # randomized layouts move the arm too
        for t in frames(meta, d, tcp, pp):
            if np.linalg.norm(tcp[t] - grasp_point(pp[t])) < NEAR_M and pp[t][2] < pp[0][2] + 0.005:   # still lying there
                q.append(qq[t])
                pip.append(np.concatenate([pp[t][:3] + shift, pp[t][3:]]))
                tcps.append(tcp[t] + shift)
                src.append(str(ep))
                kinds.append(kind)
                neigh.append(neighbours.frame_poses(meta, d["object_poses"][t]))

    def until_close(start=0):
        def f(meta, d, tcp, pp):
            s = int(meta.get("prefix_steps", 0)) if start == "prefix" else 0
            closes = np.nonzero(d["action"][s:, 7] > 0.9)[0]
            end = s + int(closes[0]) if len(closes) else len(tcp)
            return range(end)
        return f

    for run in ("primitives_v2", "primitives_v3", "primitives_v4"):
        for ep in sorted(glob.glob(str(DATA / run / "pick_up_pipette" / "ep_*"))):
            add(ep, "expert", until_close())
    for ep in sorted(glob.glob(str(DATA / "recovery_v1" / "pick_up_pipette" / "ep_*"))):
        add(ep, "recovery", until_close("prefix"))
    for run in ("pilot100_2999", "v2v4_3999", "v2v4_3999_fixed"):
        for ep in sorted(glob.glob(str(DATA / "eval" / run / "pick_up_pipette" / "ep_*"))):
            add(ep, "eval", lambda meta, d, tcp, pp: range(len(tcp)))
    kinds = np.array(kinds)
    np.savez(OUT, q=np.array(q, np.float32), pipette=np.array(pip, np.float32), tcp=np.array(tcps, np.float32),
             source=np.array(src), kind=kinds, n_expert=int((kinds == "expert").sum()),
             neigh=np.array(neigh, np.float32), neigh_names=np.array(neighbours.NAMES))
    print(f"{len(q)} start states ({', '.join(f'{k} {int((kinds == k).sum())}' for k in ('expert', 'recovery', 'eval'))})"
          f" -> {OUT}")
    p = np.array(pip)
    d = np.linalg.norm(np.array(tcps) - np.array([grasp_point(x) for x in p]), axis=1)
    print(f"pipette x {p[:, 0].min():.3f}..{p[:, 0].max():.3f}, y {p[:, 1].min():.3f}..{p[:, 1].max():.3f}, "
          f"z {p[:, 2].min():.4f}..{p[:, 2].max():.4f}; TCP-grasp distance median {np.median(d):.3f}, max {d.max():.3f}")
    up = np.array([quat_to_matrix(x[3:7])[2, 2] for x in p])
    print(f"pipette local z vs world up: min cos {up.min():.3f}")


if __name__ == "__main__":
    main()

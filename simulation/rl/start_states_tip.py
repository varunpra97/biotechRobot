r"""Start-state library for the attach-tip RL task (mini milestone m6): the arm
holding the pipette nozzle-down near the tip it is about to load, from
recorded expert attach_tip episodes (any layout: the task is kinematic, see
tip_attach/env.py).

Frames: before the tip attached, the nozzle end within NEAR_M of the target
tip's collar top. Each entry: joint positions (7 + 2 fingers), the pipette's
pose in the hand frame (in-hand transform: [x, y, z, qw, qx, qy, qz]) at that
frame, the target tip's collar top and axis (shifted to the default arm spot), the rack's pose
(x, y, yaw deg) and the tip's (row, col).

Usage (any venv with numpy, from the BiotechRobot folder):
    python biotech-lab-robotics\simulation\rl\start_states_tip.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

import neighbours

HERE = Path(__file__).resolve().parent
DATA = Path(r"C:\Users\varun\workspace\BiotechRobot\data")
OUT = HERE / "start_states_tip.npz"
RUNS = ("primitives_pipette_M6", "primitives_v2", "primitives_v3", "primitives_v4", "primitives", "primitives_preview3", "primitives_preview_pip",
        "test_free_pipette")
NEAR_M = 0.08
TIP_LENGTH = 0.060                # lab_scene.TIP_LENGTH


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def T(p):
    M = np.eye(4)
    M[:3, :3] = quat_to_matrix(p[3:7])
    M[:3, 3] = p[:3]
    return M


def main():
    q, inhand, top, up, src, rack, tip_rc = [], [], [], [], [], [], []
    for run in RUNS:
        for ep in sorted(glob.glob(str(DATA / run / "attach_tip" / "ep_*"))):
            ep = Path(ep)
            meta = json.loads((ep / "meta.json").read_text())
            if not meta.get("success"):
                continue
            names = meta.get("object_names") or []
            tip = f"tip_{meta['args']['tip']}"
            if tip not in names or "pipette_0" not in names:
                continue
            d = np.load(ep / "steps.npz")
            P = d["object_poses"]
            shift = neighbours.arm_shift(meta)                # randomized layouts move the arm too
            lay = meta.get("layout") if isinstance(meta.get("layout"), dict) else {}
            rx, ry, ryaw = (lay.get("platform") or {}).get("tip_rack", neighbours.DEFAULT_PLATFORM["tip_rack"])
            label = meta["args"]["tip"]
            ip, it = names.index("pipette_0"), names.index(tip)
            qq = np.concatenate([d["joint_position"][:, :7], d["finger_position_m"].reshape(len(P), -1)[:, :2]], axis=1)
            for t in range(len(P)):
                if d["tip_attached"][t] > 0.5:
                    break
                Tt = T(P[t, it])
                u = Tt[:3, 2]
                c = Tt[:3, 3] + u * TIP_LENGTH
                Tp = T(P[t, ip])
                if np.linalg.norm(Tp[:3, 3] - c) > NEAR_M:          # pipette origin = nozzle end
                    continue
                Th = np.linalg.inv(T(d["hand_pose"][t])) @ Tp
                q.append(qq[t])
                inhand.append(np.concatenate([Th[:3, 3], matrix_to_quat(Th[:3, :3])]))
                top.append(c + shift)
                up.append(u)
                src.append(str(ep))
                rack.append([rx + shift[0], ry + shift[1], ryaw])
                tip_rc.append(["ABCDEFGHIJ".index(label[0]), int(label[1:]) - 1])
    np.savez(OUT, q=np.array(q, np.float32), inhand=np.array(inhand, np.float32), top=np.array(top, np.float32),
             up=np.array(up, np.float32), source=np.array(src), rack=np.array(rack, np.float32),
             tip_rc=np.array(tip_rc, np.int64))
    print(f"{len(q)} start states from {len(set(src))} episodes -> {OUT}")
    if inhand:
        ih = np.array(inhand)
        print("in-hand pipette offset (m) range:", np.round(ih[:, :3].min(0), 4).tolist(), np.round(ih[:, :3].max(0), 4).tolist())


if __name__ == "__main__":
    main()

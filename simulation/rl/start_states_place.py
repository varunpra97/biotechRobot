r"""Start-state library for the tube-place RL task (mini milestone m4): the
arm holding a tube above the rack, from recorded expert place episodes
(place_tube_into_holder) with the rack at its default spot.

Frames: gripper closed on the tube, the tube's tip not yet in a bore (more
than NOT_IN_BORE_M above its resting height) and within NEAR_M of the target
slot's resting point. Each entry: joint positions (7 + 2 fingers), the held
tube's pose ([x, y, z, qw, qx, qy, qz], world == env-local), the target slot
(row, col) -- the slot the episode places into.

Usage (any venv with numpy, from the BiotechRobot folder):
    python biotech-lab-robotics\simulation\rl\start_states_place.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(r"C:\Users\varun\workspace\BiotechRobot\data")
OUT = HERE / "start_states_place.npz"
RUNS = ("primitives_place_M4", "primitives_tube_picks", "primitives_v2", "primitives_v3", "primitives_v4", "test_tube_picks", "primitives")
NEAR_M = 0.15
NOT_IN_BORE_M = 0.01
HOLDER_XY = (-0.05, 1.65)         # default layout (tube_grasp.env_cfg)
PITCH = 0.0615
TUBE_REST_Z = 0.94 + 0.012


def slot_rest(r, c):
    return np.array([HOLDER_XY[0] + (c - 2) * PITCH, HOLDER_XY[1] + (r - 1.5) * PITCH, TUBE_REST_Z])


def main():
    q, tube, slot, src = [], [], [], []
    skipped = {"layout": 0, "no_tube": 0}
    for run in RUNS:
        for ep in sorted(glob.glob(str(DATA / run / "place_tube_into_holder" / "ep_*"))):
            ep = Path(ep)
            meta = json.loads((ep / "meta.json").read_text())
            lay = meta.get("layout")
            if isinstance(lay, str):
                lay = None                                  # truncated/unknown: check the holder from the poses below
            holder = (lay or {}).get("platform", {}).get("holder")
            if holder is not None and np.linalg.norm(np.array(holder[:2]) - HOLDER_XY) > 1e-3:
                skipped["layout"] += 1
                continue
            label = meta["args"]["slot"]
            rc = ("ABCD".index(label[0]), int(label[1:]) - 1)
            target = slot_rest(*rc)
            names = meta.get("object_names") or []
            d = np.load(ep / "steps.npz")
            tubes = [i for i, n in enumerate(names) if n.startswith("tube")]
            if not tubes:
                skipped["no_tube"] += 1
                continue
            P = d["object_poses"]
            k = max(tubes, key=lambda i: np.linalg.norm(P[:, i, :3] - P[0, i, :3], axis=1).max())
            tp = P[:, k]
            if np.linalg.norm(tp[-1, :3] - target) > 0.003:   # did not end in that slot here: holder elsewhere
                skipped["layout"] += 1
                continue
            qq = np.concatenate([d["joint_position"][:, :7], d["finger_position_m"].reshape(len(tp), -1)[:, :2]], axis=1)
            closed = d["action"][:, 7] > 0.9
            for t in range(len(tp)):
                if (closed[t] and tp[t, 2] > TUBE_REST_Z + NOT_IN_BORE_M
                        and np.linalg.norm(tp[t, :3] - target) < NEAR_M):
                    q.append(qq[t])
                    tube.append(tp[t])
                    slot.append(rc)
                    src.append(str(ep))
    np.savez(OUT, q=np.array(q, np.float32), tube=np.array(tube, np.float32), slot=np.array(slot, np.int64),
             source=np.array(src))
    t = np.array(tube)
    print(f"{len(q)} start states from {len(set(src))} episodes -> {OUT}; skipped {skipped}")
    if len(t):
        s = np.array(slot)
        print("slots:", sorted({f'{"ABCD"[r]}{c + 1}' for r, c in s}))
        print(f"tube tip height above rest {(t[:, 2] - TUBE_REST_Z).min():.3f}..{(t[:, 2] - TUBE_REST_Z).max():.3f} m")


if __name__ == "__main__":
    main()

"""Start states for the tube-grasp RL task, computed in the lab simulator.

The RL task (tube_grasp_env.py) starts with the open hand NEAR a target tube
and must finish the grasp: centre over it, descend, close, lift. This script
builds the lab scene in its fixed (pilot) arrangement and, for every holder
slot, samples hand poses around the slot's grasp pose:
  - horizontal offset within START_XY_M, height START_Z_M above the grasp
    point, wrist turned within START_YAW_DEG and tilted within START_TILT_DEG
    of the expert's grasp frame,
  - plus the VLA's real near misses: every recorded tube-pick eval trial
    that ended within REAL_NEAR_M of its tube (hand pose relative to the tube,
    re-applied at every slot),
solves the arm's joint angles with the planner's calibrated analytic IK
(seeded from home), and keeps a pose only if it has joint-limit margin and
the arm is collision-free there (teleport + PhysX contact check).

Output: data/rl/tube_grasp_starts.npz
  q        [N, 7]  arm joint angles [rad]
  slot     [N]     slot index r * 5 + c
  offset   [N, 3]  hand (TCP) position minus the slot's grasp point [m]
  real     [N]     1 if from a real eval near miss
  grasp    [20, 3] grasp TCP point per slot in the lab's world frame [m]
  tube_tip [20, 3] tube tip (bottom) per slot [m]
  arm_base [3]     arm base position [m]; Q_HOME, closing axis, grasp depth

Usage (lab Isaac venv, from the BiotechRobot folder):
    .venv-isaac\\Scripts\\python.exe biotech-lab-robotics\\simulation\\rl\\precompute_starts.py [--per-slot 400]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parents[1] / "sim"
DATA = Path(r"C:\Users\varun\workspace\BiotechRobot\data")
START_XY_M = 0.05
START_Z_M = (0.02, 0.12)
START_YAW_DEG = 25.0
START_TILT_DEG = 5.0
REAL_NEAR_M = 0.08
MIN_MARGIN_RAD = 0.1


def rot(axis, deg):
    import numpy as np
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = axis
    return np.array([[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-slot", type=int, default=400, help="sampled candidates per slot")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, str(SIM))
    import collect_primitives as cp
    import layout

    layout.sample = lambda rng, tries=None: (layout.default_layout(), 0)
    out_dir = DATA / "rl"
    out_dir.mkdir(parents=True, exist_ok=True)
    rargs = argparse.Namespace(out=str(out_dir / "scene"), seed=args.seed, chain=0, max_episodes=1,
                               prior_valid=0, prior_chained=0, force_chained=False, pick_slot=None, block=None)
    (out_dir / "scene" / "chains").mkdir(parents=True, exist_ok=True)
    r = cp.ChainRunner(rargs)
    r.layout["empty_slots"] = []                 # all 20 slots filled, as the RL scene
    r.build()

    import numpy as np
    from expert import GRASP_DEPTH, Z_DOWN
    from planner import frame_from_axis

    ls, ex, pl = r.ls, r.ex, r.planner
    rng = np.random.default_rng(args.seed)
    q_home = np.asarray(ls.Q_HOME, float)
    R0 = frame_from_axis(Z_DOWN, ex.closing_y)             # the expert's grasp frame
    any_tube = next(n for n in r.env.objects if n.startswith("tube_"))
    tube_len = ex.tube_len(any_tube)
    tip_z = ex.holder_floor
    grasp, tips = [], []
    for s in range(20):
        x, y = ls.holder_slot_xy(s // 5, s % 5)
        tips.append([x, y, tip_z])
        grasp.append([x, y, tip_z + tube_len - GRASP_DEPTH])
    grasp, tips = np.array(grasp), np.array(tips)

    # the VLA's real near misses, as hand poses relative to their tube's grasp point
    real = []
    for row in json.loads((out_dir / "eval_tube_end_states.json").read_text()):
        tcp = np.array(row["tcp"][:3])
        tube = np.array(row["tube_start"])
        off = tcp - (tube + [0, 0, tube_len - GRASP_DEPTH])
        if np.hypot(off[0], off[1]) < REAL_NEAR_M and np.linalg.norm(np.array(row["tube_end"]) - tube) < 0.01:
            real.append(off)
    print(f"[starts] {len(real)} real near-miss offsets", flush=True)

    def try_pose(q_seed, pos, R):
        q, ok, *_ = pl.solve(pos, R, q_seed)
        if not ok or pl.limit_margin(q) < MIN_MARGIN_RAD:
            return None
        pl.arm.set_joint_positions(q)
        pl.app.update()
        if pl.collisions():
            return None
        return q[:7].copy()

    Q, SLOT, OFF, REAL = [], [], [], []
    for s in range(20):
        kept = 0
        cands = [(o, True) for o in real] + [(None, False)] * args.per_slot
        for off, is_real in cands:
            if off is None:
                off = np.array([rng.uniform(-START_XY_M, START_XY_M), rng.uniform(-START_XY_M, START_XY_M),
                                rng.uniform(*START_Z_M)])
            R = rot((0, 0, 1), rng.uniform(-START_YAW_DEG, START_YAW_DEG)) @ \
                rot((1, 0, 0), rng.uniform(-START_TILT_DEG, START_TILT_DEG)) @ R0
            q = try_pose(q_home, grasp[s] + off, R)
            if q is not None:
                Q.append(q); SLOT.append(s); OFF.append(off); REAL.append(is_real)
                kept += 1
        print(f"[starts] slot {s}: {kept}/{len(cands)} kept", flush=True)
    pl.arm.set_joint_positions(q_home)
    pl.app.update()
    np.savez(out_dir / "tube_grasp_starts.npz", q=np.array(Q), slot=np.array(SLOT), offset=np.array(OFF),
             real=np.array(REAL), grasp=grasp, tube_tip=tips, arm_base=np.array(ls.ARM_POS),
             q_home=q_home[:7], closing_y=np.asarray(ex.closing_y), grasp_depth=GRASP_DEPTH,
             tube_len=tube_len, holder_top=ex.holder_top)
    print(f"[starts] saved {len(Q)} start states -> {out_dir / 'tube_grasp_starts.npz'}", flush=True)
    r.app.close()


if __name__ == "__main__":
    main()

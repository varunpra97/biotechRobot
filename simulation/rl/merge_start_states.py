"""Merge the two start-state libraries into start_states_v2.npz (the format
tube_grasp/env.py reads):

  start_states.npz              recorded states: expert approaches (first
                                n_expert rows) + failed fixed-layout eval
                                trials -- 15 of 20 slots, skewed (B1 ~1/3)
  data/rl/tube_grasp_starts.npz (precompute_starts.py) poses sampled evenly
                                around all 20 slots' grasp points (+-5 cm,
                                2-12 cm above, +-25 deg wrist, +-5 deg tilt),
                                IK-solved and collision-checked in the lab
                                scene, incl. the VLA's real near misses

Usage (any Python with numpy):
    python biotech-lab-robotics\\simulation\\rl\\merge_start_states.py
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GRIPPER_OPEN = 0.012

import sys

REC = sys.argv[sys.argv.index("--recorded") + 1] if "--recorded" in sys.argv else "start_states.npz"
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "start_states_v2.npz"
a = np.load(HERE / REC)
b = np.load(Path(r"C:\Users\varun\workspace\BiotechRobot\data\rl\tube_grasp_starts.npz"))
assert np.allclose(a["grasp"], b["grasp"], atol=1e-4), "grasp points differ between the libraries"

n = len(b["q"])
q = np.concatenate([b["q"], np.full((n, 2), GRIPPER_OPEN)], axis=1).astype(np.float32)
slot = np.stack([b["slot"] // 5, b["slot"] % 5], axis=1).astype(np.int32)
tcp = (b["grasp"][b["slot"]] + b["offset"]).astype(np.float32)
source = np.where(b["real"], "eval_near_miss_resampled", "uniform_near_grasp")
out = dict(q=np.concatenate([a["q"], q]), slot=np.concatenate([a["slot"], slot]),
           tcp=np.concatenate([a["tcp"], tcp]), source=np.concatenate([a["source"], source]),
           n_expert=a["n_expert"], grasp=a["grasp"])
np.savez(HERE / OUT, **out)
print(f"{OUT}: {len(a['q'])} recorded + {n} sampled = {len(out['q'])} states, "
      f"{len({tuple(s) for s in out['slot'].tolist()})} slots")

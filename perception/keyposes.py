"""Task-space keyposes for tray pick-and-place (ChainedDiffuser-style).

The high-level policy reasons over KEYPOSES, not pixels::

    images -> keypose estimator -> initial keyposes
           -> keypose diffusion  -> refined keypose sequence
           -> trajectory diffusion -> joint actions

A keypose is an end-effector pose (position + quaternion) at a task
bottleneck. For tray pick-and-place the bottleneck structure is fixed::

    pre-grasp -> grasp -> lift -> transit -> pre-place -> place

Poses are derived analytically from the tray pose and goal pose. The
*learned* component is the vision -> pose estimator (trained supervised
from sim ground truth); keyposes themselves stay analytic so they are
always physically grounded — no diffusion hallucination at the
bottleneck level. The diffusion models then handle the multimodal parts:
which exact keypose sequence, and how to move between keyposes.
"""
from __future__ import annotations

import numpy as np

# Gripper pointing straight down: 180 deg about X. MuJoCo quat order (w,x,y,z).
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])

KEYPOSE_NAMES = ["pre_grasp", "grasp", "lift", "transit", "pre_place", "place"]


def compute_keyposes(tray_pos, goal_pos,
                     approach_h: float = 0.10,
                     grasp_h: float = 0.02,
                     lift_h: float = 0.16):
    """Analytic keypose sequence from tray pose and goal pose.

    Returns list of dicts: {"name", "pos" (3,), "quat" (4,)}.
    This is the "algorithm" converting scene state into initial keyposes.
    In deployment, tray_pos/goal_pos come from the learned vision estimator;
    in sim training they come from ground truth.
    """
    tray_pos = np.asarray(tray_pos, dtype=float).ravel()[:3]
    goal_pos = np.asarray(goal_pos, dtype=float).ravel()[:3]
    transit_xy = (tray_pos[:2] + goal_pos[:2]) / 2.0
    items = [
        ("pre_grasp", tray_pos + np.array([0, 0, approach_h])),
        ("grasp",     tray_pos + np.array([0, 0, grasp_h])),
        ("lift",      tray_pos + np.array([0, 0, lift_h])),
        ("transit",   np.array([transit_xy[0], transit_xy[1],
                                tray_pos[2] + lift_h])),
        ("pre_place", goal_pos + np.array([0, 0, approach_h])),
        ("place",     goal_pos + np.array([0, 0, grasp_h])),
    ]
    return [{"name": n, "pos": p, "quat": DOWN_QUAT.copy()} for n, p in items]


def keyposes_to_array(keyposes) -> np.ndarray:
    """(K, 7) array of [pos xyz, quat wxyz] — the diffusion target format."""
    return np.stack(
        [np.concatenate([k["pos"], k["quat"]]) for k in keyposes], axis=0
    )

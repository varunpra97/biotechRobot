"""Position-only damped least-squares IK against Isaac Sim's live physics
Jacobian.

The Jacobian layout was verified empirically against this project's Franka
asset (not assumed): `Articulation.get_jacobians()` returns shape
(1, n_bodies, 6, 15). Columns 0:6 are an implicit free-root block (present
even though the arm's base is effectively stationary); columns 6:15 are the
9 actuated DOFs, in `dof_names` order (joint1..joint7, finger_joint1,
finger_joint2). Verified by perturbing joint2 by a known delta and checking
which Jacobian column's predicted displacement matched the measured hand
displacement -- column 7 (= 6 + joint index 1) matched to within 0.0005 m,
every other column was off by 0.01-0.03 m.

This solver is deliberately position-only (3-DOF task error), not full
6-DOF pose matching: the tasks here (hover above / dip into a tube opening)
only constrain where the gripper is, not its exact orientation.

Because PhysX only writes updated link transforms back to the USD stage on
a physics step, each IK iteration needs the caller to advance the
simulation between setting joint positions and reading the resulting hand
pose / Jacobian -- that's what `step_fn` is for.
"""
from __future__ import annotations

import numpy as np

JACOBIAN_ROOT_COLUMNS = 6  # free-root block to skip; see module docstring


def solve_ik(arm, hand_body_index, target_pos, q_init, joint_limits,
             get_hand_pos, step_fn, n_arm_joints=7, max_iters=150,
             tol=5e-3, damping=0.05, max_step=0.15, steps_per_iter=2):
    """Solve for the n_arm_joints config that puts the hand at target_pos.

    q_init: full (9,) joint vector (7 arm + 2 finger) to warm-start from;
        only the first n_arm_joints are updated, fingers pass through
        unchanged.
    joint_limits: (9, 2) array of [lower, upper] per DOF (from
        Articulation.get_dof_limits()[0]).
    get_hand_pos: () -> np.ndarray[3], current hand world position.
    step_fn: () -> None, advances the simulation by one frame.

    Returns (q, converged, iters, final_error).
    """
    q = np.array(q_init, dtype=float).copy()
    err_norm = float("inf")

    for it in range(max_iters):
        arm.set_joint_positions(q)
        arm.set_joint_position_targets(q)
        for _ in range(steps_per_iter):
            step_fn()

        hand_pos = get_hand_pos()[:3]
        err = np.asarray(target_pos) - hand_pos
        err_norm = float(np.linalg.norm(err))
        if err_norm < tol:
            return q, True, it, err_norm

        J = np.asarray(arm.get_jacobians())[0][hand_body_index]  # (6, 15)
        lo = JACOBIAN_ROOT_COLUMNS
        J_pos = J[:3, lo:lo + n_arm_joints]  # (3, n_arm_joints) linear part

        JJt = J_pos @ J_pos.T + (damping ** 2) * np.eye(3)
        dq = J_pos.T @ np.linalg.solve(JJt, err)

        # Clamp the step: an unclamped DLS step can be huge near a poorly
        # conditioned Jacobian (seen empirically: >2.5 rad on the very
        # first iteration from the folded home pose), which overshoots the
        # local linearization and causes oscillation instead of
        # convergence. Capping it keeps each step small enough that the
        # Jacobian stays a valid local approximation.
        dq_norm = np.linalg.norm(dq)
        if dq_norm > max_step:
            dq = dq / dq_norm * max_step

        q_arm = q[:n_arm_joints] + dq
        q[:n_arm_joints] = np.clip(
            q_arm, joint_limits[:n_arm_joints, 0], joint_limits[:n_arm_joints, 1])

    return q, False, max_iters, err_norm

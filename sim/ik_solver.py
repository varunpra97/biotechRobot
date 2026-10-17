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


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rotation_error(R_target, R_current):
    """World-frame axis-angle vector rotating R_current onto R_target."""
    R = R_target @ R_current.T
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - th < 1e-6:  # 180 deg: axis from the symmetric part instead
        w, U = np.linalg.eigh((R + np.eye(3)) / 2.0)
        return np.pi * U[:, -1]
    return th / (2.0 * np.sin(th)) * v


def solve_ik_pose(arm, hand_body_index, target_pos, target_R, q_init, joint_limits,
                  get_tcp_pose, step_fn, tcp_local, n_arm_joints=7, max_iters=300,
                  pos_tol=2e-4, rot_tol=2e-3, damping=0.01, max_step=0.1,
                  rot_weight=0.1, steps_per_iter=2, null_gain=0.05):
    """6-DOF damped least-squares IK: put the TCP at target_pos AND the
    hand at orientation target_R (world).

    Needed (vs. solve_ik's position-only control) wherever the gripper's
    orientation matters physically: grasping a vertical tube needs the
    fingers vertical around it, and inserting it into a 45-deg rotor bore
    needs the tube -- i.e. the hand -- coaxial with the bore. Position-only
    IK lets orientation drift freely between waypoints.

    Jacobian layout verified empirically (cad_source/probe_franka_kinematics.py,
    finite differences): rows 0:3 are the hand ORIGIN's linear velocity,
    rows 3:6 the world-frame angular velocity, columns 6:15 the 9 DOFs.
    The TCP's linear Jacobian is the origin's plus omega x r:
    J_v_tcp = J_v - [r]x J_w, r = tcp - hand origin.

    rot_weight (m/rad) balances orientation against position error in the
    stacked least-squares step; tolerances are checked unweighted.

    get_tcp_pose: () -> (tcp_pos, R_hand).
    tcp_local: the TCP's offset in the hand frame (lab_scene.TCP_LOCAL).
    Returns (q, converged, iters, pos_err_m, rot_err_rad).
    """
    q = np.array(q_init, dtype=float).copy()
    pos_err = rot_err = float("inf")
    lo = JACOBIAN_ROOT_COLUMNS
    for it in range(max_iters):
        arm.set_joint_positions(q)
        arm.set_joint_position_targets(q)
        for _ in range(steps_per_iter):
            step_fn()

        tcp, R = get_tcp_pose()
        e_p = np.asarray(target_pos) - tcp
        e_r = rotation_error(target_R, R)
        pos_err, rot_err = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
        if pos_err < pos_tol and rot_err < rot_tol:
            return q, True, it, pos_err, rot_err

        J = np.asarray(arm.get_jacobians())[0][hand_body_index]
        Jv = J[:3, lo:lo + n_arm_joints]
        Jw = J[3:6, lo:lo + n_arm_joints]
        Jv_tcp = Jv - _skew(R @ np.asarray(tcp_local)) @ Jw
        Jst = np.vstack([Jv_tcp, rot_weight * Jw])
        e = np.concatenate([e_p, rot_weight * e_r])
        J_pinv = Jst.T @ np.linalg.inv(Jst @ Jst.T + damping ** 2 * np.eye(6))
        dq = J_pinv @ e
        # The Franka has 7 joints for this 6-DOF task; spend the spare DOF
        # (the task's null space) pulling joints toward mid-range. Without
        # it, a joint pinned at its limit by clipping stalls the solve
        # mid-path (seen: 0.9mm residual halfway through a transit).
        lo_l, hi_l = joint_limits[:n_arm_joints, 0], joint_limits[:n_arm_joints, 1]
        mid, span = (lo_l + hi_l) / 2, (hi_l - lo_l)
        centering = -null_gain * (q[:n_arm_joints] - mid) / (span / 2) ** 2
        dq += (np.eye(n_arm_joints) - J_pinv @ Jst) @ centering
        n = np.linalg.norm(dq)
        if n > max_step:
            dq *= max_step / n
        q[:n_arm_joints] = np.clip(q[:n_arm_joints] + dq,
                                   joint_limits[:n_arm_joints, 0],
                                   joint_limits[:n_arm_joints, 1])
    return q, False, max_iters, pos_err, rot_err

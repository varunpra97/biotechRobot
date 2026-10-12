"""Damped least-squares IK: task-space keypose -> joint configuration.

Used offline to convert analytic keyposes into joint-space demonstrations.
The learned policies never run IK at deployment — the low-level trajectory
diffuser outputs joint actions directly.
"""
from __future__ import annotations

import numpy as np


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Rotation vector for a (near-identity) error quaternion."""
    w = float(np.clip(q[0], -1.0, 1.0))
    v = q[1:4]
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1e-12, 1.0 - w * w))
    if s < 1e-6 or angle < 1e-9:
        return np.zeros(3)
    return v / s * angle


def ik_dls(model, data, target_pos, target_quat,
           site_name: str = "ee", q_init=None,
           max_iter: int = 300, pos_tol: float = 3e-3,
           damping: float = 1e-2, rot_w: float = 0.5,
           max_step: float = 0.1, max_restarts: int = 20, seed: int = 0,
           orient_refine: bool = True):
    """Solve IK for the 6 arm DOFs. Returns (q, pos_err, quat_achieved).

    Two-tier: (1) position-only solve with adaptive restarts until `pos_tol`
    (position is what grasping needs); (2) optional orientation refinement
    that is kept only if it preserves position accuracy. The returned
    achieved quaternion is what demos record, so training data always
    reflects reachable keyposes.
    """
    import mujoco

    rng = np.random.default_rng(seed)
    lo, hi = model.jnt_range[:6, 0], model.jnt_range[:6, 1]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

    def attempt(q0, iters, rw):
        q, err = _ik_single(model, data, target_pos, target_quat, site_name,
                            q0, iters, 1e-4, damping, rw, max_step, lo, hi)
        return q, err

    # Tier 1: position-only, adaptive restarts.
    best_q, best_err = None, float("inf")
    if q_init is not None:
        best_q, best_err = attempt(np.clip(q_init, lo, hi), max_iter, 0.0)
    n_restart = 0
    while best_err > pos_tol and n_restart < max_restarts:
        q, err = attempt(rng.uniform(lo, hi), 200, 0.0)
        n_restart += 1
        if err < best_err:
            best_q, best_err = q, err

    # Tier 2: orientation refinement, kept only if position holds.
    if orient_refine and best_err <= pos_tol:
        q2, err2 = attempt(best_q, 200, rot_w)
        if err2 <= max(pos_tol, best_err * 1.5):
            best_q, best_err = q2, err2

    data.qpos[:6] = best_q
    mujoco.mj_forward(model, data)
    q_ach = np.zeros(4)
    mujoco.mju_mat2Quat(q_ach, data.site_xmat[site_id].flatten())
    return best_q, best_err, q_ach


def _ik_single(model, data, target_pos, target_quat, site_name, q_init,
               max_iter, tol, damping, rot_w, max_step, lo, hi):
    import mujoco

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    data.qpos[:6] = np.clip(q_init, lo, hi)
    target_pos = np.asarray(target_pos, dtype=float).ravel()[:3]
    target_quat = np.asarray(target_quat, dtype=float).ravel()[:4]

    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        ep = target_pos - data.site_xpos[site_id]
        cur_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cur_quat, data.site_xmat[site_id].flatten())
        eq = np.zeros(4)
        mujoco.mju_mulQuat(eq, target_quat, _quat_conj(cur_quat))
        er = _quat_to_axis_angle(eq)
        e = np.concatenate([ep, rot_w * er])
        if float(np.linalg.norm(e)) < tol:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, :6], rot_w * jacr[:, :6]])  # (6, 6)
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
        data.qpos[:6] = np.clip(
            data.qpos[:6] + np.clip(dq, -max_step, max_step), lo, hi)

    mujoco.mj_forward(model, data)
    err = float(np.linalg.norm(target_pos - data.site_xpos[site_id]))
    return data.qpos[:6].copy(), err

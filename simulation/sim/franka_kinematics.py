"""Analytic Franka Panda kinematics (numpy) for planning without touching
the physics simulation.

The planner used to solve IK by teleporting the articulation and stepping
physics every iteration to read the hand pose and Jacobian back. Mid-
episode that disturbs the world it's planning in (the finger pads shove a
held object every time the hand moves inside a solve), so IK here is pure
math: standard Panda DH kinematics (Craig convention, Franka's published
parameters), with the base pose and the flange-to-hand transform calibrated
once against the live articulation (calibrate()), so FK matches the
simulator rather than an assumption.
"""
from __future__ import annotations

import math

import numpy as np

# Franka Emika Panda, modified DH (Craig): a_{i-1}, d_i, alpha_{i-1}
DH_A = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
DH_D = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0]
DH_ALPHA = [0.0, -math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2]
FLANGE_D = 0.107


def _mdh(a, d, alpha, theta):
    ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
    return np.array([[ct, -st, 0, a],
                     [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca],
                     [0, 0, 0, 1.0]])


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _rot_err(R_t, R):
    Rd = R_t @ R.T
    c = np.clip((np.trace(Rd) - 1) / 2, -1, 1)
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([Rd[2, 1] - Rd[1, 2], Rd[0, 2] - Rd[2, 0], Rd[1, 0] - Rd[0, 1]])
    if math.pi - th < 1e-6:
        w, U = np.linalg.eigh((Rd + np.eye(3)) / 2)
        return math.pi * U[:, -1]
    return th / (2 * math.sin(th)) * v


class FrankaKinematics:
    def __init__(self, T_world_base, T_flange_hand, tcp_local, limits):
        self.T_wb = np.asarray(T_world_base, float)
        self.T_fh = np.asarray(T_flange_hand, float)
        self.tcp_local = np.asarray(tcp_local, float)
        self.limits = np.asarray(limits, float)[:7]

    # ------------------------------------------------------------------ FK
    def frames(self, q):
        """World frames of joints 1..7 (their z axes are the joint axes)."""
        T = self.T_wb.copy()
        out = []
        for i in range(7):
            T = T @ _mdh(DH_A[i], DH_D[i], DH_ALPHA[i], q[i])
            out.append(T.copy())
        return out

    def hand(self, q):
        T7 = self.frames(q)[-1]
        Tf = T7.copy()
        Tf[:3, 3] = T7[:3, 3] + T7[:3, 2] * FLANGE_D
        return Tf @ self.T_fh

    def tcp(self, q):
        H = self.hand(q)
        return H[:3, 3] + H[:3, :3] @ self.tcp_local, H[:3, :3]

    def jacobian_tcp(self, q):
        fr = self.frames(q)
        p, _ = self.tcp(q)
        J = np.zeros((6, 7))
        for i, T in enumerate(fr):
            z, o = T[:3, 2], T[:3, 3]
            J[:3, i] = np.cross(z, p - o)
            J[3:, i] = z
        return J

    # ------------------------------------------------------------------ IK
    def ik(self, pos, R, q0, max_iters=1500, pos_tol=2e-4, rot_tol=2e-3,
           damping=0.01, max_step=0.2, rot_weight=0.1, null_gain=0.05):
        """6-DOF damped least squares at the TCP, with null-space pull toward
        mid-range joint angles (the arm has one redundant DOF).
        Returns (q7, ok, pos_err, rot_err)."""
        q = np.array(q0[:7], float)
        lo, hi = self.limits[:, 0], self.limits[:, 1]
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        pe = re = float("inf")
        for _ in range(max_iters):
            p, Rc = self.tcp(q)
            e_p = np.asarray(pos) - p
            e_r = _rot_err(R, Rc)
            pe, re = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
            if pe < pos_tol and re < rot_tol:
                return q, True, pe, re
            J = self.jacobian_tcp(q)
            Jw = np.vstack([J[:3], rot_weight * J[3:]])
            e = np.concatenate([e_p, rot_weight * e_r])
            Jp = Jw.T @ np.linalg.inv(Jw @ Jw.T + damping ** 2 * np.eye(6))
            dq = Jp @ e + (np.eye(7) - Jp @ Jw) @ (-null_gain * (q - mid) / half ** 2)
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n
            q = np.clip(q + dq, lo, hi)
        return q, False, pe, re


def calibrate(arm, simulation_app, hand_pose_fn, tcp_local, n=12, seed=0):
    """Fit the base pose and flange->hand transform to the live articulation:
    teleport to n random in-limit configurations, read the hand pose back,
    and solve T_world_hand = T_wb * F(q) * T_fh. Returns (FrankaKinematics,
    worst residual in meters) -- the residual is the check that the DH model
    really is this robot."""
    import lab_scene

    rng = np.random.default_rng(seed)
    limits = np.asarray(arm.get_dof_limits())[0]
    q_saved = np.asarray(arm.get_joint_positions())[0].copy()
    yaw = math.radians(90.0)
    T_wb = np.eye(4)
    T_wb[:3, :3] = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    T_wb[:3, 3] = lab_scene.ARM_POS
    rough = FrankaKinematics(T_wb, np.eye(4), tcp_local, limits)
    samples = []
    tries = 0
    while len(samples) < n and tries < 20 * n:
        tries += 1
        q = q_saved.copy()
        q[:7] = np.clip(lab_scene.Q_HOME[:7] + rng.uniform(-0.5, 0.5, 7), limits[:7, 0], limits[:7, 1])
        # keep well clear of the bench and everything on it
        if min(T[2, 3] for T in rough.frames(q[:7])[2:]) < lab_scene.BENCH_Z + 0.25:
            continue
        arm.set_joint_positions(q)
        arm.set_joint_position_targets(q)
        simulation_app.update()
        simulation_app.update()
        if np.abs(np.asarray(arm.get_joint_positions())[0][:7] - q[:7]).max() > 1e-4:
            continue  # something pushed the arm -- not a clean sample
        h = hand_pose_fn()
        T = np.eye(4)
        T[:3, :3] = lab_scene.quat_to_matrix(h[3:7])
        T[:3, 3] = h[:3]
        samples.append((q[:7].copy(), T))
    arm.set_joint_positions(q_saved)
    arm.set_joint_position_targets(q_saved)
    simulation_app.update()
    if len(samples) < 6:
        raise RuntimeError(f"only {len(samples)} clean calibration samples")

    # base from the scene (ARM_POS, 90 deg yaw); fit T_fh, then check.
    kin = FrankaKinematics(T_wb, np.eye(4), tcp_local, limits)
    fh = []
    for q, T in samples:
        T7 = kin.frames(q)[-1]
        Tf = T7.copy()
        Tf[:3, 3] = T7[:3, 3] + T7[:3, 2] * FLANGE_D
        fh.append(np.linalg.inv(Tf) @ T)
    t = np.mean([X[:3, 3] for X in fh], axis=0)
    U, _, Vt = np.linalg.svd(sum(X[:3, :3] for X in fh))
    Rm = U @ Vt
    T_fh = np.eye(4)
    T_fh[:3, :3], T_fh[:3, 3] = Rm, t
    kin.T_fh = T_fh
    worst = max(float(np.linalg.norm(kin.hand(q)[:3, 3] - T[:3, 3])) for q, T in samples)
    worst_rot = max(float(np.linalg.norm(_rot_err(T[:3, :3], kin.hand(q)[:3, :3]))) for q, T in samples)
    return kin, worst, worst_rot

"""The RL tube-grasp teacher (rl/tube_grasp, trained in Isaac Lab) driving our
Isaac Sim lab env: it takes over a tube pick from wherever the arm is and
finishes the grasp and a 3 cm upright lift.

It sees exactly what it was trained on (rl/tube_grasp/env.py, 29 values):
TCP minus the target tube's grasp point (rim - 12 mm along its axis) x10, the
tube's tilt / 30 deg, its lift above the bore floor x10, its velocity, the
finger positions x25, the gripper command, arm joints minus home, joint
velocities x0.1, its last action. Its action -- a TCP move (dx, dy, dz) of up
to 1 cm and open / close -- becomes a joint command the same way: one damped
least-squares IK step on the TCP (analytic Jacobian), the hand's orientation
held at the hand-over, the joint goal accumulated on the env's command and at
most MAX_LEAD ahead of the measured arm, sent as lab_env's joint-velocity
action (the env interpolates it over the step, as Isaac Lab's env did).

    fin = TeacherFinisher(env, planner.kin, tube_path, policy_path)
    fin.start()              # at the hand-over
    while ...: env.step(fin.action())
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import lab_env
import lab_scene

POLICY = Path(__file__).resolve().parents[1] / "rl" / "teacher_policy.pt"
TUBE_REST_Z = lab_scene.BENCH_Z + 0.04 + lab_scene.HOLDER_BORE_FLOOR_MM * lab_scene.MM_TO_M   # 0.952
TUBE_LEN, GRASP_DEPTH = 0.058, 0.012
ACTION_STEP_M = 0.01
IK_DAMPING = 0.05
MAX_LEAD = 0.1
OPEN_CMD = 1.0 - 0.012 / lab_scene.GRIPPER_STROKE        # 12 mm per finger, the expert's tube opening


def _rot_error(R_des, R):
    """Axis-angle vector rotating R onto R_des (world frame)."""
    E = R_des @ R.T
    c = np.clip((np.trace(E) - 1) / 2, -1.0, 1.0)
    ang = math.acos(c)
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    return v / (2 * math.sin(ang)) * ang


class TeacherFinisher:
    def __init__(self, env, kin, tube_path, policy_path=POLICY, pose_fn=None):
        """pose_fn(prim_path) -> [x, y, z, qw, qx, qy, qz] world pose of the
        target tube (default: the sim's exact pose, lab_scene.get_prim_world_pose;
        the vision pipeline's PerceivedPose(localizer) is a drop-in)."""
        import torch
        self.torch = torch
        self.env, self.kin, self.tube_path = env, kin, tube_path
        self.pose_fn = pose_fn or lab_scene.get_prim_world_pose
        self.policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
        self.q_home = np.asarray(lab_scene.Q_HOME[:7], float)
        lim = np.asarray(env.limits)[:7]
        self.q_lo, self.q_hi = lim[:, 0], lim[:, 1]
        self.R_des = None
        self.last = np.zeros(4)
        self.closed = False
        self.prev_tip = None

    def _tube(self):
        p = np.asarray(self.pose_fn(self.tube_path), float)
        R = lab_scene.quat_to_matrix(p[3:7])
        axis = R[:, 2]
        grasp = p[:3] + axis * (TUBE_LEN - GRASP_DEPTH)
        tilt = math.degrees(math.acos(np.clip(axis[2], -1.0, 1.0)))
        return p[:3], grasp, tilt

    def _joints(self):
        q = np.asarray(self.env.arm.get_joint_positions())[0]
        qd = np.asarray(self.env.arm.get_joint_velocities())[0]
        return q, qd

    def status(self):
        """(TCP - grasp distance m, lift m, tilt deg, finger gap m)."""
        q, _ = self._joints()
        tcp, _ = self.kin.tcp(q[:7])
        tip, grasp, tilt = self._tube()
        return float(np.linalg.norm(tcp - grasp)), float(tip[2] - TUBE_REST_Z), tilt, float(q[7] + q[8])

    def start(self):
        """Hand-over: hold the hand's current orientation from here on."""
        q, _ = self._joints()
        _, self.R_des = self.kin.tcp(q[:7])
        self.last[:] = 0.0
        self.closed = False
        self.prev_tip = None

    def observation(self):
        q, qd = self._joints()
        tcp, _ = self.kin.tcp(q[:7])
        tip, grasp, tilt = self._tube()
        vel = np.zeros(3) if self.prev_tip is None else (tip - self.prev_tip) * lab_env.CONTROL_HZ
        self.prev_tip = tip.copy()
        obs = np.concatenate([
            (tcp - grasp) * 10.0, [tilt / 30.0], [(tip[2] - TUBE_REST_Z) * 10.0], vel,
            q[7:9] * 25.0, [float(self.closed)], q[:7] - self.q_home, qd[:7] * 0.1, self.last,
        ])
        return np.clip(np.nan_to_num(obs), -10.0, 10.0).astype(np.float32)

    def action(self):
        """The teacher's next action as a lab_env action vector."""
        torch = self.torch
        with torch.inference_mode():
            a = self.policy(torch.from_numpy(self.observation()).unsqueeze(0))[0].numpy().astype(float)
        a = np.clip(a, -1.0, 1.0)
        self.last = a.copy()
        return self._ik_action(a)

    def lift_action(self):
        """Scripted: straight up 1 cm this step, gripper closed (after the teacher's lift)."""
        return self._ik_action(np.array([0.0, 0.0, 1.0, 1.0]))

    def hold_action(self):
        return self._ik_action(np.array([0.0, 0.0, 0.0, 1.0]))

    def _ik_action(self, a):
        q, _ = self._joints()
        tcp, R = self.kin.tcp(q[:7])
        err = np.concatenate([a[:3] * ACTION_STEP_M, _rot_error(self.R_des, R)])
        J = self.kin.jacobian_tcp(q[:7])
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING ** 2 * np.eye(6), err)
        q_cmd = np.asarray(self.env.q_cmd[:7], float)
        lead = np.clip(q_cmd + dq - q[:7], -MAX_LEAD, MAX_LEAD)
        q_goal = np.clip(q[:7] + lead, self.q_lo, self.q_hi)
        self.closed = bool(a[3] > 0.0)
        v = lab_env.delta_to_droid_velocity(q_goal - q_cmd)
        return self.env.action(v, gripper=1.0 if self.closed else OPEN_CMD)


# ------------------------------------------------------------------ tray
TRAY_POLICY = Path(__file__).resolve().parents[1] / "rl" / "tray_policy.pt"
TRAY_GRASP_Z = 0.005                               # expert.Expert.TRAY_GRASP_Z
TRAY_OPEN_CMD = 1.0 - 0.020 / lab_scene.GRIPPER_STROKE   # expert.Expert.TRAY_OPEN (20 mm per finger)
YAW_STEP_RAD = 0.1
ROT_ERR_MAX = 0.2


def _down_frame(yaw):
    """Hand pointing down, closing axis (hand y) at this yaw (world)."""
    y = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.stack([np.cross(y, z), y, z], axis=1)


class TrayFinisher(TeacherFinisher):
    """The RL tray-grasp teacher (rl/tray_grasp): grasp the sample tray across
    its width and lift it level (the generator's pick_up_tray goal). Same
    observation (32) and action (dx, dy, dz, dyaw, grip) as its Isaac Lab env."""

    def __init__(self, env, kin, tray_path, policy_path=TRAY_POLICY, pose_fn=None):
        super().__init__(env, kin, tray_path, policy_path, pose_fn)
        self.last = np.zeros(5)
        self.yaw_des = 0.0

    def _tray(self):
        p = np.asarray(self.pose_fn(self.tube_path), float)
        R = lab_scene.quat_to_matrix(p[3:7])
        tray_yaw = math.atan2(R[1, 0], R[0, 0])
        tilt = math.degrees(math.acos(np.clip(R[2, 2], -1.0, 1.0)))
        return p[:3], p[:3] + np.array([0.0, 0.0, TRAY_GRASP_Z]), tray_yaw, tilt

    def _closing_yaw(self, R):
        return math.atan2(R[1, 1], R[0, 1])

    def _yaw_error(self, R, tray_yaw):
        e = self._closing_yaw(R) - (tray_yaw + math.pi / 2)
        return math.atan2(math.sin(2 * e), math.cos(2 * e)) / 2

    def status(self):
        q, _ = self._joints()
        tcp, _ = self.kin.tcp(q[:7])
        pos, grasp, _, tilt = self._tray()
        return float(np.linalg.norm(tcp - grasp)), float(pos[2] - (lab_scene.BENCH_Z + 0.04)), tilt, float(q[7] + q[8])

    def start(self):
        q, _ = self._joints()
        _, R = self.kin.tcp(q[:7])
        self.yaw_des = self._closing_yaw(R)
        self.last = np.zeros(5)
        self.closed = False
        self.prev_tip = None

    def observation(self):
        q, qd = self._joints()
        tcp, R = self.kin.tcp(q[:7])
        pos, grasp, tray_yaw, tilt = self._tray()
        ye = self._yaw_error(R, tray_yaw)
        vel = np.zeros(3) if self.prev_tip is None else (pos - self.prev_tip) * lab_env.CONTROL_HZ
        self.prev_tip = pos.copy()
        obs = np.concatenate([
            (tcp - grasp) * 10.0, [math.sin(2 * ye), math.cos(2 * ye)], [tilt / 30.0],
            [(pos[2] - (lab_scene.BENCH_Z + 0.04)) * 10.0], vel, q[7:9] * 25.0, [float(self.closed)],
            q[:7] - self.q_home, qd[:7] * 0.1, self.last,
        ])
        return np.clip(np.nan_to_num(obs), -10.0, 10.0).astype(np.float32)

    def action(self):
        torch = self.torch
        with torch.inference_mode():
            a = self.policy(torch.from_numpy(self.observation()).unsqueeze(0))[0].numpy().astype(float)
        a = np.clip(a, -1.0, 1.0)
        self.last = a.copy()
        self.yaw_des += a[3] * YAW_STEP_RAD
        return self._tray_ik(a[:3], a[4] > 0.0)

    def lift_action(self):
        return self._tray_ik(np.array([0.0, 0.0, 1.0]), True)

    def hold_action(self):
        return self._tray_ik(np.zeros(3), True)

    def _tray_ik(self, dxyz, close):
        q, _ = self._joints()
        tcp, R = self.kin.tcp(q[:7])
        rot = np.clip(_rot_error(_down_frame(self.yaw_des), R), -ROT_ERR_MAX, ROT_ERR_MAX)
        err = np.concatenate([np.asarray(dxyz) * ACTION_STEP_M, rot])
        J = self.kin.jacobian_tcp(q[:7])
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING ** 2 * np.eye(6), err)
        q_cmd = np.asarray(self.env.q_cmd[:7], float)
        lead = np.clip(q_cmd + dq - q[:7], -MAX_LEAD, MAX_LEAD)
        q_goal = np.clip(q[:7] + lead, self.q_lo, self.q_hi)
        self.closed = bool(close)
        return self.env.action(lab_env.delta_to_droid_velocity(q_goal - q_cmd),
                               gripper=1.0 if self.closed else TRAY_OPEN_CMD)


# ------------------------------------------------------------------ pipette
PIPETTE_POLICY = Path(__file__).resolve().parents[1] / "rl" / "pipette_policy.pt"
PIPETTE_GRASP_Y = 0.11                             # pipette_expert.PIPETTE_GRASP_Y
PIPETTE_GRASP_BELOW_AXIS = 0.006                   # pipette_expert.PIPETTE_GRASP_BELOW_AXIS
PIPETTE_OPEN_CMD = 1.0 - 0.016 / lab_scene.GRIPPER_STROKE   # pipette_expert.PIPETTE_OPEN_CMD


class PipetteFinisher(TeacherFinisher):
    """The RL pipette-grasp teacher (rl/pipette_grasp): grasp the lying
    pipette across its body at the expert's grasp point and lift it 3 cm
    without turning it; pipette_expert.turn_pipette_ready then turns it
    nozzle-down. Same observation (32) and action (dx, dy, dz, grip) as its
    Isaac Lab env: the hand's orientation servos onto the expert's grasp
    frame (at most ROT_ERR_MAX per step), frozen once the gripper closes."""

    def __init__(self, env, kin, pipette_path, policy_path=PIPETTE_POLICY, pose_fn=None):
        super().__init__(env, kin, pipette_path, policy_path, pose_fn)
        self.z_start = np.array([0.0, 0.0, 1.0])
        self.p_start = np.zeros(3)

    def _pipette(self):
        p = np.asarray(self.pose_fn(self.tube_path), float)
        R = lab_scene.quat_to_matrix(p[3:7])
        lx, ly, lz = R[:, 0], R[:, 1], R[:, 2]
        grasp = p[:3] + ly * PIPETTE_GRASP_Y - lz * PIPETTE_GRASP_BELOW_AXIS
        z = -lz                                         # planner.frame_from_axis(-lz, +lx)
        y = lx - (lx @ z) * z
        y /= np.linalg.norm(y)
        frame = np.stack([np.cross(y, z), y, z], axis=1)
        turn = math.degrees(math.acos(np.clip(lz @ self.z_start, -1.0, 1.0)))
        return p[:3], grasp, frame, turn

    def status(self):
        """(TCP - grasp distance m, lift m since the hand-over, turn deg since the hand-over, finger gap m)."""
        q, _ = self._joints()
        tcp, _ = self.kin.tcp(q[:7])
        pos, grasp, _, turn = self._pipette()
        return float(np.linalg.norm(tcp - grasp)), float(pos[2] - self.p_start[2]), turn, float(q[7] + q[8])

    def start(self):
        p = np.asarray(self.pose_fn(self.tube_path), float)
        self.p_start = p[:3].copy()
        self.z_start = lab_scene.quat_to_matrix(p[3:7])[:, 2].copy()
        self.R_des = self._pipette()[2]
        self.last = np.zeros(4)
        self.closed = False
        self.prev_tip = None

    def observation(self):
        q, qd = self._joints()
        tcp, R = self.kin.tcp(q[:7])
        pos, grasp, _, turn = self._pipette()          # R_des: as of the last action (as Isaac Lab's obs)
        vel = np.zeros(3) if self.prev_tip is None else (pos - self.prev_tip) * lab_env.CONTROL_HZ
        self.prev_tip = pos.copy()
        obs = np.concatenate([
            (tcp - grasp) * 10.0, _rot_error(self.R_des, R), [turn / 30.0], [(pos[2] - self.p_start[2]) * 10.0], vel,
            q[7:9] * 25.0, [float(self.closed)], q[:7] - self.q_home, qd[:7] * 0.1, self.last,
        ])
        return np.clip(np.nan_to_num(obs), -10.0, 10.0).astype(np.float32)

    def _ik_action(self, a):
        q, _ = self._joints()
        _, R = self.kin.tcp(q[:7])
        if not self.closed:                             # (Isaac Lab: the frame is refreshed before each IK step)
            self.R_des = self._pipette()[2]
        rot = np.clip(_rot_error(self.R_des, R), -ROT_ERR_MAX, ROT_ERR_MAX)
        err = np.concatenate([a[:3] * ACTION_STEP_M, rot])
        J = self.kin.jacobian_tcp(q[:7])
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING ** 2 * np.eye(6), err)
        q_cmd = np.asarray(self.env.q_cmd[:7], float)
        lead = np.clip(q_cmd + dq - q[:7], -MAX_LEAD, MAX_LEAD)
        q_goal = np.clip(q[:7] + lead, self.q_lo, self.q_hi)
        self.closed = bool(a[3] > 0.0)
        return self.env.action(lab_env.delta_to_droid_velocity(q_goal - q_cmd),
                               gripper=1.0 if self.closed else PIPETTE_OPEN_CMD)


# ------------------------------------------------------------------ tube place
PLACE_POLICY = Path(__file__).resolve().parents[1] / "rl" / "place_policy.pt"
HOLDER_FLOOR_Z = lab_scene.BENCH_Z + 0.04 + lab_scene.HOLDER_BORE_FLOOR_MM * lab_scene.MM_TO_M   # a tube's tip at rest


class PlaceFinisher(TeacherFinisher):
    """The RL tube-place teacher (rl/tube_place): lower the held tube into its
    slot's bore, release it and back off. Same observation (33) and action
    (dx, dy, dz, grip) as its Isaac Lab env; the hand keeps the orientation it
    has at the hand-over."""

    def __init__(self, env, kin, tube_path, slot_rc, policy_path=PLACE_POLICY, pose_fn=None):
        super().__init__(env, kin, tube_path, policy_path, pose_fn)
        self.rest = np.array([*lab_scene.holder_slot_xy(*slot_rc), HOLDER_FLOOR_Z])

    def start(self):
        super().start()
        self.closed = True
        self.last = np.array([0.0, 0.0, 0.0, 1.0])

    def status(self):
        """(tip to the slot's resting point m, tip height above it m, tilt deg, finger gap m)."""
        d, tilt, gap, _ = self.place_status()
        return float(np.linalg.norm(d)), float(d[2]), tilt, gap

    def place_status(self):
        """(tip - resting point, tilt deg, finger gap m, TCP - grasp point)."""
        q, _ = self._joints()
        tcp, _ = self.kin.tcp(q[:7])
        tip, grasp, tilt = self._tube()
        return tip - self.rest, tilt, float(q[7] + q[8]), tcp - grasp

    def observation(self):
        q, qd = self._joints()
        tcp, _ = self.kin.tcp(q[:7])
        p = np.asarray(self.pose_fn(self.tube_path), float)
        axis = lab_scene.quat_to_matrix(p[3:7])[:, 2]
        tip, grasp, tilt = self._tube()
        vel = np.zeros(3) if self.prev_tip is None else (tip - self.prev_tip) * lab_env.CONTROL_HZ
        self.prev_tip = tip.copy()
        obs = np.concatenate([
            (tip - self.rest) * 10.0, [tilt / 30.0], axis[:2], (tcp - grasp) * 10.0, vel,
            q[7:9] * 25.0, [float(self.closed)], q[:7] - self.q_home, qd[:7] * 0.1, self.last,
        ])
        return np.clip(np.nan_to_num(obs), -10.0, 10.0).astype(np.float32)


# ------------------------------------------------------------------ centrifuge insert
INSERT_POLICY = Path(__file__).resolve().parents[1] / "rl" / "insert_policy.pt"
RELEASE_TIP_S = -0.014                           # expert.RELEASE_TIP_S
INSERT_STEP_M = 0.005                            # rl/tube_insert: 5 mm per step (a held tube slips at 1 cm)
INSERT_ROT_STEP = 0.1


class InsertFinisher(TeacherFinisher):
    """The RL centrifuge-insert teacher (rl/tube_insert): move the held tube
    down the rotor bore, release it seated and back the hand off. Same
    observation (33) and action (dx, dy, dz, grip) as its Isaac Lab env: while
    the tube is held above the bore's opening the hand turns (at most
    INSERT_ROT_STEP per step) to line the tube's axis up with the bore's;
    inside the bore, and once released, it holds its orientation."""

    def __init__(self, env, kin, tube_path, bore, policy_path=INSERT_POLICY, pose_fn=None):
        super().__init__(env, kin, tube_path, policy_path, pose_fn)
        self.bore = bore                                 # centrifuge_geometry.Bore

    def start(self):
        super().start()
        self.closed = True
        self.last = np.array([0.0, 0.0, 0.0, 1.0])

    def _vs_bore(self):
        p = np.asarray(self.pose_fn(self.tube_path), float)
        axis = lab_scene.quat_to_matrix(p[3:7])[:, 2]
        tip = p[:3]
        s = self.bore.s_of(tip)
        radial = self.bore.radial_offset(tip)
        ang = math.degrees(math.acos(np.clip(axis @ self.bore.axis, -1.0, 1.0)))
        rel = tip - self.bore.at(RELEASE_TIP_S)
        return tip, axis, tip + axis * (TUBE_LEN - GRASP_DEPTH), rel, s, radial, ang

    def status(self):
        """(tip to the bore opening m, depth s m, tube-vs-bore angle deg, finger gap m)."""
        q, _ = self._joints()
        tip, _, _, _, s, _, ang = self._vs_bore()
        return float(np.linalg.norm(tip - self.bore.at(self.bore.opening_s))), s, ang, float(q[7] + q[8])

    def observation(self):
        q, qd = self._joints()
        tcp, _ = self.kin.tcp(q[:7])
        tip, _, grasp, rel, s, radial, ang = self._vs_bore()
        vel = np.zeros(3) if self.prev_tip is None else (tip - self.prev_tip) * lab_env.CONTROL_HZ
        self.prev_tip = tip.copy()
        obs = np.concatenate([
            rel * 10.0, [(s - RELEASE_TIP_S) * 10.0], [radial * 100.0], [ang / 30.0], (tcp - grasp) * 10.0, vel,
            q[7:9] * 25.0, [float(self.closed)], q[:7] - self.q_home, qd[:7] * 0.1, self.last,
        ])
        return np.clip(np.nan_to_num(obs), -10.0, 10.0).astype(np.float32)

    def _ik_action(self, a):
        q, _ = self._joints()
        _, axis, _, _, s, _, _ = self._vs_bore()
        rot = np.zeros(3)
        if self.closed and s > self.bore.opening_s + 0.002:      # aim only above the opening
            w = np.cross(axis, self.bore.axis)
            sw = np.linalg.norm(w)
            if sw > 1e-9:
                rot = np.clip(w / sw * math.atan2(sw, float(axis @ self.bore.axis)), -INSERT_ROT_STEP, INSERT_ROT_STEP)
        err = np.concatenate([a[:3] * INSERT_STEP_M, rot])
        J = self.kin.jacobian_tcp(q[:7])
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING ** 2 * np.eye(6), err)
        q_cmd = np.asarray(self.env.q_cmd[:7], float)
        lead = np.clip(q_cmd + dq - q[:7], -MAX_LEAD, MAX_LEAD)
        q_goal = np.clip(q[:7] + lead, self.q_lo, self.q_hi)
        self.closed = bool(a[3] > 0.0)
        return self.env.action(lab_env.delta_to_droid_velocity(q_goal - q_cmd), gripper=1.0 if self.closed else OPEN_CMD)


# ------------------------------------------------------------------ pipette operations (M6)
TIP_POLICY = Path(__file__).resolve().parents[1] / "rl" / "tip_policy.pt"
PIPETTING_POLICY = Path(__file__).resolve().parents[1] / "rl" / "pipetting_policy.pt"
KIN_STEP_M = 0.005                               # rl/tip_attach, rl/tip_position: 5 mm per step
KIN_ROT_STEP = 0.1
TIP_TARGET_DEPTH = 0.0065                        # tip_attach: the middle of the 4-9 mm seal window
CONTAINERS = {0: (0.058, 0.0030, 0.0008), 1: (0.0144, 0.00345, 0.0037)}   # tube, well: rim_z, rim_r, floor_z
TIP_END_R, TIP_BODY_TOP_R, TIP_COLLAR_Z = 0.00045, 0.0037, 0.052


class _KinematicFinisher(TeacherFinisher):
    """Shared by the M6 teachers: a 3-D TCP move (x KIN_STEP_M) and a hand turn (at most KIN_ROT_STEP) taking a
    pointing direction onto a goal axis, gripper closed on the pipette throughout."""

    def __init__(self, env, kin, policy_path):
        import torch
        self.torch = torch
        self.env, self.kin = env, kin
        self.policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
        self.q_home = np.asarray(lab_scene.Q_HOME[:7], float)
        lim = np.asarray(env.limits)[:7]
        self.q_lo, self.q_hi = lim[:, 0], lim[:, 1]
        self.last = np.zeros(3)
        self.closed = True
        self.prev_tip = None

    def start(self):
        self.last = np.zeros(3)

    def _pointing(self):
        """(direction to align, goal direction) -- per subclass."""
        raise NotImplementedError

    def action(self):
        torch = self.torch
        with torch.inference_mode():
            a = self.policy(torch.from_numpy(self.observation()).unsqueeze(0))[0].numpy().astype(float)
        a = np.clip(a, -1.0, 1.0)
        self.last = a.copy()
        return self._kin_action(a)

    def hold_action(self):
        return self._kin_action(np.zeros(3))

    def _kin_action(self, a):
        q, _ = self._joints()
        u, g = self._pointing()
        w = np.cross(u, g)
        sw = np.linalg.norm(w)
        rot = np.zeros(3) if sw < 1e-9 else np.clip(w / sw * math.atan2(sw, float(u @ g)), -KIN_ROT_STEP, KIN_ROT_STEP)
        err = np.concatenate([np.asarray(a) * KIN_STEP_M, rot])
        J = self.kin.jacobian_tcp(q[:7])
        dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING ** 2 * np.eye(6), err)
        q_cmd = np.asarray(self.env.q_cmd[:7], float)
        lead = np.clip(q_cmd + dq - q[:7], -MAX_LEAD, MAX_LEAD)
        q_goal = np.clip(q[:7] + lead, self.q_lo, self.q_hi)
        pip = (self.env.pipette.plunger_cmd, 0.0) if self.env.pipette else (0.0, 0.0)
        return self.env.action(lab_env.delta_to_droid_velocity(q_goal - q_cmd), gripper=1.0, pipette=pip)


class TipAttachFinisher(_KinematicFinisher):
    """The RL attach-tip teacher (rl/tip_attach): the pipette's nozzle onto the target tip's axis and 4-9 mm into
    its collar, where the pipette model (sim/pipette.py) locks the tip on. Observation (23) as its env, from the
    pipette's actual nozzle (env.pipette.nozzle())."""

    def __init__(self, env, kin, tip_path, policy_path=TIP_POLICY):
        super().__init__(env, kin, policy_path)
        Tt = lab_scene.quat_to_matrix(np.asarray(lab_scene.get_prim_world_pose(tip_path), float)[3:7])
        p = np.asarray(lab_scene.get_prim_world_pose(tip_path), float)
        self.up = Tt[:, 2]
        self.top = p[:3] + self.up * lab_scene.TIP_LENGTH

    def _pointing(self):
        _, n_dir = self.env.pipette.nozzle()
        return n_dir, -self.up

    def status(self):
        """(nozzle end to the target point m, depth into the collar m, radial m, angle deg)."""
        end, n_dir = self.env.pipette.nozzle()
        rel = self.top - end
        depth = float(rel @ self.up)
        radial = float(np.linalg.norm(rel - depth * self.up))
        ang = math.degrees(math.acos(np.clip(float(-self.up @ n_dir), -1, 1)))
        return float(np.linalg.norm(end - (self.top - TIP_TARGET_DEPTH * self.up))), depth, radial, ang

    def observation(self):
        q, qd = self._joints()
        end, n_dir = self.env.pipette.nozzle()
        target = self.top - TIP_TARGET_DEPTH * self.up
        _, depth, radial, ang = self.status()
        obs = np.concatenate([(end - target) * 10.0, [(depth - TIP_TARGET_DEPTH) * 100.0], [radial * 100.0], [ang / 30.0],
                              q[:7] - self.q_home, qd[:7] * 0.1, self.last])
        return np.clip(np.nan_to_num(obs), -10.0, 10.0).astype(np.float32)


class TipPositionFinisher(_KinematicFinisher):
    """The RL tip-positioning teacher (rl/tip_position): the attached tip's end to the aspirate / dispense point in
    a tube (kind 0) or tray well (kind 1), clear of its wall and floor. Observation (26) as its env, from the
    attached tip's actual pose (origin = its end, local +z up the tip)."""

    def __init__(self, env, kin, tip_path, origin, axis, kind, target, policy_path=PIPETTING_POLICY):
        super().__init__(env, kin, policy_path)
        self.tip_path = tip_path
        self.origin, self.axis = np.asarray(origin, float), np.asarray(axis, float)
        self.kind, self.target = int(kind), np.asarray(target, float)

    def _tip(self):
        p = np.asarray(lab_scene.get_prim_world_pose(self.tip_path), float)
        return p[:3], lab_scene.quat_to_matrix(p[3:7])[:, 2]

    def _pointing(self):
        return self._tip()[1], self.axis

    def geometry(self):
        """(target error vector, radial, height - target height, height above the rim, clearance at the rim, angle)."""
        end, up = self._tip()
        rim_z, rim_r, _ = CONTAINERS[self.kind]
        d = end - self.origin
        h = float(d @ self.axis)
        radial = float(np.linalg.norm(d - h * self.axis))
        cos = float(up @ self.axis)
        L = max((rim_z - h) / max(cos, 0.1), 0.0)
        at_rim = end + L * up - self.origin
        r_rim = float(np.linalg.norm(at_rim - (at_rim @ self.axis) * self.axis))
        tip_r = TIP_END_R + (TIP_BODY_TOP_R - TIP_END_R) * min(L, TIP_COLLAR_Z) / TIP_COLLAR_Z
        th = float((self.target - self.origin) @ self.axis)
        ang = math.degrees(math.acos(np.clip(cos, -1, 1)))
        return end - self.target, radial, h - th, h - rim_z, rim_r - (r_rim + tip_r), ang

    def status(self):
        """(tip end to the target m, height - target height m, angle deg, clearance at the rim m)."""
        rel, radial, dh, _, gap, ang = self.geometry()
        return float(np.linalg.norm(rel)), dh, ang, gap

    def at_target(self, radial_m=0.0005, height_m=0.0015, angle_deg=3.0, clearance_m=0.0001):
        """rl/tip_position's success test."""
        _, radial, dh, _, gap, ang = self.geometry()
        return radial < radial_m and abs(dh) < height_m and ang < angle_deg and gap > clearance_m

    def observation(self):
        q, qd = self._joints()
        rel, radial, dh, above, gap, ang = self.geometry()
        obs = np.concatenate([rel * 10.0, [radial * 100.0], [dh * 100.0], [above * 10.0], [np.clip(gap * 1000.0, -5, 5)],
                              [ang / 30.0], [float(self.kind)], q[:7] - self.q_home, qd[:7] * 0.1, self.last])
        return np.clip(np.nan_to_num(obs), -10.0, 10.0).astype(np.float32)


# ------------------------------------------------------------------ joint-space tube pick (no IK)
JOINT_POLICY = Path(__file__).resolve().parents[1] / "rl" / "tube_joint_policy.pt"
JOINT_STEP_RAD = 0.02                            # rl/tube_grasp_joint: joint_step_rad (must match: run 6 on)
JOINT_LIFT_GOAL = 0.06


class JointTeacherFinisher(TeacherFinisher):
    """The joint-space tube-pick teacher (rl/tube_grasp_joint): the policy's 8 outputs are changes of the 7 joint
    targets and the gripper, sent to the arm as they are -- no IK, no held orientation, no scripted lift (it lifts
    the full JOINT_LIFT_GOAL itself). Observation (39) as its env: TeacherFinisher's 29 with the 8-D last action,
    plus the hand's z and y axes."""

    def __init__(self, env, kin, tube_path, policy_path=JOINT_POLICY, pose_fn=None):
        super().__init__(env, kin, tube_path, policy_path, pose_fn)
        self.last = np.zeros(8)

    def start(self):
        self.last = np.zeros(8)
        self.closed = False
        self.prev_tip = None

    def observation(self):
        base = super().observation()                   # 25 + last action (8)
        q, _ = self._joints()
        _, R = self.kin.tcp(q[:7])
        return np.clip(np.concatenate([base, R[:, 2], R[:, 1]]), -10.0, 10.0).astype(np.float32)

    def action(self):
        torch = self.torch
        with torch.inference_mode():
            a = self.policy(torch.from_numpy(self.observation()).unsqueeze(0))[0].numpy().astype(float)
        a = np.clip(a, -1.0, 1.0)
        self.last = a.copy()
        return self._joint_action(a[:7] * JOINT_STEP_RAD, a[7] > 0.0)

    def hold_action(self):
        return self._joint_action(np.zeros(7), True)

    def lift_action(self):                           # nothing scripted: hold
        return self.hold_action()

    def _joint_action(self, dq, close):
        q, _ = self._joints()
        q_cmd = np.asarray(self.env.q_cmd[:7], float)
        lead = np.clip(q_cmd + dq - q[:7], -MAX_LEAD, MAX_LEAD)
        q_goal = np.clip(q[:7] + lead, self.q_lo, self.q_hi)
        self.closed = bool(close)
        return self.env.action(lab_env.delta_to_droid_velocity(q_goal - q_cmd), gripper=1.0 if self.closed else OPEN_CMD)

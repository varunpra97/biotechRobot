"""Tray-grasp RL task (mini milestone m2): from up to ~60 cm away, grasp the
sample tray across its width and lift it 5.5 cm level.

Action (15 Hz): TCP translation (dx, dy, dz) x 1 cm, hand yaw change x 0.1
rad, gripper (> 0 closes, else open to TRAY_OPEN per finger). The hand is
held pointing down; one damped least-squares IK step per control step,
accumulated on the previous joint command (as tube_grasp).

Observation (32): (TCP - grasp point) x10, sin/cos of 2 x (closing-axis yaw
error vs the tray's width axis), tray tilt / 30, lift x10, tray velocity,
finger positions x25, gripper command, arm joints - home, joint velocities
x0.1, last action.

Reward: the tube_grasp template (reach at 10 cm / 2 cm / 5 mm, progress,
grasp, lift x level, tilt penalty, success + rest-of-episode payout, failure)
plus yaw alignment.

Resets: the tray at a recorded pose and the arm at a recorded state (expert
tray approaches and pi0.5's failed tray trials, up to 60 cm away), or the arm
at home and the tray at its default spot with small jitter. With
cfg.neighbours (full milestone M2) the pipette, tip rack and waste bin stand
where that entry recorded them, and moving any of them is a failure.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import compute_pose_error, quat_apply, quat_from_matrix, skew_symmetric_matrix

from neighbours import DISTURBED_M as NB_DISTURBED_M, Neighbours

from .env_cfg import BENCH_TOP, Q_HOME, START_STATES, TCP_LOCAL, TRAY_GRASP_Z, TRAY_OPEN, TRAY_XY, TrayGraspEnvCfg


class TrayGraspEnv(DirectRLEnv):
    cfg: TrayGraspEnvCfg

    def __init__(self, cfg: TrayGraspEnvCfg, render_mode: str | None = None, **kwargs):
        self.nb = Neighbours("tray") if cfg.neighbours else None
        super().__init__(cfg, render_mode, **kwargs)
        dev, n = self.device, self.num_envs
        self.arm_ids = self._robot.find_joints("joint[1-7]")[0]
        self.finger_ids = self._robot.find_joints("finger_joint.*")[0]
        self.hand_idx = self._robot.find_bodies("hand")[0][0]
        self.jac_idx = self.hand_idx - 1 if self._robot.is_fixed_base else self.hand_idx
        self.jac_cols = list(self.arm_ids) if self._robot.is_fixed_base else [i + 6 for i in self.arm_ids]
        self.tcp_local = torch.tensor(TCP_LOCAL, device=dev).repeat(n, 1)
        self.q_home = torch.tensor(Q_HOME, device=dev)
        limits = self._robot.data.soft_joint_pos_limits.torch[0, self.arm_ids]
        self.q_lo, self.q_hi = limits[:, 0], limits[:, 1]

        self.q_cmd = torch.zeros(n, 7, device=dev)
        self.q_goal = torch.zeros(n, 7, device=dev)
        self.substep = 0
        self.grip_closed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.yaw_des = torch.zeros(n, device=dev)
        self.fresh = torch.full((n,), 2, dtype=torch.long, device=dev)
        self.actions = torch.zeros(n, 5, device=dev)
        self.prev_actions = torch.zeros(n, 5, device=dev)
        self.succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.failed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.d_prev = torch.full((n,), -1.0, device=dev)
        self.tray_start = torch.zeros(n, 3, device=dev)
        c = cfg
        self.max_step_reward = c.rew_reach + c.rew_near + c.rew_fine + c.rew_yaw + c.rew_grasp + c.rew_lift

        lib = np.load(START_STATES)
        self.lib_q = torch.tensor(lib["q"], device=dev)                 # (M, 9)
        t = lib["tray"]                                                 # (M, 7) xyz + wxyz
        self.lib_tray = torch.tensor(np.concatenate([t[:, :3], t[:, 4:7], t[:, 3:4]], 1), device=dev)   # xyz + xyzw
        if self.nb is not None:
            self.nb.init_buffers(n, dev, lib["neigh"])
        print(f"[tray_grasp] {len(self.lib_q)} library start states ({int(lib['n_expert'])} expert)", flush=True)

    # ------------------------------------------------------------- scene
    def _setup_scene(self):
        import isaaclab.sim as sim_utils
        from pxr import Usd, UsdPhysics
        self._robot = Articulation(self.cfg.robot)
        self._tray = RigidObject(self.cfg.tray)
        self.cfg.bench.spawn.func("/World/envs/env_0/Bench", self.cfg.bench.spawn, translation=self.cfg.bench.init_state.pos)
        stage = get_current_stage()                       # visible finger pads collide (as tube_grasp)
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot")):
            if prim.GetName() in ("left_finger", "right_finger"):
                f1 = prim.GetChild("finger_1")
                if f1.IsValid():
                    f1.SetInstanceable(False)
                    pad = f1.GetChild("finger_1")
                    if pad.IsValid() and pad.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(pad).GetCollisionEnabledAttr().Set(True)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["tray"] = self._tray
        if self.nb is not None:
            self.nb.spawn(self.scene)
        self.scene.clone_environments(copy_from_source=False)
        light = sim_utils.DomeLightCfg(intensity=self.cfg.light_intensity)
        light.func("/World/Light", light)

    # ------------------------------------------------------------- helpers
    def _tcp(self):
        hp = self._robot.data.body_pos_w.torch[:, self.hand_idx]
        hq = self._robot.data.body_quat_w.torch[:, self.hand_idx]
        return hp + quat_apply(hq, self.tcp_local), hq

    def _tray_state(self):
        pos = self._tray.data.root_pos_w.torch
        quat = self._tray.data.root_quat_w.torch                        # xyzw
        n = self.num_envs
        x_axis = quat_apply(quat, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(n, 3))
        z_axis = quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n, 3))
        tray_yaw = torch.atan2(x_axis[:, 1], x_axis[:, 0])
        tilt = torch.rad2deg(torch.acos(z_axis[:, 2].clamp(-1, 1)))
        grasp = pos + torch.tensor([0.0, 0.0, TRAY_GRASP_Z], device=self.device)
        return pos, grasp, tray_yaw, tilt

    def _closing_yaw(self, hq):
        y = quat_apply(hq, torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, 3))
        return torch.atan2(y[:, 1], y[:, 0])

    def _yaw_error(self, hq):
        """Closing axis vs the tray's width axis (tray yaw + 90 deg), mod 180 deg."""
        _, _, tray_yaw, _ = self._tray_state()
        e = self._closing_yaw(hq) - (tray_yaw + math.pi / 2)
        return torch.atan2(torch.sin(2 * e), torch.cos(2 * e)) / 2

    def _down_quat(self, yaw):
        """Hand pointing down (z = -Z), closing axis (y) at this yaw: xyzw."""
        n = yaw.shape[0]
        y = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], 1)
        z = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(n, 3)
        x = torch.cross(y, z, dim=1)
        R = torch.stack([x, y, z], dim=2)
        return quat_from_matrix(R)

    # ------------------------------------------------------------- step
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)
        tcp, hq = self._tcp()
        take = self.fresh == 1
        self.yaw_des[take] = self._closing_yaw(hq)[take]
        hold = self.fresh == 2
        self.fresh = (self.fresh - 1).clamp(min=0)
        self.yaw_des = self.yaw_des + self.actions[:, 3] * self.cfg.yaw_step_rad
        pos_des = tcp + self.actions[:, :3] * self.cfg.action_step_m
        quat_des = self._down_quat(self.yaw_des)
        J = self._robot.data.body_link_jacobian_w.torch[:, self.jac_idx, :, self.jac_cols].clone()
        r = quat_apply(hq, self.tcp_local)
        J[:, :3, :] -= torch.bmm(skew_symmetric_matrix(r), J[:, 3:, :])
        pe, re = compute_pose_error(tcp, hq, pos_des, quat_des, rot_error_type="axis_angle")
        re = re.clamp(-0.2, 0.2)                         # re-orient a tilted start gradually
        err = torch.cat([pe, re], dim=1).unsqueeze(-1)
        lam = self.cfg.ik_damping ** 2 * torch.eye(6, device=self.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2) + lam, err)).squeeze(-1)
        q_now = self._robot.data.joint_pos.torch[:, self.arm_ids]
        lead = (self.q_cmd + dq - q_now).clamp(-self.cfg.max_cmd_lead, self.cfg.max_cmd_lead)
        self.q_goal = torch.clamp(q_now + lead, self.q_lo, self.q_hi)
        self.q_goal[hold] = self.q_cmd[hold]
        self.grip_closed = (self.actions[:, 4] > 0.0) & ~hold
        self.substep = 0

    def _apply_action(self):
        self.substep += 1
        t = self.substep / self.cfg.decimation
        q = self.q_cmd + t * (self.q_goal - self.q_cmd)
        fingers = torch.where(self.grip_closed, 0.0, TRAY_OPEN).unsqueeze(-1).expand(-1, 2).contiguous()
        self._robot.set_joint_position_target_index(target=q, joint_ids=self.arm_ids)
        self._robot.set_joint_position_target_index(target=fingers, joint_ids=self.finger_ids)
        if self.substep == self.cfg.decimation:
            self.q_cmd[:] = self.q_goal

    # ------------------------------------------------------------- MDP
    def _get_observations(self) -> dict:
        tcp, hq = self._tcp()
        pos, grasp, _, tilt = self._tray_state()
        ye = self._yaw_error(hq)
        jp = self._robot.data.joint_pos.torch
        jv = self._robot.data.joint_vel.torch
        obs = torch.cat([
            (tcp - grasp) * 10.0,
            torch.sin(2 * ye).unsqueeze(-1), torch.cos(2 * ye).unsqueeze(-1),
            (tilt / 30.0).unsqueeze(-1),
            ((pos[:, 2] - BENCH_TOP - self.scene.env_origins[:, 2]) * 10.0).unsqueeze(-1),
            self._tray.data.root_lin_vel_w.torch,
            jp[:, self.finger_ids] * 25.0,
            self.grip_closed.float().unsqueeze(-1),
            jp[:, self.arm_ids] - self.q_home,
            jv[:, self.arm_ids] * 0.1,
            self.actions,
        ], dim=-1)
        return {"policy": torch.nan_to_num(obs).clamp(-10.0, 10.0)}

    def _status(self):
        tcp, hq = self._tcp()
        pos, grasp, _, tilt = self._tray_state()
        d = torch.linalg.norm(tcp - grasp, dim=-1)
        gap = self._robot.data.joint_pos.torch[:, self.finger_ids].sum(-1)
        held = (self.grip_closed & (gap > self.cfg.hold_gap_m[0]) & (gap < self.cfg.hold_gap_m[1])
                & (d < self.cfg.hold_max_dist_m))
        lift = (pos[:, 2] - self.scene.env_origins[:, 2] - BENCH_TOP).clamp(min=0.0)
        shoved = torch.linalg.norm((pos - self.tray_start)[:, :2], dim=-1) > self.cfg.pushed_m
        fail = ((tilt > self.cfg.knocked_tilt_deg) | shoved) & ~held
        if self.nb is not None:                             # the generator fails a trial if anything else moved
            fail |= self.nb.moved() > NB_DISTURBED_M
        success = held & (lift > self.cfg.lift_goal) & (tilt < self.cfg.success_tilt_deg)
        return d, held, lift, tilt, fail, success, self._yaw_error(hq)

    def _get_dones(self):
        _, _, _, _, fail, success, _ = self._status()
        self.succeeded |= success
        self.failed |= fail
        return success | fail, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        d, held, lift, tilt, fail, success, ye = self._status()
        level = (1.0 - tilt / c.level_zero_deg).clamp(0.0, 1.0)
        progress = torch.where(self.d_prev >= 0, self.d_prev - d, torch.zeros_like(d))
        self.d_prev = d.detach().clone()
        r = (c.rew_reach * (1 - torch.tanh(d / 0.10)) + c.rew_near * (1 - torch.tanh(d / 0.02))
             + c.rew_fine * (1 - torch.tanh(d / 0.005)) + c.rew_progress * progress
             + c.rew_yaw * torch.cos(2 * ye) * (1 - torch.tanh(d / 0.05))
             + c.rew_grasp * held.float() + c.rew_lift * held.float() * (lift / c.lift_goal).clamp(max=1.0) * level
             + c.rew_tilt * held.float() * (tilt - c.success_tilt_deg).clamp(min=0.0)
             + c.rew_success * success.float()
             + success.float() * self.max_step_reward * (self.max_episode_length - self.episode_length_buf).float()
             + c.rew_fail * fail.float()
             + c.rew_early_close * (self.grip_closed & (d > 0.015)).float()
             + c.rew_action_rate * ((self.actions - self.prev_actions) ** 2).sum(-1))
        log = self.extras.setdefault("log", {})
        log["Metrics/dist_cm"] = (d * 100).mean().item()
        log["Metrics/grasped"] = held.float().mean().item()
        log["Metrics/lift_cm"] = (lift * 100).mean().item()
        log["Metrics/held_tilt_deg"] = tilt[held].mean().item() if held.any() else 0.0
        log["Metrics/yaw_err_deg"] = torch.rad2deg(ye.abs()).mean().item()
        if self.nb is not None:
            log["Metrics/neighbour_moved_mm"] = (self.nb.moved() * 1000).mean().item()
        return r

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        log = self.extras.setdefault("log", {})
        log["Metrics/success_rate"] = self.succeeded[env_ids].float().mean().item()
        log["Metrics/fail_rate"] = self.failed[env_ids].float().mean().item()
        self.succeeded[env_ids] = False
        self.failed[env_ids] = False
        super()._reset_idx(env_ids)
        k, dev = len(env_ids), self.device

        from_lib = torch.rand(k, device=dev) < self.cfg.library_p
        pick = torch.randint(0, len(self.lib_q), (k,), device=dev)
        q = torch.zeros(k, 9, device=dev)
        q[:, :7] = self.q_home
        q[:, 7:] = TRAY_OPEN
        q[from_lib] = self.lib_q[pick[from_lib]]
        q[:, :7] += self.cfg.joint_noise * (2 * torch.rand(k, 7, device=dev) - 1)
        q[:, :7] = torch.clamp(q[:, :7], self.q_lo, self.q_hi)
        ids = self.arm_ids + self.finger_ids
        self._robot.write_joint_position_to_sim_index(position=q, joint_ids=ids, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids, env_ids=env_ids)
        self._robot.set_joint_position_target_index(target=q, joint_ids=ids, env_ids=env_ids)

        # tray: the library entry's recorded pose, else the default spot with jitter
        jit = self.cfg.home_tray_jitter_m * (2 * torch.rand(k, 2, device=dev) - 1)
        yaw = torch.deg2rad(self.cfg.home_tray_jitter_deg * (2 * torch.rand(k, device=dev) - 1))
        pose = torch.zeros(k, 7, device=dev)
        pose[:, 0] = TRAY_XY[0] + jit[:, 0]
        pose[:, 1] = TRAY_XY[1] + jit[:, 1]
        pose[:, 2] = BENCH_TOP
        pose[:, 5] = torch.sin(yaw / 2)
        pose[:, 6] = torch.cos(yaw / 2)
        pose[from_lib] = self.lib_tray[pick[from_lib]]
        pose[:, :3] += self.scene.env_origins[env_ids]
        self._tray.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self._tray.write_root_velocity_to_sim_index(root_velocity=torch.zeros(k, 6, device=dev), env_ids=env_ids)
        self.tray_start[env_ids] = pose[:, :3]
        if self.nb is not None:
            self.nb.reset(env_ids, pick, self.scene.env_origins[env_ids])

        self.q_cmd[env_ids] = q[:, :7]
        self.q_goal[env_ids] = q[:, :7]
        self.grip_closed[env_ids] = False
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.fresh[env_ids] = 2
        self.d_prev[env_ids] = -1.0

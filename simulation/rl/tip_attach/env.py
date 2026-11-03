"""Attach-tip RL task (mini milestone m6): from a recorded approach, bring
the held pipette's nozzle onto the target tip's axis and press it 4-9 mm
into the collar (the generator's attach rule: within 0.6 mm and 3 deg).

Kinematic (see env_cfg): the nozzle end and direction follow the hand
through the entry's recorded in-hand transform (the pipette's origin is its
nozzle end, nozzle pointing down its local -y).

Action (15 Hz): TCP translation (dx, dy, dz) x action_step_m. The hand turns,
at most rot_step_rad per step, so the nozzle points down the tip's axis.
One damped least-squares IK step per control step, accumulated on the
previous joint command (as tube_grasp).

Observation (23): nozzle end minus its target point (on the tip's axis,
target_depth_m into the collar) x10, the depth into the collar minus the
target x100, the distance from the tip's axis x100, the angle between
nozzle and tip axis / 30 deg, arm joints - home, joint velocities x0.1,
last action.

Reward: on-axis alignment (5 mm and 0.5 mm scales), depth only while on the
axis, progress, alignment, success (attached) + rest-of-episode payout;
failure (crushed the tip, or pressed on its rim) and jerky actions.
"""
from __future__ import annotations

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import compute_pose_error, quat_apply, quat_mul, skew_symmetric_matrix

from tube_grasp.env_cfg import Q_HOME, TCP_LOCAL

from .env_cfg import START_STATES, TipAttachEnvCfg


class TipAttachEnv(DirectRLEnv):
    cfg: TipAttachEnvCfg

    def __init__(self, cfg: TipAttachEnvCfg, render_mode: str | None = None, **kwargs):
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
        self.fresh = torch.full((n,), 2, dtype=torch.long, device=dev)
        self.actions = torch.zeros(n, 3, device=dev)
        self.prev_actions = torch.zeros(n, 3, device=dev)
        self.succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.failed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.e_prev = torch.full((n,), -1.0, device=dev)
        self.ih_pos = torch.zeros(n, 3, device=dev)                  # pipette (nozzle end) in the hand frame
        self.ih_quat = torch.zeros(n, 4, device=dev)
        self.ih_quat[:, 3] = 1.0
        self.top = torch.zeros(n, 3, device=dev)                     # target tip's collar top (world)
        self.up = torch.zeros(n, 3, device=dev)
        self.up[:, 2] = 1.0
        c = cfg
        self.max_step_reward = c.rew_radial + c.rew_radial_fine + c.rew_depth + c.rew_align

        lib = np.load(START_STATES)
        self.lib_q = torch.tensor(lib["q"], device=dev)
        ih = lib["inhand"]                                                # xyz + wxyz
        self.lib_ih = torch.tensor(np.concatenate([ih[:, :3], ih[:, 4:7], ih[:, 3:4]], 1), device=dev)   # xyz + xyzw
        self.lib_top = torch.tensor(lib["top"], device=dev)
        self.lib_up = torch.tensor(lib["up"], device=dev)
        self.lib_rack = torch.tensor(lib["rack"], device=dev)         # (M, 3) x, y, yaw deg (M6: any tip in the rack)
        print(f"[tip_attach] {len(self.lib_q)} recorded approaches", flush=True)

    def _setup_scene(self):
        import isaaclab.sim as sim_utils
        self._robot = Articulation(self.cfg.robot)
        self.cfg.bench.spawn.func("/World/envs/env_0/Bench", self.cfg.bench.spawn, translation=self.cfg.bench.init_state.pos)
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)
        light = sim_utils.DomeLightCfg(intensity=self.cfg.light_intensity)
        light.func("/World/Light", light)

    # ------------------------------------------------------------- helpers
    def _hand(self):
        return self._robot.data.body_pos_w.torch[:, self.hand_idx], self._robot.data.body_quat_w.torch[:, self.hand_idx]

    def _nozzle(self):
        """Nozzle end and pointing direction (world), and vs the tip: end - target, depth into the collar,
        radial offset, angle (deg)."""
        hp, hq = self._hand()
        end = hp + quat_apply(hq, self.ih_pos)
        pq = quat_mul(hq, self.ih_quat)
        ndir = -quat_apply(pq, torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.num_envs, 3))
        top = self.top + self.scene.env_origins
        rel = top - end
        depth = (rel * self.up).sum(-1)
        radial = (rel - depth.unsqueeze(-1) * self.up).norm(dim=-1)
        ang = torch.rad2deg(torch.acos((-self.up * ndir).sum(-1).clamp(-1, 1)))
        target = top - self.cfg.target_depth_m * self.up
        return end, ndir, end - target, depth, radial, ang

    # ------------------------------------------------------------- step
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)
        hp, hq = self._hand()
        tcp = hp + quat_apply(hq, self.tcp_local)
        _, ndir, *_ = self._nozzle()
        hold = self.fresh == 2
        self.fresh = (self.fresh - 1).clamp(min=0)
        pos_des = tcp + self.actions * self.cfg.action_step_m
        J = self._robot.data.body_link_jacobian_w.torch[:, self.jac_idx, :, self.jac_cols].clone()
        r = quat_apply(hq, self.tcp_local)
        J[:, :3, :] -= torch.bmm(skew_symmetric_matrix(r), J[:, 3:, :])
        pe, _ = compute_pose_error(tcp, hq, pos_des, hq, rot_error_type="axis_angle")
        # the rotation taking the nozzle direction onto the tip's axis (pointing down it)
        goal = -self.up
        w = torch.cross(ndir, goal, dim=-1)
        sw = w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        re = (w / sw * torch.atan2(sw, (ndir * goal).sum(-1, keepdim=True))).clamp(-self.cfg.rot_step_rad,
                                                                                   self.cfg.rot_step_rad)
        err = torch.cat([pe, re], dim=1).unsqueeze(-1)
        lam = self.cfg.ik_damping ** 2 * torch.eye(6, device=self.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2) + lam, err)).squeeze(-1)
        q_now = self._robot.data.joint_pos.torch[:, self.arm_ids]
        lead = (self.q_cmd + dq - q_now).clamp(-self.cfg.max_cmd_lead, self.cfg.max_cmd_lead)
        self.q_goal = torch.clamp(q_now + lead, self.q_lo, self.q_hi)
        self.q_goal[hold] = self.q_cmd[hold]               # first step after a reset: poses still stale
        self.substep = 0

    def _apply_action(self):
        self.substep += 1
        t = self.substep / self.cfg.decimation
        self._robot.set_joint_position_target_index(target=self.q_cmd + t * (self.q_goal - self.q_cmd),
                                                    joint_ids=self.arm_ids)
        if self.substep == self.cfg.decimation:
            self.q_cmd[:] = self.q_goal

    # ------------------------------------------------------------- MDP
    def _get_observations(self) -> dict:
        _, _, rel, depth, radial, ang = self._nozzle()
        jp = self._robot.data.joint_pos.torch
        jv = self._robot.data.joint_vel.torch
        obs = torch.cat([
            rel * 10.0,
            ((depth - self.cfg.target_depth_m) * 100.0).unsqueeze(-1),
            (radial * 100.0).unsqueeze(-1),
            (ang / 30.0).unsqueeze(-1),
            jp[:, self.arm_ids] - self.q_home,
            jv[:, self.arm_ids] * 0.1,
            self.actions,
        ], dim=-1)
        return {"policy": torch.nan_to_num(obs).clamp(-10.0, 10.0)}

    def _status(self):
        c = self.cfg
        _, _, rel, depth, radial, ang = self._nozzle()
        success = ((depth >= c.seal_depth_m[0]) & (depth <= c.seal_depth_m[1]) & (radial <= c.attach_radial_m)
                   & (ang <= c.attach_angle_deg))
        crushed = (depth > c.seal_depth_m[1]) & (radial < 0.003)
        rim = (depth > c.rim_depth_m) & (radial > c.rim_radial_m) & (radial < 0.006)
        return rel, depth, radial, ang, success & ~self._fresh_mask(), (crushed | rim) & ~success

    def _fresh_mask(self):
        return self.fresh > 0                              # poses not yet refreshed after a reset

    def _get_dones(self):
        *_, success, fail = self._status()
        self.succeeded |= success
        self.failed |= fail
        return success | fail, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        rel, depth, radial, ang, success, fail = self._status()
        e = rel.norm(dim=-1)
        progress = torch.where(self.e_prev >= 0, self.e_prev - e, torch.zeros_like(e))
        self.e_prev = e.detach().clone()
        on_axis = (radial <= c.attach_radial_m).float()
        r = (c.rew_radial * (1 - torch.tanh(radial / 0.005)) + c.rew_radial_fine * (1 - torch.tanh(radial / 0.0005))
             + c.rew_depth * on_axis * (1 - torch.tanh((depth - c.target_depth_m).abs() / 0.01))
             + c.rew_progress * progress
             + c.rew_align * torch.cos(torch.deg2rad(ang))
             + c.rew_success * success.float()
             + success.float() * self.max_step_reward * (self.max_episode_length - self.episode_length_buf).float()
             + c.rew_fail * fail.float()
             + c.rew_action_rate * ((self.actions - self.prev_actions) ** 2).sum(-1))
        log = self.extras.setdefault("log", {})
        log["Metrics/radial_mm"] = (radial * 1000).mean().item()
        log["Metrics/depth_mm"] = (depth * 1000).mean().item()
        log["Metrics/angle_deg"] = ang.mean().item()
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
        pick = torch.randint(0, len(self.lib_q), (len(env_ids),), device=self.device)
        q = self.lib_q[pick].clone()
        ids = self.arm_ids + self.finger_ids
        self._robot.write_joint_position_to_sim_index(position=q, joint_ids=ids, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids, env_ids=env_ids)
        self._robot.set_joint_position_target_index(target=q, joint_ids=ids, env_ids=env_ids)
        self.ih_pos[env_ids] = self.lib_ih[pick, :3]
        self.ih_quat[env_ids] = self.lib_ih[pick, 3:]
        self.top[env_ids] = self.lib_top[pick]
        self.up[env_ids] = self.lib_up[pick]
        k = len(env_ids)
        re = torch.nonzero(torch.rand(k, device=self.device) < self.cfg.retarget_p).flatten()
        if len(re):                                    # another tip of the same rack (lab_scene.tip_xy), same height
            r = torch.randint(0, 10, (len(re),), device=self.device).float()
            c = torch.randint(0, 10, (len(re),), device=self.device).float()
            rack = self.lib_rack[pick[re]]
            a = torch.deg2rad(rack[:, 2])
            px, py = (c - 4.5) * 0.009, (r - 4.5) * 0.009
            top = self.top[env_ids[re]].clone()
            top[:, 0] = rack[:, 0] + torch.cos(a) * px - torch.sin(a) * py
            top[:, 1] = rack[:, 1] + torch.sin(a) * px + torch.cos(a) * py
            self.top[env_ids[re]] = top
        self.q_cmd[env_ids] = q[:, :7]
        self.q_goal[env_ids] = q[:, :7]
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.fresh[env_ids] = 2
        self.e_prev[env_ids] = -1.0

"""Tip-positioning RL task (full milestone M6): from a recorded approach,
put the attached tip's end at the aspirate / dispense point inside a tube or
a tray well without touching its wall or floor (the generator's rules,
sim/pipette.py _check_clearance).

Kinematic (see env_cfg): the tip end and its axis follow the hand through
the entry's recorded in-hand transform; the container is a frame (origin,
axis) with a profile (rim height and inner radius, floor height).

Action (15 Hz): TCP translation (dx, dy, dz) x action_step_m. The hand turns,
at most rot_step_rad per step, to hold the tip along the container's axis.
One damped least-squares IK step per control step, accumulated on the
previous joint command (as tube_grasp).

Observation (26): tip end minus the target x10, its distance from the
container's axis x100, its height minus the target height x100, its height
above the rim x10, the wall clearance at the rim x1000 (clipped), the tip's
angle to the container axis / 30 deg, the container kind (0 tube, 1 well),
arm joints - home, joint velocities x0.1, last action.

Reward: reach, on-axis alignment (3 mm and 0.3 mm), height only while near
the axis, progress, alignment, success + rest-of-episode payout; failure
(the tip touched the wall, entered through it, or hit the floor), jerky
actions.
"""
from __future__ import annotations

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import compute_pose_error, quat_apply, quat_mul, skew_symmetric_matrix

from tube_grasp.env_cfg import Q_HOME, TCP_LOCAL

from .env_cfg import START_STATES, TIP_BODY_TOP_R, TIP_COLLAR_Z, TIP_END_R, TUBE, WELL, TipPositionEnvCfg


class TipPositionEnv(DirectRLEnv):
    cfg: TipPositionEnvCfg

    def __init__(self, cfg: TipPositionEnvCfg, render_mode: str | None = None, **kwargs):
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
        self.ih_pos = torch.zeros(n, 3, device=dev)                   # tip end in the hand frame
        self.ih_quat = torch.zeros(n, 4, device=dev)
        self.ih_quat[:, 3] = 1.0
        self.origin = torch.zeros(n, 3, device=dev)                   # container frame (env-local)
        self.axis = torch.zeros(n, 3, device=dev)
        self.axis[:, 2] = 1.0
        self.target = torch.zeros(n, 3, device=dev)
        self.kind = torch.zeros(n, dtype=torch.long, device=dev)
        prof = torch.tensor([[TUBE["rim_z"], TUBE["rim_r"], TUBE["floor_z"]], [WELL["rim_z"], WELL["rim_r"], WELL["floor_z"]]],
                            device=dev)
        self.prof = prof                                              # (2, 3) rim_z, rim_r, floor_z by kind
        c = cfg
        self.max_step_reward = c.rew_reach + c.rew_radial + c.rew_radial_fine + c.rew_depth + c.rew_align

        lib = np.load(START_STATES)
        self.lib_q = torch.tensor(lib["q"], device=dev)
        ih = lib["inhand"]
        self.lib_ih = torch.tensor(np.concatenate([ih[:, :3], ih[:, 4:7], ih[:, 3:4]], 1), device=dev)   # xyz + xyzw
        self.lib_origin = torch.tensor(lib["origin"], device=dev)
        self.lib_axis = torch.tensor(lib["axis"], device=dev)
        self.lib_target = torch.tensor(lib["target"], device=dev)
        self.lib_kind = torch.tensor(lib["kind"].astype(np.int64), device=dev)
        print(f"[tip_position] {len(self.lib_q)} recorded approaches ({int((lib['kind'] == 1).sum())} to wells)", flush=True)

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

    def _tip(self):
        """Tip end and its up axis (world), and vs the container: target error, radial, height, clearance at the
        rim, angle (deg), inside, failure."""
        hp, hq = self._hand()
        end = hp + quat_apply(hq, self.ih_pos)
        up = quat_apply(quat_mul(hq, self.ih_quat), torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3))
        o = self.origin + self.scene.env_origins
        ax = self.axis
        rim_z, rim_r, floor_z = self.prof[self.kind, 0], self.prof[self.kind, 1], self.prof[self.kind, 2]
        d = end - o
        h = (d * ax).sum(-1)
        radial = (d - h.unsqueeze(-1) * ax).norm(dim=-1)
        cos = (up * ax).sum(-1)
        ang = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
        L = ((rim_z - h) / cos.clamp(min=0.1)).clamp(min=0.0)               # along the tip from its end to the rim plane
        at_rim = end + L.unsqueeze(-1) * up - o
        r_rim = (at_rim - (at_rim * ax).sum(-1, keepdim=True) * ax).norm(dim=-1)
        tip_r = TIP_END_R + (TIP_BODY_TOP_R - TIP_END_R) * L.clamp(max=TIP_COLLAR_Z) / TIP_COLLAR_Z
        gap = rim_r - (r_rim + tip_r)
        inside = (h < rim_z) & (radial < rim_r)
        through_wall = (h < rim_z) & (radial >= rim_r - TIP_END_R) & (radial < rim_r + 0.006)
        fail = (inside & ((gap < 0.0) | (h < floor_z))) | through_wall
        target_h = ((self.target + self.scene.env_origins - o) * ax).sum(-1)
        return end, up, end - (self.target + self.scene.env_origins), radial, h - target_h, h - rim_z, gap, ang, inside, fail

    # ------------------------------------------------------------- step
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)
        hp, hq = self._hand()
        tcp = hp + quat_apply(hq, self.tcp_local)
        _, up, *_ = self._tip()
        hold = self.fresh == 2
        self.fresh = (self.fresh - 1).clamp(min=0)
        pos_des = tcp + self.actions * self.cfg.action_step_m
        J = self._robot.data.body_link_jacobian_w.torch[:, self.jac_idx, :, self.jac_cols].clone()
        r = quat_apply(hq, self.tcp_local)
        J[:, :3, :] -= torch.bmm(skew_symmetric_matrix(r), J[:, 3:, :])
        pe, _ = compute_pose_error(tcp, hq, pos_des, hq, rot_error_type="axis_angle")
        w = torch.cross(up, self.axis, dim=-1)                              # tip axis onto the container's
        sw = w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        re = (w / sw * torch.atan2(sw, (up * self.axis).sum(-1, keepdim=True))).clamp(-self.cfg.rot_step_rad,
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
        _, _, rel, radial, dh, above_rim, gap, ang, _, _ = self._tip()
        jp = self._robot.data.joint_pos.torch
        jv = self._robot.data.joint_vel.torch
        obs = torch.cat([
            rel * 10.0,
            (radial * 100.0).unsqueeze(-1),
            (dh * 100.0).unsqueeze(-1),
            (above_rim * 10.0).unsqueeze(-1),
            (gap * 1000.0).clamp(-5.0, 5.0).unsqueeze(-1),
            (ang / 30.0).unsqueeze(-1),
            self.kind.float().unsqueeze(-1),
            jp[:, self.arm_ids] - self.q_home,
            jv[:, self.arm_ids] * 0.1,
            self.actions,
        ], dim=-1)
        return {"policy": torch.nan_to_num(obs).clamp(-10.0, 10.0)}

    def _status(self):
        c = self.cfg
        _, _, rel, radial, dh, _, gap, ang, _, fail = self._tip()
        fresh = self.fresh > 0
        success = ((radial < c.target_radial_m) & (dh.abs() < c.target_height_m) & (ang < c.target_angle_deg)
                   & (gap > c.clearance_m) & ~fresh)
        return rel, radial, dh, gap, ang, success, fail & ~fresh

    def _get_dones(self):
        *_, success, fail = self._status()
        self.succeeded |= success
        self.failed |= fail & ~success
        return success | fail, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        rel, radial, dh, gap, ang, success, fail = self._status()
        e = rel.norm(dim=-1)
        progress = torch.where(self.e_prev >= 0, self.e_prev - e, torch.zeros_like(e))
        self.e_prev = e.detach().clone()
        near_axis = (radial < 0.001).float()
        r = (c.rew_reach * (1 - torch.tanh(e / 0.05))
             + c.rew_radial * (1 - torch.tanh(radial / 0.003)) + c.rew_radial_fine * (1 - torch.tanh(radial / 0.0003))
             + c.rew_depth * near_axis * (1 - torch.tanh(dh.abs() / 0.005))
             + c.rew_progress * progress
             + c.rew_align * torch.cos(torch.deg2rad(ang))
             + c.rew_success * success.float()
             + success.float() * self.max_step_reward * (self.max_episode_length - self.episode_length_buf).float()
             + c.rew_fail * (fail & ~success).float()
             + c.rew_action_rate * ((self.actions - self.prev_actions) ** 2).sum(-1))
        log = self.extras.setdefault("log", {})
        log["Metrics/dist_mm"] = (e * 1000).mean().item()
        log["Metrics/radial_mm"] = (radial * 1000).mean().item()
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
        self.origin[env_ids] = self.lib_origin[pick]
        self.axis[env_ids] = self.lib_axis[pick]
        self.target[env_ids] = self.lib_target[pick]
        self.kind[env_ids] = self.lib_kind[pick]
        self.q_cmd[env_ids] = q[:, :7]
        self.q_goal[env_ids] = q[:, :7]
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.fresh[env_ids] = 2
        self.e_prev[env_ids] = -1.0

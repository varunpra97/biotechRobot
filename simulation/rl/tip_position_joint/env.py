"""Joint-space tip-positioning RL task (no IK): tip_position's kinematic pipette, containers, resets and rules with
rl/joint_space.py's action (7 joints, no gripper), potential-based shaping and a success that has to hold for
success_hold_steps (the plunger stroke happens there).

Observation (42): tip_position's 26 (with the 7-D last action), the tip's and the container's axes (world), the
hand's z and y axes.
"""
from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply

from joint_space import JointSpaceMixin
from tip_position.env import TipPositionEnv

from .env_cfg import TipPositionJointEnvCfg


class TipPositionJointEnv(JointSpaceMixin, TipPositionEnv):
    cfg: TipPositionJointEnvCfg

    def __init__(self, cfg: TipPositionJointEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_setup()
        self.hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _tcp(self):
        hp, hq = self._hand()
        return hp + quat_apply(hq, self.tcp_local), hq

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if hasattr(self, "phi_prev"):
            self._reset_joint(env_ids)
            self.hold_count[env_ids] = 0

    def _get_observations(self) -> dict:
        obs = super()._get_observations()["policy"]
        _, up, *_ = self._tip()
        return {"policy": self._joint_obs(torch.cat([obs, up, self.axis], dim=-1))}

    def _status(self):
        rel, radial, dh, gap, ang, success, fail = super()._status()
        if getattr(self, "hold_count", None) is not None and self.cfg.success_hold_steps > 1:
            success = success & (self.hold_count >= self.cfg.success_hold_steps)
        return rel, radial, dh, gap, ang, success, fail

    def _get_dones(self):
        raw = TipPositionEnv._status(self)[5]
        self.hold_count = torch.where(raw, self.hold_count + 1, torch.zeros_like(self.hold_count))
        return super()._get_dones()

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        rel, radial, dh, gap, ang, success, fail = self._status()
        e = rel.norm(dim=-1)
        near_axis = (radial < 0.001).float()
        phi = (c.rew_reach * (1 - torch.tanh(e / 0.05))
               + c.rew_radial * (1 - torch.tanh(radial / 0.003)) + c.rew_radial_fine * (1 - torch.tanh(radial / 0.0003))
               + c.rew_depth * near_axis * (1 - torch.tanh(dh.abs() / 0.005))
               + c.rew_align * (1 - torch.tanh(ang / 5.0)))
        r = (self.potential(phi) + c.rew_success * success.float() + c.rew_fail * (fail & ~success).float()
             + c.rew_action_rate * self.action_rate())
        log = self.extras.setdefault("log", {})
        log["Metrics/dist_mm"] = (e * 1000).mean().item()
        log["Metrics/radial_mm"] = (radial * 1000).mean().item()
        log["Metrics/angle_deg"] = ang.mean().item()
        log["Metrics/hold_steps"] = self.hold_count.float().mean().item()
        return r

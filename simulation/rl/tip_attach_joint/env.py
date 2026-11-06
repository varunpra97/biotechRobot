"""Joint-space tip-attach RL task (no IK): tip_attach's kinematic pipette, resets and attach rule with
rl/joint_space.py's action (7 joints, no gripper) and potential-based shaping.

Observation (39): tip_attach's 23 (with the 7-D last action), the nozzle's direction and the tip's axis (world), the
hand's z and y axes.
"""
from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply

from joint_space import JointSpaceMixin
from tip_attach.env import TipAttachEnv

from .env_cfg import TipAttachJointEnvCfg


class TipAttachJointEnv(JointSpaceMixin, TipAttachEnv):
    cfg: TipAttachJointEnvCfg

    def __init__(self, cfg: TipAttachJointEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_setup()

    def _tcp(self):
        hp, hq = self._hand()
        return hp + quat_apply(hq, self.tcp_local), hq

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if hasattr(self, "phi_prev"):
            self._reset_joint(env_ids)

    def _get_observations(self) -> dict:
        obs = super()._get_observations()["policy"]
        _, ndir, *_ = self._nozzle()
        return {"policy": self._joint_obs(torch.cat([obs, ndir, self.up], dim=-1))}

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        rel, depth, radial, ang, success, fail = self._status()
        e = rel.norm(dim=-1)
        on_axis = (radial < c.attach_radial_m).float()
        phi = (c.rew_reach * (1 - torch.tanh(e / 0.02))
               + c.rew_radial * (1 - torch.tanh(radial / 0.005)) + c.rew_radial_fine * (1 - torch.tanh(radial / 0.0005))
               + c.rew_depth * on_axis * (1 - torch.tanh((depth - c.target_depth_m).abs() / 0.01))
               + c.rew_align * (1 - torch.tanh(ang / 5.0)))
        r = (self.potential(phi) + c.rew_success * success.float() + c.rew_fail * fail.float()
             + c.rew_action_rate * self.action_rate())
        log = self.extras.setdefault("log", {})
        log["Metrics/radial_mm"] = (radial * 1000).mean().item()
        log["Metrics/depth_mm"] = (depth * 1000).mean().item()
        log["Metrics/axis_err_deg"] = ang.mean().item()
        log["Metrics/end_to_target_mm"] = (e * 1000).mean().item()
        return r

"""Joint-space centrifuge-insert RL task (no IK): tube_insert's scene, resets and success test with the joint action
of rl/joint_space.py, the tube and bore axes in the observation (the alignment is the policy's now), potential-based
shaping and a success that has to hold for success_hold_steps.

Observation (49): tube_insert's 33 (with the 8-D last action), the held tube's axis and the bore's (world), the
hand's z and y axes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from joint_space import JointSpaceMixin
from tube_insert.env import TubeInsertEnv

from .env_cfg import TubeInsertJointEnvCfg


class TubeInsertJointEnv(JointSpaceMixin, TubeInsertEnv):
    cfg: TubeInsertJointEnvCfg

    def __init__(self, cfg: TubeInsertJointEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_setup()
        self.hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        deep = np.load(Path(__file__).resolve().parents[1] / "start_states_insert_deep.npz")
        t = deep["tube"]                                                    # xyz + wxyz
        self.deep_q = torch.tensor(deep["q"], device=self.device)
        self.deep_tube = torch.tensor(np.concatenate([t[:, :3], t[:, 4:7], t[:, 3:4]], 1), device=self.device)
        self.deep_bore = torch.tensor(deep["bore"], dtype=torch.long, device=self.device)
        # entry starts (run 3): the carry library's holds lined up just above the bore -- within 1.5 mm of its axis,
        # 4 deg of its direction, the tip from entry_s_m below the opening to entry_above_m above it
        b = self.lib_bore
        d = self.lib_tube[:, :3] - self.bore_point[b]
        s_lib = (d * self.bore_axis[b]).sum(-1)
        radial = (d - s_lib.unsqueeze(-1) * self.bore_axis[b]).norm(dim=-1)
        from isaaclab.utils.math import quat_apply
        tax = quat_apply(self.lib_tube[:, 3:7], torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(len(b), 3))
        ang = torch.rad2deg(torch.acos((tax * self.bore_axis[b]).sum(-1).clamp(-1, 1)))
        op = self.bore_opening[b]
        self.entry_ids = torch.nonzero((radial < 0.0015) & (ang < 4.0) & (s_lib > op - cfg.entry_s_m)
                                       & (s_lib < op + cfg.entry_above_m)).flatten()
        print(f"[tube_insert_joint] {len(self.deep_q)} in-bore start states (share {cfg.deep_p}), "
              f"{len(self.entry_ids)} entry start states (share {cfg.entry_p})", flush=True)

    def _write_hold(self, sel, q, pose, bore):
        """Reset these envs to a hold: joints (pads a little open, closing), the tube's pose, its bore."""
        q = q.clone()
        ids = self.arm_ids + self.finger_ids
        self._robot.write_joint_position_to_sim_index(position=q, joint_ids=ids, env_ids=sel)
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids, env_ids=sel)
        target = q.clone()
        target[:, 7:] = 0.0
        self._robot.set_joint_position_target_index(target=target, joint_ids=ids, env_ids=sel)
        pose = pose.clone()
        pose[:, :3] += self.scene.env_origins[sel]
        self._tube.write_root_pose_to_sim_index(root_pose=pose, env_ids=sel)
        self._tube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(sel), 6, device=self.device),
                                                    env_ids=sel)
        self.bore[sel] = bore
        self.q_cmd[sel] = q[:, :7]
        self.q_goal[sel] = q[:, :7]

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_joint(env_ids)
        self.hold_count[env_ids] = 0
        if not hasattr(self, "entry_ids"):
            return
        u = torch.rand(len(env_ids), device=self.device)
        sel = env_ids[u < self.cfg.deep_p]                 # the reverse curriculum: held inside the bore
        if len(sel):
            pick = torch.randint(0, len(self.deep_q), (len(sel),), device=self.device)
            self._write_hold(sel, self.deep_q[pick], self.deep_tube[pick], self.deep_bore[pick])
        sel = env_ids[(u >= self.cfg.deep_p) & (u < self.cfg.deep_p + self.cfg.entry_p)]   # ... lined up at the entry
        if len(sel) and len(self.entry_ids):
            pick = self.entry_ids[torch.randint(0, len(self.entry_ids), (len(sel),), device=self.device)]
            self._write_hold(sel, self.lib_q[pick], self.lib_tube[pick], self.lib_bore[pick])

    def _release_allowed(self) -> torch.Tensor:
        """The tube's tip inside the bore (below its opening): elsewhere letting go is a drop."""
        s = self._tube_state()[5]
        return s < self.bore_opening[self.bore]

    def _get_observations(self) -> dict:
        obs = super()._get_observations()["policy"]
        _, _, axis, baxis, *_ = self._tube_state()
        return {"policy": self._joint_obs(torch.cat([obs, axis, baxis], dim=-1))}

    def _raw_status(self):
        """tube_insert's status with this cfg's hold test (hold_gap_m)."""
        c = self.cfg
        tcp, _ = self._tcp()
        tp, grasp, axis, baxis, rel, s, radial, ang = self._tube_state()
        gap = self._robot.data.joint_pos.torch[:, self.finger_ids].sum(-1)
        held = self.grip_closed & (gap > c.hold_gap_m[0]) & (gap < c.hold_gap_m[1])
        on_axis = radial < c.radial_m
        in_bore = on_axis & (s < self.bore_opening[self.bore] - 0.005) & (ang < c.coaxial_deg)
        seated = on_axis & (s < c.seated_s) & (ang < c.coaxial_deg)
        released = ~self.grip_closed & (gap > c.open_gap_m)
        backed_off = (tcp - grasp).norm(dim=-1) > c.retreat_m
        success = seated & released & backed_off
        fail = ((ang > c.fallen_deg) & ~held) | (rel.norm(dim=-1) > c.lost_m)
        return rel, s, radial, ang, on_axis, in_bore, seated, released, success, fail

    def _status(self):
        rel, s, radial, ang, on_axis, in_bore, seated, released, success, fail = self._raw_status()
        if getattr(self, "hold_count", None) is not None and self.cfg.success_hold_steps > 1:
            success = success & (self.hold_count >= self.cfg.success_hold_steps)
        return rel, s, radial, ang, on_axis, in_bore, seated, released, success, fail

    def _get_dones(self):
        raw = self._raw_status()[8]                       # count before the (held) success test reads it
        self.hold_count = torch.where(raw, self.hold_count + 1, torch.zeros_like(self.hold_count))
        return super()._get_dones()

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        rel, s, radial, ang, on_axis, in_bore, seated, released, success, fail = self._status()
        from tube_insert.env_cfg import RELEASE_TIP_S
        tcp, _ = self._tcp()
        grasp = self._tube_state()[1]
        e = rel.norm(dim=-1)
        done_seated = (seated & released).float()
        backed = ((tcp - grasp).norm(dim=-1) / c.retreat_m).clamp(max=1.0)
        phi = (c.rew_far * (1 - torch.tanh(e / 0.40)) + c.rew_carry * (1 - torch.tanh(e / 0.10)) + c.rew_carry_near * (1 - torch.tanh(e / 0.02))
               + c.rew_radial * (1 - torch.tanh(radial / 0.005)) + c.rew_radial_fine * (1 - torch.tanh(radial / 0.001))
               + c.rew_coaxial * (1 - torch.tanh(ang / 10.0)) + c.rew_coaxial_wide * (1 + torch.cos(torch.deg2rad(ang))) / 2
               + c.rew_deep * on_axis.float() * (1 - torch.tanh((s - RELEASE_TIP_S).clamp(min=0.0) / 0.03))
               + c.rew_in_bore * in_bore.float() + c.rew_released * done_seated + c.rew_backoff * done_seated * backed)
        near = (s - self.bore_opening[self.bore]).abs() < c.near_m
        tube_v = self._tube.data.root_lin_vel_w.torch.norm(dim=-1)
        fast = near.float() * (tube_v - c.fast_mps).clamp(min=0.0)
        # run 3: while the held tube is lined up in the entry zone (on the axis, coaxial, between seat and
        # zone_above_m above the opening), pay its speed down the bore axis and charge the hand's speed across it --
        # tube_grasp_joint's extraction terms, reversed (run 2: in-bore starts were finished by just letting go,
        # so a push along the axis was never learned and lined-up entry starts all failed)
        tube_vel = self._tube.data.root_lin_vel_w.torch
        baxis = self.bore_axis[self.bore]
        zone = (self.grip_closed & (radial < c.zone_radial_m) & (ang < c.zone_deg)
                & (s < self.bore_opening[self.bore] + c.zone_above_m) & (s > RELEASE_TIP_S)).float()
        v_down = -(tube_vel * baxis).sum(-1)
        hand_v = self._robot.data.body_lin_vel_w.torch[:, self.hand_idx]
        across = hand_v - (hand_v * baxis).sum(-1, keepdim=True) * baxis
        insert = zone * (c.rew_insert_v * (v_down / c.insert_vmps).clamp(0.0, 1.0) + c.rew_across * across.norm(dim=-1))
        r = (self.potential(phi) + insert + c.rew_success * success.float() + c.rew_fail * fail.float()
             + c.rew_open_outside * (~self.grip_closed & ~seated).float()
             + c.rew_fast_near * fast + c.rew_action_rate * self.action_rate())
        log = self.extras.setdefault("log", {})
        log["Metrics/radial_mm"] = (radial * 1000).mean().item()
        log["Metrics/depth_to_go_mm"] = ((s - RELEASE_TIP_S) * 1000).mean().item()
        log["Metrics/axis_err_deg"] = ang.mean().item()
        log["Metrics/tip_to_release_cm"] = (e * 100).mean().item()
        log["Metrics/in_bore"] = in_bore.float().mean().item()
        log["Metrics/seated"] = seated.float().mean().item()
        log["Metrics/released_seated"] = done_seated.mean().item()
        log["Metrics/hold_steps"] = self.hold_count.float().mean().item()
        log["Metrics/in_entry_zone"] = zone.mean().item()
        return r

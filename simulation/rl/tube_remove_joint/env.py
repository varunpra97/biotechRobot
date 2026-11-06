"""Joint-space centrifuge-remove RL task (no IK): grasp a tube seated in a 45 deg rotor bore, draw it out along the
bore, lift it clear and upright. tube_insert's scene and helpers, rl/joint_space.py's action.

Observation (46): TCP - the tube's grasp point x10, the tip's position along the bore axis relative to the clear
point x10, its distance from the axis x100, tube-vs-bore axis angle / 30 deg, the tube's and the bore's axes
(world), tube velocity, fingers x25, gripper command, arm joints - home, joint velocities x0.1, last action (8),
the hand's z and y axes.

Resets: start_states_remove.npz -- approaches (gripper open, the tube seated) and, held_p of them, held frames
part-way out (the reverse curriculum).

Success: held, the tip lift_goal_m above the clear point, tilt < success_tilt_deg, for success_hold_steps.
Failure: the tube out of the bore and not held (dropped), fallen, or lost.
Reward: potential-based shaping over reach / hand along the tube / held / extracted / lifted, + success, failure,
and per-step costs (tilt once clear, sideways speed in the bore, closing early, jerky actions).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv

from joint_space import JointSpaceMixin
from tube_grasp.env_cfg import TUBE_LEN
from tube_insert.env import TubeInsertEnv

from .env_cfg import TubeRemoveJointEnvCfg

SEAT_S = -0.0157                  # a seated tube's tip along the bore axis (~ -15.75 mm)


class TubeRemoveJointEnv(JointSpaceMixin, TubeInsertEnv):
    cfg: TubeRemoveJointEnvCfg

    def __init__(self, cfg: TubeRemoveJointEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_setup()
        dev, n = self.device, self.num_envs
        lib = np.load(Path(__file__).resolve().parents[1] / "start_states_remove.npz")
        def xyzw(t):
            return torch.tensor(np.concatenate([t[:, :3], t[:, 4:7], t[:, 3:4]], 1), device=dev)
        self.rm_q = torch.tensor(lib["q"], device=dev)
        self.rm_tube = xyzw(lib["tube"])
        self.rm_seat = xyzw(lib["seat"])
        self.rm_bore = torch.tensor(lib["bore"], dtype=torch.long, device=dev)
        ph = torch.tensor(lib["phase"], dtype=torch.long, device=dev)
        self.rm_open = torch.nonzero(ph == 0).flatten()
        self.rm_held = torch.nonzero(ph == 1).flatten()
        self.hold_count = torch.zeros(n, dtype=torch.long, device=dev)
        print(f"[tube_remove_joint] {len(self.rm_open)} approach + {len(self.rm_held)} held start states "
              f"(held share {cfg.held_p})", flush=True)

    # ------------------------------------------------------------- geometry
    def _remove_state(self):
        """grasp point (TCP target, 10 mm below the rim), tip along the bore axis vs the clear point, radial, axis
        angle, tilt from vertical, tip height above the clear point."""
        tp, _, axis, baxis, _, s, radial, ang = self._tube_state()
        grasp = tp + axis * (TUBE_LEN - self.cfg.remove_grasp_depth)
        clear_s = self.bore_opening[self.bore] + self.cfg.clear_s
        clear_pt = self.bore_point[self.bore] + self.scene.env_origins + clear_s.unsqueeze(-1) * baxis
        tilt = torch.rad2deg(torch.acos(axis[:, 2].clamp(-1, 1)))
        return tp, grasp, axis, baxis, s, s - clear_s, radial, ang, tilt, tp[:, 2] - clear_pt[:, 2]

    def _raw_status(self):
        c = self.cfg
        tcp, _ = self._tcp()
        tp, grasp, axis, baxis, s, s_clear, radial, ang, tilt, lift = self._remove_state()
        d = (tcp - grasp).norm(dim=-1)
        gap = self._robot.data.joint_pos.torch[:, self.finger_ids].sum(-1)
        held = self.grip_closed & (gap > c.hold_gap_m[0]) & (gap < c.hold_gap_m[1]) & (d < c.hold_max_dist_m)
        in_bore = s < self.bore_opening[self.bore]
        success = held & (s_clear > 0) & (lift > c.lift_goal_m) & (tilt < c.success_tilt_deg)
        dist = (tp - (self.bore_point[self.bore] + self.scene.env_origins)).norm(dim=-1)
        fail = (~held & ~in_bore & (dist > 0.03)) | ((tilt > 60.0) & ~held) | (dist > c.lost_m)
        return d, held, s, s_clear, radial, ang, tilt, lift, in_bore, success, fail

    def _status(self):
        *rest, success, fail = self._raw_status()
        if getattr(self, "hold_count", None) is not None and self.cfg.success_hold_steps > 1:
            success = success & (self.hold_count >= self.cfg.success_hold_steps)
        return (*rest, success, fail)

    # ------------------------------------------------------------- MDP
    def _get_observations(self) -> dict:
        tcp, _ = self._tcp()
        tp, grasp, axis, baxis, s, s_clear, radial, ang, tilt, lift = self._remove_state()
        jp = self._robot.data.joint_pos.torch
        jv = self._robot.data.joint_vel.torch
        obs = torch.cat([
            (tcp - grasp) * 10.0,
            (s_clear * 10.0).unsqueeze(-1),
            (radial * 100.0).unsqueeze(-1),
            (ang / 30.0).unsqueeze(-1),
            axis, baxis,
            self._tube.data.root_lin_vel_w.torch,
            jp[:, self.finger_ids] * 25.0,
            self.grip_closed.float().unsqueeze(-1),
            jp[:, self.arm_ids] - self.q_home,
            jv[:, self.arm_ids] * 0.1,
            self.actions,
        ], dim=-1)
        return {"policy": self._joint_obs(torch.nan_to_num(obs))}

    def _get_dones(self):
        raw = self._raw_status()[9]
        self.hold_count = torch.where(raw, self.hold_count + 1, torch.zeros_like(self.hold_count))
        _, _, _, _, _, _, _, _, _, success, fail = self._status()
        self.succeeded |= success
        self.failed |= fail
        return success | fail, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        d, held, s, s_clear, radial, ang, tilt, lift, in_bore, success, fail = self._status()
        tp, grasp, axis, baxis, *_ = self._remove_state()
        z, _ = self._hand_axes()
        align = torch.acos((-(z * axis).sum(-1)).clamp(-1, 1))             # hand z vs down the tube's axis
        extracted = ((s - SEAT_S) / (self.bore_opening[self.bore] + c.clear_s - SEAT_S)).clamp(0.0, 1.0)
        upright = (1.0 - tilt / c.upright_zero_deg).clamp(0.0, 1.0)
        hf = held.float()
        tcp, _ = self._tcp()
        rel = tcp - grasp
        h = (rel * axis).sum(-1)                                            # height above the grasp point along the axis
        lat = (rel - h.unsqueeze(-1) * axis).norm(dim=-1)
        on_line = (lat < c.line_m).float()
        tang = torch.cross(baxis, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3), dim=-1)
        tang = tang / tang.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        _, hy = self._hand_axes()
        fade = 1 - torch.tanh(d / 0.10)
        phi = (c.rew_line * (1 - torch.tanh(lat / 0.01)) * fade + c.rew_line_fine * (1 - torch.tanh(lat / 0.002)) * fade
               + c.rew_down * on_line * (1 - torch.tanh(h.clamp(min=0.0) / 0.02))
               + c.rew_closing * ((hy * tang).sum(-1) ** 2) * fade + c.rew_far * (1 - torch.tanh(d / 0.40)) + c.rew_reach * (1 - torch.tanh(d / 0.10)) + c.rew_near * (1 - torch.tanh(d / 0.02))
               + c.rew_fine * (1 - torch.tanh(d / 0.005))
               + c.rew_align * (1 - torch.tanh(align / 0.15)) * (1 - torch.tanh(d / 0.10))
               + c.rew_grasp * hf + c.rew_extract * hf * extracted
               + c.rew_lift * hf * (s_clear > 0).float() * (lift / c.lift_goal_m).clamp(0.0, 1.0) * upright)
        hand_v = self._robot.data.body_lin_vel_w.torch[:, self.hand_idx]
        across = hand_v - (hand_v * baxis).sum(-1, keepdim=True) * baxis
        r = (self.potential(phi) + c.rew_success * success.float() + c.rew_fail * fail.float()
             + c.rew_tilt * hf * (s_clear > 0).float() * (tilt - c.success_tilt_deg).clamp(min=0.0)
             + c.rew_sideways * (hf * in_bore.float()) * across.norm(dim=-1)
             + c.rew_early_close * (self.grip_closed & (d > 0.015)).float()
             + c.rew_action_rate * self.action_rate())
        log = self.extras.setdefault("log", {})
        log["Metrics/dist_cm"] = (d * 100).mean().item()
        log["Metrics/held"] = hf.mean().item()
        log["Metrics/extracted"] = (hf * extracted).mean().item()
        log["Metrics/clear"] = (hf * (s_clear > 0).float()).mean().item()
        log["Metrics/held_tilt_deg"] = tilt[held].mean().item() if held.any() else 0.0
        log["Metrics/hand_align_deg"] = torch.rad2deg(align).mean().item()
        log["Metrics/hold_steps"] = self.hold_count.float().mean().item()
        log["Metrics/line_mm"] = (lat * 1000).mean().item()
        return r

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        log = self.extras.setdefault("log", {})
        log["Metrics/success_rate"] = self.succeeded[env_ids].float().mean().item()
        log["Metrics/fail_rate"] = self.failed[env_ids].float().mean().item()
        self.succeeded[env_ids] = False
        self.failed[env_ids] = False
        DirectRLEnv._reset_idx(self, env_ids)
        k, dev = len(env_ids), self.device
        held = torch.rand(k, device=dev) < self.cfg.held_p
        pick = torch.where(held, self.rm_held[torch.randint(0, len(self.rm_held), (k,), device=dev)],
                           self.rm_open[torch.randint(0, len(self.rm_open), (k,), device=dev)])
        q = self.rm_q[pick].clone()
        q[held, 7:] += 0.0008                             # pads a little open, closing onto the tube (tube_grasp_joint)
        ids = self.arm_ids + self.finger_ids
        self._robot.write_joint_position_to_sim_index(position=q, joint_ids=ids, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids, env_ids=env_ids)
        from tube_grasp.env_cfg import GRIPPER_OPEN
        target = q.clone()
        target[:, 7:] = torch.where(held.unsqueeze(-1), torch.zeros_like(q[:, 7:]), torch.full_like(q[:, 7:], GRIPPER_OPEN))
        self._robot.set_joint_position_target_index(target=target, joint_ids=ids, env_ids=env_ids)
        pose = torch.where(held.unsqueeze(-1), self.rm_tube[pick], self.rm_seat[pick]).clone()   # approaches: seated
        pose[:, :3] += self.scene.env_origins[env_ids]
        self._tube.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self._tube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(k, 6, device=dev), env_ids=env_ids)
        self.bore[env_ids] = self.rm_bore[pick]
        self.q_cmd[env_ids] = q[:, :7]
        self.q_goal[env_ids] = q[:, :7]
        self.grip_closed[env_ids] = held
        self.start_closed_env[env_ids] = held
        self.fresh[env_ids] = 2
        self.hold_count[env_ids] = 0
        self._reset_joint(env_ids)

"""Joint-space tray-place RL task (no IK): carry the held sample tray to a target spot, set it down level at the
target yaw, release, back off. tray_grasp's scene and helpers (tray, neighbours), rl/joint_space.py's action.

Observation (45): TCP - the tray's grasp point x10, tray xy - target x10, tray height above the bench x10, sin/cos
of the yaw error, tilt / 30 deg, sin/cos of 2 x (closing axis vs the tray's width), tray velocity, fingers x25,
gripper command, arm joints - home, joint velocities x0.1, last action (8), the hand's z and y axes.

Resets: start_states_tray_place.npz -- the arm holding the tray (expert place_tray and move_tray frames), the
neighbours where that frame had them, the episode's target (+ target_jitter_m).

Success: placed (within place_xy_m / place_yaw_deg, resting, level), released, hand backed off, for
success_hold_steps. Failure: released with the tray in the air (a drop), knocked over, a neighbour moved, lost.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv

from joint_space import JointSpaceMixin
from neighbours import DISTURBED_M as NB_DISTURBED_M
from tray_grasp.env import TrayGraspEnv
from tray_grasp.env_cfg import BENCH_TOP

from .env_cfg import TrayPlaceJointEnvCfg


class TrayPlaceJointEnv(JointSpaceMixin, TrayGraspEnv):
    cfg: TrayPlaceJointEnvCfg

    def __init__(self, cfg: TrayPlaceJointEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_setup()
        dev, n = self.device, self.num_envs
        lib = np.load(Path(__file__).resolve().parents[1] / "start_states_tray_place.npz")
        t = lib["tray"]
        self.pl_q = torch.tensor(lib["q"], device=dev)
        self.pl_tray = torch.tensor(np.concatenate([t[:, :3], t[:, 4:7], t[:, 3:4]], 1), device=dev)
        self.pl_target = torch.tensor(lib["target"], device=dev)          # x, y, yaw (rad)
        if self.nb is not None:
            nh = np.asarray(lib["neigh"], np.float32)
            self.nb.lib = torch.tensor(np.concatenate([nh[..., :3], nh[..., 4:7], nh[..., 3:4]], -1), device=dev)
        self.target = torch.zeros(n, 3, device=dev)
        self.hold_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.dropped = torch.zeros(n, dtype=torch.bool, device=dev)
        print(f"[tray_place_joint] {len(self.pl_q)} held start states, "
              f"{len(torch.unique(self.pl_target.round(decimals=3), dim=0))} targets", flush=True)

    # ------------------------------------------------------------- geometry
    def _place_state(self):
        pos, grasp, tray_yaw, tilt = self._tray_state()
        xy = pos[:, :2] - self.scene.env_origins[:, :2] - self.target[:, :2]
        z = pos[:, 2] - self.scene.env_origins[:, 2] - BENCH_TOP
        ye = torch.remainder(tray_yaw - self.target[:, 2] + math.pi, 2 * math.pi) - math.pi
        return pos, grasp, xy, z, ye, tilt

    def _raw_status(self):
        c = self.cfg
        tcp, hq = self._tcp()
        pos, grasp, xy, z, ye, tilt = self._place_state()
        d = (tcp - grasp).norm(dim=-1)
        gap = self._robot.data.joint_pos.torch[:, self.finger_ids].sum(-1)
        held = self.grip_closed & (gap > c.hold_gap_m[0]) & (gap < c.hold_gap_m[1]) & (d < c.hold_max_dist_m)
        released = ~self.grip_closed & (gap > c.release_gap_m)
        e = xy.norm(dim=-1)
        placed = ((e < c.place_xy_m) & (z.abs() < c.place_z_m) & (torch.rad2deg(ye.abs()) < c.place_yaw_deg)
                  & (tilt < c.place_tilt_deg))
        backed = d > c.backoff_m
        success = placed & released & backed
        self.dropped |= ~self.grip_closed & (z > c.drop_z_m) & ~held
        fail = self.dropped | ((tilt > c.knocked_tilt_deg) & ~held) | (e > c.lost_m)
        if self.nb is not None:
            fail |= self.nb.moved() > NB_DISTURBED_M
        return d, held, released, e, z, ye, tilt, placed, success, fail

    def _status(self):
        *rest, success, fail = self._raw_status()
        if getattr(self, "hold_count", None) is not None and self.cfg.success_hold_steps > 1:
            success = success & (self.hold_count >= self.cfg.success_hold_steps)
        return (*rest, success, fail)

    def _release_allowed(self) -> torch.Tensor:
        """The tray within release_z_m of the bench and release_xy_m of its target: elsewhere letting go is a drop."""
        _, _, xy, z, _, _ = self._place_state()
        return (z < self.cfg.release_z_m) & (xy.norm(dim=-1) < self.cfg.release_xy_m)

    # ------------------------------------------------------------- MDP
    def _get_observations(self) -> dict:
        tcp, hq = self._tcp()
        pos, grasp, xy, z, ye, tilt = self._place_state()
        ce = self._yaw_error(hq)
        jp = self._robot.data.joint_pos.torch
        jv = self._robot.data.joint_vel.torch
        obs = torch.cat([
            (tcp - grasp) * 10.0,
            xy * 10.0,
            (z * 10.0).unsqueeze(-1),
            torch.sin(ye).unsqueeze(-1), torch.cos(ye).unsqueeze(-1),
            (tilt / 30.0).unsqueeze(-1),
            torch.sin(2 * ce).unsqueeze(-1), torch.cos(2 * ce).unsqueeze(-1),
            self._tray.data.root_lin_vel_w.torch,
            jp[:, self.finger_ids] * 25.0,
            self.grip_closed.float().unsqueeze(-1),
            jp[:, self.arm_ids] - self.q_home,
            jv[:, self.arm_ids] * 0.1,
            self.actions,
        ], dim=-1)
        return {"policy": self._joint_obs(torch.nan_to_num(obs))}

    def _get_dones(self):
        raw = self._raw_status()[8]
        self.hold_count = torch.where(raw, self.hold_count + 1, torch.zeros_like(self.hold_count))
        *_, success, fail = self._status()
        self.succeeded |= success
        self.failed |= fail
        return success | fail, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        d, held, released, e, z, ye, tilt, placed, success, fail = self._status()
        yaw_deg = torch.rad2deg(ye.abs())
        done = (placed & released).float()
        phi = (c.rew_far * (1 - torch.tanh(e / 0.40)) + c.rew_carry * (1 - torch.tanh(e / 0.10)) + c.rew_carry_near * (1 - torch.tanh(e / 0.01))
               + c.rew_yaw * (1 - torch.tanh(yaw_deg / 5.0)) * (1 - torch.tanh(e / 0.05))
               + c.rew_level * (1 - torch.tanh(tilt / 2.0))
               + c.rew_lower * (e < 0.005).float() * (1 - torch.tanh(z.clamp(min=0.0) / 0.01))
               + c.rew_released * done + c.rew_backoff * done * (d / c.backoff_m).clamp(max=1.0))
        r = (self.potential(phi) + c.rew_success * success.float() + c.rew_fail * fail.float()
             + c.rew_tilt * held.float() * (tilt - 3.0).clamp(min=0.0)
             + c.rew_action_rate * self.action_rate())
        log = self.extras.setdefault("log", {})
        log["Metrics/xy_err_mm"] = (e * 1000).mean().item()
        log["Metrics/height_mm"] = (z * 1000).mean().item()
        log["Metrics/yaw_err_deg"] = yaw_deg.mean().item()
        log["Metrics/tilt_deg"] = tilt.mean().item()
        log["Metrics/held"] = held.float().mean().item()
        log["Metrics/placed_released"] = done.mean().item()
        log["Metrics/hold_steps"] = self.hold_count.float().mean().item()
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
        DirectRLEnv._reset_idx(self, env_ids)
        k, dev = len(env_ids), self.device
        pick = torch.randint(0, len(self.pl_q), (k,), device=dev)
        q = self.pl_q[pick].clone()
        q[:, 7:] += self.cfg.held_finger_open_m
        ids = self.arm_ids + self.finger_ids
        self._robot.write_joint_position_to_sim_index(position=q, joint_ids=ids, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids, env_ids=env_ids)
        target = q.clone()
        target[:, 7:] = 0.0
        self._robot.set_joint_position_target_index(target=target, joint_ids=ids, env_ids=env_ids)
        pose = self.pl_tray[pick].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        self._tray.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self._tray.write_root_velocity_to_sim_index(root_velocity=torch.zeros(k, 6, device=dev), env_ids=env_ids)
        self.tray_start[env_ids] = pose[:, :3]
        if self.nb is not None:
            self.nb.reset(env_ids, pick, self.scene.env_origins[env_ids])
        tgt = self.pl_target[pick].clone()
        tgt[:, :2] += self.cfg.target_jitter_m * (2 * torch.rand(k, 2, device=dev) - 1)
        self.target[env_ids] = tgt
        self.q_cmd[env_ids] = q[:, :7]
        self.q_goal[env_ids] = q[:, :7]
        self.grip_closed[env_ids] = True
        self.start_closed_env[env_ids] = True
        self.fresh[env_ids] = 2
        self.d_prev[env_ids] = -1.0
        self.hold_count[env_ids] = 0
        self.dropped[env_ids] = False
        self._reset_joint(env_ids)

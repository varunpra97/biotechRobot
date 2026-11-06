"""Joint-space tray pick-up RL task (no IK): tray_grasp's scene, neighbours, library resets and success test with
rl/joint_space.py's action, potential-based shaping, held starts (the reverse curriculum) and a success that has to
hold for success_hold_steps.

Observation (41): tray_grasp's 32 (with the 8-D last action) + the hand's z and y axes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from joint_space import JointSpaceMixin
from tray_grasp.env_cfg import START_STATES as OPEN_LIB, TRAY_GRASP_Z, TRAY_OPEN
from tray_grasp.env import TrayGraspEnv

from .env_cfg import TrayGraspJointEnvCfg


class TrayGraspJointEnv(JointSpaceMixin, TrayGraspEnv):
    cfg: TrayGraspJointEnvCfg

    def __init__(self, cfg: TrayGraspJointEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_setup()
        dev = self.device
        held = np.load(Path(__file__).resolve().parents[1] / "start_states_tray_held.npz")
        m = len(self.lib_q)
        t = held["tray"]
        self.lib_q = torch.cat([self.lib_q, torch.tensor(held["q"], device=dev)])
        self.lib_tray = torch.cat([self.lib_tray, torch.tensor(np.concatenate([t[:, :3], t[:, 4:7], t[:, 3:4]], 1),
                                                               device=dev)])
        if self.nb is not None:
            nh = np.asarray(held["neigh"], np.float32)
            self.nb.lib = torch.cat([self.nb.lib, torch.tensor(np.concatenate([nh[..., :3], nh[..., 4:7], nh[..., 3:4]], -1),
                                                               device=dev)])
        self.n_open_lib = m
        lib = np.load(OPEN_LIB)
        p = lib["tray"]                                               # xyz + wxyz
        w, x, y, z = p[:, 3], p[:, 4], p[:, 5], p[:, 6]
        R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                      np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                      np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)
        d = np.linalg.norm(lib["tcp"] - (p[:, :3] + np.array([0.0, 0.0, TRAY_GRASP_Z])), axis=1)
        self.pre_ids = torch.tensor(np.nonzero((d < cfg.pregrasp_m) & (np.arange(len(lib["q"])) < int(lib["n_expert"])))[0], device=dev)
        print(f"[tray_grasp_joint] {len(self.pre_ids)} pre-grasp start states (share {cfg.pregrasp_p})", flush=True)
        self.held_ids = torch.arange(m, len(self.lib_q), device=dev)
        self.hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        print(f"[tray_grasp_joint] {len(self.held_ids)} held start states (share {cfg.held_p})", flush=True)

    def _get_observations(self) -> dict:
        return {"policy": self._joint_obs(super()._get_observations()["policy"])}

    def _status(self):
        d, held, lift, tilt, fail, success, ye = super()._status()
        if getattr(self, "hold_count", None) is not None and self.cfg.success_hold_steps > 1:
            success = success & (self.hold_count >= self.cfg.success_hold_steps)
        return d, held, lift, tilt, fail, success, ye

    def _get_dones(self):
        raw = TrayGraspEnv._status(self)[5]
        self.hold_count = torch.where(raw, self.hold_count + 1, torch.zeros_like(self.hold_count))
        return super()._get_dones()

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        d, held, lift, tilt, fail, success, ye = self._status()
        level = (1.0 - tilt / c.level_zero_deg).clamp(0.0, 1.0)
        z, _ = self._hand_axes()
        off = torch.acos((-z[:, 2]).clamp(-1, 1))
        hf = held.float()
        phi = (c.rew_far * (1 - torch.tanh(d / 0.40)) + c.rew_reach * (1 - torch.tanh(d / 0.10)) + c.rew_near * (1 - torch.tanh(d / 0.02))
               + c.rew_fine * (1 - torch.tanh(d / 0.005))
               + c.rew_down * (1 + torch.cos(off)) / 2 * (1 - torch.tanh(d / 0.15))
               + c.rew_yaw * torch.cos(2 * ye) * (1 - torch.tanh(d / 0.05))
               + c.rew_grasp * hf + c.rew_lift * hf * (lift / c.lift_goal).clamp(max=1.0) * level)
        hand_v = self._robot.data.body_lin_vel_w.torch[:, self.hand_idx].norm(dim=-1)
        near_open = (d < c.near_speed_m) & ~self.grip_closed
        r = (self.potential(phi) + c.rew_success * success.float() + c.rew_fail * fail.float()
             + c.rew_tilt * hf * (tilt - c.success_tilt_deg).clamp(min=0.0)
             + c.rew_early_close * (self.grip_closed & (d > 0.015)).float()
             + c.rew_near_speed * near_open.float() * (hand_v - c.near_speed_mps).clamp(min=0.0)
             + c.rew_action_rate * self.action_rate())
        log = self.extras.setdefault("log", {})
        log["Metrics/dist_cm"] = (d * 100).mean().item()
        log["Metrics/grasped"] = hf.mean().item()
        log["Metrics/lift_cm"] = (lift * 100).mean().item()
        log["Metrics/held_tilt_deg"] = tilt[held].mean().item() if held.any() else 0.0
        log["Metrics/yaw_err_deg"] = torch.rad2deg(ye.abs()).mean().item()
        log["Metrics/hand_off_vertical_deg"] = torch.rad2deg(off).mean().item()
        log["Metrics/hold_steps"] = self.hold_count.float().mean().item()
        if self.nb is not None:
            log["Metrics/neighbour_moved_mm"] = (self.nb.moved() * 1000).mean().item()
        return r

    def _reset_idx(self, env_ids):
        if hasattr(self, "n_open_lib"):                   # tray_grasp's draw: approach entries only
            full_q, full_tray = self.lib_q, self.lib_tray
            self.lib_q, self.lib_tray = full_q[:self.n_open_lib], full_tray[:self.n_open_lib]
            try:
                super()._reset_idx(env_ids)
            finally:
                self.lib_q, self.lib_tray = full_q, full_tray
        else:
            super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        dev = self.device
        self.start_closed_env[env_ids] = False
        self.hold_count[env_ids] = 0
        if hasattr(self, "held_ids"):
            u = torch.rand(len(env_ids), device=dev)
            sel = env_ids[(u >= self.cfg.held_p) & (u < self.cfg.held_p + self.cfg.pregrasp_p)]
            if len(sel) and len(self.pre_ids):              # pre-grasp: the expert's last frames before the close, open
                pick = self.pre_ids[torch.randint(0, len(self.pre_ids), (len(sel),), device=dev)]
                q = self.lib_q[pick].clone()
                ids = self.arm_ids + self.finger_ids
                self._robot.write_joint_position_to_sim_index(position=q, joint_ids=ids, env_ids=sel)
                self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids, env_ids=sel)
                target = q.clone()
                target[:, 7:] = TRAY_OPEN
                self._robot.set_joint_position_target_index(target=target, joint_ids=ids, env_ids=sel)
                pose = self.lib_tray[pick].clone()
                pose[:, :3] += self.scene.env_origins[sel]
                self._tray.write_root_pose_to_sim_index(root_pose=pose, env_ids=sel)
                self._tray.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(sel), 6, device=dev),
                                                            env_ids=sel)
                self.tray_start[sel] = pose[:, :3]
                pass
                if self.nb is not None:
                    self.nb.reset(sel, pick, self.scene.env_origins[sel])
                self.q_cmd[sel] = q[:, :7]
                self.q_goal[sel] = q[:, :7]
                self.grip_closed[sel] = False
            sel = env_ids[u < self.cfg.held_p]
            if len(sel):
                pick = self.held_ids[torch.randint(0, len(self.held_ids), (len(sel),), device=dev)]
                q = self.lib_q[pick].clone()
                q[:, 7:] += self.cfg.held_finger_open_m
                ids = self.arm_ids + self.finger_ids
                self._robot.write_joint_position_to_sim_index(position=q, joint_ids=ids, env_ids=sel)
                self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids, env_ids=sel)
                target = q.clone()
                target[:, 7:] = 0.0
                self._robot.set_joint_position_target_index(target=target, joint_ids=ids, env_ids=sel)
                pose = self.lib_tray[pick].clone()
                pose[:, :3] += self.scene.env_origins[sel]
                self._tray.write_root_pose_to_sim_index(root_pose=pose, env_ids=sel)
                self._tray.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(sel), 6, device=dev),
                                                            env_ids=sel)
                self.tray_start[sel] = pose[:, :3]
                if self.nb is not None:
                    self.nb.reset(sel, pick, self.scene.env_origins[sel])
                self.q_cmd[sel] = q[:, :7]
                self.q_goal[sel] = q[:, :7]
                self.grip_closed[sel] = True
                self.start_closed_env[sel] = True
        if hasattr(self, "phi_prev"):
            self._reset_joint(env_ids)

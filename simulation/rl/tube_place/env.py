"""Tube-place RL task (mini milestone m4): from a recorded hold above the
rack, lower the tube into its slot's bore (8 mm; 0.5 mm clearance), release
it upright on the bore floor and back the hand off.

Same scene, action, IK and hand orientation hold as tube_grasp (its env is
the base class). The target tube is the one belonging to the target slot;
it starts between the pads as recorded, the other 19 tubes in their bores.

Observation (33): tube tip minus the slot's resting point x10, tilt / 30,
the tube axis' xy, TCP minus the tube's grasp point x10 (how it sits in the
hand), tube velocity, finger positions x25, gripper command, arm joints -
home, joint velocities x0.1, last action.

Reward: xy alignment over the bore (1 cm and 1 mm scales), descent only
while aligned, progress, in-bore and released-in-bore terms, success
(released upright on the floor, hand backed off) + rest-of-episode payout;
penalties for tilt while held, opening outside the bore, failure (released
anywhere but at the bore floor, knocked over, neighbour disturbed) and jerky
actions.
"""
from __future__ import annotations

import numpy as np
import torch

from isaaclab.utils.math import quat_apply

from tube_grasp.env import TubeGraspEnv
from tube_grasp.env_cfg import GRASP_DEPTH, SLOTS, TUBE_LEN

from .env_cfg import PLACE_STATES, TubePlaceEnvCfg


class TubePlaceEnv(TubeGraspEnv):
    cfg: TubePlaceEnvCfg

    def __init__(self, cfg: TubePlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        c, dev = cfg, self.device
        self.max_step_reward = c.rew_xy + c.rew_xy_fine + c.rew_down + c.rew_down_fine + c.rew_in_bore + c.rew_released
        lib = np.load(PLACE_STATES)
        # holds over the target slot, or above the standing tubes' rims: the expert carries the tube with its
        # tip at rim height, which over a slot that is empty in the recording but filled here would collide
        t, s = lib["tube"], lib["slot"]
        rest = self.slot_rest[torch.tensor(s[:, 0] * 5 + s[:, 1])].cpu().numpy()
        keep = (np.linalg.norm(t[:, :2] - rest[:, :2], axis=1) < 0.03) | (t[:, 2] - rest[:, 2] > TUBE_LEN + 0.01)
        t, s = t[keep], s[keep]
        self.lib_q = torch.tensor(lib["q"][keep], device=dev)                             # (M, 9)
        self.lib_tube = torch.tensor(np.concatenate([t[:, :3], t[:, 4:7], t[:, 3:4]], 1), device=dev)   # xyz + xyzw
        self.lib_slot = torch.tensor(s[:, 0] * 5 + s[:, 1], dtype=torch.long, device=dev)
        rc = torch.tensor(SLOTS, device=dev)
        # neighbours (8-connected) of each slot, for retargeting
        self.neigh = [[j for j in range(len(SLOTS)) if j != i and (rc[j] - rc[i]).abs().max() <= 1] for i in range(len(SLOTS))]
        self.e_prev = torch.full((self.num_envs,), -1.0, device=dev)
        self.dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=dev)
        print(f"[tube_place] {len(self.lib_q)} recorded holds (of {len(keep)})", flush=True)

    # ------------------------------------------------------------- helpers
    def _place_state(self):
        pos, quat, tp, grasp, tilt = self._tube_state()
        rest = self.slot_rest[self.target] + self.scene.env_origins
        return pos, quat, tp, grasp, tilt, tp - rest

    # ------------------------------------------------------------- step
    def _pre_physics_step(self, actions: torch.Tensor):
        hold = self.fresh == 2
        was_closed = self.grip_closed.clone()
        super()._pre_physics_step(actions)
        self.grip_closed |= hold                          # keep holding through the first (stale-pose) step
        # opening the gripper anywhere but with the tip down at the bore floor drops the tube: a failure.
        # (The first teacher run learned to let go at once: a tube dropped from a low hold sometimes landed
        # in the bore, and a success pays out the rest of the episode.)
        _, xy, tilt, *_ = self._status()
        rel = self._place_state()[-1]
        low = (xy < self.cfg.in_bore_xy_m) & (rel[:, 2].abs() < self.cfg.in_bore_z_m)
        if self.cfg.scripted_release:                     # M4: the finisher's script lets go; any opening drops it
            low = torch.zeros_like(low)
        self.dropped |= was_closed & ~self.grip_closed & ~low

    # ------------------------------------------------------------- MDP
    def _get_observations(self) -> dict:
        tcp, _ = self._tcp()
        _, quat, tp, grasp, tilt, rel = self._place_state()
        tq = torch.gather(quat, 1, self.target.view(-1, 1, 1).expand(-1, 1, 4)).squeeze(1)
        axis = quat_apply(tq, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3))
        tvel = torch.gather(self._tubes.data.body_lin_vel_w.torch, 1,
                            self.target.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
        jp = self._robot.data.joint_pos.torch
        jv = self._robot.data.joint_vel.torch
        obs = torch.cat([
            rel * 10.0,
            (tilt / 30.0).unsqueeze(-1),
            axis[:, :2],
            (tcp - grasp) * 10.0,
            tvel,
            jp[:, self.finger_ids] * 25.0,
            self.grip_closed.float().unsqueeze(-1),
            jp[:, self.arm_ids] - self.q_home,
            jv[:, self.arm_ids] * 0.1,
            self.actions,
        ], dim=-1)
        return {"policy": torch.nan_to_num(obs).clamp(-10.0, 10.0)}

    def _status(self):
        c = self.cfg
        tcp, _ = self._tcp()
        pos, quat, tp, grasp, tilt, rel = self._place_state()
        xy = rel[:, :2].norm(dim=-1)
        in_bore = (xy < c.in_bore_xy_m) & (rel[:, 2].abs() < c.in_bore_z_m) & (tilt < c.in_bore_tilt_deg)
        gap = self._robot.data.joint_pos.torch[:, self.finger_ids].sum(-1)
        released = ~self.grip_closed & (gap > c.open_gap_m)
        backed_off = (tcp[:, 2] - grasp[:, 2]) > c.retreat_m
        success = in_bore & released & backed_off
        if c.scripted_release:                            # M4: seated while held; the generator's script releases
            success = (self.grip_closed & (rel[:, 2].abs() < c.seated_z_m) & (xy < c.seated_xy_m)
                       & (tilt < c.seated_tilt_deg))
        # neighbours: as tube_grasp (moved > 3 mm or tilted > 10 deg)
        rest = self.slot_rest.unsqueeze(0) + self.scene.env_origins.unsqueeze(1)
        moved = torch.linalg.norm(pos - rest, dim=-1)
        up = quat_apply(quat.reshape(-1, 4), torch.tensor([0.0, 0.0, 1.0], device=self.device)
                        .expand(quat.shape[0] * quat.shape[1], 3)).reshape(quat.shape[0], quat.shape[1], 3)
        tilts = torch.rad2deg(torch.acos(up[..., 2].clamp(-1, 1)))
        other = torch.ones_like(moved, dtype=torch.bool)
        other[torch.arange(self.num_envs, device=self.device), self.target] = False
        disturbed = (((moved > c.neighbour_move_m) | (tilts > c.neighbour_tilt_deg)) & other).any(-1)
        fail = (disturbed | (tilt > c.knocked_tilt_deg) | (rel.norm(dim=-1) > c.lost_m) | self.dropped
                | (rel[:, 2] > c.too_high_m))
        return rel, xy, tilt, in_bore, released, success, fail

    def _get_dones(self):
        *_, success, fail = self._status()
        self.succeeded |= success
        self.failed |= fail
        return success | fail, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        rel, xy, tilt, in_bore, released, success, fail = self._status()
        e = rel.norm(dim=-1)
        progress = torch.where(self.e_prev >= 0, self.e_prev - e, torch.zeros_like(e))
        self.e_prev = e.detach().clone()
        over_bore = (xy < c.in_bore_xy_m).float()
        held = self.grip_closed
        # alignment pays only near the rack, descent over the bore from up to ~10 cm (the second run lined up
        # within 1 mm but lifted the tube 20 cm and hovered there, collecting the xy terms)
        h = rel[:, 2].clamp(min=0.0)
        near = 1 - torch.tanh(h / c.align_height_m)
        r = (near * (c.rew_xy * (1 - torch.tanh(xy / 0.01)) + c.rew_xy_fine * (1 - torch.tanh(xy / 0.001)))
             + c.rew_down * over_bore * (1 - torch.tanh(h / 0.10)) + c.rew_down_fine * over_bore * (1 - torch.tanh(h / 0.01))
             + c.rew_progress * progress
             + c.rew_in_bore * in_bore.float() + c.rew_released * (in_bore & released).float()
             + c.rew_tilt * held.float() * (tilt - c.in_bore_tilt_deg).clamp(min=0.0)
             + c.rew_success * success.float()
             + success.float() * self.max_step_reward * (self.max_episode_length - self.episode_length_buf).float()
             + c.rew_fail * fail.float()
             + c.rew_open_outside * (~held & ~in_bore).float()
             + c.rew_action_rate * ((self.actions - self.prev_actions) ** 2).sum(-1))
        log = self.extras.setdefault("log", {})
        log["Metrics/xy_mm"] = (xy * 1000).mean().item()
        log["Metrics/height_mm"] = (rel[:, 2] * 1000).mean().item()
        log["Metrics/in_bore"] = in_bore.float().mean().item()
        log["Metrics/released_in_bore"] = (in_bore & released).float().mean().item()
        log["Metrics/held_tilt_deg"] = tilt[held].mean().item() if held.any() else 0.0
        return r

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        log = self.extras.setdefault("log", {})
        log["Metrics/success_rate"] = self.succeeded[env_ids].float().mean().item()
        log["Metrics/fail_rate"] = self.failed[env_ids].float().mean().item()
        self.succeeded[env_ids] = False
        self.failed[env_ids] = False
        super(TubeGraspEnv, self)._reset_idx(env_ids)
        k, dev = len(env_ids), self.device

        pick = torch.randint(0, len(self.lib_q), (k,), device=dev)
        q = self.lib_q[pick].clone()
        slot = self.lib_slot[pick].clone()
        held = self.lib_tube[pick].clone()
        # a neighbouring slot instead (the held tube is that slot's; the recorded slot gets its own tube back),
        # only for holds above the standing tubes' tops
        high = held[:, 2] - self.slot_rest[slot, 2] > TUBE_LEN + 0.01
        retarget = torch.nonzero((torch.rand(k, device=dev) < self.cfg.retarget_p) & high).flatten().tolist()
        for i in retarget:
            nb = self.neigh[int(slot[i])]
            slot[i] = nb[int(torch.randint(0, len(nb), (1,)))]
        held[:, :3] += self.scene.env_origins[env_ids]

        rest = self.slot_rest.unsqueeze(0) + self.scene.env_origins[env_ids].unsqueeze(1)          # (k, 20, 3)
        poses = torch.cat([rest, torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(k, len(SLOTS), 4)], -1).clone()
        poses[torch.arange(k, device=dev), slot] = held
        self._tubes.write_body_pose_to_sim_index(body_poses=poses, env_ids=env_ids)
        self._tubes.write_body_velocity_to_sim_index(body_velocities=torch.zeros(k, len(SLOTS), 6, device=dev),
                                                     env_ids=env_ids)
        ids = self.arm_ids + self.finger_ids
        self._robot.write_joint_position_to_sim_index(position=q, joint_ids=ids, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids, env_ids=env_ids)
        target = q.clone()
        target[:, 7:] = 0.0                               # pads squeezing the tube
        self._robot.set_joint_position_target_index(target=target, joint_ids=ids, env_ids=env_ids)
        self.target[env_ids] = slot
        self.q_cmd[env_ids] = q[:, :7]
        self.q_goal[env_ids] = q[:, :7]
        self.grip_closed[env_ids] = True
        self.actions[env_ids] = 0.0
        self.actions[env_ids, 3] = 1.0
        self.prev_actions[env_ids] = self.actions[env_ids]
        self.fresh[env_ids] = 2
        self.e_prev[env_ids] = -1.0
        self.dropped[env_ids] = False


__all__ = ["TubePlaceEnv", "TUBE_LEN", "GRASP_DEPTH"]

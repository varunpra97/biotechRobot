"""Joint-space tube-pick RL task (the no-IK pilot).

Action (15 Hz, 8): a change of each arm joint's target (x joint_step_rad) and
the gripper (> 0 closes, else open to GRIPPER_OPEN). The new targets are
accumulated on the previous command (at most max_cmd_lead ahead of the
measured arm) and interpolated over the step's 16 physics updates, as
tube_grasp does -- but there is no IK: the policy itself decides every joint,
so the hand's orientation, the grasp and the whole lift are learned.

Observation (39): tube_grasp's (TCP - grasp point, tilt, lift, tube
velocity, fingers, gripper command, joints - home, joint velocities, last
action) with the 8-D last action, plus the hand's z and y axes (world), since
orientation is the policy's to control now.

Resets: as tube_grasp's (M1 library, perceived pose), and held_p of them from
an expert hold (start_states_held.npz) so the lift is learned too.

Reward: tube_grasp's, with the full lift as the goal, plus the hand held
vertical near the tube, a grasp-ready bonus (open, within 4 mm, hand
within 10 deg of vertical), a reward for closing there, a descent term while
centred and vertical, and a 1.5 cm-scale fine reach.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from isaaclab.utils.math import quat_apply

from tube_grasp.env import TubeGraspEnv

from .env_cfg import TubeGraspJointEnvCfg


class TubeGraspJointEnv(TubeGraspEnv):
    cfg: TubeGraspJointEnvCfg

    def __init__(self, cfg: TubeGraspJointEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.actions = torch.zeros(n, 8, device=dev)
        self.prev_actions = torch.zeros(n, 8, device=dev)
        self.max_step_reward = (self.max_step_reward + cfg.rew_down + cfg.rew_ready + cfg.rew_close_ready
                                + cfg.rew_descend + cfg.rew_fine_wide + cfg.rew_extract)
        held = np.load(Path(__file__).resolve().parents[1] / "start_states_held.npz")
        t = held["tube"]                                                # xyz + wxyz
        self.held_q = torch.tensor(held["q"], device=dev)
        self.held_tube = torch.tensor(np.concatenate([t[:, :3], t[:, 4:7], t[:, 3:4]], 1), device=dev)   # xyz + xyzw
        self.held_slot = torch.tensor(held["slot"][:, 0] * 5 + held["slot"][:, 1], dtype=torch.long, device=dev)
        self.held_start = torch.zeros(n, dtype=torch.bool, device=dev)
        self.phi_prev = torch.full((n,), float("nan"), device=dev)        # potential of the previous step (nan: none)
        # domain randomization (TubeGraspJointDREnvCfg; all off by default): control latency of 0..dr_latency_steps
        # steps per episode, per-episode joint-encoder and TCP offsets, per-step joint noise
        L = getattr(cfg, "dr_latency_steps", 0)
        self.act_buf = torch.zeros(n, L + 1, 8, device=dev)
        self.delay = torch.zeros(n, dtype=torch.long, device=dev)
        self.q_off = torch.zeros(n, 7, device=dev)
        self.tcp_off = torch.zeros(n, 3, device=dev)
        self.hold_count = torch.zeros(n, dtype=torch.long, device=dev)    # steps in a row the success test has held
        self.tilt_prev = torch.full((n,), float("nan"), device=dev)       # held tube's tilt last step (nan: none)
        print(f"[tube_grasp_joint] {len(self.held_q)} held start states (share {cfg.held_p})", flush=True)

    def _hand_axes(self):
        _, hq = self._tcp()
        n = self.num_envs
        z = quat_apply(hq, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n, 3))
        y = quat_apply(hq, torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(n, 3))
        return z, y

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)
        # latency (DR): the arm gets the action of `delay` steps ago; the observation keeps the policy's latest one
        self.act_buf = torch.roll(self.act_buf, 1, dims=1)
        self.act_buf[:, 0] = self.actions
        a = self.act_buf[torch.arange(self.num_envs, device=self.device), self.delay]
        hold = self.fresh == 2                            # first step after a reset: poses still stale
        self.fresh = (self.fresh - 1).clamp(min=0)
        q_now = self._robot.data.joint_pos.torch[:, self.arm_ids]
        target = self.q_cmd + a[:, :7] * self.cfg.joint_step_rad
        lead = (target - q_now).clamp(-self.cfg.max_cmd_lead, self.cfg.max_cmd_lead)
        self.q_goal = torch.clamp(q_now + lead, self.q_lo, self.q_hi)
        self.q_goal[hold] = self.q_cmd[hold]
        self.grip_closed = ((a[:, 7] > 0.0) & ~hold) | (hold & self.held_start)   # a held start stays closed
        self.substep = 0

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        dev = self.device
        self.held_start[env_ids] = False
        if hasattr(self, "act_buf"):                      # DR draws for the new episode
            k, c = len(env_ids), self.cfg
            self.act_buf[env_ids] = 0.0
            self.delay[env_ids] = torch.randint(0, self.act_buf.shape[1], (k,), device=dev)
            self.q_off[env_ids] = getattr(c, "dr_q_offset_rad", 0.0) * (2 * torch.rand(k, 7, device=dev) - 1)
            self.tcp_off[env_ids] = getattr(c, "dr_tcp_offset_m", 0.0) * (2 * torch.rand(k, 3, device=dev) - 1)
        if hasattr(self, "phi_prev"):
            self.phi_prev[env_ids] = float("nan")
            self.hold_count[env_ids] = 0
            self.tilt_prev[env_ids] = float("nan")
        sel = env_ids[torch.rand(len(env_ids), device=dev) < self.cfg.held_p]
        if not len(sel):
            return
        if hasattr(self, "act_buf"):
            self.act_buf[sel] = 0.0
            self.act_buf[sel, :, 7] = 1.0                 # a held start: the queued gripper commands are "closed"
        pick = torch.randint(0, len(self.held_q), (len(sel),), device=dev)
        q = self.held_q[pick].clone()
        # the pads start a little open and close onto the tube: teleported straight onto it they start inside its
        # collider, and PhysX squeezed past and twisted the tube (gap 6.5 -> 4.6 mm, 7 deg) in ~half of the starts
        q[:, 7:] += self.cfg.held_finger_open_m
        ids = self.arm_ids + self.finger_ids
        self._robot.write_joint_position_to_sim_index(position=q, joint_ids=ids, env_ids=sel)
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids, env_ids=sel)
        target = q.clone()
        target[:, 7:] = 0.0                               # pads squeezing the tube
        self._robot.set_joint_position_target_index(target=target, joint_ids=ids, env_ids=sel)
        slot = self.held_slot[pick]
        # every tube back in its bore, then the held one where the expert held it
        from tube_grasp.env_cfg import SLOTS
        rest = self.slot_rest.unsqueeze(0) + self.scene.env_origins[sel].unsqueeze(1)
        poses = torch.cat([rest, torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(len(sel), len(SLOTS), 4)], -1).clone()
        held = self.held_tube[pick].clone()
        held[:, :3] += self.scene.env_origins[sel]
        poses[torch.arange(len(sel), device=dev), slot] = held
        self._tubes.write_body_pose_to_sim_index(body_poses=poses, env_ids=sel)
        self._tubes.write_body_velocity_to_sim_index(body_velocities=torch.zeros(len(sel), len(SLOTS), 6, device=dev),
                                                     env_ids=sel)
        self.target[sel] = slot
        self.q_cmd[sel] = q[:, :7]
        self.q_goal[sel] = q[:, :7]
        self.grip_closed[sel] = True
        self.actions[sel] = 0.0
        self.actions[sel, 7] = 1.0
        self.prev_actions[sel] = self.actions[sel]
        self.held_start[sel] = True

    def _status(self):
        """tube_grasp's, but success only once its test has held success_hold_steps in a row (run 10)."""
        d, grasped, lift, fail, success = super()._status()
        if getattr(self, "hold_count", None) is not None and self.cfg.success_hold_steps > 1:
            success = success & (self.hold_count >= self.cfg.success_hold_steps)
        return d, grasped, lift, fail, success

    def _get_dones(self):
        if self.cfg.success_hold_steps > 1:              # count before the (held) success test reads it
            raw = TubeGraspEnv._status(self)[4]
            self.hold_count = torch.where(raw, self.hold_count + 1, torch.zeros_like(self.hold_count))
        return super()._get_dones()

    def _get_observations(self) -> dict:
        obs = super()._get_observations()["policy"].clone()
        c = self.cfg
        if getattr(c, "dr_q_offset_rad", 0.0) or getattr(c, "dr_q_noise_rad", 0.0):
            # sensor DR on tube_grasp's layout: [0:3] (TCP - grasp) x10, [11:18] joints - home, [18:25] joint vel x0.1
            n = self.num_envs
            obs[:, 0:3] += self.tcp_off * 10.0
            obs[:, 11:18] += self.q_off + c.dr_q_noise_rad * torch.randn(n, 7, device=self.device)
            obs[:, 18:25] += 0.1 * c.dr_qd_noise * torch.randn(n, 7, device=self.device)
        z, y = self._hand_axes()
        return {"policy": torch.cat([obs, z, y], dim=-1).clamp(-10.0, 10.0)}

    def _potential(self):
        """Phi(s): how far along the pick this state is (reach, hand vertical, centred and down, held, lifted)."""
        c = self.cfg
        d, grasped, lift, fail, success = self._status()
        _, _, _, grasp_pt, tilt = self._tube_state()
        z, _ = self._hand_axes()
        off = torch.acos((-z[:, 2]).clamp(-1, 1))
        upright_hand = off < torch.deg2rad(torch.tensor(c.ready_deg, device=self.device))
        tcp, _ = self._tcp()
        rel = tcp - grasp_pt
        centred = (rel[:, :2].norm(dim=-1) < c.centred_m) & upright_hand
        upright = (1.0 - tilt / c.upright_zero_deg).clamp(0.0, 1.0)
        phi = (c.rew_reach * (1 - torch.tanh(d / 0.10)) + c.rew_near * (1 - torch.tanh(d / 0.02))
               + c.rew_fine * (1 - torch.tanh(d / 0.005)) + c.rew_fine_wide * (1 - torch.tanh(d / 0.015))
               + c.rew_down * (1 - torch.tanh(off / c.down_scale_rad)) * (1 - torch.tanh(d / 0.10))
               + c.rew_descend * centred.float() * (1 - torch.tanh(rel[:, 2].clamp(min=0.0) / 0.01))
               + c.rew_grasp * grasped.float()
               + c.rew_lift * grasped.float() * (lift / c.lift_goal).clamp(max=1.0) * upright)
        return phi, d, grasped, lift, tilt, fail, success, off

    def _get_rewards(self) -> torch.Tensor:
        if self.cfg.potential_shaping:
            return self._potential_rewards()
        c = self.cfg
        r = super()._get_rewards()
        d = self._status()[0]
        z, _ = self._hand_axes()
        off = torch.acos((-z[:, 2]).clamp(-1, 1))           # hand's angle from straight down (rad)
        upright_hand = off < torch.deg2rad(torch.tensor(c.ready_deg, device=self.device))
        ready = (d < c.ready_m) & upright_hand
        log = self.extras.setdefault("log", {})
        log["Metrics/hand_off_vertical_deg"] = torch.rad2deg(off).mean().item()
        tcp, _ = self._tcp()
        rel = tcp - self._tube_state()[3]
        centred = (rel[:, :2].norm(dim=-1) < c.centred_m) & upright_hand
        # extraction: while the held tube is still in its bore, pay its upward speed and charge sideways hand speed
        _, grasped, lift, _, _ = self._status()
        in_bore = grasped & (lift < c.in_bore_lift_m)
        tvel = torch.gather(self._tubes.data.body_lin_vel_w.torch, 1, self.target.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
        hand_v = self._robot.data.body_lin_vel_w.torch[:, self.hand_idx]
        extract = (in_bore.float() * (c.rew_extract * (tvel[:, 2] / c.extract_vz).clamp(0.0, 1.0)
                                      + c.rew_sideways * hand_v[:, :2].norm(dim=-1)))
        return (r + extract + c.rew_down * (1 - torch.tanh(off / c.down_scale_rad)) * (1 - torch.tanh(d / 0.10))
                + c.rew_ready * (ready & ~self.grip_closed).float() + c.rew_close_ready * (ready & self.grip_closed).float()
                + c.rew_descend * centred.float() * (1 - torch.tanh(rel[:, 2].clamp(min=0.0) / 0.01))
                + c.rew_fine_wide * (1 - torch.tanh(d / 0.015)))

    def _potential_rewards(self) -> torch.Tensor:
        """Potential-based shaping (Ng et al. 1999): the shaping terms pay only for progress, Phi(s') - Phi(s), so
        standing still earns nothing -- runs 1-7 paid them every step and learned to hover, wait open or let go
        rather than risk a -20 failure that also ended that income. The optimal policy is unchanged by this
        shaping; the steady payoffs are success, and the penalties stay per step."""
        c = self.cfg
        phi, d, grasped, lift, tilt, fail, success, off = self._potential()
        # gamma * Phi(s') - Phi(s), with PPO's gamma (0.99): standing still even costs a little
        shaping = torch.where(torch.isnan(self.phi_prev), torch.zeros_like(phi), 0.99 * phi - self.phi_prev)
        self.phi_prev = phi.detach().clone()
        r = (shaping + c.rew_success_potential * success.float() + c.rew_fail * fail.float()
             + c.rew_tilt * grasped.float() * (tilt - c.success_tilt_deg).clamp(min=0.0)
             + c.rew_early_close * (self.grip_closed & (d > 0.015)).float()
             + c.rew_action_rate * ((self.actions - self.prev_actions) ** 2).sum(-1))
        if c.rew_tilt_up or c.rew_near_speed:            # run 10: smooth holds, no fast open sweeps near the tube
            # the held tube's tilt increase over the step (its angular velocity is no use: PhysX reports ~15-30 rad/s
            # of contact jitter for the squeezed tube, mostly about its own axis, so a per-rad/s cost was ~-7/step)
            rise = torch.where(torch.isnan(self.tilt_prev), torch.zeros_like(tilt), (tilt - self.tilt_prev).clamp(min=0.0))
            self.tilt_prev = torch.where(grasped, tilt, torch.full_like(tilt, float("nan")))
            hand_v = self._robot.data.body_lin_vel_w.torch[:, self.hand_idx].norm(dim=-1)
            near_open = (d < c.near_speed_m) & ~self.grip_closed
            pen_spin = c.rew_tilt_up * grasped.float() * rise
            pen_near = c.rew_near_speed * near_open.float() * (hand_v - c.near_speed_mps).clamp(min=0.0)
            r = r + pen_spin + pen_near
            log = self.extras.setdefault("log", {})
            log["Metrics/pen_tilt_up"] = pen_spin.mean().item()
            log["Metrics/pen_near_speed"] = pen_near.mean().item()
            log["Metrics/shaping"] = shaping.mean().item()
        log = self.extras.setdefault("log", {})
        log["Metrics/dist_cm"] = (d * 100).mean().item()
        log["Metrics/grasped"] = grasped.float().mean().item()
        log["Metrics/lift_cm"] = (lift * 100).mean().item()
        log["Metrics/held_tilt_deg"] = tilt[grasped].mean().item() if grasped.any() else 0.0
        log["Metrics/hand_off_vertical_deg"] = torch.rad2deg(off).mean().item()
        log["Metrics/hold_steps"] = self.hold_count.float().mean().item()
        return r

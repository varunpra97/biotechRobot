"""Tube-grasp RL task: from near a tube in the rack, grasp it and lift it.

Action (15 Hz): TCP translation (dx, dy, dz) in [-1, 1] x action_step_m and a
gripper command (> 0 closes, else open to GRIPPER_OPEN per finger). The hand
keeps the orientation it had at the episode's start. Each step one damped
least-squares IK step on the TCP gives the arm's joint target, interpolated
over the step's 16 physics updates (as lab_env.LabEnv.step does).

Observation (privileged state; a vision student comes later): TCP minus the
target tube's grasp point (its rim - GRASP_DEPTH along its axis), the tube's
tilt, lift and velocity, finger positions, the gripper command, arm joint
positions (rel. home) and velocities, the last action.

Reward: reach (3 cm and 5 mm scales) + grasp + lift progress, a success bonus
(lifted LIFT_GOAL, tilt < 5 deg, held), failure penalties (any other tube
moved > 3 mm or tilted > 10 deg; the target knocked over), small penalties
for closing far from the tube and for jerky actions.

Resets: all 20 tubes back in their bores; the arm either at a recorded
near-grasp state (start_states.npz: expert approaches, and failed policy
trials with the fingertips above the tube tops) or at home with a random
target slot.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import compute_pose_error, quat_apply, skew_symmetric_matrix

from .env_cfg import (GRASP_DEPTH, GRIPPER_OPEN, Q_HOME, SLOTS, START_STATES, TCP_LOCAL, TUBE_LEN, TUBE_REST_Z,
                      TubeGraspEnvCfg, slot_xy)


class TubeGraspEnv(DirectRLEnv):
    cfg: TubeGraspEnvCfg

    def __init__(self, cfg: TubeGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev, n = self.device, self.num_envs
        self.arm_ids = self._robot.find_joints("joint[1-7]")[0]
        self.finger_ids = self._robot.find_joints("finger_joint.*")[0]
        self.hand_idx = self._robot.find_bodies("hand")[0][0]
        self.jac_idx = self.hand_idx - 1 if self._robot.is_fixed_base else self.hand_idx
        # Jacobian columns of the arm joints (a floating base adds 6 leading columns)
        self.jac_cols = list(self.arm_ids) if self._robot.is_fixed_base else [i + 6 for i in self.arm_ids]
        self.tcp_local = torch.tensor(TCP_LOCAL, device=dev).repeat(n, 1)
        self.q_home = torch.tensor(Q_HOME, device=dev)
        limits = self._robot.data.soft_joint_pos_limits.torch[0, self.arm_ids]
        self.q_lo, self.q_hi = limits[:, 0], limits[:, 1]

        self.slot_rest = torch.tensor([[*slot_xy(r, c), TUBE_REST_Z] for r, c in SLOTS], device=dev)   # (20, 3)
        self.target = torch.zeros(n, dtype=torch.long, device=dev)
        self.q_cmd = torch.zeros(n, 7, device=dev)          # joint target at the start of this step
        self.q_goal = torch.zeros(n, 7, device=dev)         # ... and at its end
        self.substep = 0
        self.grip_closed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.quat_des = torch.zeros(n, 4, device=dev)
        # steps since reset: body poses refresh only after a physics step, so
        # the arm holds still on the first step and takes the hand's
        # orientation (held for the episode) on the second
        self.fresh = torch.full((n,), 2, dtype=torch.long, device=dev)
        self.actions = torch.zeros(n, 4, device=dev)
        self.prev_actions = torch.zeros(n, 4, device=dev)
        self.succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.failed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.d_prev = torch.full((n,), -1.0, device=dev)       # last step's distance (-1: none yet)
        self.max_step_reward = cfg.rew_reach + cfg.rew_near + cfg.rew_fine + cfg.rew_grasp + cfg.rew_lift

        lib = np.load(cfg.start_states or START_STATES)
        keep = np.arange(len(lib["q"])) < int(lib["n_expert"])
        keep |= lib["tcp"][:, 2] > cfg.library_min_tcp_z
        self.lib_q = torch.tensor(lib["q"][keep], device=dev)                       # (M, 9)
        self.lib_slot = torch.tensor(lib["slot"][keep][:, 0] * 5 + lib["slot"][keep][:, 1], dtype=torch.long, device=dev)
        print(f"[tube_grasp] {len(self.lib_q)} library start states "
              f"({int(keep[:int(lib['n_expert'])].sum())} expert)", flush=True)

    # ------------------------------------------------------------- scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._tubes = RigidObjectCollection(self.cfg.tubes)
        self.cfg.bench.spawn.func("/World/envs/env_0/Bench", self.cfg.bench.spawn,
                                  translation=self.cfg.bench.init_state.pos)
        self.cfg.holder.spawn.func("/World/envs/env_0/Holder", self.cfg.holder.spawn,
                                   translation=self.cfg.holder.init_state.pos)
        self._fix_finger_pads()
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_object_collections["tubes"] = self._tubes
        self.scene.clone_environments(copy_from_source=False)
        import isaaclab.sim as sim_utils
        light = sim_utils.DomeLightCfg(intensity=self.cfg.light_intensity)     # rendering only
        light.func("/World/Light", light)

    def _fix_finger_pads(self):
        """As lab_scene.configure_franka_physics: the visible pads collide
        (the asset collides only through small hidden boxes 2.5 mm inside)."""
        from pxr import Usd, UsdPhysics
        stage = get_current_stage()
        root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        for prim in Usd.PrimRange(root):
            if prim.GetName() in ("left_finger", "right_finger"):
                f1 = prim.GetChild("finger_1")
                if f1.IsValid():
                    f1.SetInstanceable(False)
                    pad = f1.GetChild("finger_1")
                    if pad.IsValid() and pad.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(pad).GetCollisionEnabledAttr().Set(True)
                        print(f"[tube_grasp] pad collision on: {pad.GetPath()}", flush=True)
                    else:
                        print(f"[tube_grasp] pad NOT found under {f1.GetPath()}: "
                              f"{[c.GetName() for c in f1.GetChildren()]}", flush=True)

    # ------------------------------------------------------------- helpers
    def _hand(self):
        return (self._robot.data.body_pos_w.torch[:, self.hand_idx], self._robot.data.body_quat_w.torch[:, self.hand_idx])

    def _tcp(self):
        hp, hq = self._hand()
        return hp + quat_apply(hq, self.tcp_local), hq

    def _tube_state(self):
        pos = self._tubes.data.body_pos_w.torch                     # (N, 20, 3)
        quat = self._tubes.data.body_quat_w.torch                   # (N, 20, 4) xyzw
        idx = self.target.view(-1, 1, 1)
        tp = torch.gather(pos, 1, idx.expand(-1, 1, 3)).squeeze(1)
        tq = torch.gather(quat, 1, idx.expand(-1, 1, 4)).squeeze(1)
        axis = quat_apply(tq, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3))
        grasp = tp + axis * (TUBE_LEN - GRASP_DEPTH)
        tilt = torch.rad2deg(torch.acos(axis[:, 2].clamp(-1, 1)))
        return pos, quat, tp, grasp, tilt

    # ------------------------------------------------------------- step
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)
        tcp, hq = self._tcp()
        take = self.fresh == 1
        self.quat_des[take] = hq[take]
        hold = self.fresh == 2
        self.fresh = (self.fresh - 1).clamp(min=0)
        pos_des = tcp + self.actions[:, :3] * self.cfg.action_step_m
        # one damped least-squares IK step on the TCP (world frame)
        J = self._robot.data.body_link_jacobian_w.torch[:, self.jac_idx, :, self.jac_cols].clone()  # (N, 6, 7)
        r = quat_apply(hq, self.tcp_local)
        J[:, :3, :] -= torch.bmm(skew_symmetric_matrix(r), J[:, 3:, :])
        pe, re = compute_pose_error(tcp, hq, pos_des, self.quat_des, rot_error_type="axis_angle")
        err = torch.cat([pe, re], dim=1).unsqueeze(-1)
        lam = self.cfg.ik_damping ** 2 * torch.eye(6, device=self.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2) + lam, err)).squeeze(-1)
        # accumulate on the previous command, as lab_env integrates q_cmd: the
        # arm's PD lags its target by ~0.1 s, and re-basing on the measured
        # joints each step held the TCP to ~40% of the commanded speed; the
        # command may lead the measured arm by at most MAX_LEAD (a blocked arm
        # must not build up a stale command to lunge at)
        q_now = self._robot.data.joint_pos.torch[:, self.arm_ids]
        lead = (self.q_cmd + dq - q_now).clamp(-self.cfg.max_cmd_lead, self.cfg.max_cmd_lead)
        self.q_goal = torch.clamp(q_now + lead, self.q_lo, self.q_hi)
        self.q_goal[hold] = self.q_cmd[hold]
        self.grip_closed = (self.actions[:, 3] > 0.0) & ~hold
        self.substep = 0

    def _apply_action(self):
        self.substep += 1
        t = self.substep / self.cfg.decimation
        q = self.q_cmd + t * (self.q_goal - self.q_cmd)
        fingers = torch.where(self.grip_closed, 0.0, GRIPPER_OPEN).unsqueeze(-1).expand(-1, 2).contiguous()
        self._robot.set_joint_position_target_index(target=q, joint_ids=self.arm_ids)
        self._robot.set_joint_position_target_index(target=fingers, joint_ids=self.finger_ids)
        if self.substep == self.cfg.decimation:
            self.q_cmd[:] = self.q_goal

    # ------------------------------------------------------------- MDP
    def _perceived(self, tp, grasp, tilt, tvel):
        """The target tube as the wrist-camera localizer reports it (cfg.perception, full milestone M1): one
        control step late, with a per-episode bias, per-step noise, more noise while the tube is in the hand,
        and tilt noise. Rewards and success keep using the true state."""
        c, n, dev = self.cfg, self.num_envs, self.device
        cur = torch.cat([tp, grasp, tilt.unsqueeze(-1), tvel], -1)
        if not hasattr(self, "_perc_prev"):
            self._perc_prev = cur.clone()
            self._perc_bias = torch.zeros(n, 3, device=dev)
        fresh = self.episode_length_buf <= 1
        self._perc_prev[fresh] = cur[fresh]                 # no stale pose from the previous episode
        prev, self._perc_prev = self._perc_prev, cur.clone()
        in_hand = self.grip_closed.unsqueeze(-1).float()
        noise = (self._perc_bias + c.perc_step_m * torch.randn(n, 3, device=dev)
                 + in_hand * c.perc_in_hand_m * torch.randn(n, 3, device=dev))
        tilt_p = (prev[:, 6] + c.perc_tilt_deg * torch.randn(n, device=dev)).abs()
        return prev[:, 0:3] + noise, prev[:, 3:6] + noise, tilt_p, prev[:, 7:10]

    def _get_observations(self) -> dict:
        tcp, _ = self._tcp()
        _, _, tp, grasp, tilt = self._tube_state()
        tvel = torch.gather(self._tubes.data.body_lin_vel_w.torch, 1,
                            self.target.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
        if self.cfg.perception:
            tp, grasp, tilt, tvel = self._perceived(tp, grasp, tilt, tvel)
        jp = self._robot.data.joint_pos.torch
        jv = self._robot.data.joint_vel.torch
        obs = torch.cat([
            (tcp - grasp) * 10.0,                                     # dm
            (tilt / 30.0).unsqueeze(-1),
            ((tp[:, 2] - TUBE_REST_Z - self.scene.env_origins[:, 2]) * 10.0).unsqueeze(-1),
            tvel,
            jp[:, self.finger_ids] * 25.0,
            self.grip_closed.float().unsqueeze(-1),
            jp[:, self.arm_ids] - self.q_home,
            jv[:, self.arm_ids] * 0.1,
            self.actions,
        ], dim=-1)
        return {"policy": torch.nan_to_num(obs).clamp(-10.0, 10.0)}

    def _status(self):
        tcp, _ = self._tcp()
        pos, quat, tp, grasp, tilt = self._tube_state()
        d = torch.linalg.norm(tcp - grasp, dim=-1)
        gap = self._robot.data.joint_pos.torch[:, self.finger_ids].sum(-1)
        # a hold anywhere on the tube's upper part: pads closed on a 7 mm tube
        # read a 4.5-7.5 mm gap (5.2-6.4 measured); a hand that coasted on after
        # the policy stopped can hold the tube up to ~1.5 cm below the grasp point
        grasped = (self.grip_closed & (gap > self.cfg.hold_gap_m[0]) & (gap < self.cfg.hold_gap_m[1])
                   & (d < self.cfg.hold_max_dist_m))
        lift = (tp[:, 2] - self.scene.env_origins[:, 2] - TUBE_REST_Z).clamp(min=0.0)
        rest = self.slot_rest.unsqueeze(0) + self.scene.env_origins.unsqueeze(1)                 # (N, 20, 3)
        moved = torch.linalg.norm(pos - rest, dim=-1)                                          # (N, 20)
        up = quat_apply(quat.reshape(-1, 4), torch.tensor([0.0, 0.0, 1.0], device=self.device)
                        .expand(quat.shape[0] * quat.shape[1], 3)).reshape(quat.shape[0], quat.shape[1], 3)
        tilts = torch.rad2deg(torch.acos(up[..., 2].clamp(-1, 1)))
        other = torch.ones_like(moved, dtype=torch.bool)
        other[torch.arange(self.num_envs, device=self.device), self.target] = False
        disturbed = ((moved > self.cfg.neighbour_move_m) | (tilts > self.cfg.neighbour_tilt_deg)) & other
        knocked = (tilt > self.cfg.knocked_tilt_deg) & ~grasped
        success = grasped & (lift > self.cfg.lift_goal) & (tilt < self.cfg.success_tilt_deg)
        return d, grasped, lift, disturbed.any(-1) | knocked, success

    def _get_dones(self):
        _, _, _, fail, success = self._status()
        self.succeeded |= success
        self.failed |= fail
        terminated = success | fail
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        d, grasped, lift, fail, success = self._status()
        # the first teacher lifted tubes 3+ cm but at 20-30 deg (yanked out at
        # an angle), and only success cared about tilt: reward uprightness
        _, _, _, _, tilt = self._tube_state()
        upright = (1.0 - tilt / c.upright_zero_deg).clamp(0.0, 1.0)
        progress = torch.where(self.d_prev >= 0, self.d_prev - d, torch.zeros_like(d))
        self.d_prev = d.detach().clone()
        r = (c.rew_reach * (1 - torch.tanh(d / 0.10)) + c.rew_near * (1 - torch.tanh(d / 0.02))
             + c.rew_fine * (1 - torch.tanh(d / 0.005)) + c.rew_progress * progress
             + c.rew_grasp * grasped.float()
             + c.rew_lift * grasped.float() * (lift / c.lift_goal).clamp(max=1.0) * upright
             + c.rew_tilt * grasped.float() * (tilt - c.success_tilt_deg).clamp(min=0.0)
             + c.rew_success * success.float() + c.rew_fail * fail.float()
             # a success ends the episode: pay the rest of it at the best per-step
             # rate, or holding the tube just short of the lift goal (reward every
             # step until the time limit) beats finishing -- the first teacher
             # run learned exactly that
             + success.float() * self.max_step_reward * (self.max_episode_length - self.episode_length_buf).float()
             + c.rew_early_close * (self.grip_closed & (d > 0.015)).float()
             + c.rew_action_rate * ((self.actions - self.prev_actions) ** 2).sum(-1))
        log = self.extras.setdefault("log", {})
        log["Metrics/dist_cm"] = (d * 100).mean().item()
        log["Metrics/grasped"] = grasped.float().mean().item()
        log["Metrics/lift_cm"] = (lift * 100).mean().item()
        log["Metrics/held_tilt_deg"] = (tilt[grasped].mean().item() if grasped.any() else 0.0)
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
        k = len(env_ids)
        dev = self.device

        # tubes back in their bores, at rest
        rest = self.slot_rest.unsqueeze(0) + self.scene.env_origins[env_ids].unsqueeze(1)          # (k, 20, 3)
        quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev).expand(k, len(SLOTS), 4)
        self._tubes.write_body_pose_to_sim_index(body_poses=torch.cat([rest, quat], -1), env_ids=env_ids)
        self._tubes.write_body_velocity_to_sim_index(body_velocities=torch.zeros(k, len(SLOTS), 6, device=dev),
                                                     env_ids=env_ids)

        # arm: a recorded near-grasp state, or home with a random slot
        from_lib = torch.rand(k, device=dev) < self.cfg.library_p
        pick = torch.randint(0, len(self.lib_q), (k,), device=dev)
        q = torch.zeros(k, 9, device=dev)
        q[:, :7] = self.q_home
        q[:, 7:] = GRIPPER_OPEN
        slot = torch.randint(0, len(SLOTS), (k,), device=dev)
        q[from_lib] = self.lib_q[pick[from_lib]]
        slot[from_lib] = self.lib_slot[pick[from_lib]]
        q[:, :7] += self.cfg.joint_noise * (2 * torch.rand(k, 7, device=dev) - 1)
        q[:, :7] = torch.clamp(q[:, :7], self.q_lo, self.q_hi)
        ids = self.arm_ids + self.finger_ids
        self._robot.write_joint_position_to_sim_index(position=q, joint_ids=ids, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids, env_ids=env_ids)
        self._robot.set_joint_position_target_index(target=q, joint_ids=ids, env_ids=env_ids)
        self.target[env_ids] = slot
        if self.cfg.perception and hasattr(self, "_perc_bias"):
            self._perc_bias[env_ids] = self.cfg.perc_bias_m * torch.randn(k, 3, device=dev)
        self.q_cmd[env_ids] = q[:, :7]
        self.q_goal[env_ids] = q[:, :7]
        self.grip_closed[env_ids] = False
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.fresh[env_ids] = 2
        self.d_prev[env_ids] = -1.0

"""Pipette-grasp RL task (mini milestone m3): from up to ~60 cm away, grasp
the electronic pipette lying on the bench (across its body, at the expert's
grasp point) and lift it 3 cm without turning it in the hand. The
generator's planner then turns it nozzle-down (pipette_expert.turn_pipette_ready).

Action (15 Hz): TCP translation (dx, dy, dz) x 1 cm, gripper (> 0 closes,
else open to PIPETTE_OPEN per finger). The hand's orientation servos onto
the expert's grasp frame -- approach along the pipette's -z, closing axis
along its +x (the sign that holds the nozzle toward the wrist_side camera)
-- at most rot_step_rad per step, and is frozen once the gripper closes. One
damped least-squares IK step per control step, accumulated on the previous
joint command (as tube_grasp).

Observation (32): (TCP - grasp point) x10, the hand's orientation error
(axis-angle, rad), how far the pipette has turned from its start / 30 deg,
lift x10, pipette velocity, finger positions x25, gripper command, arm
joints - home, joint velocities x0.1, last action.

Reward: the tube_grasp template, with a staged approach (reach at 10 cm, onto
the approach line along the pipette's +z at 2 cm / 3 mm, down it only while
on it, 5 mm fine reach), progress, orientation, grasp, lift x steady, turn penalty, success + rest-of-episode
payout, failure).

Resets: the pipette at a recorded pose (+ small xy jitter) with the arm at
the recorded state (expert pick-ups, recovery episodes, pi0.5's failed
trials, up to 60 cm away), or the arm at home. With cfg.neighbours (full
milestone M3) the tray, tip rack and waste bin stand where that entry recorded
them (shifted with the arm), and moving any of them is a failure.
"""
from __future__ import annotations

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import compute_pose_error, matrix_from_quat, quat_apply, quat_from_matrix, skew_symmetric_matrix

from neighbours import DISTURBED_M as NB_DISTURBED_M, Neighbours

from .env_cfg import GRASP_BELOW_AXIS, GRASP_Y, PIPETTE_OPEN, Q_HOME, START_STATES, TCP_LOCAL, PipetteGraspEnvCfg


class PipetteGraspEnv(DirectRLEnv):
    cfg: PipetteGraspEnvCfg

    def __init__(self, cfg: PipetteGraspEnvCfg, render_mode: str | None = None, **kwargs):
        self.nb = Neighbours("pipette") if cfg.neighbours else None
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
        self.grip_closed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.quat_des = torch.zeros(n, 4, device=dev)
        self.quat_des[:, 3] = 1.0
        self.fresh = torch.full((n,), 2, dtype=torch.long, device=dev)
        self.actions = torch.zeros(n, 4, device=dev)
        self.prev_actions = torch.zeros(n, 4, device=dev)
        self.succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.failed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.d_prev = torch.full((n,), -1.0, device=dev)
        self.pip_start = torch.zeros(n, 3, device=dev)
        self.z_start = torch.zeros(n, 3, device=dev)
        self.z_start[:, 2] = 1.0
        c = cfg
        self.max_step_reward = (c.rew_reach + c.rew_line + c.rew_line_fine + c.rew_down + c.rew_fine + c.rew_align
                                + c.rew_ready + c.rew_close_ready + c.rew_grasp + c.rew_lift)

        lib = np.load(START_STATES)
        self.lib_q = torch.tensor(lib["q"], device=dev)                 # (M, 9)
        t = lib["pipette"]                                              # (M, 7) xyz + wxyz
        self.lib_pip = torch.tensor(np.concatenate([t[:, :3], t[:, 4:7], t[:, 3:4]], 1), device=dev)   # xyz + xyzw
        if self.nb is not None:
            self.nb.init_buffers(n, dev, lib["neigh"])
        kinds = lib["kind"]
        print(f"[pipette_grasp] {len(self.lib_q)} library start states "
              f"({', '.join(f'{k} {int((kinds == k).sum())}' for k in np.unique(kinds))})", flush=True)

    # ------------------------------------------------------------- scene
    def _setup_scene(self):
        import isaaclab.sim as sim_utils
        from pxr import Usd, UsdPhysics
        self._robot = Articulation(self.cfg.robot)
        self._pipette = RigidObject(self.cfg.pipette)
        self.cfg.bench.spawn.func("/World/envs/env_0/Bench", self.cfg.bench.spawn, translation=self.cfg.bench.init_state.pos)
        stage = get_current_stage()                       # visible finger pads collide (as tube_grasp)
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/Robot")):
            if prim.GetName() in ("left_finger", "right_finger"):
                f1 = prim.GetChild("finger_1")
                if f1.IsValid():
                    f1.SetInstanceable(False)
                    pad = f1.GetChild("finger_1")
                    if pad.IsValid() and pad.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(pad).GetCollisionEnabledAttr().Set(True)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["pipette"] = self._pipette
        if self.nb is not None:
            self.nb.spawn(self.scene)
        self.scene.clone_environments(copy_from_source=False)
        light = sim_utils.DomeLightCfg(intensity=self.cfg.light_intensity)
        light.func("/World/Light", light)

    # ------------------------------------------------------------- helpers
    def _tcp(self):
        hp = self._robot.data.body_pos_w.torch[:, self.hand_idx]
        hq = self._robot.data.body_quat_w.torch[:, self.hand_idx]
        return hp + quat_apply(hq, self.tcp_local), hq

    def _pip_state(self):
        """Pipette position, grasp point, the expert's grasp frame (xyzw), how far it has turned (deg)."""
        pos = self._pipette.data.root_pos_w.torch
        R = matrix_from_quat(self._pipette.data.root_quat_w.torch)     # columns: pipette x, y, z
        lx, ly, lz = R[..., 0], R[..., 1], R[..., 2]
        grasp = pos + ly * GRASP_Y - lz * GRASP_BELOW_AXIS
        z = -lz                                                         # planner.frame_from_axis(-lz, +lx)
        y = lx - (lx * z).sum(-1, keepdim=True) * z
        y = y / y.norm(dim=-1, keepdim=True)
        frame = quat_from_matrix(torch.stack([torch.cross(y, z, dim=-1), y, z], dim=-1))
        turn = torch.rad2deg(torch.acos((lz * self.z_start).sum(-1).clamp(-1, 1)))
        return pos, grasp, frame, turn

    def _rot_err(self, tcp, hq):
        _, re = compute_pose_error(tcp, hq, tcp, self.quat_des, rot_error_type="axis_angle")
        return re

    # ------------------------------------------------------------- step
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)
        tcp, hq = self._tcp()
        _, _, frame, _ = self._pip_state()
        open_ = ~self.grip_closed
        self.quat_des[open_] = frame[open_]                # frozen once closed
        hold = self.fresh == 2
        self.fresh = (self.fresh - 1).clamp(min=0)
        pos_des = tcp + self.actions[:, :3] * self.cfg.action_step_m
        J = self._robot.data.body_link_jacobian_w.torch[:, self.jac_idx, :, self.jac_cols].clone()
        r = quat_apply(hq, self.tcp_local)
        J[:, :3, :] -= torch.bmm(skew_symmetric_matrix(r), J[:, 3:, :])
        pe, re = compute_pose_error(tcp, hq, pos_des, self.quat_des, rot_error_type="axis_angle")
        re = re.clamp(-self.cfg.rot_step_rad, self.cfg.rot_step_rad)
        err = torch.cat([pe, re], dim=1).unsqueeze(-1)
        lam = self.cfg.ik_damping ** 2 * torch.eye(6, device=self.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2) + lam, err)).squeeze(-1)
        q_now = self._robot.data.joint_pos.torch[:, self.arm_ids]
        lead = (self.q_cmd + dq - q_now).clamp(-self.cfg.max_cmd_lead, self.cfg.max_cmd_lead)
        self.q_goal = torch.clamp(q_now + lead, self.q_lo, self.q_hi)
        self.q_goal[hold] = self.q_cmd[hold]               # first step after a reset: poses still stale
        self.grip_closed = (self.actions[:, 3] > 0.0) & ~hold
        self.substep = 0

    def _apply_action(self):
        self.substep += 1
        t = self.substep / self.cfg.decimation
        q = self.q_cmd + t * (self.q_goal - self.q_cmd)
        fingers = torch.where(self.grip_closed, 0.0, PIPETTE_OPEN).unsqueeze(-1).expand(-1, 2).contiguous()
        self._robot.set_joint_position_target_index(target=q, joint_ids=self.arm_ids)
        self._robot.set_joint_position_target_index(target=fingers, joint_ids=self.finger_ids)
        if self.substep == self.cfg.decimation:
            self.q_cmd[:] = self.q_goal

    # ------------------------------------------------------------- MDP
    def _get_observations(self) -> dict:
        tcp, hq = self._tcp()
        pos, grasp, _, turn = self._pip_state()
        jp = self._robot.data.joint_pos.torch
        jv = self._robot.data.joint_vel.torch
        obs = torch.cat([
            (tcp - grasp) * 10.0,
            self._rot_err(tcp, hq),
            (turn / 30.0).unsqueeze(-1),
            ((pos[:, 2] - self.pip_start[:, 2]) * 10.0).unsqueeze(-1),
            self._pipette.data.root_lin_vel_w.torch,
            jp[:, self.finger_ids] * 25.0,
            self.grip_closed.float().unsqueeze(-1),
            jp[:, self.arm_ids] - self.q_home,
            jv[:, self.arm_ids] * 0.1,
            self.actions,
        ], dim=-1)
        return {"policy": torch.nan_to_num(obs).clamp(-10.0, 10.0)}

    def _status(self):
        tcp, hq = self._tcp()
        pos, grasp, _, turn = self._pip_state()
        d = torch.linalg.norm(tcp - grasp, dim=-1)
        gap = self._robot.data.joint_pos.torch[:, self.finger_ids].sum(-1)
        held = (self.grip_closed & (gap > self.cfg.hold_gap_m[0]) & (gap < self.cfg.hold_gap_m[1])
                & (d < self.cfg.hold_max_dist_m))
        lift = (pos[:, 2] - self.pip_start[:, 2]).clamp(min=0.0)
        shoved = torch.linalg.norm((pos - self.pip_start)[:, :2], dim=-1) > self.cfg.pushed_m
        fail = ((turn > self.cfg.knocked_turn_deg) | shoved) & ~held
        if self.nb is not None:                             # the generator fails a trial if anything else moved
            fail |= self.nb.moved() > NB_DISTURBED_M
        success = held & (lift > self.cfg.lift_goal) & (turn < self.cfg.success_turn_deg)
        rot = self._rot_err(tcp, hq).norm(dim=-1)
        return d, held, lift, turn, fail, success, rot

    def _get_dones(self):
        _, _, _, _, fail, success, _ = self._status()
        self.succeeded |= success
        self.failed |= fail
        return success | fail, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        d, held, lift, turn, fail, success, rot = self._status()
        steady = (1.0 - turn / c.steady_zero_deg).clamp(0.0, 1.0)
        progress = torch.where(self.d_prev >= 0, self.d_prev - d, torch.zeros_like(d))
        self.d_prev = d.detach().clone()
        # staged approach (the first run hovered ~10 cm off and never grasped): onto the approach line through
        # the grasp point along the pipette's +z, then down it -- the way the expert and the oracle do it
        tcp, _ = self._tcp()
        _, grasp, _, _ = self._pip_state()
        lz = matrix_from_quat(self._pipette.data.root_quat_w.torch)[..., 2]
        rel = tcp - grasp
        h = (rel * lz).sum(-1)
        lat = (rel - h.unsqueeze(-1) * lz).norm(dim=-1)
        on_line = (lat < c.line_m).float()
        ready = (d < c.ready_m) & (lat < c.line_m)
        r = (c.rew_reach * (1 - torch.tanh(d / 0.10))
             + c.rew_line * (1 - torch.tanh(lat / 0.02)) + c.rew_line_fine * (1 - torch.tanh(lat / 0.003))
             + c.rew_down * on_line * (1 - torch.tanh(h.clamp(min=0.0) / 0.015))
             + c.rew_fine * (1 - torch.tanh(d / 0.01)) + c.rew_progress * progress
             # the second run lined up on the approach line and stopped 2 cm above the grasp point: the
             # last 2 cm paid little and risked nudging the pipette; make the grasp-ready state and closing
             # there worth taking
             + c.rew_ready * ready.float() + c.rew_close_ready * (ready & self.grip_closed).float()
             + c.rew_align * torch.cos(rot.clamp(max=torch.pi)) * (1 - torch.tanh(d / 0.05))
             + c.rew_grasp * held.float() + c.rew_lift * held.float() * (lift / c.lift_goal).clamp(max=1.0) * steady
             + c.rew_turn * held.float() * (turn - c.success_turn_deg).clamp(min=0.0)
             + c.rew_success * success.float()
             + success.float() * self.max_step_reward * (self.max_episode_length - self.episode_length_buf).float()
             + c.rew_fail * fail.float()
             + c.rew_early_close * (self.grip_closed & (d > 0.015)).float()
             + c.rew_action_rate * ((self.actions - self.prev_actions) ** 2).sum(-1))
        log = self.extras.setdefault("log", {})
        log["Metrics/dist_cm"] = (d * 100).mean().item()
        log["Metrics/grasped"] = held.float().mean().item()
        log["Metrics/lift_cm"] = (lift * 100).mean().item()
        log["Metrics/held_turn_deg"] = turn[held].mean().item() if held.any() else 0.0
        log["Metrics/rot_err_deg"] = torch.rad2deg(rot).mean().item()
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
        super()._reset_idx(env_ids)
        k, dev = len(env_ids), self.device

        from_lib = torch.rand(k, device=dev) < self.cfg.library_p
        pick = torch.randint(0, len(self.lib_q), (k,), device=dev)
        q = torch.zeros(k, 9, device=dev)
        q[:, :7] = self.q_home
        q[:, 7:] = PIPETTE_OPEN
        q[from_lib] = self.lib_q[pick[from_lib]]
        q[:, :7] += self.cfg.joint_noise * (2 * torch.rand(k, 7, device=dev) - 1)
        q[:, :7] = torch.clamp(q[:, :7], self.q_lo, self.q_hi)
        ids = self.arm_ids + self.finger_ids
        self._robot.write_joint_position_to_sim_index(position=q, joint_ids=ids, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids, env_ids=env_ids)
        self._robot.set_joint_position_target_index(target=q, joint_ids=ids, env_ids=env_ids)

        # pipette: the entry's recorded (settled) pose + small xy jitter (home starts take a random entry's pose)
        pose = self.lib_pip[pick].clone()
        pose[:, :2] += self.cfg.pose_jitter_m * (2 * torch.rand(k, 2, device=dev) - 1)
        pose[:, :3] += self.scene.env_origins[env_ids]
        self._pipette.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self._pipette.write_root_velocity_to_sim_index(root_velocity=torch.zeros(k, 6, device=dev), env_ids=env_ids)
        self.pip_start[env_ids] = pose[:, :3]
        self.z_start[env_ids] = matrix_from_quat(pose[:, 3:7])[..., 2]
        if self.nb is not None:
            self.nb.reset(env_ids, pick, self.scene.env_origins[env_ids])

        self.q_cmd[env_ids] = q[:, :7]
        self.q_goal[env_ids] = q[:, :7]
        self.grip_closed[env_ids] = False
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.fresh[env_ids] = 2
        self.d_prev[env_ids] = -1.0

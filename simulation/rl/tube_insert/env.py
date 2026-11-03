"""Centrifuge-insert RL task (mini milestone m5): from a recorded hold near
rotor bore 2's entry, move the tube down the 45 deg bore (8 mm, ~0.5 mm
clearance), release it seated and back the hand off.

Action (15 Hz): TCP translation (dx, dy, dz) x 1 cm, gripper (> 0 closes,
else open to GRIPPER_OPEN per finger). While the tube is held above the
bore's opening the hand turns, at most rot_step_rad per step, so the tube's
axis lines up with the bore's (the expert's insert aims the same way from
the tube's measured in-hand pose); inside the bore, and once released, the
hand holds its orientation. One damped least-squares
IK step per control step, accumulated on the previous joint command (as
tube_grasp).

Observation (33): tube tip minus its release point on the bore axis x10, the
depth still to go along the axis x10, the tip's distance from the axis x100,
the angle between tube and bore axes / 30 deg, TCP minus the tube's grasp
point x10 (how it sits in the hand), tube velocity, finger positions x25,
gripper command, arm joints - home, joint velocities x0.1, last action.

Reward: on-axis alignment (5 mm and 1 mm scales), depth only while on the
axis, progress, coaxiality, in-bore and released-seated terms, success
(released seated and coaxial, hand backed off) + rest-of-episode payout;
penalties for opening before seated, failure (dropped, fell over, lost)
and jerky actions.
"""
from __future__ import annotations

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import compute_pose_error, quat_apply, skew_symmetric_matrix

from tube_grasp.env_cfg import GRASP_DEPTH, TUBE_LEN, TubeGraspEnvCfg

from .env_cfg import BORES, GRIPPER_OPEN, Q_HOME, RELEASE_TIP_S, START_STATES, TCP_LOCAL, TubeInsertEnvCfg

HOLD_GAP = TubeGraspEnvCfg().hold_gap_m


class TubeInsertEnv(DirectRLEnv):
    cfg: TubeInsertEnvCfg

    def __init__(self, cfg: TubeInsertEnvCfg, render_mode: str | None = None, **kwargs):
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

        b = np.load(BORES)
        self.bore_point = torch.tensor(b["point"], dtype=torch.float32, device=dev)      # (6, 3), env-local
        self.bore_axis = torch.tensor(b["axis"], dtype=torch.float32, device=dev)        # (6, 3), up/out
        self.bore_opening = torch.tensor(b["opening_s"], dtype=torch.float32, device=dev)
        self.bore = torch.zeros(n, dtype=torch.long, device=dev)

        self.q_cmd = torch.zeros(n, 7, device=dev)
        self.q_goal = torch.zeros(n, 7, device=dev)
        self.substep = 0
        self.grip_closed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.fresh = torch.full((n,), 2, dtype=torch.long, device=dev)
        self.actions = torch.zeros(n, 4, device=dev)
        self.prev_actions = torch.zeros(n, 4, device=dev)
        self.succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.failed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.e_prev = torch.full((n,), -1.0, device=dev)
        c = cfg
        self.max_step_reward = (c.rew_radial + c.rew_radial_fine + c.rew_deep + c.rew_coaxial + c.rew_in_bore
                                + c.rew_released)

        lib = np.load(START_STATES)
        self.lib_q = torch.tensor(lib["q"], device=dev)
        t = lib["tube"]                                                     # xyz + wxyz
        self.lib_tube = torch.tensor(np.concatenate([t[:, :3], t[:, 4:7], t[:, 3:4]], 1), device=dev)
        self.lib_bore = torch.tensor(lib["bore"], dtype=torch.long, device=dev)
        print(f"[tube_insert] {len(self.lib_q)} recorded holds, bores {sorted(set(lib['bore'].tolist()))}", flush=True)

    # ------------------------------------------------------------- scene
    def _setup_scene(self):
        import isaaclab.sim as sim_utils
        from pxr import Usd, UsdPhysics
        self._robot = Articulation(self.cfg.robot)
        self._tube = RigidObject(self.cfg.tube)
        self.cfg.bench.spawn.func("/World/envs/env_0/Bench", self.cfg.bench.spawn, translation=self.cfg.bench.init_state.pos)
        self.cfg.centrifuge.spawn.func("/World/envs/env_0/Centrifuge", self.cfg.centrifuge.spawn,
                                       translation=self.cfg.centrifuge.init_state.pos)
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
        self.scene.rigid_objects["tube"] = self._tube
        self.scene.clone_environments(copy_from_source=False)
        light = sim_utils.DomeLightCfg(intensity=self.cfg.light_intensity)
        light.func("/World/Light", light)

    # ------------------------------------------------------------- helpers
    def _tcp(self):
        hp = self._robot.data.body_pos_w.torch[:, self.hand_idx]
        hq = self._robot.data.body_quat_w.torch[:, self.hand_idx]
        return hp + quat_apply(hq, self.tcp_local), hq

    def _tube_state(self):
        """Tip (the tube's origin), grasp point, axis, and vs the bore: tip - release point, depth s, radial, angle."""
        tp = self._tube.data.root_pos_w.torch
        axis = quat_apply(self._tube.data.root_quat_w.torch,
                          torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3))
        grasp = tp + axis * (TUBE_LEN - GRASP_DEPTH)
        point = self.bore_point[self.bore] + self.scene.env_origins
        baxis = self.bore_axis[self.bore]
        d = tp - point
        s = (d * baxis).sum(-1)
        radial = (d - s.unsqueeze(-1) * baxis).norm(dim=-1)
        rel = tp - (point + RELEASE_TIP_S * baxis)
        ang = torch.rad2deg(torch.acos((axis * baxis).sum(-1).clamp(-1, 1)))
        return tp, grasp, axis, baxis, rel, s, radial, ang

    # ------------------------------------------------------------- step
    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)
        tcp, hq = self._tcp()
        _, _, axis, baxis, _, s, _, _ = self._tube_state()
        hold = self.fresh == 2
        self.fresh = (self.fresh - 1).clamp(min=0)
        pos_des = tcp + self.actions[:, :3] * self.cfg.action_step_m
        J = self._robot.data.body_link_jacobian_w.torch[:, self.jac_idx, :, self.jac_cols].clone()
        r = quat_apply(hq, self.tcp_local)
        J[:, :3, :] -= torch.bmm(skew_symmetric_matrix(r), J[:, 3:, :])
        pe, _ = compute_pose_error(tcp, hq, pos_des, hq, rot_error_type="axis_angle")
        # the rotation taking the held tube's axis onto the bore's, while the tube is outside the bore; inside
        # it (and once released) the hand holds its orientation, as the expert's insert does: re-aiming
        # against the bore walls' push tilted the tube further each step until it jammed
        w = torch.cross(axis, baxis, dim=-1)
        sw = w.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        ang = torch.atan2(sw, (axis * baxis).sum(-1, keepdim=True))
        aim = self.grip_closed & (s > self.bore_opening[self.bore] + 0.002)
        re = torch.where(aim.unsqueeze(-1), w / sw * ang, torch.zeros_like(w))
        re = re.clamp(-self.cfg.rot_step_rad, self.cfg.rot_step_rad)
        err = torch.cat([pe, re], dim=1).unsqueeze(-1)
        lam = self.cfg.ik_damping ** 2 * torch.eye(6, device=self.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(J @ J.transpose(1, 2) + lam, err)).squeeze(-1)
        q_now = self._robot.data.joint_pos.torch[:, self.arm_ids]
        lead = (self.q_cmd + dq - q_now).clamp(-self.cfg.max_cmd_lead, self.cfg.max_cmd_lead)
        self.q_goal = torch.clamp(q_now + lead, self.q_lo, self.q_hi)
        self.q_goal[hold] = self.q_cmd[hold]               # first step after a reset: poses still stale
        self.grip_closed = (self.actions[:, 3] > 0.0) | hold
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
    def _get_observations(self) -> dict:
        tcp, _ = self._tcp()
        tp, grasp, axis, baxis, rel, s, radial, ang = self._tube_state()
        jp = self._robot.data.joint_pos.torch
        jv = self._robot.data.joint_vel.torch
        obs = torch.cat([
            rel * 10.0,
            ((s - RELEASE_TIP_S) * 10.0).unsqueeze(-1),
            (radial * 100.0).unsqueeze(-1),
            (ang / 30.0).unsqueeze(-1),
            (tcp - grasp) * 10.0,
            self._tube.data.root_lin_vel_w.torch,
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
        tp, grasp, axis, baxis, rel, s, radial, ang = self._tube_state()
        gap = self._robot.data.joint_pos.torch[:, self.finger_ids].sum(-1)
        held = self.grip_closed & (gap > HOLD_GAP[0]) & (gap < HOLD_GAP[1])
        on_axis = radial < c.radial_m
        in_bore = on_axis & (s < self.bore_opening[self.bore] - 0.005) & (ang < c.coaxial_deg)
        seated = on_axis & (s < c.seated_s) & (ang < c.coaxial_deg)
        released = ~self.grip_closed & (gap > c.open_gap_m)
        backed_off = (tcp - grasp).norm(dim=-1) > c.retreat_m
        success = seated & released & backed_off
        fail = ((ang > c.fallen_deg) & ~held) | (rel.norm(dim=-1) > c.lost_m)
        return rel, s, radial, ang, on_axis, in_bore, seated, released, success, fail

    def _get_dones(self):
        *_, success, fail = self._status()
        self.succeeded |= success
        self.failed |= fail
        return success | fail, self.episode_length_buf >= self.max_episode_length - 1

    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        rel, s, radial, ang, on_axis, in_bore, seated, released, success, fail = self._status()
        e = rel.norm(dim=-1)
        progress = torch.where(self.e_prev >= 0, self.e_prev - e, torch.zeros_like(e))
        self.e_prev = e.detach().clone()
        r = (c.rew_radial * (1 - torch.tanh(radial / 0.005)) + c.rew_radial_fine * (1 - torch.tanh(radial / 0.001))
             + c.rew_deep * on_axis.float() * (1 - torch.tanh((s - RELEASE_TIP_S).clamp(min=0.0) / 0.03))
             + c.rew_progress * progress
             + c.rew_coaxial * torch.cos(torch.deg2rad(ang))
             + c.rew_in_bore * in_bore.float() + c.rew_released * (seated & released).float()
             + c.rew_success * success.float()
             + success.float() * self.max_step_reward * (self.max_episode_length - self.episode_length_buf).float()
             + c.rew_fail * fail.float()
             + c.rew_open_outside * (~self.grip_closed & ~seated).float()
             + c.rew_action_rate * ((self.actions - self.prev_actions) ** 2).sum(-1))
        log = self.extras.setdefault("log", {})
        log["Metrics/radial_mm"] = (radial * 1000).mean().item()
        log["Metrics/depth_to_go_mm"] = ((s - RELEASE_TIP_S) * 1000).mean().item()
        log["Metrics/axis_err_deg"] = ang.mean().item()
        log["Metrics/in_bore"] = in_bore.float().mean().item()
        log["Metrics/seated"] = seated.float().mean().item()
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

        pick = torch.randint(0, len(self.lib_q), (k,), device=dev)
        q = self.lib_q[pick].clone()
        ids = self.arm_ids + self.finger_ids
        self._robot.write_joint_position_to_sim_index(position=q, joint_ids=ids, env_ids=env_ids)
        self._robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), joint_ids=ids, env_ids=env_ids)
        target = q.clone()
        target[:, 7:] = 0.0                               # pads squeezing the tube
        self._robot.set_joint_position_target_index(target=target, joint_ids=ids, env_ids=env_ids)
        pose = self.lib_tube[pick].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        self._tube.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self._tube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(k, 6, device=dev), env_ids=env_ids)
        self.bore[env_ids] = self.lib_bore[pick]
        self.q_cmd[env_ids] = q[:, :7]
        self.q_goal[env_ids] = q[:, :7]
        self.grip_closed[env_ids] = True
        self.actions[env_ids] = 0.0
        self.actions[env_ids, 3] = 1.0
        self.prev_actions[env_ids] = self.actions[env_ids]
        self.fresh[env_ids] = 2
        self.e_prev[env_ids] = -1.0

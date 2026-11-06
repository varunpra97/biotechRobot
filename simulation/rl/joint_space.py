"""Joint-space control for any of the RL skill envs (the no-IK recipe from tube_grasp_joint, made reusable).

Mix in ahead of an IK skill env -- ``class TubeInsertJointEnv(JointSpaceMixin, TubeInsertEnv)`` -- whose cfg sets the
fields listed below. The skill env keeps its scene, resets, success test and (unless it overrides them) its
rewards; the mixin replaces its action interface:

Action (8, or 7 without a gripper): a change of each arm joint's target (x joint_step_rad) and the gripper (> its
threshold closes). The targets accumulate on the previous command, at most max_cmd_lead ahead of the measured arm,
interpolated over the step's physics updates by the skill env's own ``_apply_action`` -- no IK, no held orientation:
the policy decides every joint. ``grip_threshold`` below 0 biases the untrained gripper towards closed (run 9 of the
tube pick: a coin-flip gripper never kept a hold long enough to learn the lift).

Observation: the skill env's (which carries ``self.actions``, now the joint action) plus the hand's z and y axes.

Rewards: ``potential(phi)`` turns a skill's progress potential into Phi(s') - Phi(s) shaping (Ng et al. 1999), so
standing still earns nothing -- the per-step level terms of runs 1-7 of the tube pick paid hovering and waiting.
"""
from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply


# A skill whose job ends in a release overrides _release_allowed(): outside it the gripper stays closed whatever the
# policy outputs (insert run 1: with the untrained gripper biased closed instead, an open was never sampled and the
# policy learned to hold still, seated tubes included).
#
# the fields a joint-space skill cfg sets (on the skill's own cfg class; configclass and multiple inheritance don't mix):
#   joint_step_rad = 0.02     max change of a joint target per 15 Hz step (0.3 rad/s)
#   joint_gripper = True      8th action = gripper; False: 7 actions, the gripper as the skill env leaves it
#   grip_threshold = 0.0      the gripper closes when its action is above this
#   start_closed = False      the first step after a reset keeps the gripper closed (a held start)
#   shaping_gamma = 1.0       Phi(s') - Phi(s); 0.99 (PPO's gamma) drains (1 - gamma) Phi per waiting step, which
#                             pays a policy that can't finish yet to back away from the target


class JointSpaceMixin:
    def _joint_setup(self):
        """Call at the end of the skill env's __init__ (the subclass's)."""
        n, dev = self.num_envs, self.device
        a = 8 if self.cfg.joint_gripper else 7
        self.actions = torch.zeros(n, a, device=dev)
        self.prev_actions = torch.zeros(n, a, device=dev)
        self.phi_prev = torch.full((n,), float("nan"), device=dev)
        # per env: the first step after a reset keeps the gripper closed (a held start); envs that mix held and open
        # starts set it in their reset, the others take cfg.start_closed
        self.start_closed_env = torch.full((n,), bool(self.cfg.start_closed), dtype=torch.bool, device=dev)

    def _hand_axes(self):
        _, hq = self._tcp()
        n = self.num_envs
        z = quat_apply(hq, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n, 3))
        y = quat_apply(hq, torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(n, 3))
        return z, y

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)
        hold = self.fresh == 2                            # first step after a reset: poses still stale
        self.fresh = (self.fresh - 1).clamp(min=0)
        q_now = self._robot.data.joint_pos.torch[:, self.arm_ids]
        target = self.q_cmd + self.actions[:, :7] * self.cfg.joint_step_rad
        lead = (target - q_now).clamp(-self.cfg.max_cmd_lead, self.cfg.max_cmd_lead)
        self.q_goal = torch.clamp(q_now + lead, self.q_lo, self.q_hi)
        self.q_goal[hold] = self.q_cmd[hold]
        if self.cfg.joint_gripper:
            closed = self.actions[:, 7] > self.cfg.grip_threshold
            # release interlock: a closed gripper opens only where the skill allows it (e.g. the tube inside its bore,
            # the tray resting at its spot) -- letting go anywhere else is always a drop
            closed = closed | (self.grip_closed & ~self._release_allowed())
            self.grip_closed = torch.where(hold, self.start_closed_env, closed)
        self.substep = 0

    def _release_allowed(self) -> torch.Tensor:
        """Where a closed gripper may open (default: anywhere). Skills that must let go override it."""
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _reset_joint(self, env_ids):
        """Call after the skill env's _reset_idx: clear the joint action and the shaping memory."""
        self.actions[env_ids] = 0.0
        if self.cfg.joint_gripper:
            self.actions[env_ids, 7] = self.start_closed_env[env_ids].float()
        self.prev_actions[env_ids] = self.actions[env_ids]
        self.phi_prev[env_ids] = float("nan")

    def _joint_obs(self, obs: torch.Tensor) -> torch.Tensor:
        z, y = self._hand_axes()
        return torch.cat([obs, z, y], dim=-1).clamp(-10.0, 10.0)

    def potential(self, phi: torch.Tensor) -> torch.Tensor:
        """gamma * Phi(s') - Phi(s); 0 on the first step of an episode."""
        shaping = torch.where(torch.isnan(self.phi_prev), torch.zeros_like(phi),
                              self.cfg.shaping_gamma * phi - self.phi_prev)
        self.phi_prev = phi.detach().clone()
        return shaping

    def action_rate(self) -> torch.Tensor:
        return ((self.actions - self.prev_actions) ** 2).sum(-1)

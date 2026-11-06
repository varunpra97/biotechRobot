"""Joint-space tray placement (no IK): holding the sample tray, the policy commands the 7 arm joints and the gripper
to carry it to a target spot on the bench, set it down level at the target yaw, release it and back the hand off
(the generator's place_tray test: within 3 mm, 3 deg, resting, tilt < 1 deg, nothing else moved). The scene is
tray_grasp's (bench, tray, the pipette / tip rack / waste bin as neighbours); the recorded expert placements give
the held starts and their targets (start_states_tray_place.npz).
"""
from __future__ import annotations

from isaaclab.utils.configclass import configclass

from tray_grasp.env_cfg import TrayGraspEnvCfg


@configclass
class TrayPlaceJointEnvCfg(TrayGraspEnvCfg):
    episode_length_s = 12.0          # 180 steps: carry, align, lower, release, back off
    action_space = 8
    observation_space = 45
    # joint space (rl/joint_space.py)
    joint_step_rad = 0.02
    joint_gripper = True
    grip_threshold = 0.0             # neutral, with the release interlock (env._release_allowed; insert run 1)
    release_z_m = 0.01               # the gripper may open only with the tray this close to the bench ...
    release_xy_m = 0.02              # ... and its target
    start_closed = True
    # Phi(s') - Phi(s), undiscounted: with gamma 0.99 waiting cost (1 - gamma) Phi per step, ~-18 per episode near the
    # target and nothing far away, and remove run 1's approaches learned to swing the arm away (85 cm)
    shaping_gamma = 1.0
    rew_far = 3.0                    # 1 - tanh(distance / 40 cm): a pull from the far starts (up to 30-60 cm)
    held_finger_open_m = 0.0008
    target_jitter_m = 0.01           # the recorded target, moved up to this in x and y
    # the generator's test (expert.place_tray)
    place_xy_m = 0.003
    place_z_m = 0.0005               # resting: the tray's origin within this of the bench
    place_yaw_deg = 3.0
    place_tilt_deg = 1.0
    release_gap_m = 0.028            # fingers opened past the 24 mm tray
    backoff_m = 0.02                 # TCP this far from the tray's grasp point once released
    drop_z_m = 0.005                 # released with the tray higher than this: a drop (failure)
    success_hold_steps = 5
    lost_m = 0.6
    # potential Phi(s)
    rew_carry = 3.0                  # 1 - tanh(xy error / 10 cm)
    rew_carry_near = 3.0             # ... / 1 cm
    rew_yaw = 3.0                    # 1 - tanh(|yaw error| / 5 deg), near the target
    rew_level = 2.0                  # 1 - tanh(tilt / 2 deg)
    rew_lower = 4.0                  # within 5 mm of the target in xy: 1 - tanh(height / 1 cm)
    rew_released = 6.0               # released, placed
    rew_backoff = 4.0                # released, placed, x min(|TCP - grasp point| / backoff_m, 1)
    # terminal and per-step terms
    rew_success = 100.0
    rew_fail = -20.0
    rew_tilt = -0.1                  # per degree of tray tilt beyond 3 deg while held
    rew_action_rate = -0.005


__all__ = ["TrayPlaceJointEnvCfg"]

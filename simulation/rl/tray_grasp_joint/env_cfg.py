"""Joint-space tray pick-up (no IK): the policy commands the 7 arm joints and the gripper to grasp the sample tray
across its 24 mm width and lift it 5.5 cm level, from up to ~60 cm away. Otherwise as full milestone M2
(tray_grasp's: neighbours, the start library of expert approaches and pi0.5 failures). The IK teacher held the hand
pointing down and turned only its yaw; here the whole hand orientation is learned.
"""
from __future__ import annotations

from isaaclab.utils.configclass import configclass

from tray_grasp.env_cfg import TrayGraspEnvCfg


@configclass
class TrayGraspJointEnvCfg(TrayGraspEnvCfg):
    episode_length_s = 12.0          # 180 steps from up to ~60 cm
    action_space = 8
    observation_space = 41           # tray_grasp's 32 with the 8-D last action, + the hand's z and y axes
    # joint space (rl/joint_space.py)
    joint_step_rad = 0.02
    joint_gripper = True
    grip_threshold = -0.8            # biased towards closed (tube pick run 9), z = -2 at std 0.4
    start_closed = False             # held starts set it per env
    # Phi(s') - Phi(s), undiscounted: with gamma 0.99 waiting cost (1 - gamma) Phi per step, ~-18 per episode near the
    # target and nothing far away, and remove run 1's approaches learned to swing the arm away (85 cm)
    shaping_gamma = 1.0
    rew_far = 3.0                    # 1 - tanh(distance / 40 cm): a pull from the far starts (up to 30-60 cm)
    held_p = 0.3                     # share of resets from an expert hold (start_states_tray_held.npz)
    held_finger_open_m = 0.0008
    # pre-grasp curriculum: this share of resets from the expert's frames within pregrasp_m of the grasp point,
    # gripper open (pipette runs 1-3: held starts 100%, approaches stalled 3-6 cm out with the hand misoriented --
    # only ~5% of the library's approaches start within 10 cm)
    pregrasp_p = 0.3
    pregrasp_m = 0.04
    # potential Phi(s): reach, hand down, closing axis across the tray, held, lift x level
    rew_reach = 2.0                  # 1 - tanh(d / 10 cm)
    rew_near = 2.0                   # ... / 2 cm
    rew_fine = 4.0                   # ... / 5 mm
    rew_down = 3.0                   # hand z along -Z: (1 + cos angle) / 2, faded over 15 cm (a tanh(angle / 0.15 rad)
                                     # term is flat beyond ~30 deg: pipette run 1 stalled with its hand turned ~164 deg)
    rew_yaw = 3.0                    # cos(2 x closing-axis yaw error), near the tray
    rew_grasp = 5.0
    rew_lift = 10.0
    # terminal and per-step terms
    success_hold_steps = 10
    rew_success = 100.0
    rew_tilt = -0.1
    rew_early_close = -0.2
    rew_near_speed = -2.0            # per m/s of open-hand TCP speed beyond near_speed_mps within near_speed_m
    near_speed_mps = 0.15
    near_speed_m = 0.05
    rew_action_rate = -0.005


__all__ = ["TrayGraspJointEnvCfg"]

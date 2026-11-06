"""Joint-space pipette pick-up (no IK): the policy commands the 7 arm joints and the gripper to grasp the electronic
pipette lying on the bench across its body (the expert's grasp point and frame) and lift it 3 cm without turning
it in the hand, from up to ~60 cm away. Otherwise as full milestone M3 (pipette_grasp's: neighbours, the start
library of expert, recovery and pi0.5 states). The IK teacher servoed the hand onto the grasp frame itself; here
the orientation is learned (the frame error stays in the observation).
"""
from __future__ import annotations

from isaaclab.utils.configclass import configclass

from pipette_grasp.env_cfg import PipetteGraspEnvCfg


@configclass
class PipetteGraspJointEnvCfg(PipetteGraspEnvCfg):
    episode_length_s = 12.0          # 180 steps from up to ~60 cm
    action_space = 8
    observation_space = 42           # pipette_grasp's 32 with the 8-D last action, + the hand's z and y axes
    # joint space (rl/joint_space.py)
    joint_step_rad = 0.02
    joint_gripper = True
    grip_threshold = -0.8            # biased towards closed (tube pick run 9), z = -2 at std 0.4
    start_closed = False             # held starts set it per env
    # Phi(s') - Phi(s), undiscounted: with gamma 0.99 waiting cost (1 - gamma) Phi per step, ~-18 per episode near the
    # target and nothing far away, and remove run 1's approaches learned to swing the arm away (85 cm)
    shaping_gamma = 1.0
    rew_far = 3.0                    # 1 - tanh(distance / 40 cm): a pull from the far starts (up to 30-60 cm)
    held_p = 0.3                     # share of resets from an expert hold (start_states_pipette_held.npz)
    held_finger_open_m = 0.0008
    # pre-grasp curriculum: this share of resets from the expert's frames within pregrasp_m of the grasp point,
    # gripper open (pipette runs 1-3: held starts 100%, approaches stalled 3-6 cm out with the hand misoriented --
    # only ~5% of the library's approaches start within 10 cm)
    pregrasp_p = 0.3
    pregrasp_m = 0.04
    # potential Phi(s): pipette_grasp's level terms (reach, approach line, down it, fine, frame, held, lift x steady)
    rew_reach = 2.0
    rew_line = 2.0
    rew_line_fine = 3.0
    rew_down = 3.0
    rew_fine = 4.0
    # hand onto the expert's grasp frame: (1 + cos(frame error)) / 2, faded over align_fade_m. Run 1 used
    # 1 - tanh(error / 0.2 rad) faded over 5 cm: flat beyond ~30 deg, and its approaches stopped 5.7 cm off with the
    # hand turned ~164 deg the wrong way (held starts 100%, approaches 0-16%)
    rew_align = 2.0
    align_fade_m = 0.15
    # run 3 (run 2's approaches still hovered on the approach line 4.8 cm above the grasp point, the hand's approach axis
    # 45 deg and closing axis 97 deg off): each axis gets its own broad term, (1 + cos angle) / 2, and the descent a
    # 6 cm-scale term next to the 1.5 cm one (flat at 4.8 cm)
    rew_axis_z = 3.0
    rew_axis_y = 3.0
    rew_down_wide = 3.0
    rew_grasp = 5.0
    rew_lift = 10.0
    # terminal and per-step terms
    success_hold_steps = 10
    rew_success = 100.0
    rew_turn = -0.1
    rew_early_close = -0.2
    rew_near_speed = -2.0            # per m/s of open-hand TCP speed beyond near_speed_mps within near_speed_m
    near_speed_mps = 0.15
    near_speed_m = 0.05
    rew_action_rate = -0.005


__all__ = ["PipetteGraspJointEnvCfg"]

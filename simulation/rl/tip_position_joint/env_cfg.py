"""Joint-space tip positioning for aspirate / dispense (no IK): the policy commands the 7 arm joints to put the
attached tip's end at the aspirate / dispense point inside a tube or a tray well, coaxial and clear of the wall
and floor (the generator's rules). Otherwise as tip_position (kinematic pipette, recorded approaches up to 12 cm).
The plunger stays scripted, as with the IK teacher; the IK teacher also held the tip along the container's axis
itself, which is learned here.
"""
from __future__ import annotations

from isaaclab.utils.configclass import configclass

from tip_position.env_cfg import TipPositionEnvCfg


@configclass
class TipPositionJointEnvCfg(TipPositionEnvCfg):
    episode_length_s = 8.0           # 120 steps from up to 12 cm
    action_space = 7                 # 7 joint-target changes (x joint_step_rad); the pipette stays held
    observation_space = 42           # tip_position's 26 with the 7-D last action, + tip and container axes, hand z/y
    # joint space (rl/joint_space.py)
    joint_step_rad = 0.01            # 0.15 rad/s: a 0.5 mm target
    joint_gripper = False
    grip_threshold = 0.0
    start_closed = True
    # Phi(s') - Phi(s), undiscounted: with gamma 0.99 waiting cost (1 - gamma) Phi per step, ~-18 per episode near the
    # target and nothing far away, and remove run 1's approaches learned to swing the arm away (85 cm)
    shaping_gamma = 1.0
    success_hold_steps = 5           # the plunger stroke needs the tip to stay put
    # potential Phi(s): reach, on the axis (3 mm / 0.3 mm), height while near it, tip along the axis
    rew_reach = 3.0                  # 1 - tanh(|tip end - target| / 5 cm)
    rew_radial = 3.0
    rew_radial_fine = 4.0
    rew_depth = 4.0                  # 1 - tanh(|height - target| / 5 mm), only near the axis (radial < 1 mm)
    rew_align = 4.0                  # 1 - tanh(angle / 5 deg)
    rew_success = 100.0
    rew_action_rate = -0.005


__all__ = ["TipPositionJointEnvCfg"]

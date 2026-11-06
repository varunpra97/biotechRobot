"""Joint-space tip attach (no IK): the policy commands the 7 arm joints to bring the held pipette's nozzle onto the
target tip's axis and press it 4-9 mm into the collar (the generator's attach rule: within 0.6 mm and 3 deg).
Otherwise as full milestone M6 (tip_attach's TipAttachM6EnvCfg: any of the rack's 100 tips, kinematic pipette).
The IK teacher turned the hand so the nozzle pointed down the tip's axis; here that is learned too.
"""
from __future__ import annotations

from isaaclab.utils.configclass import configclass

from tip_attach.env_cfg import TipAttachM6EnvCfg


@configclass
class TipAttachJointEnvCfg(TipAttachM6EnvCfg):
    episode_length_s = 8.0           # 120 steps from up to 8 cm
    action_space = 7                 # 7 joint-target changes (x joint_step_rad); the pipette stays held
    observation_space = 39           # tip_attach's 23 with the 7-D last action, + nozzle and tip axes, hand z and y
    # joint space (rl/joint_space.py)
    joint_step_rad = 0.01            # 0.15 rad/s: a 0.6 mm attach window
    joint_gripper = False
    grip_threshold = 0.0
    start_closed = True
    # Phi(s') - Phi(s), undiscounted: with gamma 0.99 waiting cost (1 - gamma) Phi per step, ~-18 per episode near the
    # target and nothing far away, and remove run 1's approaches learned to swing the arm away (85 cm)
    shaping_gamma = 1.0
    # potential Phi(s): near the target, on the tip's axis (5 mm / 0.5 mm), depth while on it, nozzle along the axis
    rew_reach = 3.0                  # 1 - tanh(|nozzle end - target| / 2 cm)
    rew_radial = 3.0
    rew_radial_fine = 4.0
    rew_depth = 4.0                  # 1 - tanh(|depth - target| / 1 cm), only while on the axis (radial < 0.6 mm)
    rew_align = 4.0                  # 1 - tanh(angle / 5 deg)
    rew_success = 100.0
    rew_action_rate = -0.005


__all__ = ["TipAttachJointEnvCfg"]

"""Tube-place RL task config (mini milestone m4): a tube held above the rack
(from recorded expert place episodes) goes down into its slot's 8 mm bore,
is released, and the hand backs off. Same scene, robot, physics and control
as tube_grasp.
"""
from __future__ import annotations

from pathlib import Path

from isaaclab.utils.configclass import configclass

from tube_grasp.env_cfg import TubeGraspEnvCfg

SIM = Path(__file__).resolve().parents[2]
PLACE_STATES = str(SIM / "rl" / "start_states_place.npz")


@configclass
class TubePlaceEnvCfg(TubeGraspEnvCfg):
    episode_length_s = 8.0           # 120 steps from up to 15 cm away
    observation_space = 33

    # resets: every start is a recorded hold (a held tube can't start from home)
    joint_noise = 0.0                # the tube sits between the pads as recorded
    retarget_p = 0.5                 # place into a neighbouring slot instead of the recorded one
    # success: tip on the bore floor, upright, released, hand backed off
    in_bore_xy_m = 0.002             # tip within this of the slot's axis (0.5 mm bore clearance at the top)
    in_bore_z_m = 0.004              # ... and this of its resting height
    in_bore_tilt_deg = 5.0
    open_gap_m = 0.009               # pads apart (> the 7 mm tube): released
    retreat_m = 0.02                 # TCP this far above the tube's grasp point once released
    # rewards
    rew_xy = 1.0                     # 1 - tanh(xy / 1 cm)
    rew_xy_fine = 2.0                # 1 - tanh(xy / 1 mm)
    rew_down = 2.0                   # 1 - tanh(height above rest / 10 cm), only while over the bore (xy < 2 mm)
    rew_down_fine = 2.0              # ... / 1 cm
    align_height_m = 0.08            # the xy terms fade out as the tip rises above rest (tanh scale)
    too_high_m = 0.12                # tip this far above rest: failure (the holds start at most 9 cm up)
    rew_progress = 20.0              # per metre the tip got closer to its resting point
    rew_in_bore = 3.0
    rew_released = 3.0               # released in the bore
    rew_tilt = -0.05                 # per degree of held-tube tilt beyond in_bore_tilt_deg
    rew_success = 50.0               # + the rest of the episode at the best per-step rate
    rew_fail = -20.0                 # dropped / knocked over / a neighbour disturbed
    rew_open_outside = -0.5          # gripper open while the tube is not in its bore
    rew_action_rate = -0.01
    lost_m = 0.25                    # tube this far from its slot: failure
    knocked_tilt_deg = 25.0
    scripted_release = False
    seated_z_m = 0.001
    seated_xy_m = 0.0015
    seated_tilt_deg = 3.0


__all__ = ["TubePlaceEnvCfg", "TubePlaceM4EnvCfg", "PLACE_STATES"]


@configclass
class TubePlaceM4EnvCfg(TubePlaceEnvCfg):
    """Full milestone M4: holds over all 20 slots (start_states_place.npz incl. data/primitives_place_M4); the
    teacher's goal is the tube seated while held (tip within seated_z_m of the bore floor, on the axis, upright);
    the generator's finisher then opens and backs off by script (sim/eval_place_finisher.py --scripted-release:
    5/5 with the m4 teacher)."""
    scripted_release = True
    retarget_p = 0.5



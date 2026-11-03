"""Attach-tip RL task config (mini milestone m6): the held pipette's nozzle
pressed onto a tip in the rack, from recorded expert approaches. A minimal
port of the generator's attach rule (sim/pipette.py): the tip locks on when
the nozzle end is 4-9 mm inside its collar, within 0.6 mm and 3 deg of its
axis. As in the generator, tips don't collide with the pipette (the fit is
the rule), so the task is kinematic: the nozzle follows the hand through the
recorded in-hand transform; the scene is the robot and the bench.
"""
from __future__ import annotations

from pathlib import Path

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from tube_grasp.env_cfg import Q_HOME, TubeGraspEnvCfg

SIM = Path(__file__).resolve().parents[2]
START_STATES = str(SIM / "rl" / "start_states_tip.npz")


@configclass
class TipAttachEnvCfg(DirectRLEnvCfg):
    decimation = 16                  # 15 Hz control over 240 Hz physics
    episode_length_s = 6.0           # 90 steps from up to 8 cm away
    action_space = 3                 # dx, dy, dz (x action_step_m)
    observation_space = 23
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=16)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=True)

    robot = TubeGraspEnvCfg().robot
    bench = TubeGraspEnvCfg().bench
    light_intensity = 700.0

    # actions / IK: the hand turns, at most rot_step_rad per step, to point the nozzle down the tip's axis
    action_step_m = 0.005
    rot_step_rad = 0.1
    ik_damping = 0.05
    max_cmd_lead = 0.1
    # the attach rule (pipette.py)
    seal_depth_m = (0.004, 0.009)
    attach_radial_m = 0.0006
    attach_angle_deg = 3.0
    target_depth_m = 0.0065          # aim: the middle of the seal window
    rim_radial_m = 0.0015            # pressing > rim_depth_m into the collar this far off its axis: hits the tip
    rim_depth_m = 0.001
    # rewards
    rew_radial = 1.0                 # 1 - tanh(radial / 5 mm)
    rew_radial_fine = 2.0            # 1 - tanh(radial / 0.5 mm)
    rew_depth = 2.0                  # 1 - tanh(|depth - target| / 1 cm), only while on the axis (radial < 0.6 mm)
    rew_progress = 20.0              # per metre the nozzle end got closer to its target point
    rew_align = 0.5                  # cos(nozzle vs tip axis)
    rew_success = 50.0               # + the rest of the episode at the best per-step rate
    rew_fail = -20.0                 # crushed the tip (> 9 mm) or pressed on its rim
    rew_action_rate = -0.01
    retarget_p = 0.0                 # M6: share of resets that target another tip of the rack

    def __post_init__(self):
        self.robot.init_state.joint_pos = {**{f"joint{i + 1}": Q_HOME[i] for i in range(7)}, "finger_joint.*": 0.0094}


@configclass
class TipAttachM6EnvCfg(TipAttachEnvCfg):
    """Full milestone M6: any of the rack's 100 tips (half the resets retarget the recorded approach to another tip;
    the library is shifted to the default arm spot)."""
    retarget_p = 0.5
    episode_length_s = 8.0


__all__ = ["TipAttachEnvCfg", "TipAttachM6EnvCfg", "START_STATES"]

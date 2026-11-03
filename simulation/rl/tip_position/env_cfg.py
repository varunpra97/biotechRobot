"""Tip-positioning RL task config (full milestone M6: aspirate and dispense):
put the attached tip's end at the aspirate / dispense point in a tube or a
tray well, from recorded expert approaches. A port of the generator's rules
(sim/pipette.py, pipette_expert.py): where the tip crosses the container's
rim plane its outer surface must stay inside the inner wall, and its end
must stay above the floor. As in the generator the tip and the pipette don't
collide with the containers (the rules decide), so the task is kinematic:
the tip end follows the hand through the recorded in-hand transform; the
scene is the robot and the bench. The plunger stays scripted (the generator
finisher holds the reached point while it moves).
"""
from __future__ import annotations

from pathlib import Path

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from tube_grasp.env_cfg import Q_HOME, TubeGraspEnvCfg

SIM = Path(__file__).resolve().parents[2]
START_STATES = str(SIM / "rl" / "start_states_pipetting.npz")
# container profiles (sim/liquids.py): inner radius at the rim, floor and rim heights above the container origin
TUBE = {"rim_z": 0.058, "rim_r": 0.0030, "floor_z": 0.0008}
WELL = {"rim_z": 0.0144, "rim_r": 0.00345, "floor_z": 0.0037}
TIP_END_R, TIP_BODY_TOP_R, TIP_COLLAR_Z = 0.00045, 0.0037, 0.052     # pipette.tip_outer_r


@configclass
class TipPositionEnvCfg(DirectRLEnvCfg):
    decimation = 16                  # 15 Hz control over 240 Hz physics
    episode_length_s = 8.0           # 120 steps from up to 12 cm away
    action_space = 3                 # dx, dy, dz (x action_step_m)
    observation_space = 26
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=16)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=True)

    robot = TubeGraspEnvCfg().robot
    bench = TubeGraspEnvCfg().bench
    light_intensity = 700.0

    # actions / IK: the hand turns, at most rot_step_rad per step, to hold the tip along the container's axis
    action_step_m = 0.005
    rot_step_rad = 0.1
    ik_damping = 0.05
    max_cmd_lead = 0.1
    # success: at the target point, coaxial, clear of the wall
    target_radial_m = 0.0005         # tip end this close to the container's axis...
    target_height_m = 0.0015         # ... and this close to the target height
    target_angle_deg = 3.0
    clearance_m = 0.0001             # outer surface at the rim at least this far inside the wall
    # rewards
    rew_reach = 1.0                  # 1 - tanh(d / 5 cm)
    rew_radial = 1.0                 # 1 - tanh(radial / 3 mm)
    rew_radial_fine = 2.0            # 1 - tanh(radial / 0.3 mm)
    rew_depth = 2.0                  # 1 - tanh(|height - target| / 5 mm), only while near the axis (radial < 1 mm)
    rew_progress = 20.0
    rew_align = 0.5
    rew_success = 50.0               # + the rest of the episode at the best per-step rate
    rew_fail = -20.0                 # the tip touched the wall or the floor
    rew_action_rate = -0.01

    def __post_init__(self):
        self.robot.init_state.joint_pos = {**{f"joint{i + 1}": Q_HOME[i] for i in range(7)}, "finger_joint.*": 0.0094}


__all__ = ["TipPositionEnvCfg", "START_STATES", "TUBE", "WELL", "TIP_END_R", "TIP_BODY_TOP_R", "TIP_COLLAR_Z"]

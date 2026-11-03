"""Pipette-grasp RL task config (mini milestone m3): grasp the electronic
pipette lying on the bench and lift it 3 cm, from up to ~60 cm away (pi0.5's
pipette attempts end far from it). The generator's planner then turns it
nozzle-down to the ready spot (pipette_expert.turn_pipette_ready). Same
robot, physics and control as tube_grasp; the scene is the bench and the
pipette only.
"""
from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from tube_grasp.env_cfg import BENCH_TOP, Q_HOME, TCP_LOCAL, TubeGraspEnvCfg

SIM = Path(__file__).resolve().parents[2]
PIPETTE_USD = str(SIM / "rl" / "assets" / "pipette.usda")
START_STATES = str(SIM / "rl" / "start_states_pipette.npz")
GRASP_Y = 0.11                      # pipette_expert.PIPETTE_GRASP_Y: grasp point along the pipette from the nozzle end
GRASP_BELOW_AXIS = 0.006            # pipette_expert.PIPETTE_GRASP_BELOW_AXIS
PIPETTE_OPEN = 0.016                # pipette_expert.PIPETTE_OPEN: m per finger


@configclass
class PipetteGraspEnvCfg(DirectRLEnvCfg):
    decimation = 16                  # 15 Hz control over 240 Hz physics
    episode_length_s = 10.0          # 150 steps: starts up to ~60 cm away
    action_space = 4                 # dx, dy, dz (x action_step_m), gripper (> 0 closes)
    observation_space = 32
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=16)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=3.0, replicate_physics=True)

    robot = TubeGraspEnvCfg().robot
    bench = TubeGraspEnvCfg().bench
    pipette = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Pipette",
        spawn=sim_utils.UsdFileCfg(
            usd_path=PIPETTE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                                                         max_depenetration_velocity=0.5, sleep_threshold=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.21, 1.58, BENCH_TOP + 0.041)),
    )
    light_intensity = 700.0

    # actions / IK (as tube_grasp); the hand's orientation servos onto the
    # expert's grasp frame (approach along -pipette z, closing axis along +pipette x),
    # at most rot_step_rad per step, frozen once the gripper closes
    action_step_m = 0.01
    rot_step_rad = 0.2
    ik_damping = 0.05
    max_cmd_lead = 0.1
    # resets
    library_p = 0.8                  # recorded start states (expert, recovery, pi0.5 failures), else home
    joint_noise = 0.01
    pose_jitter_m = 0.005            # pipette xy jitter around the recorded pose
    # rewards (tube_grasp template)
    rew_reach = 1.0                  # 1 - tanh(d / 10 cm)
    rew_line = 1.0                   # 1 - tanh(distance from the approach line / 2 cm)
    rew_line_fine = 2.0              # ... / 3 mm
    rew_down = 2.0                   # 1 - tanh(height above the grasp point along the line / 1.5 cm), while on it
    line_m = 0.005                   # "on the line"
    rew_fine = 4.0                   # 1 - tanh(d / 1 cm)
    rew_ready = 2.0                  # within ready_m of the grasp point and on the line
    rew_close_ready = 2.0            # ... with the gripper closed
    ready_m = 0.005
    rew_progress = 20.0              # per metre closer
    rew_align = 0.5                  # hand orientation onto the grasp frame, near the grasp point
    rew_grasp = 3.0
    rew_lift = 10.0                  # x lift / lift_goal x steady, while held
    rew_turn = -0.1                  # per degree the pipette has turned beyond success_turn_deg while held
    rew_success = 50.0               # + the rest of the episode at the best per-step rate
    rew_fail = -20.0
    rew_early_close = -0.05
    rew_action_rate = -0.01
    steady_zero_deg = 20.0           # lift reward scales 1 -> 0 as the pipette turns 0 -> this (vs its start)
    lift_goal = 0.03
    success_turn_deg = 10.0
    hold_gap_m = (0.015, 0.021)      # pipette_expert: finger gap holding the ~19 mm body
    hold_max_dist_m = 0.02
    knocked_turn_deg = 30.0
    pushed_m = 0.05                  # pipette shoved this far without being held: failure
    neighbours = True                # M3: the tray, tip rack and waste bin around it (neighbours.py)

    def __post_init__(self):
        self.robot.init_state.joint_pos = {**{f"joint{i + 1}": Q_HOME[i] for i in range(7)},
                                           "finger_joint.*": PIPETTE_OPEN}


__all__ = ["PipetteGraspEnvCfg", "GRASP_Y", "GRASP_BELOW_AXIS", "PIPETTE_OPEN", "START_STATES", "TCP_LOCAL", "Q_HOME",
           "BENCH_TOP"]

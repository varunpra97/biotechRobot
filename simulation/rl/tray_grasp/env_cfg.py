"""Tray-grasp RL task config (mini milestone m2): pick up the sample tray and
lift it level, from up to ~60 cm away (pi0.5's tray attempts end 17-67 cm
from it). Same robot, physics and control as tube_grasp; the scene is the
bench and the tray only (no rack/tubes: faster).
"""
from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from tube_grasp.env_cfg import BENCH_TOP, Q_HOME, TCP_LOCAL, TubeGraspEnvCfg

SIM = Path(__file__).resolve().parents[2]
TRAY_USD = str(SIM / "rl" / "assets" / "tray.usda")
START_STATES = str(SIM / "rl" / "start_states_tray.npz")
TRAY_XY = (0.30, 1.73)              # lab_scene.SAMPLE_TRAY_XY (default layout)
TRAY_GRASP_Z = 0.005                # expert.Expert.TRAY_GRASP_Z: TCP this far above the bench at the grasp
TRAY_OPEN = 0.020                   # expert.Expert.TRAY_OPEN: m per finger
TRAY_WIDTH = 0.024                  # gripped across its 24 mm width


@configclass
class TrayGraspEnvCfg(DirectRLEnvCfg):
    decimation = 16                  # 15 Hz control over 240 Hz physics
    episode_length_s = 10.0          # 150 steps: starts up to ~60 cm away
    action_space = 5                 # dx, dy, dz (x action_step_m), dyaw (x yaw_step_rad), gripper (> 0 closes)
    observation_space = 32
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=16)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=3.0, replicate_physics=True)

    robot = TubeGraspEnvCfg().robot
    bench = TubeGraspEnvCfg().bench
    tray = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Tray",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TRAY_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                                                         max_depenetration_velocity=0.5, sleep_threshold=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.015),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*TRAY_XY, BENCH_TOP)),
    )
    light_intensity = 700.0

    # actions / IK (as tube_grasp)
    action_step_m = 0.01
    yaw_step_rad = 0.1
    ik_damping = 0.05
    max_cmd_lead = 0.1
    # resets
    library_p = 0.8                  # recorded start states (expert approaches + pi0.5 failures), else home
    home_tray_jitter_m = 0.01
    home_tray_jitter_deg = 5.0
    joint_noise = 0.01
    # rewards (tube_grasp template + yaw alignment)
    rew_reach = 1.0                  # 1 - tanh(d / 10 cm)
    rew_near = 1.0                   # 1 - tanh(d / 2 cm)
    rew_fine = 2.0                   # 1 - tanh(d / 5 mm)
    rew_progress = 20.0              # per metre closer
    rew_yaw = 0.5                    # alignment of the closing axis with the tray's width (cos 2*error)
    rew_grasp = 3.0
    rew_lift = 10.0                  # x lift / lift_goal x level, while held
    rew_tilt = -0.1                  # per degree of tray tilt beyond success_tilt_deg while held
    rew_success = 50.0               # + the rest of the episode at the best per-step rate
    rew_fail = -20.0
    rew_early_close = -0.05
    rew_action_rate = -0.01
    level_zero_deg = 10.0            # lift reward scales 1 -> 0 as the tray tilts 0 -> this
    lift_goal = 0.055                # generator's pick_up_tray success: lifted > TRAY_LIFT - 0.005
    success_tilt_deg = 3.0           # ... and tilt < 3 deg
    hold_gap_m = (0.020, 0.0265)     # finger gap holding the 24 mm-wide tray
    hold_max_dist_m = 0.02
    knocked_tilt_deg = 20.0
    pushed_m = 0.05                  # tray shoved this far without being held: failure
    neighbours = True                # M2: the pipette, tip rack and waste bin around it (neighbours.py)

    def __post_init__(self):
        self.robot.init_state.joint_pos = {**{f"joint{i + 1}": Q_HOME[i] for i in range(7)}, "finger_joint.*": TRAY_OPEN}


__all__ = ["TrayGraspEnvCfg", "TRAY_GRASP_Z", "TRAY_OPEN", "TRAY_WIDTH", "TRAY_XY", "START_STATES", "TCP_LOCAL", "Q_HOME",
           "BENCH_TOP"]

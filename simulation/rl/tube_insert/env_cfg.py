"""Centrifuge-insert RL task config (mini milestone m5): a tube held near the
entry of rotor bore 2 (the recorded expert inserts at the default centrifuge
spot) goes down the 45 deg, 8 mm bore and is released seated. Robot, physics
and control as tube_grasp; the scene is the bench, the centrifuge body (exact
triangle-mesh collider) and the held tube.
"""
from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from tube_grasp.env_cfg import BENCH_TOP, GRIPPER_OPEN, Q_HOME, TCP_LOCAL, TUBE_USD, TubeGraspEnvCfg

SIM = Path(__file__).resolve().parents[2]
CENTRIFUGE_USD = str(SIM / "rl" / "assets" / "centrifuge.usda")
BORES = str(SIM / "rl" / "assets" / "centrifuge_bores.npz")
START_STATES = str(SIM / "rl" / "start_states_insert.npz")
RELEASE_TIP_S = -0.014              # expert.RELEASE_TIP_S: tube tip along the bore axis at release (seat ~ -15.75 mm)


@configclass
class TubeInsertEnvCfg(DirectRLEnvCfg):
    decimation = 16                  # 15 Hz control over 240 Hz physics
    episode_length_s = 8.0           # 120 steps from up to 8 cm away
    action_space = 4                 # dx, dy, dz (x action_step_m), gripper (> 0 closes)
    observation_space = 33
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=16)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=3.0, replicate_physics=True)

    robot = TubeGraspEnvCfg().robot
    bench = TubeGraspEnvCfg().bench
    centrifuge = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Centrifuge",
        spawn=sim_utils.UsdFileCfg(usd_path=CENTRIFUGE_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),     # the asset is in env-local coordinates
    )
    tube = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Tube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TUBE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                                                         max_depenetration_velocity=0.5, sleep_threshold=0.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0015),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 1.9, BENCH_TOP + 0.2)),
    )
    light_intensity = 700.0

    # actions / IK (as tube_grasp); the hand turns so the held tube's axis lines up with the bore's
    # (at most rot_step_rad per step), frozen once released
    action_step_m = 0.005            # 7.5 cm/s max: a held tube slips in the pads at 15 cm/s (and pops out at the floor)
    rot_step_rad = 0.1
    ik_damping = 0.05
    max_cmd_lead = 0.1
    # success: released, coaxial, seated, hand backed off
    seated_s = RELEASE_TIP_S + 0.004     # tip at least this deep along the axis
    coaxial_deg = 5.0
    radial_m = 0.0015
    open_gap_m = 0.009
    retreat_m = 0.02                     # TCP this far from the tube's grasp point once released
    # rewards
    rew_radial = 1.0                     # 1 - tanh(radial / 5 mm)
    rew_radial_fine = 2.0                # 1 - tanh(radial / 1 mm)
    rew_deep = 2.0                       # 1 - tanh(depth to go / 3 cm), only while on the axis (radial < 1.5 mm)
    rew_progress = 20.0                  # per metre the tip got closer to its release point
    rew_coaxial = 0.5                    # cos(tube axis vs bore axis)
    rew_in_bore = 3.0                    # inside the bore (below the opening), on the axis
    rew_released = 3.0                   # released seated
    rew_success = 50.0                   # + the rest of the episode at the best per-step rate
    rew_fail = -20.0                     # dropped / fell over / lost
    rew_open_outside = -0.5              # gripper open while the tube is not seated
    rew_action_rate = -0.01
    fallen_deg = 30.0                    # tube axis this far off the bore's while not held: failure
    lost_m = 0.2




@configclass
class TubeInsertM5EnvCfg(TubeInsertEnvCfg):
    """Full milestone M5: bores 1-3 (reachable on the default layout), holds from up to 30 cm (the expert's carry
    from the rack included), 12 s episodes."""
    episode_length_s = 12.0


__all__ = ["TubeInsertM5EnvCfg", "TubeInsertEnvCfg", "BORES", "START_STATES", "RELEASE_TIP_S", "GRIPPER_OPEN", "Q_HOME", "TCP_LOCAL", "BENCH_TOP"]

"""Tube-grasp RL task config (Isaac Lab 3.0, direct workflow).

One environment = our lab bench in its default arrangement, in env-local
coordinates equal to sim/lab_scene.py's world coordinates: the Franka on its
base at ARM_POS (yaw 90), the 5x4 rack at its default spot with all 20 tubes
standing in their bores, the bench top. Same assets and physics as the data
generator: sim/assets/panda (gains from the USD; pads collide; force-limited
finger drives; 64 solver iterations; gravity-compensated arm), rack = exact
triangle-mesh collider, tube = 40-vertex convex hull, 1.5 g, glass friction,
240 Hz physics, 15 Hz control.
"""
from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

SIM = Path(__file__).resolve().parents[2]                    # biotech-lab-robotics/simulation
FRANKA_USD = str(SIM / "assets" / "panda" / "panda.usda")
HOLDER_USD = str(SIM / "rl" / "assets" / "holder.usda")
TUBE_USD = str(SIM / "rl" / "assets" / "tube.usda")
START_STATES = str(SIM / "rl" / "start_states.npz")

# lab_scene (default layout)
ARM_POS = (0.0, 1.1, 0.9)
Q_HOME = (0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741)
BENCH_TOP = 0.94                     # BENCH_Z + 0.04
HOLDER_XY = (-0.05, 1.65)
PITCH = 0.0615                       # HOLDER_PITCH_MM
SLOTS = [(r, c) for r in range(4) for c in range(5)]
TUBE_REST_Z = BENCH_TOP + 0.012      # tube tip on its bore floor (HOLDER_BORE_FLOOR_MM)
TUBE_LEN = 0.058
GRASP_DEPTH = 0.012                  # expert grasp: TCP 12 mm below the tube's rim
TCP_LOCAL = (0.0, 0.0, 0.1123)       # lab_scene.TCP_LOCAL, hand frame
GRIPPER_OPEN = 0.012                 # m per finger: the expert's opening around a 7 mm tube


def slot_xy(r, c):
    return HOLDER_XY[0] + (c - 2) * PITCH, HOLDER_XY[1] + (r - 1.5) * PITCH


def _yaw_quat_xyzw(deg):
    a = math.radians(deg) / 2
    return (0.0, 0.0, math.sin(a), math.cos(a))


@configclass
class TubeGraspEnvCfg(DirectRLEnvCfg):
    # 15 Hz control over 240 Hz physics (lab_env.CONTROL_HZ, lab_scene.PHYSICS_HZ)
    decimation = 16
    episode_length_s = 8.0
    action_space = 4                 # dx, dy, dz (x ACTION_STEP_M), gripper (>0 closes)
    observation_space = 29
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=16)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=3.0, replicate_physics=True)

    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=FRANKA_USD,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1.0),
            # the USD holds link0 to the world with its own fixed joint (as in the
            # data generator); PhysX then reports a floating base, so the env
            # offsets the Jacobian's joint columns by 6 (fix_root_link would
            # add a second articulation root)
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=64, solver_velocity_iteration_count=1),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ARM_POS, rot=_yaw_quat_xyzw(90.0),
            joint_pos={**{f"joint{i + 1}": Q_HOME[i] for i in range(7)}, "finger_joint.*": 0.04},
        ),
        actuators={
            # the USD's own gains (stiffness 4500/3500/2000, damping 450/350/200, max 87/12 N m)
            "arm": ImplicitActuatorCfg(joint_names_expr=["joint[1-7]"], stiffness=None, damping=None),
            # lab_scene.FINGER_STIFFNESS / FINGER_DAMPING / FINGER_MAX_FORCE
            "fingers": ImplicitActuatorCfg(joint_names_expr=["finger_joint.*"], stiffness=2.3e3, damping=60.0,
                                           effort_limit_sim=30.0),
        },
    )
    bench = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Bench",
        spawn=sim_utils.CuboidCfg(size=(1.4, 0.7, 0.04), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.3,
                                                                                 dynamic_friction=0.25,
                                                                                 restitution=0.1),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.37, 0.4))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 1.8, BENCH_TOP - 0.02)),
    )
    holder = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Holder",
        spawn=sim_utils.UsdFileCfg(usd_path=HOLDER_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*HOLDER_XY, BENCH_TOP)),
    )
    tubes = RigidObjectCollectionCfg(rigid_objects={
        f"tube_{r}{c}": RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Tube_{r}{c}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=TUBE_USD,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                    max_depenetration_velocity=0.5, sleep_threshold=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.0015),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(*slot_xy(r, c), TUBE_REST_Z)),
        ) for r, c in SLOTS})

    light_intensity = 700.0          # dome light (rendering only; videos)

    # actions
    action_step_m = 0.01             # max TCP move per 15 Hz step (0.15 m/s, the expert's FAST)
    ik_damping = 0.05
    max_cmd_lead = 0.1               # rad: joint command at most this far ahead of the measured arm
    # resets: start from recorded near-grasp states, or from home at a random slot
    library_p = 0.7
    library_min_tcp_z = BENCH_TOP + 0.075   # failed-trial states: fingertips above the tube tops
    joint_noise = 0.01
    # rewards
    rew_reach = 1.0                  # 1 - tanh(d / 10 cm): a signal from anywhere near the rack
    rew_near = 1.0                   # 1 - tanh(d / 2 cm)
    rew_fine = 2.0                   # 1 - tanh(d / 5 mm)
    rew_progress = 20.0              # per metre the TCP got closer to the grasp point this step
    rew_grasp = 3.0                  # tube between closed pads at the grasp point
    rew_lift = 10.0                  # x lift / LIFT_GOAL while grasped
    rew_success = 50.0
    rew_fail = -20.0                 # a neighbour disturbed, or the target knocked over
    rew_early_close = -0.05          # gripper closed while > 1.5 cm from the grasp point
    rew_action_rate = -0.01
    rew_tilt = -0.05                 # per degree of target-tube tilt beyond 5 deg while holding it
    upright_zero_deg = 20.0          # lift reward scales 1 -> 0 as the held tube tilts 0 -> this
    lift_goal = 0.03                 # m: expert's success test (holder_top + LIFT_CLEARANCE)
    hold_gap_m = (0.0045, 0.0075)    # finger gap of pads closed on the 7 mm tube
    hold_max_dist_m = 0.02           # TCP within this of the grasp point while holding
    success_tilt_deg = 5.0
    neighbour_move_m = 0.003
    neighbour_tilt_deg = 10.0
    knocked_tilt_deg = 25.0
    # full milestone M1 (TubeGraspM1EnvCfg): start library and the localizer's errors on the observed tube
    start_states = None              # None: START_STATES
    perception = False
    perc_bias_m = 0.001              # per episode, per axis (localizer approach error ~0.3 mm median, 1.7 mm p95)
    perc_step_m = 0.0007             # per step
    perc_in_hand_m = 0.002           # extra per step while the gripper is closed (in hand: 2 mm median, 4.6 mm p95)
    perc_tilt_deg = 1.5              # (measured in-hand tilt error ~3 deg)


@configclass
class TubeGraspM1EnvCfg(TubeGraspEnvCfg):
    """Full milestone M1: hand-over from up to 25 cm at all 20 slots (start_states_M1.npz: recorded expert
    approaches and pi0.5 attempts within 25 cm + states sampled around every slot), the perceived tube pose."""
    start_states = str(SIM / "rl" / "start_states_M1.npz")
    perception = True
    library_p = 0.85

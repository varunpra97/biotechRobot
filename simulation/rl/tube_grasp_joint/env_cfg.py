"""Joint-space tube-pick RL task (the no-IK pilot): the policy commands the
arm's 7 joints and the gripper directly -- no IK step, no fixed hand
orientation, no scripted lift. Otherwise as full milestone M1 (tube_grasp's
TubeGraspM1EnvCfg): hand-overs from up to 25 cm at all 20 slots, the tube
seen as the wrist-camera localizer reports it.
"""
from __future__ import annotations

from isaaclab.utils.configclass import configclass

from tube_grasp.env_cfg import TubeGraspM1EnvCfg


@configclass
class TubeGraspJointEnvCfg(TubeGraspM1EnvCfg):
    episode_length_s = 12.0          # 180 steps: approach from up to 25 cm, grasp, the full lift
    action_space = 8                 # 7 joint-target changes (x joint_step_rad) + gripper (> 0 closes)
    observation_space = 39           # tube_grasp's 25 + hand z and y axes (6) + last action (8)
    # max change of a joint target per 15 Hz step (0.3 rad/s). 0.04 in runs 1-5: exploration jitter that size levers a
    # held tube against its 8 mm bore (0.5 mm clearance) and knocks it over, so holding looked like failure
    joint_step_rad = 0.02
    # the generator's success test needs the tube tip 5.4 cm above its rest height (holder top + 3 cm - 4 mm):
    # the policy lifts all of it (the IK teacher lifted 3 cm and a script did the rest)
    lift_goal = 0.06
    # hand vertical near the tube: 1 - tanh(angle from straight down / down_scale_rad), faded with distance. The first
    # run (0.5 x hand_z . -Z) reached the grasp point within 1.5 mm but with the hand 28 deg off vertical, never
    # closed (a tilted grip lifts a tilted tube, which scores little), and its gripper output settled at -3
    rew_down = 2.0
    down_scale_rad = 0.15
    # run 3 reached the grasp pose (1.5 mm, hand vertical) and stayed there open: a bonus for being ready while open
    # paid every step, and the gripper output drifted to "open" (-3.7) so closing was never sampled. Only closing
    # in the grasp pose pays now (the run-4 checkpoint also resets the gripper output, see job 083)
    rew_ready = 0.0                  # open, within ready_m of the grasp point, hand within ready_deg of vertical
    rew_close_ready = 4.0            # closing there
    ready_m = 0.004
    ready_deg = 10.0
    # run 2 held the hand vertical and centred (2.5-3 mm) but stopped ~11 mm above the grasp point: the 5 mm fine
    # term pays ~nothing there and the ready bonus starts at 4 mm. Reward the descent while centred and vertical,
    # and add a wider fine term
    rew_descend = 3.0                # 1 - tanh(height above the grasp point / 1 cm), while within centred_m and vertical
    centred_m = 0.004
    rew_fine_wide = 2.0              # 1 - tanh(d / 1.5 cm)
    # reverse curriculum: run 4's first closes knocked tubes over (its arm had never moved while holding one), so
    # closing looked like failure. This share of resets starts from an expert hold (start_states_held.npz:
    # fingers just closed on the tube, in the bore or part-way up)
    held_p = 0.5
    held_finger_open_m = 0.0008      # per finger, at a held start
    # hold test: pads closed on the 7 mm tube read 3.7-7.5 mm here (tube centred within 0.6 mm and upright, pads sunk
    # deeper into the convex collider); tube_grasp's 4.5 mm lower bound scored ~a quarter of real holds as "not held"
    # -- no grasp or lift reward, and a lift of one counted as knocking it over
    hold_gap_m = (0.003, 0.0075)
    # run 8: potential-based shaping (env._potential_rewards) in place of per-step shaping terms
    potential_shaping = True
    rew_success_potential = 100.0
    # extracting the held tube: its upward speed (capped at extract_vz) while still in the bore, minus sideways hand speed
    rew_extract = 2.0
    extract_vz = 0.10                # m/s
    rew_sideways = -2.0              # per (m/s) of horizontal TCP speed while the held tube is in the bore
    in_bore_lift_m = 0.03
    rew_action_rate = -0.005
    # run 10 (generator 16/20 with run 9): its episode ended the step the tube first passed the test, so it never
    # had to keep it -- 3 trials ended 5.0-6.3 deg after a 10 cm-in-0.7 s yank (11 deg peak) and slow drift in the
    # grip, and 2 of 30 knocked the tube over in a fast open sweep near it. Success now needs the test held
    # success_hold_steps in a row, with a tilt margin under the generator's 5 deg; and the held tube's tilt rising
    # and a fast open hand near the tube cost per step
    success_hold_steps = 15          # 1 s at 15 Hz
    success_tilt_deg = 3.5           # (also where rew_tilt starts)
    rew_tilt_up = -0.5               # per degree the held tube's tilt rose over the step
    rew_near_speed = -4.0            # per m/s of open-hand TCP speed beyond near_speed_mps while within near_speed_m
    near_speed_mps = 0.10
    near_speed_m = 0.05

    def __post_init__(self):
        # 1,024+ envs of 20 tubes in their bores overflowed PhysX's default GPU collision stack (64 MB) during the
        # first resets ("Contacts have been dropped"); 512 MB covers the ~360 MB it asked for
        post = getattr(super(), "__post_init__", None)
        if post:
            post()
        from isaaclab_physx.physics import PhysxCfg
        if getattr(self.sim, "physics", None) is None:
            self.sim.physics = PhysxCfg()
        self.sim.physics.gpu_collision_stack_size = 2 ** 29


@configclass
class TubeGraspJointDREnvCfg(TubeGraspJointEnvCfg):
    """Domain-randomized fine-tune of the joint-space tube pick, the first step towards a physical arm: none of the
    runs before randomized the robot or the physics beyond the camera-pose noise.

    Per env, once at start-up (Isaac Lab events): arm joint stiffness x0.75-1.25 and damping x0.75-1.5 (joint
    impedance tuning differs from the sim's), finger stiffness x0.6-1.4 (grip force), friction of the finger pads
    (static 0.5-1.2; PhysX averages it with the tube's). Not the tubes' own friction or mass: Isaac Lab's
    material and mass events don't support the tubes' RigidObjectCollection. Per episode: control latency 0-1 steps (0-67 ms at 15 Hz),
    joint-encoder offsets up to 3 mrad, a TCP offset up to 2 mm (calibration); per step: joint noise 1 mrad, joint
    velocity noise 0.02 rad/s."""
    dr_latency_steps = 1
    dr_q_offset_rad = 0.003
    dr_q_noise_rad = 0.001
    dr_qd_noise = 0.02
    dr_tcp_offset_m = 0.002

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.envs.mdp import events as ev
        from isaaclab.managers import EventTermCfg, SceneEntityCfg

        @configclass
        class DREvents:
            arm_gains = EventTermCfg(func=ev.randomize_actuator_gains, mode="startup", params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-7]"]),
                "stiffness_distribution_params": (0.75, 1.25), "damping_distribution_params": (0.75, 1.5),
                "operation": "scale", "distribution": "uniform"})
            finger_gains = EventTermCfg(func=ev.randomize_actuator_gains, mode="startup", params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["finger_joint.*"]),
                "stiffness_distribution_params": (0.6, 1.4), "damping_distribution_params": (0.8, 1.2),
                "operation": "scale", "distribution": "uniform"})
            pad_friction = EventTermCfg(func=ev.randomize_rigid_body_material, mode="startup", params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[".*finger.*"]),
                "static_friction_range": (0.5, 1.2), "dynamic_friction_range": (0.4, 1.0),
                "restitution_range": (0.0, 0.05), "num_buckets": 64, "make_consistent": True})

        self.events = DREvents()


__all__ = ["TubeGraspJointEnvCfg", "TubeGraspJointDREnvCfg"]

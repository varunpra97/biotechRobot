"""Joint-space centrifuge remove (no IK): from an approach up to 35 cm out (or, as a reverse curriculum, already
holding the tube part-way out), the policy commands the 7 arm joints and the gripper to grasp a tube seated in a
45 deg rotor bore, draw it out along the bore and lift it clear of the rotor, upright. The scene is tube_insert's
(bench, centrifuge with its exact collider, one tube); the recorded expert removals give the starts
(start_states_remove.npz).
"""
from __future__ import annotations

from isaaclab.utils.configclass import configclass

from tube_insert.env_cfg import TubeInsertEnvCfg


@configclass
class TubeRemoveJointEnvCfg(TubeInsertEnvCfg):
    episode_length_s = 12.0          # 180 steps: approach from up to 35 cm, grasp, extract, lift
    action_space = 8                 # 7 joint-target changes (x joint_step_rad) + gripper
    observation_space = 46
    # joint space (rl/joint_space.py)
    joint_step_rad = 0.015
    joint_gripper = True
    grip_threshold = -0.8            # untrained gripper biased towards closed (tube pick run 9), z = -2 at std 0.4
    start_closed = False             # held starts set it per env
    # Phi(s') - Phi(s), undiscounted: with gamma 0.99 waiting cost (1 - gamma) Phi per step, ~-18 per episode near the
    # target and nothing far away, and remove run 1's approaches learned to swing the arm away (85 cm)
    shaping_gamma = 1.0
    rew_far = 3.0                    # 1 - tanh(distance / 40 cm): a pull from the far starts (up to 30-60 cm)
    # share of resets from a held, part-extracted frame. 0.4 in runs 1-2: held starts went to 86% but the policy
    # carried their pull-out-and-lift motion into the approaches (deterministic: hand 1.2 m away at timeout)
    held_p = 0.2
    # geometry and tests
    remove_grasp_depth = 0.010       # expert.REMOVE_GRASP_DEPTH: TCP 10 mm below the rim
    clear_s = 0.015                  # tip this far beyond the bore's opening: out (expert's tip_clear_s)
    lift_goal_m = 0.02               # tip this far above the clear point (height) for success
    success_tilt_deg = 8.0           # expert's test
    success_hold_steps = 10
    hold_gap_m = (0.003, 0.0075)
    hold_max_dist_m = 0.015
    lost_m = 0.40
    # potential Phi(s)
    rew_reach = 3.0                  # 1 - tanh(|TCP - grasp point| / 10 cm)
    rew_near = 3.0                   # ... / 2 cm
    rew_fine = 3.0                   # ... / 5 mm
    rew_align = 3.0                  # hand z along -tube axis: 1 - tanh(angle / 0.15 rad), faded with distance
    # run 3: the expert's approach -- onto the tube's axis above the grasp point, then down it -- (a scripted test
    # motion straight to the grasp point knocked half the tubes out of their bores), closing along the rotor's
    # tangent (expert.closing_for_bore; any other closing axis hits the rotor)
    rew_line = 3.0                   # 1 - tanh(distance from the tube's axis line / 1 cm), faded with distance
    rew_line_fine = 3.0              # ... / 2 mm
    rew_down = 3.0                   # while on the line (lat < line_m): 1 - tanh(height above the grasp point / 2 cm)
    line_m = 0.003
    rew_closing = 3.0                # (hand y . rotor tangent)^2, faded with distance
    rew_grasp = 5.0                  # held
    rew_extract = 8.0                # held, x tip progress from seated to clear along the bore axis
    rew_lift = 6.0                   # held and clear, x lift / lift_goal_m (capped), x upright
    upright_zero_deg = 20.0
    # terminal and per-step terms
    rew_success = 100.0
    rew_fail = -10.0                 # dropped out of the bore, fallen, lost (-20 in runs 1-2: retreating looked better
                                     # than trying)
    rew_tilt = -0.05                 # per degree of held-tube tilt beyond success_tilt_deg once clear
    rew_sideways = -2.0              # per m/s of TCP speed across the bore axis while the held tube is still in it
    rew_early_close = -0.2           # gripper closed more than 1.5 cm from the grasp point
    rew_action_rate = -0.005

    def __post_init__(self):
        post = getattr(super(), "__post_init__", None)
        if post:
            post()
        from isaaclab_physx.physics import PhysxCfg
        if getattr(self.sim, "physics", None) is None:
            self.sim.physics = PhysxCfg()
        self.sim.physics.gpu_collision_stack_size = 2 ** 28


__all__ = ["TubeRemoveJointEnvCfg"]

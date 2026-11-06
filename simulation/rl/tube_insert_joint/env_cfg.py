"""Joint-space centrifuge insert (no IK): the policy commands the 7 arm joints and the gripper to carry a held tube
down a 45 deg rotor bore (8 mm, ~0.5 mm clearance), release it seated and back the hand off. Otherwise as full
milestone M5 (tube_insert's TubeInsertM5EnvCfg): bores 1-3, holds from up to 30 cm. The IK teacher aimed the
tube's axis along the bore's itself; here that alignment is learned too.
"""
from __future__ import annotations

from isaaclab.utils.configclass import configclass

from tube_insert.env_cfg import TubeInsertM5EnvCfg


@configclass
class TubeInsertJointEnvCfg(TubeInsertM5EnvCfg):
    episode_length_s = 12.0          # 180 steps: carry from up to 30 cm, align, insert, release, back off
    action_space = 8                 # 7 joint-target changes (x joint_step_rad) + gripper
    observation_space = 49           # tube_insert's 33 with the 8-D last action, + tube and bore axes, hand z and y
    # joint space (rl/joint_space.py)
    joint_step_rad = 0.015           # 0.22 rad/s: a held tube slips in the pads at ~15 cm/s hand speed
    joint_gripper = True
    # run 1 biased the gripper towards closed (opens below -1.5): no open was ever sampled, the policy held still for
    # 300 iterations, seated tubes included. Run 2: neutral, with the release interlock (env._release_allowed: open
    # only with the tip inside the bore) so that the carry can't drop the tube
    grip_threshold = 0.0
    start_closed = True
    # Phi(s') - Phi(s), undiscounted: with gamma 0.99 waiting cost (1 - gamma) Phi per step, ~-18 per episode near the
    # target and nothing far away, and remove run 1's approaches learned to swing the arm away (85 cm)
    shaping_gamma = 1.0
    rew_far = 3.0                    # 1 - tanh(distance / 40 cm): a pull from the far starts (up to 30-60 cm)
    # reverse curriculum: this share of resets starts from a held tube already inside the bore
    # (start_states_insert_deep.npz, expert frames from just inside down to seated), so the release and the
    # back-off are met early
    deep_p = 0.2
    # run 3: and this share from the carry library's holds lined up at the bore's entry (env: within 1.5 mm and 4 deg
    # of the axis, the tip from entry_s_m below to entry_above_m above the opening)
    entry_p = 0.4
    entry_s_m = 0.004
    entry_above_m = 0.02
    # run 3: in the entry zone (on the axis within zone_radial_m and zone_deg, from the seat to zone_above_m above the
    # opening, held): the tube's speed down the axis (capped at insert_vmps) pays, the hand's speed across it costs
    rew_insert_v = 2.0
    insert_vmps = 0.05
    rew_across = -2.0
    zone_radial_m = 0.002
    zone_deg = 6.0
    zone_above_m = 0.025
    # hold test: pads on the 7 mm tube read 3.8-5.8 mm here; tube_insert's (4.5, 7.5) mm called ~half of real holds
    # "not held", so a tube carried tilted (up to 48 deg off the bore axis in the M5 library) counted as fallen
    hold_gap_m = (0.003, 0.0075)
    # "lost": tip this far from its release point. tube_insert's 0.2 m failed the M5 library's holds from 20-30 cm
    # at the first step
    lost_m = 0.40
    # potential Phi(s) (env._potential): carry, on-axis, coaxial, depth, in bore, released seated, backed off
    rew_carry = 4.0                  # 1 - tanh(|tip - release point| / 10 cm)
    rew_carry_near = 3.0             # 1 - tanh(|tip - release point| / 2 cm)
    rew_radial = 2.0                 # 1 - tanh(radial / 5 mm)
    rew_radial_fine = 3.0            # 1 - tanh(radial / 1 mm)
    rew_coaxial = 4.0                # 1 - tanh(axis angle / 10 deg)
    rew_coaxial_wide = 3.0           # (1 + cos axis angle) / 2: a slope at the carry starts' 20-48 deg, where the term above is flat
    rew_deep = 4.0                   # 1 - tanh(depth to go / 3 cm), only on the axis
    rew_in_bore = 4.0
    rew_released = 6.0               # released seated
    rew_backoff = 4.0                # released seated, x min(|TCP - grasp point| / retreat_m, 1)
    # terminal and per-step terms
    success_hold_steps = 5           # the success test (released seated, coaxial, backed off) held this long
    rew_success = 100.0
    rew_fail = -20.0                 # dropped / fell over / lost
    rew_open_outside = -0.2          # gripper open while the tube is not seated
    rew_fast_near = -2.0             # per m/s of tube speed beyond fast_mps within near_m of the bore opening
    fast_mps = 0.08
    near_m = 0.04
    rew_action_rate = -0.005

    def __post_init__(self):
        post = getattr(super(), "__post_init__", None)
        if post:
            post()
        from isaaclab_physx.physics import PhysxCfg
        if getattr(self.sim, "physics", None) is None:
            self.sim.physics = PhysxCfg()
        self.sim.physics.gpu_collision_stack_size = 2 ** 28


__all__ = ["TubeInsertJointEnvCfg"]

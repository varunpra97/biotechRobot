"""PPO (rsl_rl) settings for the joint-space centrifuge-insert teacher: tube_grasp's with a smaller initial
action noise (0.4): inside the 0.5 mm-clearance bore a held tube is levered out by ~1 cm of hand drift, which
random joint targets at 0.6 reach in a few seconds (diagnostic: 52% of in-bore holds lost at noise 0.1 x 4 s)."""
from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from tube_grasp.agents import TubeGraspPPORunnerCfg


@configclass
class TubeInsertJointPPORunnerCfg(TubeGraspPPORunnerCfg):
    experiment_name = "tube_insert_joint"
    max_iterations = 3000
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.4),
    )

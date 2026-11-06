"""PPO (rsl_rl) settings for the joint-space tip attach teacher: tube_grasp's, initial action noise 0.4."""
from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from tube_grasp.agents import TubeGraspPPORunnerCfg


@configclass
class TipAttachJointPPORunnerCfg(TubeGraspPPORunnerCfg):
    experiment_name = "tip_attach_joint"
    max_iterations = 3000
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.4),
    )

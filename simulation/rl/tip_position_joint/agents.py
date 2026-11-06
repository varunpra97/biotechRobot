"""PPO (rsl_rl) settings for the joint-space tip positioning teacher: tube_grasp's, initial action noise 0.3 (starts inside a tube touch its wall under 0.4 within 2-3 steps)."""
from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from tube_grasp.agents import TubeGraspPPORunnerCfg


@configclass
class TipPositionJointPPORunnerCfg(TubeGraspPPORunnerCfg):
    experiment_name = "tip_position_joint"
    max_iterations = 3000
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3),
    )

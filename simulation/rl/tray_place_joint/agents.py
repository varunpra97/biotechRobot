"""PPO (rsl_rl) settings for the joint-space tray placement teacher: tube_grasp's, initial action noise 0.4."""
from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from tube_grasp.agents import TubeGraspPPORunnerCfg


@configclass
class TrayPlaceJointPPORunnerCfg(TubeGraspPPORunnerCfg):
    experiment_name = "tray_place_joint"
    max_iterations = 3000
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.4),
    )

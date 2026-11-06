"""PPO (rsl_rl) settings for the joint-space tube-pick teacher: tube_grasp's, a longer run."""
from isaaclab.utils.configclass import configclass

from tube_grasp.agents import TubeGraspPPORunnerCfg


@configclass
class TubeGraspJointPPORunnerCfg(TubeGraspPPORunnerCfg):
    experiment_name = "tube_grasp_joint"
    max_iterations = 4000

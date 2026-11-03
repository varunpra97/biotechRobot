"""PPO (rsl_rl) settings for the tube-place teacher (as tube_grasp)."""
from isaaclab.utils.configclass import configclass

from tube_grasp.agents import TubeGraspPPORunnerCfg


@configclass
class TubePlacePPORunnerCfg(TubeGraspPPORunnerCfg):
    experiment_name = "tube_place"
    max_iterations = 1000

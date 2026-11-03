"""PPO (rsl_rl) settings for the pipette-grasp teacher (as tube_grasp)."""
from isaaclab.utils.configclass import configclass

from tube_grasp.agents import TubeGraspPPORunnerCfg


@configclass
class PipetteGraspPPORunnerCfg(TubeGraspPPORunnerCfg):
    experiment_name = "pipette_grasp"
    max_iterations = 1000

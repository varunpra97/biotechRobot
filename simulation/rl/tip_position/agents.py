"""PPO (rsl_rl) settings for the tip-positioning teacher (as tube_grasp)."""
from isaaclab.utils.configclass import configclass

from tube_grasp.agents import TubeGraspPPORunnerCfg


@configclass
class TipPositionPPORunnerCfg(TubeGraspPPORunnerCfg):
    experiment_name = "tip_position"
    max_iterations = 600

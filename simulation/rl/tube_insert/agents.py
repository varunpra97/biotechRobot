"""PPO (rsl_rl) settings for the centrifuge-insert teacher (as tube_grasp)."""
from isaaclab.utils.configclass import configclass

from tube_grasp.agents import TubeGraspPPORunnerCfg


@configclass
class TubeInsertPPORunnerCfg(TubeGraspPPORunnerCfg):
    experiment_name = "tube_insert"
    max_iterations = 1000

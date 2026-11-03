"""PPO (rsl_rl) settings for the tray-grasp teacher (as tube_grasp)."""
from isaaclab.utils.configclass import configclass

from tube_grasp.agents import TubeGraspPPORunnerCfg


@configclass
class TrayGraspPPORunnerCfg(TubeGraspPPORunnerCfg):
    experiment_name = "tray_grasp"
    max_iterations = 1000

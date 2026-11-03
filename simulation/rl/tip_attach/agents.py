"""PPO (rsl_rl) settings for the attach-tip teacher (as tube_grasp)."""
from isaaclab.utils.configclass import configclass

from tube_grasp.agents import TubeGraspPPORunnerCfg


@configclass
class TipAttachPPORunnerCfg(TubeGraspPPORunnerCfg):
    experiment_name = "tip_attach"
    max_iterations = 600

"""Pipette-grasp RL task (mini milestone m3), gym id "Biotech-PipetteGrasp-v0".
Train like tube_grasp with --task Biotech-PipetteGrasp-v0 --external_callback pipette_grasp.register."""
import gymnasium as gym


def register():
    gym.register(
        id="Biotech-PipetteGrasp-v0",
        entry_point="pipette_grasp.env:PipetteGraspEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "pipette_grasp.env_cfg:PipetteGraspEnvCfg",
            "rsl_rl_cfg_entry_point": "pipette_grasp.agents:PipetteGraspPPORunnerCfg",
        },
    )
    return None

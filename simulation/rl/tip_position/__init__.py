"""Tip-positioning RL task (full milestone M6: aspirate / dispense), gym id "Biotech-TipPosition-v0".
Train like tube_grasp with --task Biotech-TipPosition-v0 --external_callback tip_position.register."""
import gymnasium as gym


def register():
    gym.register(
        id="Biotech-TipPosition-v0",
        entry_point="tip_position.env:TipPositionEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tip_position.env_cfg:TipPositionEnvCfg",
            "rsl_rl_cfg_entry_point": "tip_position.agents:TipPositionPPORunnerCfg",
        },
    )
    return None

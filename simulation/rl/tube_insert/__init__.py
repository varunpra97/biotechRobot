"""Centrifuge-insert RL task (mini milestone m5), gym id "Biotech-TubeInsert-v0".
Train like tube_grasp with --task Biotech-TubeInsert-v0 --external_callback tube_insert.register."""
import gymnasium as gym


def register():
    gym.register(
        id="Biotech-TubeInsert-v0",
        entry_point="tube_insert.env:TubeInsertEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tube_insert.env_cfg:TubeInsertEnvCfg",
            "rsl_rl_cfg_entry_point": "tube_insert.agents:TubeInsertPPORunnerCfg",
        },
    )
    gym.register(                    # full milestone M5: bores 1-3, holds up to 30 cm out
        id="Biotech-TubeInsert-M5-v0",
        entry_point="tube_insert.env:TubeInsertEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tube_insert.env_cfg:TubeInsertM5EnvCfg",
            "rsl_rl_cfg_entry_point": "tube_insert.agents:TubeInsertPPORunnerCfg",
        },
    )
    return None

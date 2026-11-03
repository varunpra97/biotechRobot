"""Tube-place RL task (mini milestone m4), gym id "Biotech-TubePlace-v0".
Train like tube_grasp with --task Biotech-TubePlace-v0 --external_callback tube_place.register."""
import gymnasium as gym


def register():
    gym.register(
        id="Biotech-TubePlace-v0",
        entry_point="tube_place.env:TubePlaceEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tube_place.env_cfg:TubePlaceEnvCfg",
            "rsl_rl_cfg_entry_point": "tube_place.agents:TubePlacePPORunnerCfg",
        },
    )
    gym.register(                    # full milestone M4: seated while held, scripted release
        id="Biotech-TubePlace-M4-v0",
        entry_point="tube_place.env:TubePlaceEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tube_place.env_cfg:TubePlaceM4EnvCfg",
            "rsl_rl_cfg_entry_point": "tube_place.agents:TubePlacePPORunnerCfg",
        },
    )
    return None

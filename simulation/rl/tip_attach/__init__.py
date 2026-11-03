"""Attach-tip RL task (mini milestone m6), gym id "Biotech-TipAttach-v0".
Train like tube_grasp with --task Biotech-TipAttach-v0 --external_callback tip_attach.register."""
import gymnasium as gym


def register():
    gym.register(
        id="Biotech-TipAttach-v0",
        entry_point="tip_attach.env:TipAttachEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tip_attach.env_cfg:TipAttachEnvCfg",
            "rsl_rl_cfg_entry_point": "tip_attach.agents:TipAttachPPORunnerCfg",
        },
    )
    gym.register(                    # full milestone M6: any tip of the rack
        id="Biotech-TipAttach-M6-v0",
        entry_point="tip_attach.env:TipAttachEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tip_attach.env_cfg:TipAttachM6EnvCfg",
            "rsl_rl_cfg_entry_point": "tip_attach.agents:TipAttachPPORunnerCfg",
        },
    )
    return None

"""Tray-grasp RL task (mini milestone m2), gym id "Biotech-TrayGrasp-v0".
Train like tube_grasp with --task Biotech-TrayGrasp-v0 --external_callback tray_grasp.register."""
import gymnasium as gym


def register():
    gym.register(
        id="Biotech-TrayGrasp-v0",
        entry_point="tray_grasp.env:TrayGraspEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tray_grasp.env_cfg:TrayGraspEnvCfg",
            "rsl_rl_cfg_entry_point": "tray_grasp.agents:TrayGraspPPORunnerCfg",
        },
    )
    return None

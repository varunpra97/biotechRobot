"""Tube-grasp RL task (Isaac Lab 3.0).

Registered as gym id "Biotech-TubeGrasp-v0". Train with Isaac Lab's rsl_rl
script, which calls register() before launching the app:

    cd IsaacLab
    isaaclab.bat -p scripts\\reinforcement_learning\\rsl_rl\\train.py --task Biotech-TubeGrasp-v0 ^
        --external_callback tube_grasp.register --headless [--num_envs 256]

(PYTHONPATH must contain biotech-lab-robotics\\simulation\\rl.)
"""
import gymnasium as gym


def register():
    gym.register(
        id="Biotech-TubeGrasp-v0",
        entry_point="tube_grasp.env:TubeGraspEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tube_grasp.env_cfg:TubeGraspEnvCfg",
            "rsl_rl_cfg_entry_point": "tube_grasp.agents:TubeGraspPPORunnerCfg",
        },
    )
    gym.register(                    # full milestone M1: 25 cm hand-overs, all slots, perceived tube pose
        id="Biotech-TubeGrasp-M1-v0",
        entry_point="tube_grasp.env:TubeGraspEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tube_grasp.env_cfg:TubeGraspM1EnvCfg",
            "rsl_rl_cfg_entry_point": "tube_grasp.agents:TubeGraspPPORunnerCfg",
        },
    )
    return None

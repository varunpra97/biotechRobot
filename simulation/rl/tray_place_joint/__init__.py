"""Joint-space tray placement RL task (no IK), gym id "Biotech-TrayPlaceJoint-v0".
Train with --task Biotech-TrayPlaceJoint-v0 --external_callback tray_place_joint.register."""
import gymnasium as gym


def register():
    gym.register(
        id="Biotech-TrayPlaceJoint-v0",
        entry_point="tray_place_joint.env:TrayPlaceJointEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tray_place_joint.env_cfg:TrayPlaceJointEnvCfg",
            "rsl_rl_cfg_entry_point": "tray_place_joint.agents:TrayPlaceJointPPORunnerCfg",
        },
    )
    return None

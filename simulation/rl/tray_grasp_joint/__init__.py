"""Joint-space tray pick-up RL task (no IK), gym id "Biotech-TrayGraspJoint-v0".
Train with --task Biotech-TrayGraspJoint-v0 --external_callback tray_grasp_joint.register."""
import gymnasium as gym


def register():
    gym.register(
        id="Biotech-TrayGraspJoint-v0",
        entry_point="tray_grasp_joint.env:TrayGraspJointEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tray_grasp_joint.env_cfg:TrayGraspJointEnvCfg",
            "rsl_rl_cfg_entry_point": "tray_grasp_joint.agents:TrayGraspJointPPORunnerCfg",
        },
    )
    return None

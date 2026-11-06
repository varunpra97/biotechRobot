"""Joint-space tube-pick RL task (no-IK pilot), gym id "Biotech-TubeGraspJoint-v0".
Train with --task Biotech-TubeGraspJoint-v0 --external_callback tube_grasp_joint.register."""
import gymnasium as gym


def register():
    gym.register(
        id="Biotech-TubeGraspJoint-v0",
        entry_point="tube_grasp_joint.env:TubeGraspJointEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tube_grasp_joint.env_cfg:TubeGraspJointEnvCfg",
            "rsl_rl_cfg_entry_point": "tube_grasp_joint.agents:TubeGraspJointPPORunnerCfg",
        },
    )
    gym.register(
        id="Biotech-TubeGraspJointDR-v0",
        entry_point="tube_grasp_joint.env:TubeGraspJointEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tube_grasp_joint.env_cfg:TubeGraspJointDREnvCfg",
            "rsl_rl_cfg_entry_point": "tube_grasp_joint.agents:TubeGraspJointPPORunnerCfg",
        },
    )
    return None

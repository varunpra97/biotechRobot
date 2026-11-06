"""Joint-space pipette pick-up RL task (no IK), gym id "Biotech-PipetteGraspJoint-v0".
Train with --task Biotech-PipetteGraspJoint-v0 --external_callback pipette_grasp_joint.register."""
import gymnasium as gym


def register():
    gym.register(
        id="Biotech-PipetteGraspJoint-v0",
        entry_point="pipette_grasp_joint.env:PipetteGraspJointEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "pipette_grasp_joint.env_cfg:PipetteGraspJointEnvCfg",
            "rsl_rl_cfg_entry_point": "pipette_grasp_joint.agents:PipetteGraspJointPPORunnerCfg",
        },
    )
    return None

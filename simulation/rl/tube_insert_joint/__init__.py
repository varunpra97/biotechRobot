"""Joint-space centrifuge-insert RL task (no IK), gym id "Biotech-TubeInsertJoint-v0".
Train with --task Biotech-TubeInsertJoint-v0 --external_callback tube_insert_joint.register."""
import gymnasium as gym


def register():
    gym.register(
        id="Biotech-TubeInsertJoint-v0",
        entry_point="tube_insert_joint.env:TubeInsertJointEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tube_insert_joint.env_cfg:TubeInsertJointEnvCfg",
            "rsl_rl_cfg_entry_point": "tube_insert_joint.agents:TubeInsertJointPPORunnerCfg",
        },
    )
    return None

"""Joint-space centrifuge-remove RL task (no IK), gym id "Biotech-TubeRemoveJoint-v0".
Train with --task Biotech-TubeRemoveJoint-v0 --external_callback tube_remove_joint.register."""
import gymnasium as gym


def register():
    gym.register(
        id="Biotech-TubeRemoveJoint-v0",
        entry_point="tube_remove_joint.env:TubeRemoveJointEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tube_remove_joint.env_cfg:TubeRemoveJointEnvCfg",
            "rsl_rl_cfg_entry_point": "tube_remove_joint.agents:TubeRemoveJointPPORunnerCfg",
        },
    )
    return None

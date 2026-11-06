"""Joint-space tip attach RL task (no IK), gym id "Biotech-TipAttachJoint-v0".
Train with --task Biotech-TipAttachJoint-v0 --external_callback tip_attach_joint.register."""
import gymnasium as gym


def register():
    gym.register(
        id="Biotech-TipAttachJoint-v0",
        entry_point="tip_attach_joint.env:TipAttachJointEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tip_attach_joint.env_cfg:TipAttachJointEnvCfg",
            "rsl_rl_cfg_entry_point": "tip_attach_joint.agents:TipAttachJointPPORunnerCfg",
        },
    )
    return None

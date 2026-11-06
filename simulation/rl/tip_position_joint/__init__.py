"""Joint-space tip positioning RL task (no IK), gym id "Biotech-TipPositionJoint-v0".
Train with --task Biotech-TipPositionJoint-v0 --external_callback tip_position_joint.register."""
import gymnasium as gym


def register():
    gym.register(
        id="Biotech-TipPositionJoint-v0",
        entry_point="tip_position_joint.env:TipPositionJointEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "tip_position_joint.env_cfg:TipPositionJointEnvCfg",
            "rsl_rl_cfg_entry_point": "tip_position_joint.agents:TipPositionJointPPORunnerCfg",
        },
    )
    return None

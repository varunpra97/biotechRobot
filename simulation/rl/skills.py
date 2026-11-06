"""The RL skills (one Isaac Lab task each): name -> (package, env class, env cfg class, PPO runner cfg class).
Imported lazily (after the Isaac Lab app is up) by export_teacher.py and eval_skill.py."""
import importlib

SKILLS = {
    "tube": ("tube_grasp", "TubeGraspEnv", "TubeGraspEnvCfg", "TubeGraspPPORunnerCfg"),
    "tube_m1": ("tube_grasp", "TubeGraspEnv", "TubeGraspM1EnvCfg", "TubeGraspPPORunnerCfg"),
    "tube_joint": ("tube_grasp_joint", "TubeGraspJointEnv", "TubeGraspJointEnvCfg", "TubeGraspJointPPORunnerCfg"),
    "tube_joint_dr": ("tube_grasp_joint", "TubeGraspJointEnv", "TubeGraspJointDREnvCfg", "TubeGraspJointPPORunnerCfg"),
    "tray": ("tray_grasp", "TrayGraspEnv", "TrayGraspEnvCfg", "TrayGraspPPORunnerCfg"),
    "tray_joint": ("tray_grasp_joint", "TrayGraspJointEnv", "TrayGraspJointEnvCfg", "TrayGraspJointPPORunnerCfg"),
    "tray_place_joint": ("tray_place_joint", "TrayPlaceJointEnv", "TrayPlaceJointEnvCfg", "TrayPlaceJointPPORunnerCfg"),
    "pipette": ("pipette_grasp", "PipetteGraspEnv", "PipetteGraspEnvCfg", "PipetteGraspPPORunnerCfg"),
    "pipette_joint": ("pipette_grasp_joint", "PipetteGraspJointEnv", "PipetteGraspJointEnvCfg",
                      "PipetteGraspJointPPORunnerCfg"),
    "place": ("tube_place", "TubePlaceEnv", "TubePlaceEnvCfg", "TubePlacePPORunnerCfg"),
    "place_m4": ("tube_place", "TubePlaceEnv", "TubePlaceM4EnvCfg", "TubePlacePPORunnerCfg"),
    "insert": ("tube_insert", "TubeInsertEnv", "TubeInsertEnvCfg", "TubeInsertPPORunnerCfg"),
    "insert_m5": ("tube_insert", "TubeInsertEnv", "TubeInsertM5EnvCfg", "TubeInsertPPORunnerCfg"),
    "insert_joint": ("tube_insert_joint", "TubeInsertJointEnv", "TubeInsertJointEnvCfg", "TubeInsertJointPPORunnerCfg"),
    "remove_joint": ("tube_remove_joint", "TubeRemoveJointEnv", "TubeRemoveJointEnvCfg", "TubeRemoveJointPPORunnerCfg"),
    "tip": ("tip_attach", "TipAttachEnv", "TipAttachEnvCfg", "TipAttachPPORunnerCfg"),
    "tip_m6": ("tip_attach", "TipAttachEnv", "TipAttachM6EnvCfg", "TipAttachPPORunnerCfg"),
    "pipetting": ("tip_position", "TipPositionEnv", "TipPositionEnvCfg", "TipPositionPPORunnerCfg"),
    "tip_joint": ("tip_attach_joint", "TipAttachJointEnv", "TipAttachJointEnvCfg", "TipAttachJointPPORunnerCfg"),
    "pipetting_joint": ("tip_position_joint", "TipPositionJointEnv", "TipPositionJointEnvCfg", "TipPositionJointPPORunnerCfg"),
}


def load(name):
    """(env class, env cfg class, runner cfg class) of a skill."""
    pkg, env, cfg, runner = SKILLS[name]
    return (getattr(importlib.import_module(f"{pkg}.env"), env), getattr(importlib.import_module(f"{pkg}.env_cfg"), cfg),
            getattr(importlib.import_module(f"{pkg}.agents"), runner))

"""The RL skills (one Isaac Lab task each): name -> (package, env class, env cfg class, PPO runner cfg class).
Imported lazily (after the Isaac Lab app is up) by export_teacher.py and eval_skill.py."""
import importlib

SKILLS = {
    "tube": ("tube_grasp", "TubeGraspEnv", "TubeGraspEnvCfg", "TubeGraspPPORunnerCfg"),
    "tube_m1": ("tube_grasp", "TubeGraspEnv", "TubeGraspM1EnvCfg", "TubeGraspPPORunnerCfg"),
    "tray": ("tray_grasp", "TrayGraspEnv", "TrayGraspEnvCfg", "TrayGraspPPORunnerCfg"),
    "pipette": ("pipette_grasp", "PipetteGraspEnv", "PipetteGraspEnvCfg", "PipetteGraspPPORunnerCfg"),
    "place": ("tube_place", "TubePlaceEnv", "TubePlaceEnvCfg", "TubePlacePPORunnerCfg"),
    "place_m4": ("tube_place", "TubePlaceEnv", "TubePlaceM4EnvCfg", "TubePlacePPORunnerCfg"),
    "insert": ("tube_insert", "TubeInsertEnv", "TubeInsertEnvCfg", "TubeInsertPPORunnerCfg"),
    "insert_m5": ("tube_insert", "TubeInsertEnv", "TubeInsertM5EnvCfg", "TubeInsertPPORunnerCfg"),
    "tip": ("tip_attach", "TipAttachEnv", "TipAttachEnvCfg", "TipAttachPPORunnerCfg"),
    "tip_m6": ("tip_attach", "TipAttachEnv", "TipAttachM6EnvCfg", "TipAttachPPORunnerCfg"),
    "pipetting": ("tip_position", "TipPositionEnv", "TipPositionEnvCfg", "TipPositionPPORunnerCfg"),
}


def load(name):
    """(env class, env cfg class, runner cfg class) of a skill."""
    pkg, env, cfg, runner = SKILLS[name]
    return (getattr(importlib.import_module(f"{pkg}.env"), env), getattr(importlib.import_module(f"{pkg}.env_cfg"), cfg),
            getattr(importlib.import_module(f"{pkg}.agents"), runner))

"""Export a trained RL teacher (--task: a skills.SKILLS name) to TorchScript (actor + its observation
normalizer), for use outside Isaac Lab (sim/teacher_finisher.py in the eval).

Usage (from the IsaacLab folder, PYTHONPATH=...\\simulation\\rl):
    python ..\\biotech-lab-robotics\\simulation\\rl\\export_teacher.py --checkpoint logs\\...\\model_N.pt
        --out ..\\biotech-lab-robotics\\simulation\\rl\\teacher_policy.pt --headless
"""
import argparse
import importlib.metadata as metadata
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher  # noqa: E402

from skills import SKILLS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--task", choices=sorted(SKILLS), default="tube")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from skills import load  # noqa: E402

Env, EnvCfg, RunnerCfg = load(args.task)
cfg = EnvCfg()
cfg.scene.num_envs = 2
env = Env(cfg)
venv = RslRlVecEnvWrapper(env)
agent = handle_deprecated_rsl_rl_cfg(RunnerCfg(), metadata.version("rsl-rl-lib"))
runner = OnPolicyRunner(venv, agent.to_dict(), log_dir=None, device=env.device)
runner.load(args.checkpoint)
out = Path(args.out)
tmp = out.parent / "_export"
runner.export_policy_to_jit(path=str(tmp), filename="policy.pt")
shutil.move(str(tmp / "policy.pt"), str(out))
shutil.rmtree(tmp, ignore_errors=True)
# sanity: the exported policy matches the runner's deterministic policy on the current observation
pol = runner.get_inference_policy(device=env.device)
obs = venv.get_observations()
jit = torch.jit.load(str(out), map_location=env.device)
with torch.inference_mode():
    a1 = pol(obs)
    a2 = jit(obs["policy"])
print(f"exported -> {out}; obs dim {obs['policy'].shape[-1]}, max |runner - jit| = {float((a1 - a2).abs().max()):.2e}",
      flush=True)
env.close()
app.close()

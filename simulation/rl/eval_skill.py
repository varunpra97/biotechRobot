r"""Evaluate a trained RL teacher in its Isaac Lab task (deterministic actions):
success / failure / timeout over N episodes from the task's own resets.

Usage (from the IsaacLab folder, PYTHONPATH=...\simulation\rl):
    python ..\biotech-lab-robotics\simulation\rl\eval_skill.py --task place
        --checkpoint logs\rsl_rl\tube_place\<run>\model_N.pt [--episodes 100] --headless
"""
import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher  # noqa: E402

from skills import SKILLS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=list(SKILLS), required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--episodes", type=int, default=100, help="one per env")
ap.add_argument("--out", default=None, help="append the result as one JSON line here")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from skills import load  # noqa: E402

Env, EnvCfg, RunnerCfg = load(args.task)
cfg = EnvCfg()
cfg.scene.num_envs = args.episodes
env = Env(cfg)
venv = RslRlVecEnvWrapper(env)
agent = handle_deprecated_rsl_rl_cfg(RunnerCfg(), metadata.version("rsl-rl-lib"))
runner = OnPolicyRunner(venv, agent.to_dict(), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)

# one episode per env (num_envs = episodes): counting the first N to finish would favour quick outcomes.
# Each env's outcome is read in _get_dones, before the auto-reset clears it.
outcome = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)     # -1 running, 0 timeout, 1 success, 2 fail
_get_dones = env._get_dones


def get_dones():
    term, trunc = _get_dones()
    new = (outcome < 0) & (term | trunc)
    outcome[new & env.succeeded] = 1
    outcome[new & env.failed & ~env.succeeded] = 2
    outcome[new & ~env.succeeded & ~env.failed] = 0
    return term, trunc


env._get_dones = get_dones
obs = venv.get_observations()
while (outcome < 0).any():
    with torch.inference_mode():
        obs, _, _, _ = venv.step(policy(obs))
res = {"success": int((outcome == 1).sum()), "fail": int((outcome == 2).sum()), "timeout": int((outcome == 0).sum())}
total = sum(res.values())
out = {"task": args.task, "checkpoint": args.checkpoint, "episodes": total, **res,
       "success_rate": round(res["success"] / max(1, total), 3)}
print("EVAL", json.dumps(out), flush=True)
if args.out:
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(out) + "\n")
env.close()
app.close()

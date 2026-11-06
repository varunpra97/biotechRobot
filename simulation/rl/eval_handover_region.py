r"""Where can the joint-space tube-pick teacher take over? Its Isaac Lab success (deterministic) against the
hand's starting offset from the grasp point, over N approach starts (no held starts): the generator hands
over as soon as pi0.5 brings the hand inside the region this finds.

Usage (from the IsaacLab folder, PYTHONPATH=...\simulation\rl):
    python ..\biotech-lab-robotics\simulation\rl\eval_handover_region.py
        --checkpoint logs\rsl_rl\tube_grasp_joint\<run>\model_N.pt [--episodes 1024] --out region.json --headless
"""
import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--episodes", type=int, default=1024, help="one per env")
ap.add_argument("--out", required=True, help="per-episode start offsets and outcomes (JSON)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from skills import load  # noqa: E402

Env, EnvCfg, RunnerCfg = load("tube_joint")
cfg = EnvCfg()
cfg.scene.num_envs = args.episodes
cfg.held_p = 0.0                                  # approaches only: the hand-over is always open
env = Env(cfg)
venv = RslRlVecEnvWrapper(env)
agent = handle_deprecated_rsl_rl_cfg(RunnerCfg(), metadata.version("rsl-rl-lib"))
runner = OnPolicyRunner(venv, agent.to_dict(), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)

n = env.num_envs
outcome = torch.full((n,), -1, dtype=torch.long, device=env.device)   # -1 running, 0 timeout, 1 success, 2 fail
start = torch.full((n, 3), float("nan"), device=env.device)          # TCP - grasp point at the first live step
_get_dones = env._get_dones


def get_dones():
    term, trunc = _get_dones()
    live = (outcome < 0) & torch.isnan(start[:, 0]) & (env.fresh == 0)   # poses are fresh from the 2nd step on
    if live.any():
        tcp, _ = env._tcp()
        start[live] = (tcp - env._tube_state()[3])[live]
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
rel = start.cpu().numpy()
res = outcome.cpu().numpy()
d = np.linalg.norm(rel, axis=1)
rows = [{"rel": r.tolist(), "d": float(x), "outcome": int(o)} for r, x, o in zip(rel, d, res)]
Path(args.out).write_text(json.dumps({"checkpoint": args.checkpoint, "episodes": rows}))
print(f"{'start |TCP-grasp|':>20} {'n':>5} {'success':>8} {'fail':>6} {'timeout':>8}")
for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 0.08), (0.08, 0.12), (0.12, 0.16), (0.16, 0.20), (0.20, 0.25), (0.25, 1.0)]:
    m = (d >= lo) & (d < hi)
    if m.any():
        print(f"{lo * 100:7.0f} - {hi * 100:3.0f} cm {m.sum():12d} {(res[m] == 1).mean():8.3f} {(res[m] == 2).mean():6.3f} "
              f"{(res[m] == 0).mean():8.3f}", flush=True)
env.close()
app.close()

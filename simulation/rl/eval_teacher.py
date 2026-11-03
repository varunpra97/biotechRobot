"""Evaluate a trained tube-grasp teacher (deterministic actions) and explain
its outcomes: success / failure / timeout per episode, and for every step
where the target tube is lifted >= LIFT_GOAL the tilt, finger gap and
TCP-to-grasp distance -- which success condition blocks it.

Usage (from the IsaacLab folder, PYTHONPATH=...\\simulation\\rl):
    python ..\\biotech-lab-robotics\\simulation\\rl\\eval_teacher.py --checkpoint logs\\rsl_rl\\tube_grasp\\<run>\\model_N.pt
        [--num_envs 128] [--episodes 256] [--library_p 1.0] --headless
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--num_envs", type=int, default=128)
ap.add_argument("--episodes", type=int, default=256)
ap.add_argument("--library_p", type=float, default=None, help="override the share of recorded start states")
ap.add_argument("--start_states", default=None, help="start-state library .npz (default: env_cfg.START_STATES)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from tube_grasp.agents import TubeGraspPPORunnerCfg  # noqa: E402
import tube_grasp.env as tg_env  # noqa: E402
from tube_grasp.env import TubeGraspEnv  # noqa: E402

if args.start_states:
    tg_env.START_STATES = args.start_states
from tube_grasp.env_cfg import TubeGraspEnvCfg  # noqa: E402

cfg = TubeGraspEnvCfg()
cfg.scene.num_envs = args.num_envs
if args.library_p is not None:
    cfg.library_p = args.library_p
env = TubeGraspEnv(cfg)
venv = RslRlVecEnvWrapper(env)
import importlib.metadata as metadata  # noqa: E402

from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg  # noqa: E402

agent_cfg = handle_deprecated_rsl_rl_cfg(TubeGraspPPORunnerCfg(), metadata.version("rsl-rl-lib"))
runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)

obs = venv.get_observations()
outcomes = {"success": 0, "fail": 0, "timeout": 0}
lifted_stats = []                     # (tilt, gap, d, grasped) at steps with lift >= goal
done_eps = 0
while done_eps < args.episodes:
    with torch.inference_mode():
        act = policy(obs)
    d, grasped, lift, fail, success = env._status()
    _, _, _, _, tilt = env._tube_state()
    gap = env._robot.data.joint_pos.torch[:, env.finger_ids].sum(-1)
    m = lift >= cfg.lift_goal
    for i in torch.nonzero(m).flatten().tolist():
        lifted_stats.append((float(tilt[i]), float(gap[i]) * 1000, float(d[i]) * 1000, bool(grasped[i]),
                             bool(env.grip_closed[i])))
    obs, _, dones, extras = venv.step(act)
    fin = torch.nonzero(dones).flatten().tolist()
    for i in fin:
        if env.succeeded[i] if False else False:
            pass
    # outcome of the finished envs (flags are cleared on reset, so read the reset log)
    if fin:
        log = env.extras.get("log", {})
        k = len(fin)
        s = round(log.get("Metrics/success_rate", 0.0) * k)
        f = round(log.get("Metrics/fail_rate", 0.0) * k)
        outcomes["success"] += s
        outcomes["fail"] += f
        outcomes["timeout"] += k - s - f
        done_eps += k
print(f"episodes {done_eps}: {outcomes}  success rate {outcomes['success'] / max(1, done_eps):.2f}", flush=True)
if lifted_stats:
    import numpy as np
    a = np.array([s[:3] for s in lifted_stats])
    g = np.array([s[3] for s in lifted_stats])
    c = np.array([s[4] for s in lifted_stats])
    print(f"steps with the tube lifted >= {cfg.lift_goal * 100:.0f} cm: {len(a)}; of them held-test true {g.mean():.2f}, "
          f"gripper closed {c.mean():.2f}", flush=True)
    for name, col in (("tilt deg", 0), ("finger gap mm", 1), ("TCP-grasp mm", 2)):
        q = np.percentile(a[:, col], [5, 25, 50, 75, 95]).round(2)
        print(f"  {name}: p5/25/50/75/95 {q.tolist()}", flush=True)
    print(f"  fail tilt>={cfg.success_tilt_deg}: {(a[:, 0] >= cfg.success_tilt_deg).mean():.2f}, "
          f"gap outside {cfg.hold_gap_m}: {((a[:, 1] / 1000 <= cfg.hold_gap_m[0]) | (a[:, 1] / 1000 >= cfg.hold_gap_m[1])).mean():.2f}, "
          f"d >= {cfg.hold_max_dist_m * 1000:.0f} mm: {(a[:, 2] / 1000 >= cfg.hold_max_dist_m).mean():.2f}", flush=True)
env.close()
app.close()

"""Smoke test of the tube-grasp env (tube_grasp/): build a few envs, run a
scripted oracle (move to the grasp point from the observation, descend,
close, lift) and report reward / success / failure, so the task is shown to
be solvable before PPO trains on it. Prints the finger gap, TCP-to-grasp
distance and tube height while closing and lifting.

Usage (from the IsaacLab folder, PYTHONPATH=...\\simulation\\rl):
    python ..\\biotech-lab-robotics\\simulation\\rl\\oracle_smoke_test.py --num_envs 16 --steps 240 --headless
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=8)
ap.add_argument("--steps", type=int, default=150)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from tube_grasp.env import TubeGraspEnv  # noqa: E402
from tube_grasp.env_cfg import TubeGraspEnvCfg  # noqa: E402

cfg = TubeGraspEnvCfg()
cfg.scene.num_envs = args.num_envs
env = TubeGraspEnv(cfg)
obs, _ = env.reset()
n = env.num_envs
phase = torch.zeros(n, dtype=torch.long, device=env.device)      # 0 above, 1 descend, 2 close, 3 lift
timer = torch.zeros(n, dtype=torch.long, device=env.device)
ret = torch.zeros(n, device=env.device)
done_eps = 0
for t in range(args.steps):
    rel = obs["policy"][:, :3] / 10.0                           # TCP - grasp point (m)
    a = torch.zeros(n, 4, device=env.device)
    above = rel.clone()
    above[:, 2] -= 0.03                                         # first to 3 cm above the grasp point
    a[:, :3] = torch.where((phase == 0).unsqueeze(-1), -above, -rel) / env.cfg.action_step_m
    a[:, 3] = -1.0
    phase = torch.where((phase == 0) & (above.norm(dim=-1) < 0.003), 1, phase)
    phase = torch.where((phase == 1) & (rel.norm(dim=-1) < 0.0015), 2, phase)
    closing = phase >= 2
    a[closing, :3] = 0.0
    a[closing, 3] = 1.0
    timer = torch.where(closing, timer + 1, timer)
    phase = torch.where((phase == 2) & (timer > 8), 3, phase)
    a[phase == 3, 2] = 0.5
    watch = (phase == 2) & (timer <= 9)
    if watch.any():
        st = env._status()
        tcp, _ = env._tcp()
        _, _, tp, grasp, tilt = env._tube_state()
        e = int(torch.nonzero(watch)[0])
        print(f"      closing env {e} timer {int(timer[e])}: TCP z {float(tcp[e, 2]):.4f}, grasp pt z {float(grasp[e, 2]):.4f}, "
              f"tube tip z {float(tp[e, 2]):.4f}, |TCP-grasp| {float(st[0][e]) * 1000:.1f} mm, "
              f"gap {float(env._robot.data.joint_pos.torch[e, env.finger_ids].sum()) * 1000:.2f} mm, tilt {float(tilt[e]):.1f}",
              flush=True)
    obs, rew, term, trunc, _ = env.step(a.clamp(-1, 1))
    ret += rew
    fin = term | trunc
    if fin.any():
        done_eps += int(fin.sum())
        phase[fin] = 0
        timer[fin] = 0
    if t % 10 == 0:
        st = env._status()
        gap = env._robot.data.joint_pos.torch[:, env.finger_ids].sum(-1)
        lifting = phase == 3
        if lifting.any():
            print(f"      lifting envs: finger gap (mm) {(gap[lifting] * 1000).cpu().numpy().round(2)}, "
                  f"|TCP-grasp| (mm) {(st[0][lifting] * 1000).cpu().numpy().round(1)}, lift (mm) "
                  f"{(st[2][lifting] * 1000).cpu().numpy().round(1)}", flush=True)
        print(f"t={t:3d} mean |TCP-grasp| {st[0].mean() * 100:.2f} cm, grasped {st[1].float().mean():.2f}, "
              f"lift {st[2].mean() * 100:.2f} cm, success {st[4].float().mean():.2f}, fail {st[3].float().mean():.2f}, "
              f"phases {torch.bincount(phase, minlength=4).tolist()}", flush=True)
print("log:", {k: round(float(v), 3) for k, v in env.extras.get("log", {}).items()})
print(f"episodes finished: {done_eps}, mean return so far {ret.mean():.1f}")
env.close()
app.close()

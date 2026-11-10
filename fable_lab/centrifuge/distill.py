r"""Distil a hand-written centrifuge controller (controllers.py, Jacobian servo) into a network that maps the env's
own observation to the joint action -- the RL policies' interface, and no Jacobian or IK at run time. Supervised
imitation with DAgger (Ross et al. 2011), no PPO: round 0 records the controller; in later rounds the network drives
(a shrinking share of envs still driven by the controller) and the controller, following along on the same states,
labels every step with what it would have done. The network is trained on everything gathered so far.

The network sees only the observation. Inside it, before the MLP, a few fixed geometric features are computed from
that observation at millimetre scale (the tip's offset from the bore axis as a vector, the axis error as a vector,
...): the observation carries them at 1 mm = 0.01, too coarse for a 0.7 mm clearance to be regressed from directly.

Usage (from the IsaacLab folder):
    python ..\biotech-lab-robotics\fable_lab\centrifuge\distill.py --task insert_joint --out insert_net.pt --headless
"""
import argparse
import json
import sys
import time
from pathlib import Path

RL = Path(__file__).resolve().parents[2] / "simulation" / "rl"      # the skill envs live there
sys.path.insert(0, str(RL))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["insert_joint", "remove_joint"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--num_envs", type=int, default=1024)
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--steps", type=int, default=200, help="env steps per round")
ap.add_argument("--train_steps", type=int, default=2500)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--max_samples", type=int, default=3_000_000)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--log", default=None)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from controllers import CONTROLLERS  # noqa: E402
from net import PolicyNet  # noqa: E402
from skills import load  # noqa: E402

Env, EnvCfg, _ = load(args.task)
cfg = EnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
env = Env(cfg)
n, dev = env.num_envs, env.device
ctrl = CONTROLLERS[args.task](env)
net = PolicyNet(args.task, cfg.observation_space).to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

done_stats = {}
_get_dones = env._get_dones
student = torch.zeros(n, dtype=torch.bool, device=dev)      # envs the network is driving this episode


def get_dones():
    term, trunc = _get_dones()
    end = term | trunc
    for who, m in (("student", student), ("teacher", ~student)):
        st = done_stats.setdefault(who, [0, 0, 0])
        st[0] += int((end & m & env.succeeded).sum())
        st[1] += int((end & m & env.failed & ~env.succeeded).sum())
        st[2] += int((end & m & ~env.succeeded & ~env.failed).sum())
    return term, trunc


env._get_dones = get_dones
OBS, ACT = [], []
total = 0
obs = env.reset()[0]["policy"]
log = []
for rnd in range(args.rounds):
    beta = 1.0 if rnd == 0 else max(0.0, 0.5 - 0.15 * (rnd - 1))      # share of episodes the controller drives
    student[:] = torch.rand(n, device=dev) >= beta
    done_stats.clear()
    net.eval()
    t0 = time.time()
    for _ in range(args.steps):
        with torch.inference_mode():
            a_t = ctrl.act()
            a_s = net(obs)
            OBS.append(obs.clone())
            ACT.append(a_t.clone())
            obs = env.step(torch.where(student.unsqueeze(-1), a_s, a_t))[0]["policy"]
        fresh = torch.nonzero(env.episode_length_buf == 0).flatten()
        ctrl.reset(fresh)
        student[fresh] = torch.rand(len(fresh), device=dev) >= beta
    total += args.steps * n
    while total > args.max_samples and len(OBS) > args.steps:          # drop the oldest steps
        OBS.pop(0), ACT.pop(0)
        total -= n
    X, Y = torch.cat(OBS), torch.cat(ACT)                   # (made outside inference mode: ordinary tensors)
    if rnd == 0:
        net.set_norm(X)
    net.train()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.train_steps, eta_min=1e-4)
    for g in opt.param_groups:
        g["lr"] = 1e-3
    with torch.inference_mode(False), torch.enable_grad():
        Xc, Yc = X, Y
        for it in range(args.train_steps):
            idx = torch.randint(0, len(Xc), (args.batch,), device=dev)
            joints, grip = net.raw(Xc[idx])
            loss_j = nn.functional.smooth_l1_loss(joints, Yc[idx, :7], beta=0.05)
            loss_g = nn.functional.binary_cross_entropy_with_logits(grip, (Yc[idx, 7] > 0).float())
            loss = loss_j + 0.2 * loss_g
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    row = {"round": rnd, "beta": beta, "samples": len(X), "loss_joints": round(float(loss_j), 5),
           "loss_grip": round(float(loss_g), 5), "rollout_s": round(time.time() - t0, 1),
           **{k: {"success": v[0], "fail": v[1], "timeout": v[2],
                  "rate": round(v[0] / max(1, sum(v)), 3)} for k, v in done_stats.items()}}
    log.append(row)
    print("DISTILL", json.dumps(row), flush=True)
    net.eval()
    torch.jit.script(net).save(args.out)                    # the latest network, every round
if args.log:
    Path(args.log).write_text(json.dumps(log, indent=1))
env.close()
app.close()

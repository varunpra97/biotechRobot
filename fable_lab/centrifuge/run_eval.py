r"""Evaluate a hand-written controller (controllers.py) or its distilled network (distill.py) in the real-centrifuge
joint-space envs, the way eval_skill.py evaluates a PPO checkpoint: one episode per env, the env's own resets,
success / fail / timeout from the env's own tests. Adds the outcome by start kind and where failures ended.

Usage (from the IsaacLab folder):
    python ..\biotech-lab-robotics\fable_lab\centrifuge\run_eval.py --task insert_joint --episodes 100 --headless
        [--policy net.pt] [--seed 0] [--out results.jsonl] [--set name=value] [--param name=value] [--show 8]
"""
import argparse
import json
import sys
from pathlib import Path

RL = Path(__file__).resolve().parents[2] / "simulation" / "rl"      # the skill envs live there
sys.path.insert(0, str(RL))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["insert_joint", "remove_joint"], required=True)
ap.add_argument("--episodes", type=int, default=100, help="one per env")
ap.add_argument("--policy", default=None, help="distilled network (TorchScript, obs -> action) instead of the controller")
ap.add_argument("--checkpoint", default=None, help="a PPO checkpoint (rsl_rl) instead: the RL baseline on the same starts")
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--out", default=None, help="append the result as one JSON line here")
ap.add_argument("--set", action="append", default=[], help="env cfg overrides, name=value")
ap.add_argument("--param", action="append", default=[], help="controller parameter overrides, name=value")
ap.add_argument("--show", type=int, default=8, help="print this many non-successes")
ap.add_argument("--trace", default="", help="env ids (comma separated) whose state is printed every --trace-every steps")
ap.add_argument("--trace-every", type=int, default=5)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from isaaclab.utils.math import quat_apply  # noqa: E402

from controllers import CONTROLLERS  # noqa: E402
from skills import load  # noqa: E402

Env, EnvCfg, RunnerCfg = load(args.task)
cfg = EnvCfg()
cfg.scene.num_envs = args.episodes
if args.seed is not None:
    cfg.seed = args.seed
for kv in args.set:
    k, v = kv.split("=")
    setattr(cfg, k, type(getattr(cfg, k))(v))
env = Env(cfg)
n, dev = env.num_envs, env.device
ctrl = CONTROLLERS[args.task](env, **{k: float(v) for k, v in (kv.split("=") for kv in args.param)})
net = torch.jit.load(args.policy, map_location=dev).eval() if args.policy else None
venv = ppo = None
if args.checkpoint:
    import importlib.metadata as metadata
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    venv = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(venv, handle_deprecated_rsl_rl_cfg(RunnerCfg(), metadata.version("rsl-rl-lib")).to_dict(),
                            log_dir=None, device=dev)
    runner.load(args.checkpoint)
    ppo = runner.get_inference_policy(device=dev)
insert = args.task == "insert_joint"

outcome = torch.full((n,), -1, dtype=torch.long, device=dev)       # -1 running, 0 timeout, 1 success, 2 fail
steps = torch.zeros(n, dtype=torch.long, device=dev)
start = {}
last = {}
_get_dones = env._get_dones


def snapshot():
    if insert:
        rel, s, radial, ang, on_axis, in_bore, seated, released, _, _ = env._raw_status()
        tcp, hq = env._tcp()
        tp, grasp, axis, baxis, *_ = env._tube_state()
        gap = env._robot.data.joint_pos.torch[:, env.finger_ids].sum(-1)
        yh = quat_apply(hq, torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3))
        tang = torch.cross(baxis, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3), dim=-1)
        tang = tang / tang.norm(dim=-1, keepdim=True)
        twist = torch.rad2deg(torch.acos((yh * tang).sum(-1).abs().clamp(max=1.0)))
        return {"e_cm": rel.norm(dim=-1) * 100, "s_mm": s * 1000, "radial_mm": radial * 1000, "ang_deg": ang,
                "closed": env.grip_closed.float(), "seated": seated.float(), "released": released.float(),
                "phase": ctrl.phase.float(), "retries": ctrl.retries.float(), "gap_mm": gap * 1000, "twist_deg": twist,
                "grip_up_mm": ((tcp - grasp) * axis).sum(-1) * 1000}
    d, held, s, s_clear, radial, ang, tilt, lift, in_bore, _, _ = env._raw_status()
    return {"d_cm": d * 100, "held": held.float(), "s_clear_mm": s_clear * 1000, "tilt_deg": tilt, "lift_mm": lift * 1000,
            "closed": env.grip_closed.float(), "in_bore": in_bore.float(), "phase": ctrl.phase.float()}


def get_dones():
    snap = snapshot()
    first = (outcome < 0) & (env.fresh == 0) & (steps == 0)
    live = (outcome < 0) & (env.fresh == 0)
    for k, v in snap.items():
        start.setdefault(k, torch.full((n,), float("nan"), device=dev))[first] = v[first]
    steps[live] += 1
    in_phase[live, ctrl.phase[live]] += 1
    if insert:                                             # the state when the gripper was told to open, and the widest gap after
        rel_now = live & (ctrl.phase == ctrl.RELEASE) & torch.isnan(at_rel.setdefault("s_mm", torch.full((n,), float("nan"), device=dev)))
        for k, v in snap.items():
            at_rel.setdefault(k, torch.full((n,), float("nan"), device=dev))[rel_now] = v[rel_now]
        opened = live & (ctrl.phase >= ctrl.RELEASE)
        gap_max[opened] = torch.maximum(gap_max, snap["gap_mm"])[opened]
    term, trunc = _get_dones()
    new = (outcome < 0) & (term | trunc)
    outcome[new & env.succeeded] = 1
    outcome[new & env.failed & ~env.succeeded] = 2
    outcome[new & ~env.succeeded & ~env.failed] = 0
    for k, v in snap.items():
        last.setdefault(k, torch.zeros(n, device=dev))[new] = v[new]
    return term, trunc


at_rel = {}
in_phase = torch.zeros(n, len(ctrl.NAMES), dtype=torch.long, device=dev)
gap_max = torch.zeros(n, device=dev)
env._get_dones = get_dones
obs = venv.get_observations() if ppo is not None else env.reset()[0]
trace = [int(i) for i in args.trace.split(",") if i]
t = 0
while (outcome < 0).any():
    with torch.inference_mode():
        a = ctrl.act()                                     # always stepped: keeps the phase for the diagnostics
        if net is not None:
            a = net(obs["policy"])
        if trace and t % args.trace_every == 0:
            snap = snapshot()
            speed = env._tube.data.root_lin_vel_w.torch.norm(dim=-1)
            for i in trace:
                if outcome[i] < 0:
                    print(f"  t {t:3d} env {i} {ctrl.NAMES[int(ctrl.phase[i])]:8s} "
                          + " ".join(f"{k} {v[i]:.2f}" for k, v in snap.items() if k != "phase")
                          + f" tube_v_cm_s {speed[i] * 100:.1f} a " + " ".join(f"{x:+.1f}" for x in a[i, :7].tolist()), flush=True)
        t += 1
        if ppo is not None:
            obs, _, _, _ = venv.step(ppo(obs))
            ctrl.reset(torch.nonzero(env.episode_length_buf == 0).flatten())
            continue
        obs, _, _, _, _ = env.step(a)
    ctrl.reset(torch.nonzero(env.episode_length_buf == 0).flatten())

res = {"success": int((outcome == 1).sum()), "fail": int((outcome == 2).sum()), "timeout": int((outcome == 0).sum())}
out = {"task": args.task, "solver": "PPO checkpoint" if ppo is not None else "distilled network" if net is not None else "controller",
       "policy": args.checkpoint or args.policy,
       "seed": args.seed, "episodes": n, **res, "success_rate": round(res["success"] / n, 3), "kinds": {}}
if insert:
    inb = start["s_mm"] < 16.0
    e0 = start["e_cm"]
    kinds = (("in-bore starts", inb), ("carry < 2 cm", ~inb & (e0 < 2)), ("carry 2-6 cm", ~inb & (e0 >= 2) & (e0 < 6)),
             ("carry 6-15 cm", ~inb & (e0 >= 6) & (e0 < 15)), ("carry > 15 cm", ~inb & (e0 >= 15)))
else:
    held0 = start["closed"] > 0.5
    d0 = start["d_cm"]
    kinds = (("held starts", held0), ("approach < 5 cm", ~held0 & (d0 < 5)), ("approach 5-15 cm", ~held0 & (d0 >= 5) & (d0 < 15)),
             ("approach > 15 cm", ~held0 & (d0 >= 15)))
for name, m in kinds:
    k = int(m.sum())
    if k:
        row = {"n": k, "success": int((outcome[m] == 1).sum()), "fail": int((outcome[m] == 2).sum()),
               "timeout": int((outcome[m] == 0).sum())}
        out["kinds"][name] = row
        print(f"{name:18s} n {k:4d}  success {row['success']:4d}  fail {row['fail']:4d}  timeout {row['timeout']:4d}", flush=True)
ok = outcome == 1
if ok.any():
    print(f"successes: steps median {int(steps[ok].median())}, max {int(steps[ok].max())}; per phase median / max: "
          + ", ".join(f"{nm} {int(in_phase[ok, k].median())}/{int(in_phase[ok, k].max())}" for k, nm in enumerate(ctrl.NAMES)),
          flush=True)
names = ctrl.NAMES
for i in torch.nonzero(~ok).flatten().tolist()[:args.show]:
    what = "FAIL" if outcome[i] == 2 else "TIMEOUT"
    print(f"  env {i} {what} after {int(steps[i])} steps, phase {names[int(last['phase'][i])]}: "
          + ", ".join(f"{k} {start[k][i]:.1f}->{last[k][i]:.1f}" for k in last if k != "phase"), flush=True)
if insert and at_rel:
    rel = ~torch.isnan(at_rel["s_mm"])
    stuck = rel & (gap_max < 9.5)
    print(f"released {int(rel.sum())}, fingers never opened past 9.5 mm: {int(stuck.sum())}", flush=True)
    for name, m in (("opened", rel & ~stuck), ("stuck", stuck)):
        if m.any():
            print(f"  {name:6s} at release: " + ", ".join(
                f"{k} {at_rel[k][m].min():.1f}/{at_rel[k][m].median():.1f}/{at_rel[k][m].max():.1f}"
                for k in ("s_mm", "radial_mm", "ang_deg", "gap_mm", "twist_deg", "grip_up_mm")), flush=True)
    for i in torch.nonzero(stuck).flatten().tolist()[:12]:
        print(f"    env {i}: " + ", ".join(f"{k} {at_rel[k][i]:.2f}" for k in ("s_mm", "radial_mm", "ang_deg", "gap_mm", "twist_deg", "grip_up_mm"))
              + f", widest gap after {gap_max[i]:.1f}", flush=True)
print("EVAL", json.dumps(out), flush=True)
if args.out:
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(out) + "\n")
env.close()
app.close()

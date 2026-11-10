"""Record the hand-written controller (or its distilled network) in a centrifuge env: one env, viewer camera placed
per episode relative to the target bore, N episodes back to back, unedited, each frame labelled with the start kind,
the controller's phase, the step and the outcome. (After the parent session's scratchpad record_skill.py.)

  python record.py --task insert_joint --episodes 4 --out clip.mp4 --results clip.jsonl [--policy net.pt]
      [--set deep_p=0.0] --headless --enable_cameras
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
ap.add_argument("--episodes", type=int, default=4)
ap.add_argument("--out", required=True)
ap.add_argument("--results", default=None)
ap.add_argument("--policy", default=None)
ap.add_argument("--title", default="")
ap.add_argument("--label", default="hand-written controller (Jacobian servo, joint commands)")
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--eye", type=float, nargs=3, default=(-0.5, -0.45, 0.4))
ap.add_argument("--lookat", type=float, nargs=3, default=(0.0, 0.0, 0.1))
ap.add_argument("--set", action="append", default=[])
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
app = AppLauncher(args).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from controllers import CONTROLLERS  # noqa: E402
from skills import load  # noqa: E402

Env, EnvCfg, _ = load(args.task)
cfg = EnvCfg()
cfg.scene.num_envs = 1
if args.seed is not None:
    cfg.seed = args.seed
for kv in args.set:
    k, v = kv.split("=")
    setattr(cfg, k, type(getattr(cfg, k))(v))
c0 = np.load(RL / "assets" / "centrifuge_bores.npz")["point"][0]
cfg.viewer.origin_type = "world"
cfg.viewer.eye = tuple(float(x) for x in c0 + np.array(args.eye))
cfg.viewer.lookat = tuple(float(x) for x in c0 + np.array(args.lookat))
cfg.viewer.resolution = (1280, 720)
env = Env(cfg, render_mode="rgb_array")
ctrl = CONTROLLERS[args.task](env)
net = torch.jit.load(args.policy, map_location=env.device).eval() if args.policy else None
insert = args.task == "insert_joint"

ended = {"flag": False, "outcome": None}
_gd = env._get_dones


def gd():
    term, trunc = _gd()
    if bool(term[0] | trunc[0]):
        ended["flag"] = True
        ended["outcome"] = "SUCCESS" if bool(env.succeeded[0]) else "FAIL" if bool(env.failed[0]) else "TIMEOUT"
    return term, trunc


env._get_dones = gd


def start_kind():
    if insert:
        s = env._tube_state()[5][0].item()
        op = env.bore_opening[env.bore[0]].item()
        return ("start: tube held inside the bore" if s < op else
                "start: tube held at the bore entry" if s < op + 0.03 else "start: carrying the tube to the bore")
    return "start: holding the tube in the bore" if bool(env.start_closed_env[0]) else "start: approach (gripper open)"


W, H = 1280, 720
w = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
GREEN, RED, AMBER, WHITE = (80, 220, 80), (90, 90, 255), (60, 200, 230), (255, 255, 255)


def put(img, x, y, text, sc, col):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)


obs, _ = env.reset()
rows = []
for ep in range(args.episodes):
    frames, phases, kind, t = [], [], None, 0
    ended["flag"] = False
    while True:
        with torch.inference_mode():
            a = ctrl.act()
            if net is not None:
                a = net(obs["policy"])
            phases.append(ctrl.NAMES[int(ctrl.phase[0])])
            obs, _, _, _, _ = env.step(a)
        ctrl.reset(torch.nonzero(env.episode_length_buf == 0).flatten())
        t += 1
        if kind is None and t >= 2:
            kind = start_kind()
            tp = (env.bore_point[env.bore[0]] + env.scene.env_origins[0]).cpu().numpy()
            from isaacsim.core.rendering_manager import ViewportManager
            ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=list(map(float, tp + np.array(args.eye))),
                                            target=list(map(float, tp + np.array(args.lookat))))
        img = env.render()
        if img is not None:
            frames.append(cv2.cvtColor(np.asarray(img)[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
        if ended["flag"]:
            break
    out = ended["outcome"]
    rows.append({"episode": ep, "start": kind, "outcome": out, "steps": t})
    print(f"episode {ep}: {kind} -> {out} ({t} steps)", flush=True)
    col = GREEN if out == "SUCCESS" else RED if out == "FAIL" else AMBER
    for i, f in enumerate(frames + frames[-1:] * 10):
        f = cv2.resize(f, (W, H))
        j = min(i, len(frames) - 1)
        put(f, 16, 40, args.title, 0.9, WHITE)
        put(f, 16, 78, f"episode {ep + 1}/{args.episodes}   {kind}", 0.7, WHITE)
        put(f, 16, H - 54, f"phase: {phases[j]}" if net is None else "", 0.7, WHITE)
        put(f, 16, H - 20, f"{args.label}   step {j + 1}   result: {out}", 0.7, col)
        w.write(f)
w.release()
if args.results:
    with open(args.results, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
env.close()
app.close()

"""Video of a trained tube-grasp teacher: env 0 seen from a camera beside the
rack, the policy acting deterministically, a caption per frame (start type,
step, TCP-to-grasp distance, lift, tilt, outcome).

Usage (from the IsaacLab folder, PYTHONPATH=...\\simulation\\rl):
    python ..\\biotech-lab-robotics\\simulation\\rl\\teacher_video.py --checkpoint logs\\...\\model_N.pt
        --out ..\\data\\rl_teacher.mp4 [--episodes 8] [--library_p 0.5] --headless --enable_cameras
"""
import argparse
import importlib.metadata as metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--episodes", type=int, default=8)
ap.add_argument("--library_p", type=float, default=0.5)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from tube_grasp.agents import TubeGraspPPORunnerCfg  # noqa: E402
from tube_grasp.env import TubeGraspEnv  # noqa: E402
from tube_grasp.env_cfg import HOLDER_XY, TubeGraspEnvCfg  # noqa: E402

cfg = TubeGraspEnvCfg()
cfg.scene.num_envs = 1
cfg.library_p = args.library_p
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = 0
cfg.viewer.eye = (HOLDER_XY[0] + 0.55, HOLDER_XY[1] - 0.55, 1.42)
cfg.viewer.lookat = (HOLDER_XY[0], HOLDER_XY[1], 1.0)
cfg.viewer.resolution = (960, 540)
env = TubeGraspEnv(cfg, render_mode="rgb_array")
venv = RslRlVecEnvWrapper(env)
agent = handle_deprecated_rsl_rl_cfg(TubeGraspPPORunnerCfg(), metadata.version("rsl-rl-lib"))
runner = OnPolicyRunner(venv, agent.to_dict(), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)

writer, eps, t, results = None, 0, 0, []
obs = venv.get_observations()
start = "?"
while eps < args.episodes:
    with torch.inference_mode():
        act = policy(obs)
    if t == 0:
        start = f"slot {'ABCD'[int(env.target[0]) // 5]}{int(env.target[0]) % 5 + 1}"
    d, grasped, lift, fail, success = env._status()
    _, _, _, _, tilt = env._tube_state()
    frame = env.render()
    if frame is not None:
        img = cv2.cvtColor(np.asarray(frame)[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        text = (f"episode {eps + 1}  {start}  step {t}  |TCP-grasp| {float(d[0]) * 100:.1f} cm  lift {float(lift[0]) * 100:.1f} cm  "
                f"tilt {float(tilt[0]):.1f} deg  {'HOLDING' if bool(grasped[0]) else ''}")
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, th, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (img.shape[1], img.shape[0]))
        writer.write(img)
    obs, _, dones, _ = venv.step(act)
    t += 1
    if bool(dones[0]):
        log = env.extras.get("log", {})
        outcome = "SUCCESS" if log.get("Metrics/success_rate", 0) > 0.5 else (
            "FAIL" if log.get("Metrics/fail_rate", 0) > 0.5 else "timeout")
        results.append(outcome)
        if writer is not None and frame is not None:
            for _ in range(15):                   # hold the outcome for 1 s
                hold = img.copy()
                cv2.putText(hold, outcome, (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6, cv2.LINE_AA)
                cv2.putText(hold, outcome, (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (80, 220, 80) if outcome == "SUCCESS" else (60, 60, 230), 2, cv2.LINE_AA)
                writer.write(hold)
        eps += 1
        t = 0
if writer is not None:
    writer.release()
print("outcomes:", results, "->", args.out, flush=True)
env.close()
app.close()

"""Open-loop check of a served lab policy against recorded episodes: feed it
each episode's recorded observations (three policy cameras, state, prompt)
every --stride steps and compare the predicted action chunk with the actions
the episode actually took next. On the training episodes a model that fit
its data should match closely; a large error there points at a mismatch
between how training data and live observations reach the model.

No Isaac needed (numpy, PIL, websockets, msgpack), with the server from
training/serve_lab_policy.py running:
    .venv-isaac\\Scripts\\python.exe biotech-lab-robotics\\simulation\\sim\\check_policy_replay.py data\\primitives_v2 --episodes 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_policy import IMAGE_WH, PolicyClient  # noqa: E402

CAMS = ("platform_cam1", "wrist_top", "wrist_side")   # lab_env.POLICY_CAMERAS


def load(path):
    """As export_lerobot.load_image: the recorded JPEG, resized bicubic."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img.resize(IMAGE_WH, resample=Image.BICUBIC))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=8, help="predicted steps compared")
    ap.add_argument("--server", default="ws://localhost:8000")
    args = ap.parse_args()
    root = Path(args.root)
    s = json.loads((root / "summary.json").read_text())
    eps = [root / e["dir"] for e in s["episodes"] if e["success"]]
    step = max(1, len(eps) // args.episodes)
    client = PolicyClient(args.server)
    rows = []
    for ep in eps[::step][:args.episodes]:
        meta = json.loads((ep / "meta.json").read_text())
        d = np.load(ep / "steps.npz")
        act = d["action"]
        errs, base = [], []
        for t in range(0, len(act) - args.horizon, args.stride):
            obs = {
                "observation/exterior_image_1_left": load(ep / "images" / f"{CAMS[0]}_{t:05d}.jpg"),
                "observation/wrist_image_left": load(ep / "images" / f"{CAMS[1]}_{t:05d}.jpg"),
                "observation/wrist_image_right": load(ep / "images" / f"{CAMS[2]}_{t:05d}.jpg"),
                "observation/joint_position": d["joint_position"][t].astype(np.float32),
                "observation/gripper_position": np.asarray(d["gripper_position"][t], np.float32).reshape(1),
                "observation/pipette_state": np.array([d["plunger"][t], d["tip_attached"][t]], np.float32),
                "prompt": meta["prompt"],
            }
            pred = np.asarray(client.infer(obs)["actions"], float)[:args.horizon]
            true = act[t:t + args.horizon]
            errs.append(np.abs(pred - true).mean(axis=0))
            base.append(np.abs(true - true.mean(axis=0, keepdims=True)).mean(axis=0) + np.abs(true).mean(axis=0) * 0)
        e = np.mean(errs, axis=0)
        scale = np.abs(act).mean(axis=0) + 1e-6
        rows.append({"episode": f"{ep.parent.name}/{ep.name}", "prompt": meta["prompt"],
                     "mae_joint_vel": round(float(e[:7].mean()), 4), "mean_abs_joint_vel": round(float(scale[:7].mean()), 4),
                     "mae_gripper": round(float(e[7]), 4), "mae_plunger": round(float(e[8]), 4)})
        print(json.dumps(rows[-1]), flush=True)
    j = np.array([r["mae_joint_vel"] for r in rows]); m = np.array([r["mean_abs_joint_vel"] for r in rows])
    print(f"overall: joint-velocity MAE {j.mean():.4f} vs mean |joint velocity| {m.mean():.4f} "
          f"(ratio {j.mean() / m.mean():.2f}); gripper MAE {np.mean([r['mae_gripper'] for r in rows]):.4f}")


if __name__ == "__main__":
    main()

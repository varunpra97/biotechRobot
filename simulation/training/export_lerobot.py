"""Convert recorded sim episodes (sim/lab_env.EpisodeRecorder) into the
LeRobot dataset π0.5 fine-tunes on -- the layout of openpi's own DROID
converter (examples/droid/convert_droid_data_to_lerobot.py), so the DROID
recipe (joint-velocity actions, 15 Hz, exterior + wrist camera) carries over:

  exterior_image_1_left  platform_cam1 (fixed, whole platform)   240x320x3
  wrist_image_left       the episode's policy wrist camera        240x320x3
                         (meta "policy_wrist_camera": one of the three
                         wrist cameras, chosen per primitive --
                         lab_env.PRIMITIVE_WRIST_CAMERA)
  joint_position         7  rad
  gripper_position       1  0 open .. 1 closed
  pipette_state          2  plunger fraction, tip attached
  actions                10 joint velocity (7, DROID-normalized), gripper
                            position, pipette plunger, pipette eject
  task                   the episode's prompt

Images keep the cameras' 4:3 aspect (640x480 halved); openpi pads them to
224x224 itself. Only episodes whose primitive succeeded are exported. The
joint positions/velocities, poses and all six cameras stay in the raw
episodes for internal use.

Runs in openpi's environment (WSL), which has lerobot:
    cd ~/openpi
    uv run python /mnt/c/Users/varun/workspace/BiotechRobot/biotech-lab-robotics/simulation/training/export_lerobot.py \\
        --src /mnt/c/Users/varun/workspace/BiotechRobot/data/primitives
Writes $HF_LEROBOT_HOME/<repo-id> (default ~/.cache/huggingface/lerobot/local/biotech_lab).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

EXTERIOR = "platform_cam1"
IMAGE_HW = (240, 320)
FPS = 15
ACTION_DIM = 10


def load_image(path):
    img = Image.open(path).convert("RGB")
    if img.size != (IMAGE_HW[1], IMAGE_HW[0]):
        img = img.resize((IMAGE_HW[1], IMAGE_HW[0]), resample=Image.BICUBIC)
    return np.asarray(img)


def episodes(src: Path):
    for meta_path in sorted(src.glob("*/ep_*/meta.json")):
        meta = json.loads(meta_path.read_text())
        yield meta_path.parent, meta


def main():
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of <primitive>/ep_XXXX episodes")
    ap.add_argument("--repo-id", default="local/biotech_lab")
    ap.add_argument("--include-failed", action="store_true")
    args = ap.parse_args()

    out = HF_LEROBOT_HOME / args.repo_id
    if out.exists():
        shutil.rmtree(out)
    shape = (*IMAGE_HW, 3)
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=FPS,
        features={
            "exterior_image_1_left": {"dtype": "image", "shape": shape, "names": ["height", "width", "channel"]},
            "wrist_image_left": {"dtype": "image", "shape": shape, "names": ["height", "width", "channel"]},
            "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
            "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
            "pipette_state": {"dtype": "float32", "shape": (2,), "names": ["pipette_state"]},
            "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    kept, skipped, frames = 0, [], 0
    for ep_dir, meta in episodes(Path(args.src)):
        if not meta["success"] and not args.include_failed:
            skipped.append(str(ep_dir))
            continue
        d = np.load(ep_dir / "steps.npz")
        wrist = meta.get("policy_wrist_camera", "wrist")
        n = len(d["action"])
        assert d["action"].shape[1] == ACTION_DIM, ep_dir
        for t in range(n):
            ds.add_frame({
                "exterior_image_1_left": load_image(ep_dir / "images" / f"{EXTERIOR}_{t:05d}.jpg"),
                "wrist_image_left": load_image(ep_dir / "images" / f"{wrist}_{t:05d}.jpg"),
                "joint_position": d["joint_position"][t].astype(np.float32),
                "gripper_position": np.asarray(d["gripper_position"][t], np.float32).reshape(1),
                "pipette_state": np.array([d["plunger"][t], d["tip_attached"][t]], np.float32),
                "actions": d["action"][t].astype(np.float32),
                "task": meta["prompt"],
            })
        ds.save_episode()
        kept += 1
        frames += n
        print(f"{ep_dir.parent.name}/{ep_dir.name}: {n} frames, wrist camera {wrist}  '{meta['prompt']}'", flush=True)
    print(f"exported {kept} episodes, {frames} frames -> {out}")
    if skipped:
        print(f"skipped {len(skipped)} failed episodes: {skipped}")


if __name__ == "__main__":
    main()

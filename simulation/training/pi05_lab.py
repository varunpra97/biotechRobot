"""π0.5 fine-tuning config for the lab primitives (openpi), sized for an
RTX 3080 Ti (12 GB): LoRA on the language model and action expert, vision
encoder frozen, batch 8 -- the setting measured to fit (9.35 GB) -- starting
from the pi05_droid checkpoint.

Compared with openpi's pi05_droid_finetune:
  - state is 10-D: joints (7), gripper, pipette plunger, tip attached
  - actions are 10-D: DROID joint velocity (7), gripper position, plunger,
    eject (the model's 32-D action space has room; DroidOutputs would cut
    the last two off)
  - only the exterior (base) and wrist cameras, as in pi05_droid
  - DROID norm stats are reused (as openpi recommends for DROID
    fine-tuning), with dims 8-9 -- zero-width in DROID's stats, which would
    blow plunger/eject up by 1e6 -- set to their real 0..1 range
    (write_norm_stats()).

Runs only in openpi's environment (WSL):
    cd ~/openpi
    uv run python /mnt/c/Users/varun/workspace/BiotechRobot/biotech-lab-robotics/simulation/training/pi05_lab.py \\
        --exp-name lab_v1 [--steps 3000] [--batch 8]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

import flax.nnx as nnx
import numpy as np
from openpi import transforms as _transforms
from openpi.models import pi0_config
from openpi.policies.droid_policy import _parse_image
from openpi.shared import download, nnx_utils
from openpi.training import config as _config
from openpi.training import weight_loaders

REPO_ID = "local/biotech_lab"
ASSET_ID = "biotech_lab"
STATE_DIM = ACTION_DIM = 10
DROID_ASSETS = "gs://openpi-assets/checkpoints/pi05_droid/assets"


@dataclasses.dataclass(frozen=True)
class LabInputs(_transforms.DataTransformFn):
    """Dataset/robot frame -> model inputs (exterior as base camera, wrist
    as left wrist, no right wrist -- the pi05_droid layout)."""

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([
            np.asarray(data["observation/joint_position"]),
            np.atleast_1d(np.asarray(data["observation/gripper_position"])),
            np.asarray(data["observation/pipette_state"]),
        ])
        base = _parse_image(data["observation/exterior_image_1_left"])
        wrist = _parse_image(data["observation/wrist_image_left"])
        out = {
            "state": state,
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": np.zeros_like(base)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            out["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            p = data["prompt"]
            out["prompt"] = p.decode("utf-8") if isinstance(p, bytes) else p
        return out


@dataclasses.dataclass(frozen=True)
class LabOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}


@dataclasses.dataclass(frozen=True)
class LabDataConfig(_config.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        repack = _transforms.Group(inputs=[_transforms.RepackTransform({
            "observation/exterior_image_1_left": "exterior_image_1_left",
            "observation/wrist_image_left": "wrist_image_left",
            "observation/joint_position": "joint_position",
            "observation/gripper_position": "gripper_position",
            "observation/pipette_state": "pipette_state",
            "actions": "actions",
            "prompt": "prompt",
        })])
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=_transforms.Group(inputs=[LabInputs()], outputs=[LabOutputs()]),
            model_transforms=_config.ModelTransformFactory()(model_config),
        )


def write_norm_stats(assets_dir: pathlib.Path):
    """DROID's norm stats with the pipette dims given their 0..1 range."""
    src = download.maybe_download(f"{DROID_ASSETS}/droid/norm_stats.json")
    stats = json.loads(pathlib.Path(src).read_text())
    for key in ("state", "actions"):
        s = stats["norm_stats"][key]
        for i in (8, 9):
            s["q01"][i], s["q99"][i], s["mean"][i], s["std"][i] = 0.0, 1.0, 0.5, 0.5
    out = assets_dir / ASSET_ID / "norm_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2))
    return out


def make_config(exp_name: str, steps: int, batch: int, repo_id: str = REPO_ID):
    model = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=16,
                                 paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    return _config.TrainConfig(
        name="pi05_lab_lora",
        exp_name=exp_name,
        model=model,
        data=LabDataConfig(repo_id=repo_id, base_config=_config.DataConfig(prompt_from_task=True),
                           assets=_config.AssetsConfig(asset_id=ASSET_ID)),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        # LoRA weights train; the base LLM/action expert and the SigLIP vision encoder stay frozen
        freeze_filter=nnx.Any(model.get_freeze_filter(), nnx_utils.PathRegex(".*img.*")),
        ema_decay=None,
        num_train_steps=steps,
        batch_size=batch,
        save_interval=500,
        keep_period=1000,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path.home() / "openpi" / "scripts"))
    import train as openpi_train

    cfg = dataclasses.replace(make_config(args.exp_name, args.steps, args.batch, args.repo_id),
                              overwrite=args.overwrite)
    print("norm stats:", write_norm_stats(cfg.assets_dirs))
    openpi_train.main(cfg)


if __name__ == "__main__":
    main()

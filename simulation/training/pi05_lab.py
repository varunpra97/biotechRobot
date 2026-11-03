"""π0.5 fine-tuning config for the lab primitives (openpi), sized for an
RTX 3080 Ti (12 GB): LoRA on the language model and action expert, vision
encoder frozen, batch 8 -- the setting measured to fit (9.35 GB) -- starting
from the pi05_droid checkpoint.

Compared with openpi's pi05_droid_finetune:
  - state is 10-D: joints (7), gripper, pipette plunger, tip attached
  - actions are 10-D: DROID joint velocity (7), gripper position, plunger,
    eject (the model's 32-D action space has room; DroidOutputs would cut
    the last two off)
  - three cameras: exterior (base slot), wrist_top (left wrist slot) and
    wrist_side (right wrist slot, mask on). pi05_droid sends only two and
    never saw a real image in the right-wrist slot; fine-tuning learns it.
  - norm stats computed from the training dataset itself (--norm-stats ours,
    the default; compute_norm_stats()), with the pipette dims 8-9 fixed to
    their 0..1 range. DROID's stats (--norm-stats droid, as openpi suggests
    for DROID-like data) span joint velocities of +-0.45..0.9 while the
    scripted expert's span +-0.09..0.25: the normalized targets were then
    near zero, the training loss reached 0.008 by predicting ~no motion, and
    the policy failed every closed-loop trial (pilot100_v1).

Runs only in openpi's environment (WSL):
    cd ~/openpi
    uv run python /mnt/c/Users/varun/workspace/BiotechRobot/biotech-lab-robotics/simulation/training/pi05_lab.py \\
        --exp-name lab_v1 [--steps 3000] [--batch 8] [--repo-id local/biotech_lab_v2] [--resume]
Set XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 (batch 8 needs it; see the 3080 Ti
benchmark) and WANDB_MODE=offline unless wandb is logged in.
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
    """Dataset/robot frame -> model inputs: exterior camera as base,
    wrist_top as left wrist, wrist_side as right wrist (all three on)."""

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([
            np.asarray(data["observation/joint_position"]),
            np.atleast_1d(np.asarray(data["observation/gripper_position"])),
            np.asarray(data["observation/pipette_state"]),
        ])
        base = _parse_image(data["observation/exterior_image_1_left"])
        wrist_left = _parse_image(data["observation/wrist_image_left"])
        wrist_right = _parse_image(data["observation/wrist_image_right"])
        out = {
            "state": state,
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist_left, "right_wrist_0_rgb": wrist_right},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.True_},
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
            "observation/wrist_image_right": "wrist_image_right",
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


def compute_norm_stats(assets_dir: pathlib.Path, repo_id: str):
    """Norm stats from the LeRobot dataset's own state and actions (openpi
    format, 32 wide: the first 10 dims are ours, the rest DROID's, unused).
    State = joints (7), gripper, plunger, tip attached -- as LabInputs builds
    it; dims 8-9 of both are fixed to 0..1."""
    import pandas as pd
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from openpi.shared import normalize

    files = sorted((HF_LEROBOT_HOME / repo_id / "data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=["joint_position", "gripper_position", "pipette_state", "actions"])
                    for f in files])
    state = np.concatenate([np.stack(df["joint_position"]), np.stack(df["gripper_position"]).reshape(-1, 1),
                            np.stack(df["pipette_state"])], axis=1)
    actions = np.stack(df["actions"])
    droid = json.loads(pathlib.Path(download.maybe_download(f"{DROID_ASSETS}/droid/norm_stats.json")).read_text())
    out = {}
    for key, x in (("state", state), ("actions", actions)):
        d = {k: np.asarray(v, np.float32) for k, v in droid["norm_stats"][key].items()}
        q01, q99 = np.quantile(x, 0.01, axis=0), np.quantile(x, 0.99, axis=0)
        mean, std = x.mean(axis=0), x.std(axis=0)
        flat = (q99 - q01) < 1e-3                     # (near-)constant dims: keep a usable width
        q01[flat], q99[flat] = mean[flat] - 1e-3, mean[flat] + 1e-3
        q01[8:10], q99[8:10], mean[8:10], std[8:10] = 0.0, 1.0, 0.5, 0.5
        for k, v in (("q01", q01), ("q99", q99), ("mean", mean), ("std", np.maximum(std, 1e-3))):
            d[k][:10] = v
        out[key] = normalize.NormStats(mean=d["mean"], std=d["std"], q01=d["q01"], q99=d["q99"])
        print(f"{key}: q01 {np.round(q01, 3).tolist()} q99 {np.round(q99, 3).tolist()}", flush=True)
    normalize.save(assets_dir / ASSET_ID, out)
    print(f"norm stats from {len(x)} frames of {repo_id} -> {assets_dir / ASSET_ID}", flush=True)
    return assets_dir / ASSET_ID


# Vision encoder parts trainable on 12 GB (--train-vision). SigLIP's 27 blocks
# are one scanned parameter stack (pi0.py: scan=True), and training all of
# them needs ~16.5 GB (3080 Ti benchmark), so the blocks stay frozen; these
# adapt what the encoder hands the language model to the sim's images:
#   head:  the output projection (1152 -> 2048) and the final layer norm
#   embed: head + the patch embedding (its gradient runs back through the
#          frozen blocks: more activation memory)
VISION_TRAINABLE = {
    "none": None,
    "head": r".*img/(head|Transformer/encoder_norm)/.*",
    "embed": r".*img/(head|Transformer/encoder_norm|embedding|pos_embedding)(/.*)?$",
}


def make_config(exp_name: str, steps: int, batch: int, repo_id: str = REPO_ID, keep_period: int = 1000,
                train_vision: str = "none"):
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
        # (but for the VISION_TRAINABLE parts with --train-vision)
        freeze_filter=nnx.Any(model.get_freeze_filter(), vision_frozen(train_vision)),
        ema_decay=None,
        num_train_steps=steps,
        batch_size=batch,
        save_interval=500,
        keep_period=keep_period,
    )


def vision_frozen(train_vision):
    img = nnx_utils.PathRegex(".*img.*")
    pattern = VISION_TRAINABLE[train_vision]
    return img if pattern is None else nnx.All(img, nnx.Not(nnx_utils.PathRegex(pattern)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from the experiment's last checkpoint")
    ap.add_argument("--train-vision", choices=list(VISION_TRAINABLE), default="none",
                    help="also train these vision-encoder parts (see VISION_TRAINABLE)")
    ap.add_argument("--keep-period", type=int, default=1000,
                    help="checkpoints at multiples of this are kept (each is ~5.5 GB); the latest always is")
    ap.add_argument("--norm-stats", choices=["ours", "droid"], default="ours",
                    help="ours: computed from the dataset (default); droid: DROID's, pipette dims fixed")
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path.home() / "openpi" / "scripts"))
    import train as openpi_train

    cfg = dataclasses.replace(make_config(args.exp_name, args.steps, args.batch, args.repo_id, args.keep_period,
                                          args.train_vision),
                              overwrite=args.overwrite, resume=args.resume)
    if args.norm_stats == "ours":
        print("norm stats:", compute_norm_stats(cfg.assets_dirs, args.repo_id))
    else:
        print("norm stats:", write_norm_stats(cfg.assets_dirs))
    openpi_train.main(cfg)


if __name__ == "__main__":
    main()

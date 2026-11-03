"""Serve a fine-tuned lab policy (pi05_lab.py checkpoint) over openpi's
websocket protocol, for closed-loop evaluation in the Isaac sim
(sim/eval_policy.py is the client).

The policy expects, per call: observation/exterior_image_1_left,
observation/wrist_image_left (wrist_top), observation/wrist_image_right
(wrist_side) as uint8 HxWx3, observation/joint_position (7),
observation/gripper_position (1), observation/pipette_state (2) and a
prompt; it returns {"actions": (16, 10)} -- 16 steps at 15 Hz of the
10-D lab action (lab_env.py). Norm stats come from the checkpoint's assets.

Runs in openpi's environment (WSL), next to Isaac on the same 12 GB GPU, so
nothing is preallocated:
    cd ~/openpi
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python \\
        /mnt/c/Users/varun/workspace/BiotechRobot/biotech-lab-robotics/simulation/training/serve_lab_policy.py \\
        --checkpoint ~/openpi/checkpoints/pi05_lab_lora/pilot100_v1/2999 [--port 8000]
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="a step directory, e.g. .../pilot100_v1/2999")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server

    import pi05_lab

    ckpt = pathlib.Path(args.checkpoint).expanduser().resolve()
    # steps/batch only matter for training; the model, transforms and
    # norm-stat asset id are what the policy needs
    cfg = pi05_lab.make_config(ckpt.parent.name, steps=1, batch=1)
    policy = policy_config.create_trained_policy(cfg, ckpt)
    logging.info("serving %s on port %d", ckpt, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy, host="0.0.0.0", port=args.port,
        metadata={"checkpoint": str(ckpt), "action_dim": pi05_lab.ACTION_DIM, "action_horizon": 16},
    ).serve_forever()


if __name__ == "__main__":
    main()

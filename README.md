# Biotech Lab Robotics

A Franka Panda working a biotech lab bench (test tubes, centrifuge,
electronic pipette, sample tray), simulated physically in Isaac Sim to
generate training data for Physical Intelligence's π0.5 VLA, plus a pipeline
that measures a real lab scene into sim assets.

| Directory | What it is |
|---|---|
| [`simulation/`](simulation/README.md) | Isaac Sim lab scene, 15 Hz scripted primitives, episode recording, LeRobot export and the openpi fine-tuning config |
| [`vision_pipeline/`](vision_pipeline/README.md) | Real scene → measured 3D CAD: calibrated multi-view capture to meshes, labels and placements for the sim |

The Python environments (`.venv-isaac`, `.venv-cad`) and generated data
(`data/`) live in the workspace folder above this repo; CAD generation and
measurement scripts are in `../cad_source/`.

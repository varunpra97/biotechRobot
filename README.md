# Biotech Lab Robotics — Sim-to-Real Tray Handling

Sim-only policy training for 6-DOF arm tray pick-and-place in a biotech lab
(incubators, biosafety cabinets, bench instruments).

**Physics:** NVIDIA Newton (MuJoCo-Warp solver) is the target engine.
MuJoCo is the stable reference simulator during development; all assets
are MJCF, which loads in both.

**Hardware (frozen 2026-10-11):** single 6-DOF arm design
(`configs/canonical_arm.yaml`) on a pedestal mount, rigid tool extension
(pre-gripper). No morphology sampling — randomization is over initial arm
state, tray pose, and goal pose.

**Policy — ChainedDiffuser-style hierarchy** (after Zhou Xian, CoRL 2023):
1. Wrist + third-person cameras → perception net → `(tray_xyz, goal_xyz)`
2. Analytic bottlenecks → initial task-space keyposes
   (`pre_grasp → grasp → lift → transit → pre_place → place`)
3. `KeyposeDiffusion`: refines/predicts keypose sequences
4. `TrajectoryDiffusion`: connects keyposes → joint-space action chunks

See `policies/README.md`.

## Layout

- `configs/canonical_arm.yaml` — frozen arm parameters
- `embodiments/arm_generator.py` — MJCF generator (canonical design)
- `sim/camera_sim.py` — scene + randomized cameras, renders episodes
- `sim/ik.py` — damped least-squares IK (two-tier, adaptive restarts)
- `sim/make_demos.py` — keypose-annotated demo generator; validates
  IK accuracy, collision-freedom (kinematic), and torque feasibility
  (inverse dynamics)
- `sim/run_arm.py` — MuJoCo PD validation runner
- `perception/keyposes.py` — analytic task-space keypose bottlenecks
- `perception/train_pose_net.py` — 3-view CNN: images → tray/goal poses
- `policies/diffusion.py` — conditional DDPM core (shape-tested)
- `setup_windows.ps1` / `WINDOWS_SETUP.md` — RTX 3080 Ti box setup

## Windows (RTX 3080 Ti)

Unzip to `C:\Users\varun\workspace\BiotechRobot`, see `WINDOWS_SETUP.md`,
then run `setup_windows.ps1`. Generate data with `sim/make_demos.py`,
train with `policies/train_*.py`.

## Status (2026-10-11)

- [x] Canonical arm + scene, pedestal mount, tool extension
- [x] Analytic keyposes + robust IK (adaptive restarts, achieved-pose recording)
- [x] Demo generator: collision-free, torque-feasible trajectories (validated)
- [x] Diffusion model core (shapes + DDPM math validated)
- [ ] Rendered episodes on GPU (needs the 3080 Ti box)
- [ ] Perception net training
- [ ] Diffusion training loops (`train_keypose.py`, `train_traj.py`)
- [ ] Gripper + grasp contact physics
- [ ] Closed-loop evaluation

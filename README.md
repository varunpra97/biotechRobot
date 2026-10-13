# Biotech Lab Robotics — Sim-to-Real Tray Handling

Sim-only policy training for 6-DOF arm tray pick-and-place in a biotech lab
(incubators, biosafety cabinets, bench instruments).

**Physics:** NVIDIA Newton (MuJoCo-Warp solver) is the target engine.
MuJoCo is the stable reference simulator during development; all assets
are MJCF, which loads in both. Isaac Sim (PhysX) is the visual/parallel
lane for iteration and multi-instance data generation.

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
- `sim/mjcf_to_usd.py` — MuJoCo → USD exporter for Blender visualization
- `sim/isaac_hello.py` — Isaac Sim smoke test (Menagerie Franka → USD → stage)
- `sim/isaac_pick_place.py` — single Franka pick-and-place (PhysX, MuJoCo-IK keyframes)
- `sim/isaac_pick_place_50.py` — 50 parallel instances, randomized starts
- `perception/keyposes.py` — analytic task-space keypose bottlenecks
- `perception/train_pose_net.py` — 3-view CNN: images → tray/goal poses
- `policies/diffusion.py` — conditional DDPM core (shape-tested)
- `policies/train_keypose.py` — high-level: (tray_init, tray_goal) → keyposes
- `policies/train_traj.py` — low-level: keypose pair → joint chunk
- `setup_windows.ps1` / `WINDOWS_SETUP.md` — RTX 3080 Ti box setup

## How the simulation works

### The task
A Franka Panda arm picks up a blue 5.15 cm cube from `[0.45, 0]` on the
table and places it at the target `[0, 0.45]` — the pick-and-place template
that will later become tray handling (incubator ↔ biosafety cabinet ↔ bench).

> **Note on the robot model:** the Isaac Sim scripts use the **Franka Panda**
> (a ready-made, well-validated arm from MuJoCo Menagerie). The project's own
> hardware is the frozen **6-DOF** design (`configs/canonical_arm.yaml`), which
> lives on the MuJoCo side. The Franka sim proves out the pipeline — keyframe
> solving, playback, multi-instance scaling — which the canonical arm will use.

### Single-instance: `sim/isaac_pick_place.py`
Architecture: **keyframes solved offline, played back smoothly.**

1. **Joint targets are precomputed with validated MuJoCo IK** (damped
   least-squares, `sim/ik.py`), not solved live in Isaac Sim. Isaac's
   real-time IK diverged on the converted model, so the hand targets were
   solved in MuJoCo and baked in as `Q_HOME, Q_ABOVE, Q_GRASP, Q_HOLD,
   Q_LIFT, Q_PLACE, Q_RELEASE` (7 arm joints + 2 gripper fingers each).
2. **Playback is 7 phases** — move above cube → descend → grasp → lift →
   move to target → release → retract. Between keyframes, cosine smoothing
   (`s = 0.5 − 0.5·cos(πt)`, zero-velocity ends) interpolates targets set via
   `set_dof_position_targets` every step, letting PhysX drives track them.
3. After a 30-step settle, cube XY distance to target is measured.
   **Verified: 2.9 cm final error (SIM OK).**

Key fixes: MJCF→USD leaves finger prismatic drives at zero stiffness
(gripper never closed) — now permanently fixed in `assets/panda/panda.usda`
(stiffness 2000, damping 200). Isaac Sim 6.x needs `SimulationApp` before any
other `isaacsim` import, and examples assume NVIDIA Nucleus, so local
Menagerie-derived USD is used instead.

### 50-instance: `sim/isaac_pick_place_50.py`
Same task, **50 Frankas in one window**, each with a different random initial
arm/gripper configuration, all running the identical keyframe sequence
simultaneously (10×5 grid, 2.5 m spacing, one shared USD stage).
**Verified headless: 37/50 within 5 cm (mean 5.1 cm).** Misses come from
awkward random starts — realistic failure data for robust policy training.

Flags: `--n 20` (fewer instances), `--seed N` (reproducible starts),
`--same-init` (debug). GUI loops with fresh random starts each cycle.

### MuJoCo lane (reference physics)
- `embodiments/arm_generator.py` builds the canonical 6-DOF MJCF (pedestal
  0.35 m, tool 0.12 m, 160 N·m gears).
- `sim/make_demos.py` generates keypose-annotated `.npz` episodes: analytic
  keyposes → DLS IK with adaptive restarts (records *achieved* poses) →
  collision-free kinematic planning → inverse-dynamics torque check
  (< 90% of limit). **5/5 episodes pass; IK errors 0–0.1 mm.**
- `sim/run_arm.py` validates trajectories under PD control
  (`tau = -qfrc_bias` for static hold).

## Tools and frameworks

| Tool | Version | Role |
|---|---|---|
| **NVIDIA Isaac Sim** | 6.0.1 (USD + PhysX) | Visual/GPU simulation; the pick-and-place sims run here |
| **MuJoCo** | 3.15.0 | Stable reference; IK solving, demo validation, torque checks |
| **MuJoCo Menagerie** | Franka Panda (MJCF) | Ready-made arm model, converted to USD |
| **NVIDIA Newton** (+ MuJoCo-Warp) | 1.6.1 / 1.18.0 | Target engine for scaled training (needs CUDA) |
| **PyTorch** | 2.11.0+cu128 | Diffusion models, perception CNN, training |
| **USD + Blender** (`usd-core` 26.8) | — | MJCF→USD baking; Blender visualization (not live physics) |
| **Python** | 3.14 (main), 3.12.10 (`.venv-isaac`) | Separate venvs; Isaac needs 3.12 |
| **Compute** | RTX 3080 Ti 12 GB (Windows) | All GPU work via `ssh biotech-win` (Tailscale) |

**Physics in one line:** Isaac Sim is where you *see* and iterate today;
MuJoCo is the trusted reference that solved the motions; Newton
(MuJoCo-Warp) is where mass data-generation + training will happen;
Blender/USD is how you *watch* MuJoCo-side demos.

## How to run

```powershell
cd C:\Users\varun\workspace\BiotechRobot\biotech-lab-robotics
C:\Users\varun\workspace\BiotechRobot\.venv-isaac\Scripts\Activate.ps1
python sim\isaac_pick_place.py                    # GUI, loops until closed
python sim\isaac_pick_place_50.py                 # GUI, 50 instances
python sim\isaac_pick_place_50.py --headless --n 50 --seed 0
```

## Status (2026-10-12)

- [x] Canonical arm + scene, pedestal mount, tool extension
- [x] Analytic keyposes + robust IK (adaptive restarts, achieved-pose recording)
- [x] Demo generator: collision-free, torque-feasible trajectories (validated)
- [x] Diffusion model core (shapes + DDPM math validated)
- [x] Isaac Sim single pick-and-place (2.9 cm, PhysX contacts)
- [x] Isaac Sim 50-instance parallel (37/50 within 5 cm)
- [x] Training loops tested on synthetic demos (loss decreases, ckpts save)
- [ ] Rendered episodes on GPU (needs the 3080 Ti box)
- [ ] Perception net training
- [ ] Diffusion training on real demos
- [ ] Tray (SBS) grasp instead of cube
- [ ] Randomized goals + per-episode IK in Isaac
- [ ] Cameras + episode recording in Isaac
- [ ] Gripper + grasp contact physics
- [ ] Closed-loop evaluation

# Biotech Lab Robotics — Simulation Report

Written 2026-10-12 (PDT). Covers the current Isaac Sim simulation,
every tool/framework in the stack, and how the codebase is organized.

---

## 1. How the current simulation works

### The task
A Franka Panda arm picks up a blue 5.15 cm cube from `[0.45, 0]` on the
table and places it at the target `[0, 0.45]` — the pick-and-place template
that will later become tray handling (incubator ↔ biosafety cabinet ↔ bench).

> **Note on the robot model:** the Isaac Sim scripts below use the
> **Franka Panda** (a ready-made, well-validated arm from MuJoCo Menagerie)
> as the robot. The project's own hardware is a frozen **6-DOF** design
> (`configs/canonical_arm.yaml`); that design lives in the MuJoCo side of
> the repo. The Franka sim proves out the whole pipeline — keyframe solving,
> playback, multi-instance scaling — which the canonical arm will then use.

### Single-instance: `sim/isaac_pick_place.py`

Architecture: **keyframes solved offline, played back smoothly.**

1. **Joint targets are precomputed with validated MuJoCo IK** (damped
   least-squares, `sim/ik.py`), not solved live in Isaac Sim. The first
   Isaac attempts used Isaac's real-time IK and it diverged on the
   converted model — so the hand targets (above `[0.45,0,0.25]`, grasp
   `[0.45,0,0.08]`, lift `[0.45,0,0.3]`, place `[0,0.45,0.25]`) were solved
   in MuJoCo and baked into the script as constant arrays:
   `Q_HOME, Q_ABOVE, Q_GRASP, Q_HOLD, Q_LIFT, Q_PLACE, Q_RELEASE`
   (7 arm joints + 2 gripper fingers each; gripper open = 0.04, closed = 0.0).
2. **Playback is 7 phases** — move above cube (60 steps) → descend (40) →
   grasp/close fingers (20) → lift (40) → move to target (90) →
   release (20) → retract home (60). Between keyframes the script
   interpolates with cosine smoothing
   (`s = 0.5 − 0.5·cos(πt)`, zero-velocity ends) and sets the result as
   `set_dof_position_targets` every simulation step, letting Isaac's
   PhysX drives track them.
3. After a 30-step settle, it measures the cube's XY distance to target.
   `--headless` runs once and prints it; the GUI loops forever until the
   window is closed. **Verified: 2.9 cm final error (threshold 5 cm — SIM OK).**

Two real bugs were hit and fixed along the way:
- **Isaac Sim 6.x has no `isaacsim.core`** — `SimulationApp` must be
  instantiated *before* any other `isaacsim` import, and the code uses the
  restructured `isaacsim.core.experimental` API. Isaac examples also assume
  NVIDIA's Nucleus asset server, so the script uses the local Menagerie
  Franka converted to USD instead.
- **Finger drives were dead**: MJCF→USD conversion leaves the prismatic
  finger joints at zero stiffness, so the gripper never closed. The script
  sets stiffness 2000 / damping 200 on the finger drives at load (this fix is
  now also baked permanently into `assets/panda/panda.usda`).

### 50-instance: `sim/isaac_pick_place_50.py`

Same task, **50 Frankas in one window**, differing only in their initial
configuration — what the user asked to see:

- One USD stage, 50 robots added via `add_reference_to_stage` on an
  8-column grid (2.5 m spacing), each with its own cube at the same
  robot-relative offset.
- Each robot draws a **random initial arm pose** (uniform within
  conservative joint limits) and random gripper state; then all 50 run the
  *identical* keyframe sequence simultaneously — 100 steps for the opening
  phase (extra time to recover from awkward starts), the rest as above.
- Verified headless: **37/50 placed within 5 cm (mean error 5.1 cm).** The
  misses started from awkward random poses the arm couldn't recover from in
  the allotted steps — which is realistic: that's the kind of failure data a
  robust policy needs to train on.
- Two ordering lessons: base transforms must be set through the USD API
  **before** `timeline.play()` (calling `set_world_poses` before play leaves
  all 50 robots interpenetrating at the origin), and every robot's DOF
  positions/targets are initialized before the first physics step.

Flags: `--n 20` for fewer instances, `--seed N` for reproducible random
starts, `--same-init` for debugging (all robots from Q_HOME). GUI loops with
fresh random starts each cycle; `--headless` runs once and prints per-robot
errors.

---

## 2. Tools and frameworks

| Tool | Version / location | Role in this project |
|---|---|---|
| **NVIDIA Isaac Sim** | 6.0.1 (Omniverse, USD + PhysX) | Interactive/GPU simulation stage. The current pick-and-place sims run here; it's the "out-of-box sim with ready-made arms" the user asked for on 10/11. `SimulationApp` first, then `isaacsim.*` imports. |
| **MuJoCo** | ≥ 3.2.0 (CPU) | Stable reference simulator. The canonical arm was validated here (6 sampled embodiments, 2000 steps each, no NaNs), and its DLS IK (`sim/ik.py`, two-tier with adaptive restarts) solved the keyframes the Isaac scripts play back. |
| **MuJoCo Menagerie** | Franka Emika Panda (MJCF) | Ready-made Franka model. Lives at `mujoco_menagerie-main/` on the Windows box; `isaac_hello.py` converts `panda.xml` → USD, and both pick-and-place scripts load `assets/panda/panda.usda`. |
| **NVIDIA Newton** (+ MuJoCo-Warp) | planned, needs CUDA | The *target* physics engine for sim-only policy training — NVIDIA's new solver (Isaac Lab 3.0 EA supports kit-less `newton_mjwarp`). `newton` / `warp-lang` are commented out in `requirements.txt` until the 3080 Ti box runs them. |
| **USD + Blender** (`sim/mjcf_to_usd.py`) | `usd-core` | Visualization path. Bakes MuJoCo forward kinematics to USD (capsule/box/cylinder/plane/sphere, colors, Z-up): `assets/scene.usd` static, `assets/demo_0000.usd` time-sampled from a demo `.npz`, scrubbable in Blender. |
| **Python** | 3.12.10, `.venv-isaac` | Isaac needs 3.12 (its own venv, alongside the 3.14 project venv) with `isaacsim[all,extscache]==6.0.*` from `pypi.nvidia.com` (~10 GB install). |
| **Compute** | RTX 3080 Ti (12 GB) Windows box | Everything heavy runs here over SSH (`ssh biotech-win`, Tailscale). Work dir: `C:\Users\varun\workspace\BiotechRobot`. |
| **Policy math** | ChainedDiffuser (Zhou Xian, CoRL 2023) | Design basis: analytic task-space keyposes → diffusion-refined keyposes → diffusion-connected joint trajectories (Section 3). Chosen over pixel-goal imagination, which risks hallucinating impossible end states. |

**Physics choice in one line:** Isaac Sim is where you *see* and iterate the
task today; MuJoCo is the trusted reference that solved the motions; Newton
(MuJoCo-Warp) is where the mass data-generation + training will happen on
the 3080 Ti; Blender/USD is how you *watch* the MuJoCo-side demos.

---

## 3. Policy stack (what the sims are feeding)

The sims generate data for a two-level diffusion policy, following
Zhou Xian's ChainedDiffuser:

```
camera images
    │  perception/train_pose_net.py  →  (tray_xyz, goal_xyz)        [3-view CNN]
    ▼
perception/keyposes.py (analytic)    →  pre_grasp → grasp → lift → transit → pre_place → place
    │  KeyposeDiffusion  →  refined keypose seq  (6 × 7: pos + quat)
    ▼
TrajectoryDiffusion      →  joint-space action chunks  (T, 6)
```

- **`policies/diffusion.py`** — conditional DDPM core, shared 1-D
  FiLM-conditioned UNet, cosine beta schedule; shape-tested on CPU.
- **`sim/make_demos.py`** — generates keypose-annotated demo `.npz` files:
  analytic keyposes → DLS IK with adaptive restarts (records *achieved*
  poses, so the diffuser learns physically reachable sequences, not
  idealized ones) → collision-free kinematic planning → inverse-dynamics
  torque check (< 90 % of motor limit). 5/5 test episodes pass.
- **`policies/train_keypose.py` / `train_traj.py`** — implemented and tested
  end-to-end on synthetic demos (loss decreases, checkpoints save).
  Run on the 3080 Ti after real demos: `train_keypose.py` learns
  `(tray_init, tray_goal) → keypose sequence`; `train_traj.py` learns
  `(keypose_a, keypose_b) → joint chunk`.
- **Still TODO:** generating real demos on the GPU box, perception-net
  training, inference chaining, closed-loop evaluation, gripper grasp
  contact physics.

---

## 4. Codebase structure

```
biotech-lab-robotics/
├── configs/canonical_arm.yaml   # Frozen 6-DOF hardware design (pedestal mount,
│                                #   0.12 m rigid tool extension, 160 Nm gears)
├── embodiments/arm_generator.py # MJCF generator for the canonical arm
├── sim/
│   ├── isaac_hello.py           # Hello-world: Menagerie Franka → USD → stage
│   ├── isaac_pick_place.py      # Single pick-and-place (MuJoCo-IK keyframes)
│   ├── isaac_pick_place_50.py   # 50 parallel instances, randomized starts
│   ├── make_demos.py            # Keypose-annotated demo generator (.npz)
│   ├── ik.py                    # DLS IK, two-tier, adaptive restarts
│   ├── camera_sim.py            # Scene + randomized wrist/lab cameras
│   ├── run_arm.py               # MuJoCo PD trajectory validation
│   └── mjcf_to_usd.py           # MuJoCo → USD exporter for Blender
├── perception/
│   ├── keyposes.py              # Analytic task-space keypose bottlenecks
│   └── train_pose_net.py        # 3-view CNN → tray/goal poses
├── policies/
│   ├── diffusion.py             # Conditional DDPM core (1-D FiLM UNet)
│   ├── train_keypose.py         # High-level: (tray_init, tray_goal) → keyposes
│   ├── train_traj.py            # Low-level: keypose pair → joint chunk
│   └── README.md                # Policy design doc
├── assets/                      # panda.usda, scene.usd, demo .npz files
├── WINDOWS_SETUP.md / setup_windows.ps1 / requirements.txt
└── README.md                    # Project overview + status checklist
```

**Git state:** local has 4 commits (`45bcbfb` head: 50-instance script;
`71b04f9` pick-and-place; `30168fc` GUI fix; `57da1fd` hello-world on top of
initial commit). GitHub
([varunpra97/biotechRobot](https://github.com/varunpra97/biotechRobot))
holds the same code as one squashed commit `cd917b26` — the Git Data API push
collapses history; cosmetic only, content identical.

**How to run the current sims** (Isaac venv, repo root on the Windows box):
```powershell
cd C:\Users\varun\workspace\BiotechRobot\biotech-lab-robotics
C:\Users\varun\workspace\BiotechRobot\.venv-isaac\Scripts\Activate.ps1
python sim\isaac_pick_place.py        # GUI, loops until closed
python sim\isaac_pick_place_50.py     # GUI, 50 instances
python sim\isaac_pick_place_50.py --headless --n 50 --seed 0
```

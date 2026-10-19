# Biotech Lab Robotics — Isaac Sim

An Isaac Sim (6.0.1, PhysX) simulation of a biotech lab workstation: a
Franka Panda on a pedestal facing an open bench with labware. It generates
15 Hz multi-camera robot episodes as fine-tuning data for Physical
Intelligence's π0.5 VLA (openpi, `pi05_droid` checkpoint), so the physical
accuracy of what the cameras see is the main requirement.

## Scene

- **Room**: 10m x 6m x 3m, floor + walls, dome light + overhead work light.
- **Bench** (1.4m x 0.7m) with real CAD-derived props:
  - 5x4 test tube holder: parametric CAD with 8.0mm blind bores, 28mm deep
  - 20 test tubes: 7mm OD x 58mm, conical tip, standing on the bore floors
  - 10-well (5 x 2) sample tray, 52 x 24 x 14.4mm, 9mm SBS well pitch:
    narrow enough to grip across its width
  - electronic pipette (1000 µL), 100-tip rack (10 x 10, tips standing
    upright, collar up), waste bin
  - liquids in 3 tubes, drawn as volumes that fill each container's real
    inner profile (`sim/liquids.py`)
  - centrifuge, lid off, with a fixed-angle rotor: six 8.0mm bores tilted
    ~45° from vertical, conical floors
- **Arm**: Franka Panda, fixed base, on a pedestal level with the bench.
- **Cameras** (8):
  - `arm_cam0`, `arm_cam1` — behind the arm, arm + bench in view
  - `platform_cam0`, `platform_cam1` — behind the bench, looking back at the
    arm; `platform_cam1` is wide enough to see the whole platform and is the
    policy's fixed exterior camera
  - `top` — straight down over the bench
  - three wrist cameras, small 25mm modules on aluminium brackets bolted
    to the hand (part of the hand link: they move and collide with the
    gripper, and the planner checks them; `sim/lab_scene.py`
    `WRIST_CAMERAS`, brackets from `../../cad_source/make_wrist_camera_mount.py`).
    No single wrist view showed every primitive well, so:
    - `wrist_top` — beside the fingers looking straight down the approach
      axis: the grip, into the held tube, the holder slot or centrifuge
      bore right below
    - `wrist_side` — nozzle side at fingertip level: the grip on the
      pipette, the whole pipette, the tip and what it works in
    - `wrist_high` — above the gripper on the head side, biased sideways:
      the whole held object and its surroundings (tray, bench spots)
    All three are recorded; each primitive's episodes train π0.5 on the
    one that shows it best (`sim/lab_env.py` `PRIMITIVE_WRIST_CAMERA`)

## Physical accuracy

Each of these was checked against the simulator, not assumed:

- **Geometry is measured, not guessed.** Rotor bore axes, radius and
  floor come from the centrifuge mesh (`sim/centrifuge_geometry.py`). The
  tube and holder were sized to fit the rotor (0.5mm radial clearance).
- **Colliders match the geometry.** Static fixtures (holder, centrifuge)
  collide as their exact triangle meshes, so the narrow bores stay
  exact. Tubes collide as a 40-vertex convex hull. PhysX caps hulls at 64
  vertices and inflates a reduced hull, which made tubes wider than the
  bores.
- **240 Hz physics.** At 60 Hz a released 7mm tube steps through the
  rotor's 0.8mm sleeve wall in a single step.
- **Real fingertip.** The TCP is measured from the Franka finger
  geometry: hand-local z = 0.1123m, fingers open along local Y.
- **6-DOF IK.** Analytic Franka kinematics (`sim/franka_kinematics.py`),
  calibrated against the simulated arm to 0.03mm / 0.002°, with damped least
  squares and null-space joint centering.
- **Collision checks on real colliders.** PhysX overlap queries check every
  densely interpolated configuration (≤5mm / 3° apart), with held objects
  carried along.
- **Friction grasps.** Objects are held only by the finger pads' friction
  (pad colliders enabled, a force-limited spring drive, 64 solver
  iterations); nothing is welded to the hand. Targets are computed from the
  object's measured pose in the hand.
- **Pipette physics where PhysX has none.** A tip attaches when the nozzle
  is pressed 4-9mm into its collar, coaxial within 0.6mm / 3°; ejected tips
  fall under gravity; liquid moves only if the tip end is below the surface
  (else it draws air), and tip-to-wall/floor contact is logged
  (`sim/pipette.py`).

## Primitives (15 Hz, π0.5 format)

`sim/collect_primitives.py` runs every lab primitive as one physical chain
(the tube really is in the centrifuge when "remove" starts) and records each
primitive as its own episode with its own varied prompt (`sim/prompts.py`):

| Group | Primitives |
|---|---|
| Tubes | `pick_tube_from_holder`, `insert_tube_into_bore`, `remove_tube_from_bore`, `place_tube_into_holder` |
| Pipette | `pick_up_pipette`, `attach_tip`, `aspirate_from_tube`, `dispense_into_well`, `dispense_into_tube`, `eject_tip`, `return_pipette` |
| Sample tray | `pick_up_tray`, `place_tray` |

Each primitive checks its physical outcome (seat depth, tilt, volumes moved,
tip in the bin, nothing else disturbed) and the chain stops at a failure.

Control follows DROID, which π0.5 was trained on: 15 Hz joint-velocity
actions normalized to [-1, 1] (x 0.2 rad per step), plus gripper position
(0 open .. 1 closed), plus two electronic-pipette channels: plunger
fraction and tip eject. Action is 10-D; state is 10-D (7 joints, gripper,
plunger, tip attached).

Output in `data/primitives/<primitive>/ep_XXXX/`:

- `images/<camera>_<step>.jpg` — all eight cameras, every step
- `steps.npz` — joint positions and velocities, commanded joints, gripper,
  TCP and hand poses, all object poses, pipette state, actions, states
- `exterior_wrist.mp4` — the two policy cameras side by side
- `meta.json` — prompt, success, outcome measurements, action/state spec

## π0.5 fine-tuning data

`training/export_lerobot.py` converts successful episodes into a LeRobot
dataset with openpi's DROID layout: `exterior_image_1_left` (platform_cam1),
`wrist_image_left` (the episode's wrist camera, chosen per primitive), `joint_position`, `gripper_position`, `pipette_state`,
10-D `actions`, and the prompt as the task. Only those two cameras go to
the policy; the rest stay in the raw episodes. On the real robot the
controller has to feed the same wrist camera for the primitive it runs.

`training/pi05_lab.py` is the matching openpi train config for the RTX 3080
Ti: LoRA on `pi05_droid`, vision encoder frozen, batch 8. It reuses DROID's
norm stats, with the two pipette dims (zero-width in DROID) set to 0..1.

## Earlier single pick-and-place episode

`sim/collect_data_pickplace.py` generates one episode: pick `tube_r0_c2`
from the holder and place it into rotor bore 8.

1. Vertical grasp of the tube, with full-pose rigid attach while held.
2. Lift, then tilt the wrist so the tube is coaxial with the 45° bore.
3. Insert along the bore axis, with the fingers stopping ~3mm short of
   the rotor hub.
4. Open the gripper: the tube becomes a dynamic body and slides the last
   ~2mm into its seat under gravity.
5. Back out along the axis and return.

The episode is rejected, and nothing is saved, if any of these fail:

- any IK solve fails
- any configuration collides
- the tube doesn't seat: tip depth, axis within 4° of the bore's, at
  rest, contacts only with the centrifuge, ≤0.2mm penetration
- the retract moves the placed tube

Output in `data/episodes/ep_XXXX/`:

- `images/`, `depth/`, `segmentation/`: one file per camera per keyframe
  (10 keyframes)
- `trajectory.npz`:
  - keyframe joint positions and velocities, gripper
  - hand pose, TCP pose, all object poses
  - `action` (next keyframe's commanded joints)
  - the full validated dense joint path
- `meta.json`: phases, target bore geometry, grasp transform, seat
  measurements, physics settings

## Layout

| Path | What it is |
|---|---|
| `sim/lab_scene.py` | Builds the scene (shared by every script); GUI viewer + capture tool |
| `sim/collect_primitives.py` | Records every primitive as a 15 Hz episode (above) |
| `sim/lab_env.py` | 15 Hz DROID-style env: action integration, observations, episode recorder |
| `sim/expert.py`, `sim/pipette_expert.py` | Scripted primitives (tubes and tray; pipette) |
| `sim/planner.py`, `sim/franka_kinematics.py` | Collision-checked path planning; analytic Franka kinematics and IK |
| `sim/pipette.py`, `sim/liquids.py` | Electronic pipette model (tips, plunger, liquid transfer); liquid volumes |
| `sim/prompts.py` | Prompt templates per primitive |
| `sim/view_episodes.py` | Plays recorded episodes from the two policy cameras |
| `training/export_lerobot.py`, `training/pi05_lab.py` | LeRobot export; openpi LoRA fine-tune config |
| `sim/collect_data_pickplace.py` | Earlier single pick-and-place episode (keyframes) |
| `sim/collect_data.py` | Earlier task: approach and hover over a tube top (position-only IK, coarse collision check) |
| `sim/view_motion.py` | Replays a saved episode in the GUI: the validated dense path, live tube attach, same physical release |
| `sim/ik_solver.py` | Position-only and 6-DOF damped-least-squares IK |
| `sim/centrifuge_geometry.py` | Rotor bore axes and geometry, derived from the loaded mesh |
| `assets/panda/` | Franka Panda USD, converted from MuJoCo Menagerie (generated, gitignored) |
| `assets/cad/` | Scene meshes: holder, tubes, plate, pipette, centrifuge (generated, gitignored) |
| `setup_windows.ps1`, `WINDOWS_SETUP.md` | RTX 3080 Ti box setup |

`assets/cad/` meshes come from scripts in `../../cad_source/`, which is
**outside this repo**:

- `make_test_tube.py` (tube + collision hull)
- `make_sample_tray_5x2.py` (sample tray), `make_lab_consumables.py` +
  `process_consumables.py` (tips, tip rack, waste bin),
  `make_pipette_collision.py` (pipette collider from its cross-sections),
  `make_wrist_camera_mount.py` (wrist camera bracket + clearance checks)
- `make_tray_20holes.py` + `process_tray20.py` (holder)
- `process_assets.py` (downloaded CAD)

The measurement and probe scripts behind the numbers above are in
`../../cad_source/` too: `raycast_rotor_bore.py`, `gripper_rotor_clearance.py`,
`probe_*.py`.

## How to run

From `C:\Users\varun\workspace\BiotechRobot`, with the Isaac venv's Python:

```powershell
# Record every primitive as a 15 Hz episode (headless, ~35 min);
# --only tubes|pipette|tray runs one group
.venv-isaac\Scripts\python.exe biotech-lab-robotics\simulation\sim\collect_primitives.py --out data\primitives

# Watch them from the two policy cameras (space pause, n next, q quit)
.venv-isaac\Scripts\python.exe biotech-lab-robotics\simulation\sim\view_episodes.py data\primitives
.venv-isaac\Scripts\python.exe biotech-lab-robotics\simulation\sim\view_episodes.py data\primitives --primitive attach_tip
# ... or write them all to one captioned MP4
.venv-isaac\Scripts\python.exe biotech-lab-robotics\simulation\sim\view_episodes.py data\primitives --save data\primitives\all_primitives.mp4

# Generate the earlier pick-and-place episode (headless, ~3 min)
.venv-isaac\Scripts\python.exe biotech-lab-robotics\simulation\sim\collect_data_pickplace.py --out data\episodes

# Watch it: all cameras tiled, or one camera
.venv-isaac\Scripts\python.exe biotech-lab-robotics\simulation\sim\view_motion.py --multi-view
.venv-isaac\Scripts\python.exe biotech-lab-robotics\simulation\sim\view_motion.py --camera wrist_top

# Static scene: view, list cameras, capture a PNG per camera
.venv-isaac\Scripts\python.exe biotech-lab-robotics\simulation\sim\lab_scene.py --multi-view
.venv-isaac\Scripts\python.exe biotech-lab-robotics\simulation\sim\lab_scene.py --list-cameras
.venv-isaac\Scripts\python.exe biotech-lab-robotics\simulation\sim\lab_scene.py --capture --out lab_captures
```

Export to LeRobot and fine-tune, in openpi's environment (WSL):

```bash
cd ~/openpi
uv run python /mnt/c/Users/varun/workspace/BiotechRobot/biotech-lab-robotics/simulation/training/export_lerobot.py --src /mnt/c/Users/varun/workspace/BiotechRobot/data/primitives
uv run python /mnt/c/Users/varun/workspace/BiotechRobot/biotech-lab-robotics/simulation/training/pi05_lab.py --exp-name lab_v1
```

`view_motion.py` options: `--once` plays a single time; `--substeps N`
sets playback speed (default 3; higher is slower).

## Status (2026-10-18)

- [x] Lab scene with real CAD labware, 8 cameras (incl. three bracket-mounted wrist cameras)
- [x] Labware sized to fit together: tube ↔ holder ↔ centrifuge rotor bore;
  tray narrow enough to grip; tips that press-fit onto the pipette nozzle
- [x] Friction-based grasping (no rigid attach)
- [x] 15 Hz DROID-convention env with joint position and velocity logging
- [x] Primitives with physical outcome checks: tubes, pipette, sample tray
- [x] LeRobot export + openpi LoRA config (data path checked on CPU)
- [ ] Many episodes with variation: randomized layouts, sources/targets,
  volumes, lighting; 10-20% chained multi-step episodes
- [ ] Fine-tune π0.5 on the 3080 Ti and evaluate it in the sim
- [ ] Semantic segmentation labels: instance masks are saved, but objects
  carry no class labels yet
- [ ] Centrifuge lid as an articulated hinge (currently omitted)
- [ ] Bring `cad_source/` and its generated meshes under version control
- [ ] NVIDIA Newton backend: attempted, but the `isaacsim.physics.newton`
  extension crashes in this install; PhysX is used

Known caveats:

- `view_motion.py` re-simulates the release, so the seated tube can come
  to rest leaning ~1mm differently than in the recorded episode (same
  seat, same bore).
- `collect_data.py` predates the accuracy work above. It still uses
  position-only IK and a coarse bounding-box collision check.
- A 1000 µL tip can't reach far down a 6mm-bore tube before its widening
  body touches the wall, so a tube can be aspirated from only while it
  holds more than ~570 µL; `aspirate_from_tube` refuses below that.

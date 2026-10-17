# Biotech Lab Robotics — Isaac Sim

An Isaac Sim (6.0.1, PhysX) simulation of a biotech lab workstation: a
Franka Panda on a pedestal facing an open bench with labware. It generates
multi-camera robot episodes as training data for a Cosmos-style world model
and a π0-style VLA policy, so the physical accuracy of what the cameras see
is the main requirement.

## Scene

- **Room**: 10m x 6m x 3m, floor + walls, dome light + overhead work light.
- **Bench** (1.4m x 0.7m) with real CAD-derived props:
  - 5x4 test tube holder: parametric CAD with 8.0mm blind bores, 28mm deep
  - 20 test tubes: 7mm OD x 58mm, conical tip, standing on the bore floors
  - 96-well plate (SBS standard), 2 pipettes
  - centrifuge, lid off, with a fixed-angle rotor: six 8.0mm bores tilted
    ~45° from vertical, conical floors
- **Arm**: Franka Panda, fixed base, on a pedestal level with the bench.
- **Cameras** (6):
  - `arm_cam0`, `arm_cam1` — behind the arm, arm + bench in view
  - `platform_cam0`, `platform_cam1` — behind the bench, looking back at the arm
  - `top` — straight down over the bench
  - `wrist` — eye-in-hand: a child prim of the hand link, so it moves with
    the gripper like a camera bolted to the real one; mounted off the
    fingers' centerline so a grasped object doesn't block it

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
- **6-DOF IK.** Damped least squares on a Jacobian whose layout was
  verified by finite differences, with null-space joint-limit avoidance.
- **Collision checks on real colliders.** PhysX overlap queries check every
  densely interpolated configuration between keyframes, not just the
  keyframes.

## Pick-and-place task

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
| `sim/collect_data_pickplace.py` | Pick-and-place episode generator (above) |
| `sim/collect_data.py` | Earlier task: approach and hover over a tube top (position-only IK, coarse collision check) |
| `sim/view_motion.py` | Replays a saved episode in the GUI: the validated dense path, live tube attach, same physical release |
| `sim/ik_solver.py` | Position-only and 6-DOF damped-least-squares IK |
| `sim/centrifuge_geometry.py` | Rotor bore axes and geometry, derived from the loaded mesh |
| `assets/panda/` | Franka Panda USD, converted from MuJoCo Menagerie (generated, gitignored) |
| `assets/cad/` | Scene meshes: holder, tubes, plate, pipette, centrifuge (generated, gitignored) |
| `setup_windows.ps1`, `WINDOWS_SETUP.md` | RTX 3080 Ti box setup |

`assets/cad/` meshes come from scripts in `../cad_source/`, which is
**outside this repo**:

- `make_test_tube.py` (tube + collision hull)
- `make_tray_20holes.py` + `process_tray20.py` (holder)
- `process_assets.py` (downloaded CAD)

The measurement and probe scripts behind the numbers above are in
`../cad_source/` too: `raycast_rotor_bore.py`, `gripper_rotor_clearance.py`,
`probe_*.py`.

## How to run

From `C:\Users\varun\workspace\BiotechRobot`, with the Isaac venv's Python:

```powershell
# Generate the pick-and-place episode (headless, ~3 min)
.venv-isaac\Scripts\python.exe biotech-lab-robotics\sim\collect_data_pickplace.py --out data\episodes

# Watch it: all six cameras tiled, or one camera
.venv-isaac\Scripts\python.exe biotech-lab-robotics\sim\view_motion.py --multi-view
.venv-isaac\Scripts\python.exe biotech-lab-robotics\sim\view_motion.py --camera wrist

# Static scene: view, list cameras, capture a PNG per camera
.venv-isaac\Scripts\python.exe biotech-lab-robotics\sim\lab_scene.py --multi-view
.venv-isaac\Scripts\python.exe biotech-lab-robotics\sim\lab_scene.py --list-cameras
.venv-isaac\Scripts\python.exe biotech-lab-robotics\sim\lab_scene.py --capture --out lab_captures
```

`view_motion.py` options: `--once` plays a single time; `--substeps N`
sets playback speed (default 3; higher is slower).

## Status (2026-10-17)

- [x] Lab scene with real CAD labware, 6 cameras (incl. moving eye-in-hand)
- [x] Labware sized to fit together: tube ↔ holder ↔ centrifuge rotor bore
- [x] Pick-and-place into the centrifuge: 1 validated 10-keyframe
  episode. The tube seats physically at s = −15.7mm along the bore,
  2.8° from its axis, with no penetration.
- [x] Playback viewer that reproduces the validated motion, single or multi-camera
- [ ] Many episodes with variation: source tube, target bore, poses, lighting
- [ ] Friction-based grasping (the grasp is a rigid attach; only the release is simulated)
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

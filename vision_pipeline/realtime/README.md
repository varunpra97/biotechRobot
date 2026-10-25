# Real-time scene localizer (wrist cameras → object coordinates for the planner)

The offline pipeline (`../README.md`) builds the scene map once: every object
on the bench as a mesh, a world pose, and named targets. This directory tracks
that map live from the robot's cameras. Each control step it serves every
object's world coordinates to the VLA + RL planner.

```
offline map ──► SceneLocalizer ──► get_prim_world_pose(path)   (drop-in for lab_scene's)
                  ▲   ▲   ▲        target(name, "grasp" | "slot_A5" | "bore_2" | "well_B3" ...)
     wrist RGB ───┘   │   │        state_message()  (all objects, JSON)
     hand pose + q ───┘   │
     held object ─────────┘  (the planner knows what it grasped)
```

## Using it

**In-process** (needs open3d, `.venv-cad`):

```python
from scene_model import scene_from_sim          # or the offline pipeline's export
from localizer import SceneLocalizer, PerceivedPose

loc = SceneLocalizer(scene_from_sim(meta, poses))           # cameras: wrist_top, wrist_side
# every control step (15 Hz)
loc.step({"wrist_top": rgb_top, "wrist_side": rgb_side}, hand_pose7, finger_pos, q,
         held="tube_r0_c4" if grasped else None)
pose_fn = PerceivedPose(loc)                                # -> TeacherFinisher(pose_fn=...)
pose_fn("/World/Platform/Tube_r0_c4")                       # [x, y, z, qw, qx, qy, qz]
loc.target("holder", "slot_A5")                             # world xyz of the bore opening
loc.state_message()                                         # every object, see below
```

**As a server**, for the Isaac process: `.venv-isaac` has no open3d.
`client.py` and `wire.py` need only numpy and the standard library.

```bash
.venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/realtime/server.py --port 5577
```

```python
from client import LocalizerClient
loc = LocalizerClient()
loc.init({"object_names": names, "layout": layout}, poses)   # the scene map
loc.step({"wrist_top": rgb, "wrist_side": rgb}, hand_pose7, finger_pos, q, held=None)
TeacherFinisher(env, kin, tube_path, pose_fn=loc.get_prim_world_pose)
loc.info("tube_r0_c4")   # {"source": observed|tracked|parent|hand|map, "sigma_mm", "steps_since_seen"}
```

`step()` is pipelined: it sends this step's images and returns at once. The
poses you read answer the previous step (one step stale); use `wait=True` for
fresh poses that block.

`state_message()` gives, per object: `name`, `class`, `pose` (xyz + wxyz),
`sigma_mm` (1-σ position uncertainty), `observed` and `support` (whether the
cameras supported it this step, and with how many edge samples), and `source`:

| `source` | where the pose comes from |
|---|---|
| `observed` | a camera supported it this step |
| `tracked` | seen before, not this step |
| `parent` | seated, placed on its holder or rack's estimate, not seen itself |
| `hand` | held, carried with the hand |
| `map` | never seen |

It also gives `steps_since_seen`, `frame` (`world`, `parent` or `hand`), and
`targets`, every named point in world xyz:

| class | targets |
|---|---|
| tube | `tip`, `rim`, `grasp` (rim − 12 mm, the RL teacher's), `center` |
| holder | `slot_A1` … `slot_D5` (bore openings; row A nearest the arm, column 1 leftmost from the arm) |
| tip_rack / tip | `pos_A1` … `pos_J10` / `top`, `end` |
| tray | `well_A1` … `well_B5` |
| centrifuge | `bore_1` … `bore_6` (rotor bore openings), `hub` |
| pipette, waste_bin, all | `center`, `top` (+ `nozzle`, `drop`) |

A `held` claim is refused unless the object is within 3 cm of the fingertip
centre. This stops a grasp that closed on nothing from dragging the object off
with the hand.

## How it works

`scene_tracker.py` documents the method in full. In short:

- **Edge tracking:** each object's mesh is tracked against image edges.
- **Occlusion:** the robot's own hand, fingers and camera bracket are raycast
  (`robot_body.py`), so an object hidden behind the gripper is known to be
  hidden.
- **Anchoring:** each frame is solved against the object's anchor and fused
  across frames:
  - the map pose;
  - the pose relative to its parent, for a tube seated in a holder or a tip in
    the rack;
  - the pose relative to the hand, while held.

  A seated object's correction is clamped to its opening's clearance, so the
  target tube inherits the holder's accuracy even while the gripper hides it.
- **Grasp:** when the gripper closes on a tube, the tube is placed on the
  finger centre line.

Things that each made a measured difference (episode `ep_000`, every object's
map pose off by up to 3 mm):

| fix | what went wrong before |
|---|---|
| image-time hand pose (`image_lag` 0.25 step) | the recorded images are a quarter step older than the recorded hand pose; at 10 mm/step every edge was 1–2 px off (MAD 1.8 → 0.25 px) |
| creases located by surface continuity | a box's top edge with its side face in view was placed down the side face, biased by pose: objects drifted |
| anchored solve + fusion (not recursive re-centring) | a small pose-dependent bias walked weakly observed directions without bound (waste bin 10 mm) |
| seated-in-parent anchors with clearance clamp | glass and liquid columns pulled tube tilt 3–4°: the grasp point 46 mm up moved 3–5 mm |
| resolution gate (≤ 1.5 mm/px) | the 1.6 m exterior camera (3 mm/px) can't refine to sub-mm and its 9 mm-pitch tip rack is ambiguous at that scale |
| grasp centring | the closing fingers push the tube ~5 mm onto their centre line |

## Results

Nine recorded `pick_tube_from_holder` episodes
(`data/eval/finisher_fixed`) were replayed through the localizer with both
wrist cameras, RGB only, at the default settings (`stride=4`, `iters=2`).

- **Start:** every object's map pose was disturbed by up to ±3 mm and ±1° yaw.
  Seated tubes and tips sit on their disturbed parent, plus ±0.3 mm of their
  own.
- **Inputs:** only the images, hand pose, finger openings and joints, plus
  which object is held.
- **Scoring:** against the simulator's true poses. The table compares each
  object's key point (tube grasp point, holder slot openings, other objects'
  centres) at the map and at the end of the replay, for objects a camera
  supported.

| | map error (median) | tracked: median | p95 | max |
|---|---|---|---|---|
| **target tube grasp point, last 30 steps before the grasp** (270 samples) | 2–3 mm | **0.28 mm** | 1.66 mm | 2.07 mm |
| target tube in hand, after the grasp (about 190 samples) | | 2.05 mm | 4.58 mm | 5.65 mm |
| holder, worst of its 20 slot openings (9) | 3.55 mm | 0.33 mm | 1.53 mm | 2.23 mm |
| tubes, grasp points (158) | 2.83 mm | 0.36 mm | 2.07 mm | 4.58 mm |
| centrifuge (9) | 2.63 mm | 1.33 mm | 3.97 mm | 4.80 mm |
| tray (9) | 2.48 mm | 1.21 mm | 4.84 mm | 5.99 mm |
| pipette (9) | 1.49 mm | 1.21 mm | 2.15 mm | 2.43 mm |

The planner's budget is ≤ 2 mm lateral typical (≤ 3 mm p95), ≤ 3 mm
vertical and ≤ 1–2° axis. The approach meets it; in-hand tracking is
borderline.

Time per step: median 67 ms on the CPU (2 wrist cameras, 24-thread desktop;
range 56–91 ms per episode). Through the server it runs pipelined beside the
control loop.

Where it is weak:

- **Episode 5's layout:** the holder wanders 1–3 mm in every configuration
  tried, so its target ends at 1.6–2.1 mm. Not yet understood.
- **Objects the wrist cameras see only from afar:** centrifuge, tray and
  pipette improve only partly. Views coarser than 1.5 mm/px aren't used, so
  such objects stay at their map pose until the arm works near them.
- **Never refined:** the tip rack (ambiguous 9 mm grid; tips follow it) and the
  bench.
- **In hand:** the fingers hide most of the tube, so tilt is uncertain (up
  to ~3°).
- **During the final approach:** the target is usually hidden behind the
  wrist_top camera bracket and is outside wrist_side's view. Its pose is then
  the holder-anchored estimate, flagged `source: "parent"`, not a live view.

## Checks

```bash
P=.venv-cad/Scripts/python.exe; R=biotech-lab-robotics/vision_pipeline/realtime/checks
E=data/eval/finisher_fixed/pick_tube_from_holder/ep_000
$P $R/check_camera_model.py $E out.jpg                 # true tube pose drawn through the wrist camera model
$P $R/check_scene_render.py $E out.jpg 0,150,300       # robot body + every object's contours on the frames
$P $R/eval_scene_tracker.py $E --perturb-mm 3          # replay + score every object against the true poses
```

## Files

| file | what |
|---|---|
| `camera_model.py` | wrist cameras (hand pose × mount, from `lab_scene.WRIST_CAMERAS`), exterior camera, pose interpolation |
| `robot_body.py` | Franka hand / fingers / bracket meshes (MuJoCo Menagerie + the mount CAD), arm forward kinematics anchored at the hand pose |
| `scene_model.py` | scene map objects: mesh, pose, named targets, parent (seated) relations |
| `scene_tracker.py` | the tracker |
| `localizer.py` | in-process API: `SceneLocalizer`, `PerceivedPose`, `state_message()` |
| `server.py`, `client.py`, `wire.py` | the localizer as a process; numpy-only client for Isaac |
| `checks/` | camera-model overlay, render overlay, replay evaluation |

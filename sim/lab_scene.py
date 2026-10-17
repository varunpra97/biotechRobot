"""Biotech lab setup: open test-tube platform + Franka arm on a pedestal.

Room (10m x 6m x 3m), no enclosure. One open platform (1.4m x 0.7m, bench
height) holds a 3x4 test tube rack, a sample tray with wells, and 2
pipettes. One fixed Franka arm sits on a pedestal facing the platform,
base height matched to the platform so it can reach everything on it. The
arm is static -- held in its home pose, no motion -- this script only
builds and lets you view/capture the scene.

Cameras (6 total):
  - 2 behind the robot arm (arm + tube platform both in view)
  - 2 behind the tube platform (pointing back at the arm)
  - 1 straight-overhead top-down view of the platform
  - 1 eye-in-hand: a real child prim of the hand link, so it physically
    moves and rotates with the end effector every frame (like a camera
    actually bolted to the gripper on the real robot), not a position
    computed once and left static

Usage (Isaac venv, repo root on Windows box):
    python sim\\lab_scene.py                  # GUI, static scene
    python sim\\lab_scene.py --headless       # headless sanity check, no window
    python sim\\lab_scene.py --capture        # save PNGs from all 5 cameras, exit
    python sim\\lab_scene.py --camera wrist   # GUI, viewport starts at this camera
    python sim\\lab_scene.py --multi-view     # GUI, all 5 cameras tiled at once
    python sim\\lab_scene.py --list-cameras   # print camera names and exit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Layout constants
# ----------------------------------------------------------------------------
ROOM_X, ROOM_Y, ROOM_Z = 10.0, 6.0, 3.0
PLATFORM_W, PLATFORM_D = 1.4, 0.7      # open platform footprint
BENCH_Z = 0.9                          # platform / pedestal top height
PLATFORM_CX, PLATFORM_CY = 0.0, 1.8    # platform center
# Fixed arm base, facing the platform. Y chosen so the rack (the farthest
# task objects, ~0.2-0.6m into the platform from its near edge) fall within
# the Franka's ~0.85m kinematic reach -- verified via sim/ik_solver.py: at
# the previous ARM_POS y=0.9, the nearest rack tube was ~1.05m from the
# hand's home position and IK could not converge on it at all.
ARM_POS = (0.0, 1.1, BENCH_Z)

FRANKA_USD = str(
    Path(r"C:\Users\varun\workspace\BiotechRobot")
    / "biotech-lab-robotics" / "assets" / "panda" / "panda.usda"
)

# Franka home pose (7 arm joints + 2 fingers) -- the arm is static, held here.
Q_HOME = np.array([0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04])

HAND_PATH = "/World/Arm/Geometry/link0/link1/link2/link3/link4/link5/link6/link7/hand"


def get_prim_world_pose(path):
    """World pose [x,y,z, qw,qx,qy,qz] of a prim, or None if it doesn't exist."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Gf, UsdGeom
    import omni.timeline

    prim = stage_utils.get_current_stage().GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        omni.timeline.get_timeline_interface().get_current_time())
    tf = Gf.Transform(m)
    t = tf.GetTranslation()
    q = tf.GetRotation().GetQuat()
    return np.array([t[0], t[1], t[2], q.GetReal(), *q.GetImaginary()])


def set_prim_world_position(path, world_pos):
    """Set a prim's world position via a direct translate, keeping it at
    identity orientation (position-only kinematic update -- used for the
    "tube rides with the gripper" attach during pick-and-place; the tube
    has no authored rotation to begin with, so this doesn't lose anything,
    and a tube staying upright while carried is a reasonable look)."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Gf, UsdGeom

    prim = stage_utils.get_current_stage().GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(world_pos[0]), float(world_pos[1]), float(world_pos[2])))


LEFT_FINGER_PATH = HAND_PATH + "/left_finger"
RIGHT_FINGER_PATH = HAND_PATH + "/right_finger"
# Measured (sim/dev notes): the finger pad's lowest point sits ~0.045m below
# the left_finger/right_finger body origin (verified by traversing the pad's
# collision boxes and taking their lowest world Z at the home pose). Padded
# slightly (measured ~0.04m) so this stays conservative -- err on the side
# of stopping short of a surface, not through it.
FINGER_PAD_DEPTH = 0.045


def get_fingertip_pos():
    """World position of the gripper's control point: the midpoint between
    the two finger bodies, offset down by FINGER_PAD_DEPTH to approximate
    the actual pad tips (not the finger mount frames). This is what IK
    targets should be defined against for any task that touches/approaches
    an object -- the hand frame origin is ~8cm above the real contact
    point, and targeting it directly drives the fingers through whatever
    the hand is "above" (this was the root cause of a real, visually
    confirmed interpenetration bug in the first version of this task)."""
    lf = get_prim_world_pose(LEFT_FINGER_PATH)
    rf = get_prim_world_pose(RIGHT_FINGER_PATH)
    if lf is None or rf is None:
        return None
    mid = (lf[:3] + rf[:3]) / 2.0
    mid[2] -= FINGER_PAD_DEPTH
    return mid


MM_TO_M = 0.001

CAD_DIR = str(
    Path(r"C:\Users\varun\workspace\BiotechRobot")
    / "biotech-lab-robotics" / "assets" / "cad"
)
TUBE_HOLDER_OBJ = f"{CAD_DIR}/tube_holder.obj"      # 5x4 bore rack, 25.98mm holes
SAMPLE_TRAY_OBJ = f"{CAD_DIR}/sample_tray.obj"      # real 96-well plate (SBS standard)
PIPETTE_OBJ = f"{CAD_DIR}/pipette.obj"
CENTRIFUGE_OBJ = f"{CAD_DIR}/centrifuge.obj"
TEST_TUBE_OBJ = f"{CAD_DIR}/test_tube.obj"          # procedural hollow tube, 11mm OD (see cad_source/make_test_tube.py)

# Measured directly from the holder mesh (cad_source/rack -- circle-fit on
# cross-sections, not guessed from its misleading GrabCAD title): 20 bores
# in a clean 5x4 grid, local to the holder's own mesh origin.
HOLDER_HOLE_X_MM = [-123.0, -61.5, 0.0, 61.5, 123.0]
HOLDER_HOLE_Y_MM = [-92.3, -30.8, 30.7, 92.2]
HOLDER_THICKNESS_MM = 40.0

# Rigid-body physics materials, per material-type friction/restitution
# guidance (glass/polypropylene labware: low restitution, friction
# ~0.15-0.25). Applied via GeomPrim.apply_physics_materials(), not just
# collision shape -- collision alone has no friction/bounce behavior.
PHYSICS_MATERIALS = {
    "glass": dict(static_friction=0.2, dynamic_friction=0.2, restitution=0.05),
    "plastic": dict(static_friction=0.3, dynamic_friction=0.25, restitution=0.1),
    "metal": dict(static_friction=0.4, dynamic_friction=0.35, restitution=0.15),
}
_material_cache = {}


def _get_physics_material(name):
    if name not in _material_cache:
        from isaacsim.core.experimental.materials import RigidBodyMaterial

        spec = PHYSICS_MATERIALS[name]
        _material_cache[name] = RigidBodyMaterial(
            paths=f"/World/PhysicsMaterials/{name}",
            static_frictions=spec["static_friction"],
            dynamic_frictions=spec["dynamic_friction"],
            restitutions=spec["restitution"],
        )
    return _material_cache[name]


# ----------------------------------------------------------------------------
# Scene construction
# ----------------------------------------------------------------------------
def load_mesh_prim(prim_path, obj_path, pos, yaw_deg=0.0, pitch_deg=0.0,
                    roll_deg=0.0, color=None, collision=True, material="plastic"):
    """Load a triangle mesh from an OBJ file as a USD mesh prim, with
    PhysX convex-decomposition collision and a bound rigid-body physics
    material (friction/restitution) -- real CAD-derived geometry instead
    of a box/cylinder approximation."""
    import trimesh
    from pxr import Gf, UsdGeom
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.prims import GeomPrim
    from pxr import UsdPhysics

    mesh = trimesh.load(obj_path, force="mesh")
    stage = stage_utils.get_current_stage()
    usd_mesh = UsdGeom.Mesh.Define(stage, prim_path)
    usd_mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in mesh.vertices])
    usd_mesh.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    usd_mesh.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist())
    if color:
        usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    xf = UsdGeom.Xformable(usd_mesh)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if roll_deg:
        xf.AddRotateXOp().Set(float(roll_deg))
    if pitch_deg:
        xf.AddRotateYOp().Set(float(pitch_deg))
    if yaw_deg:
        xf.AddRotateZOp().Set(float(yaw_deg))

    if collision:
        geom = GeomPrim(paths=prim_path, apply_collision_apis=True)
        UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath(prim_path)) \
            .GetApproximationAttr().Set("convexDecomposition")
        geom.apply_physics_materials(_get_physics_material(material))

    return prim_path, tuple(float(v) for v in mesh.extents)


def load_centrifuge_open(prim_path_prefix, obj_path, pos, roll_deg=90.0,
                          hinge_angle_deg=35.0, include_lid=False,
                          color_body=None, color_lid=None, collision=True,
                          material="metal"):
    """Load the centrifuge mesh split into (body+rotor) and (lid), with the
    lid rotated open about a hinge pivot at its own bottom-back edge --
    it stays attached/touching the body there and swings up and back,
    instead of being translated away into a detached-looking floating
    piece (the first version of this did that; visibly wrong, fixed here).
    Not a real hinge joint/articulation yet, just a static open pose --
    per-user, fine for now, a real hinge is future work once this needs
    to actually animate.

    The mesh is a single fused STEP/STL export, but trimesh.split() finds
    26 separate connected components inside it. Which ones are the lid
    (prim index 23, plus two small lid-top details at 0/1) vs. the body
    and its internal rotor was confirmed visually, not guessed -- see
    cad_source/test_centrifuge_parts.py, a color-coded isolated render.
    These indices are specific to this exact mesh export.
    """
    import math
    import numpy as np
    import trimesh
    from pxr import Gf, UsdGeom, UsdPhysics
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.prims import GeomPrim

    mesh = trimesh.load(obj_path, force="mesh")
    parts = mesh.split(only_watertight=False)

    LID_IDXS = {23, 0, 1}
    rx = math.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]])

    all_rot = mesh.vertices @ Rx.T
    z_off = -all_rot[:, 2].min()  # shared, so body and lid stay aligned

    # Hinge pivot: the lid's own bottom-back edge (its min-Z, max-Y point,
    # in the rolled+leveled frame) -- the edge that touches the body's rim
    # when the lid is closed, so rotating about it keeps that edge in
    # contact while the rest of the lid swings up and open. This is a
    # bounding-box-corner approximation, not the true contact seam, so a
    # visible gap can remain after rotation (confirmed by the user against
    # a render -- not caught by the bounding-box-only check used during
    # development); the snap-to-contact step below corrects for that
    # regardless of how accurate this pivot guess was.
    lid_sub = [p for i, p in enumerate(parts) if i in LID_IDXS]
    lid_rot = trimesh.util.concatenate(lid_sub).vertices @ Rx.T
    lid_rot[:, 2] += z_off
    pivot = np.array([
        (lid_rot[:, 0].min() + lid_rot[:, 0].max()) / 2,
        lid_rot[:, 1].max(),
        lid_rot[:, 2].min(),
    ])
    theta = math.radians(hinge_angle_deg)

    body_sub = [p for i, p in enumerate(parts) if i not in LID_IDXS]
    body_verts_leveled = trimesh.util.concatenate(body_sub).vertices @ Rx.T
    body_verts_leveled[:, 2] += z_off
    # Rotation about a pivot-centered axis parallel to world X (Rodrigues'
    # formula specialized to axis=(1,0,0)): x unchanged, y/z rotate.
    def _hinge_rotate(v):
        d = v - pivot
        # The dome top sits at negative-dy/positive-dz relative to this
        # back-bottom pivot; this sign combination is what swings it
        # UP and back (verified against the first version, which had the
        # signs flipped and rotated the lid DOWN through the bench instead
        # -- caught by checking its world bounding box, not just a render).
        dy = d[:, 1] * math.cos(theta) + d[:, 2] * math.sin(theta)
        dz = d[:, 2] * math.cos(theta) - d[:, 1] * math.sin(theta)
        return np.stack([d[:, 0], dy, dz], axis=1) + pivot

    def _snap_to_contact(lid_verts, body_verts, gap_target=0.001):
        """Translate lid_verts as a rigid body so its single nearest point
        to body_verts ends up gap_target away (~touching), closing
        whatever gap the (approximate) hinge pivot left open. Brute-force
        pairwise distance -- lid/body vertex counts here (hundreds to a
        few thousand) make that fine, no need for a KD-tree."""
        d = np.linalg.norm(lid_verts[:, None, :] - body_verts[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        closest_gap = body_verts[j] - lid_verts[i]
        dist = np.linalg.norm(closest_gap)
        if dist <= gap_target:
            return lid_verts
        shift = closest_gap * (1.0 - gap_target / dist)
        return lid_verts + shift

    def _build(idx_set, suffix, is_lid, color):
        sub = [p for i, p in enumerate(parts) if i in idx_set]
        combined = trimesh.util.concatenate(sub)
        verts = combined.vertices @ Rx.T
        verts[:, 2] += z_off
        if is_lid:
            verts = _hinge_rotate(verts)
            verts = _snap_to_contact(verts, body_verts_leveled)

        path = f"{prim_path_prefix}{suffix}"
        stage = stage_utils.get_current_stage()
        um = UsdGeom.Mesh.Define(stage, path)
        um.CreatePointsAttr([Gf.Vec3f(*v) for v in verts])
        um.CreateFaceVertexCountsAttr([3] * len(combined.faces))
        um.CreateFaceVertexIndicesAttr(combined.faces.flatten().tolist())
        if color:
            um.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        xf = UsdGeom.Xformable(um)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))

        if collision:
            geom = GeomPrim(paths=path, apply_collision_apis=True)
            UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath(path)) \
                .GetApproximationAttr().Set("convexDecomposition")
            geom.apply_physics_materials(_get_physics_material(material))
        return path

    body_idxs = set(range(len(parts))) - LID_IDXS
    body_path = _build(body_idxs, "Body", False, color_body)
    # Several attempts at a static "open lid resting against the body" pose
    # (translate-aside, then hinge-rotate, then snap-to-contact + a
    # shallower angle) still read as visually off per user review -- lid
    # dropped entirely for now rather than keep iterating on a cosmetic
    # detail. The body alone already shows the rotor (nothing was ever
    # covering it), which is what "open" mainly needs to convey.
    lid_path = _build(LID_IDXS, "Lid", True, color_lid) if include_lid else None
    return body_path, lid_path


def _box(path, size, pos, color=None, collision=True):
    """A static box. Has a real collision collider by default -- scene boxes
    previously had none at all (pure visual geometry), which was the root
    cause of a confirmed interpenetration bug: nothing physically stopped
    the gripper from passing straight through object geometry."""
    from isaacsim.core.experimental.objects import Cube
    from isaacsim.core.experimental.prims import GeomPrim

    cube = Cube(
        paths=path,
        positions=np.array(pos, dtype=float),
        sizes=1.0,
        scales=np.array(size, dtype=float),
        colors=color or "white",
    )
    if collision:
        GeomPrim(paths=path, apply_collision_apis=True)
    return cube


def build_platform(cx, cy):
    """Open test-tube platform: support stand, bench top, real CAD-derived
    props (tube holder + 20 test tubes, 96-well plate, 2 pipettes), work
    light.

    Returns {name: {"path": prim_path, "size": (w, d, h)}} for the loggable
    task objects (tubes, pipettes, tray) -- not the static fixtures (stand,
    bench, holder body, light). "size" is the object's authored full
    bounding box, used for state-aware metadata and for computing
    task-space targets relative to an object without re-deriving geometry
    by hand at call sites.
    """
    root = "/World/Platform"
    _box(f"{root}/Stand", (PLATFORM_W, PLATFORM_D, BENCH_Z),
         (cx, cy, BENCH_Z / 2), "gray")
    _box(f"{root}/Bench", (PLATFORM_W, PLATFORM_D, 0.04),
         (cx, cy, BENCH_Z + 0.02), "white")

    objects = {}
    pz = BENCH_Z + 0.04  # bench top surface (world Z); everything sits on this

    # ---- tube holder (real 5x4-bore rack) + 20 real hollow test tubes ----
    # Placed left-of-center and toward the near (arm-facing) edge so at
    # least the nearest rows are within the arm's ~0.85m reach.
    holder_x, holder_y = cx - 0.05, cy - 0.15
    holder_path = f"{root}/TubeHolder"
    load_mesh_prim(holder_path, TUBE_HOLDER_OBJ, (holder_x, holder_y, pz),
                   color=(0.85, 0.85, 0.85), material="plastic")

    holder_top_z = pz + HOLDER_THICKNESS_MM * MM_TO_M
    tube_insertion = 0.01  # m, how far the tube's base sits below the holder's top
    for r, y_mm in enumerate(HOLDER_HOLE_Y_MM):
        for c, x_mm in enumerate(HOLDER_HOLE_X_MM):
            tx = holder_x + x_mm * MM_TO_M
            ty = holder_y + y_mm * MM_TO_M
            tz = holder_top_z - tube_insertion
            path = f"{root}/Tube_r{r}_c{c}"
            _, tube_extents = load_mesh_prim(path, TEST_TUBE_OBJ, (tx, ty, tz),
                           color=(0.75, 0.9, 0.95), material="glass")
            objects[f"tube_r{r}_c{c}"] = {"path": path, "size": tube_extents}

    # ---- 2 pipettes (real model, lying flat) ----
    # Mesh origin isn't at its base (local Z spans -0.041 to +0.015), so
    # without this offset most of the pipette sat below the bench surface.
    pipette_extents = (0.0188, 0.181, 0.056)
    pipette_z_offset = 0.041
    for i, off in enumerate((-0.2, -0.28)):
        px, py = cx + 0.15, cy + off
        path = f"{root}/Pipette{i}"
        load_mesh_prim(path, PIPETTE_OBJ, (px, py, pz + pipette_z_offset),
                       color=(0.45, 0.5, 0.55), material="plastic")
        objects[f"pipette_{i}"] = {"path": path, "size": pipette_extents}

    # ---- sample tray (real 96-well plate, SBS standard dimensions) ----
    tray_extents = (0.128, 0.0855, 0.0146)
    tx, ty = cx + 0.35, cy - 0.1
    tray_path = f"{root}/SampleTray"
    load_mesh_prim(tray_path, SAMPLE_TRAY_OBJ, (tx, ty, pz),
                   color=(0.6, 0.6, 0.9), material="plastic")
    objects["tray"] = {"path": tray_path, "size": tray_extents}

    # ---- centrifuge, to the LEFT of the test tube holder, upright. ----
    # ---- Body only (no lid prim) -- see load_centrifuge_open's docstring ----
    # Upright extents after the roll are (0.143, 0.197, 0.189) -- placed
    # clear of the holder's left edge (holder_x - 0.158) with a gap.
    cf_extents = (0.143, 0.197, 0.189)
    cf_x = holder_x - 0.158 - 0.08 - cf_extents[0] / 2
    cf_y = holder_y
    centrifuge_body, _ = load_centrifuge_open(
        f"{root}/Centrifuge", CENTRIFUGE_OBJ, (cf_x, cf_y, pz),
        color_body=(0.95, 0.95, 0.95))
    objects["centrifuge_body"] = {"path": centrifuge_body, "size": cf_extents}

    # overhead work light
    from pxr import UsdLux, UsdGeom, Gf
    import isaacsim.core.experimental.utils.stage as _su
    _stage = _su.get_current_stage()
    _light = UsdLux.RectLight.Define(_stage, f"{root}/WorkLight")
    _light.GetIntensityAttr().Set(1200.0)
    _light.GetWidthAttr().Set(1.6)
    _light.GetHeightAttr().Set(1.0)
    _lx = UsdGeom.Xformable(_light)
    _lx.ClearXformOpOrder()
    _lx.AddTranslateOp().Set(Gf.Vec3d(cx, cy, BENCH_Z + 1.3))

    return objects


def build_scene(simulation_app):
    """Build room + platform + arm + cameras; settle the arm into its home
    pose. Returns (arm: Articulation, cams: {name: (prim_path, Camera)},
    objects: {name: {"path": prim_path, "size": (w, d, h)}}).

    Shared by lab_scene.py's own CLI (viewing/capture) and sim/collect_data.py
    (data generation), so both always build the exact same scene.
    """
    from pxr import Gf, UsdGeom
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.experimental.objects import DomeLight
    from isaacsim.core.prims import Articulation

    # ---------------- room ----------------
    _box("/World/Floor", (ROOM_X, ROOM_Y, 0.1), (0, 0, -0.05), "lightgray")
    _box("/World/WallN", (ROOM_X, 0.1, ROOM_Z), (0, ROOM_Y / 2, ROOM_Z / 2), "white")
    _box("/World/WallS", (ROOM_X, 0.1, ROOM_Z), (0, -ROOM_Y / 2, ROOM_Z / 2), "white")
    _box("/World/WallE", (0.1, ROOM_Y, ROOM_Z), (ROOM_X / 2, 0, ROOM_Z / 2), "white")
    _box("/World/WallW", (0.1, ROOM_Y, ROOM_Z), (-ROOM_X / 2, 0, ROOM_Z / 2), "white")
    DomeLight("/World/DomeLight").set_intensities(1500)

    # ---------------- platform ----------------
    objects = build_platform(PLATFORM_CX, PLATFORM_CY)

    # ---------------- fixed arm on pedestal ----------------
    # Pedestal top at BENCH_Z so the arm base is level with the platform.
    _box("/World/Pedestal", (0.5, 0.5, BENCH_Z),
         (ARM_POS[0], ARM_POS[1], BENCH_Z / 2), "darkgray")
    add_reference_to_stage(usd_path=FRANKA_USD, prim_path="/World/Arm")

    def set_prim_pose(path, pos, yaw_deg):
        prim = stage_utils.get_current_stage().GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddRotateZOp().Set(float(yaw_deg))

    # Arm base yawed 90 deg so local +x (reach direction) points in world +y,
    # towards the platform.
    set_prim_pose("/World/Arm", ARM_POS, 90.0)

    # ---------------- cameras ----------------
    import omni.replicator.core as rep
    from isaacsim.sensors.camera import Camera

    def aim(name, eye, target, res, focal_length=24.0):
        rep.create.camera(position=tuple(eye), look_at=tuple(target), name=name,
                          focal_length=focal_length)
        prim_path = f"/Replicator/{name}_Xform/{name}"
        cam = Camera(prim_path, resolution=res)
        cam.initialize()
        return (prim_path, cam)

    # ---------------- physics: settle the arm into its home pose first, so ----
    # ---------------- the wrist camera can be aimed from its real hand pose --
    import omni.timeline
    omni.timeline.get_timeline_interface().play()
    for _ in range(5):
        simulation_app.update()

    arm = Articulation("/World/Arm")
    arm.set_joint_positions(Q_HOME)
    arm.set_joint_position_targets(Q_HOME)
    for _ in range(10):
        simulation_app.update()

    cams = {}
    # 2 behind the robot arm: arm + tube platform both in view
    for j, sx in enumerate((-1.3, 1.3)):
        eye = (ARM_POS[0] + sx, ARM_POS[1] - 1.4, 1.7)
        target = (PLATFORM_CX, PLATFORM_CY, BENCH_Z + 0.2)
        cams[f"arm_cam{j}"] = aim(f"arm_cam{j}", eye, target, (640, 480))
    # 2 behind the tube platform, looking back at the arm
    for j, sx in enumerate((-1.0, 1.0)):
        eye = (PLATFORM_CX + sx, PLATFORM_CY + PLATFORM_D / 2 + 0.6, BENCH_Z + 0.9)
        target = (ARM_POS[0], ARM_POS[1], ARM_POS[2] + 0.3)
        cams[f"platform_cam{j}"] = aim(f"platform_cam{j}", eye, target,
                                       (640, 480), focal_length=18.0)
    # 1 straight overhead top-down view of the whole platform.
    top_eye = (PLATFORM_CX, PLATFORM_CY, BENCH_Z + 1.6)
    top_target = (PLATFORM_CX, PLATFORM_CY, BENCH_Z)
    cams["top"] = aim("top", top_eye, top_target, (640, 480), focal_length=20.0)

    # 1 real eye-in-hand camera: a Camera prim parented directly under the
    # hand link (not a free-floating prim with a position computed once),
    # so it inherits the hand's transform every frame through USD's own
    # parent-child hierarchy -- same as a camera physically bolted to the
    # gripper on the real arm would move with it. Previous version of this
    # camera only computed its position once from the home-pose transform,
    # which was fine while the arm was static but silently went stale as
    # soon as a task (e.g. pick-and-place) actually moved the arm.
    #
    # Hand's local frame: local +Z points from the hand origin toward the
    # fingers (the gripper's approach axis) -- verified empirically by
    # transforming the finger links' world position into the hand's local
    # frame (pure +Z offset, no X/Y component).
    #
    # Mounting dead-center on that axis was tried first and rejected: at an
    # actual grasp (fingers closed on a tube) the camera's own forward path
    # runs straight between the fingers, so it ends up looking directly
    # into the finger pad / grasped object at point-blank range (confirmed
    # empirically -- that placement rendered almost solid black/occluded
    # when tested at a real grasp pose, not just at the empty home pose).
    # A real eye-in-hand camera is mounted to the side of the gripper body
    # for the same reason. Offsetting sideways along local X and yawing
    # back in towards the approach axis keeps the grasp point framed
    # without the camera's own view path running through the gripper's
    # closing gap -- also verified empirically against a real grasp pose
    # (descend_to_grasp), comparing centered vs. offset/yawed placements.
    from pxr import Gf, UsdGeom
    stage = stage_utils.get_current_stage()
    eih_path = f"{HAND_PATH}/EyeInHandCamera"
    usd_cam = UsdGeom.Camera.Define(stage, eih_path)
    usd_cam.CreateFocalLengthAttr(24.0)
    usd_cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 1000.0))
    xf = UsdGeom.Xformable(usd_cam)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.07, 0.0, 0.04))
    # USD cameras look down their own local -Z with +Y up. RotateY first
    # yaws the view 30 deg back toward the approach axis (in the prim's own
    # unrotated frame), then RotateX(180) flips forward from the hand's
    # local -Z (back into the arm) to its local +Z (toward the fingers) --
    # USD applies xform ops in the order added, each in the frame left by
    # the previous one.
    xf.AddRotateYOp().Set(-30.0)
    xf.AddRotateXOp().Set(180.0)
    cams["wrist"] = (eih_path, Camera(eih_path, resolution=(640, 480)))
    cams["wrist"][1].initialize()

    return arm, cams, objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--capture", action="store_true",
                    help="save PNGs from all cameras and exit")
    ap.add_argument("--out", default=r"C:\Users\varun\workspace\BiotechRobot\lab_captures")
    ap.add_argument("--camera", default=None,
                    help="GUI mode: set viewport to this camera on startup")
    ap.add_argument("--multi-view", action="store_true",
                    help="GUI mode: tile all cameras in one window, viewed simultaneously")
    ap.add_argument("--list-cameras", action="store_true",
                    help="print camera names and exit")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless or args.capture})

    from pxr import Sdf

    arm, cams, objects = build_scene(simulation_app)
    print(f"scene ready: 1 platform, 1 arm (static), {len(cams)} cameras")

    out_dir = Path(args.out)

    def capture():
        from PIL import Image

        out_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(8):
            simulation_app.update()
        for name, (prim_path, cam) in cams.items():
            rgb = cam.get_current_frame()["rgb"]
            Image.fromarray(rgb).save(str(out_dir / f"{name}.png"))
        print(f"captured {len(cams)} images -> {out_dir}")

    if args.list_cameras:
        print("Available cameras:")
        for name in sorted(cams.keys()):
            print(f"  {name}")
        simulation_app.close()
        return

    if args.capture:
        capture()
        print("CAPTURE DONE:", out_dir)
    elif args.headless:
        for _ in range(60):
            simulation_app.update()
        print("HEADLESS RUN OK")
    else:
        import omni.kit.viewport.utility as vpu

        if args.multi_view:
            # Tile every camera into its own window, viewed simultaneously.
            # Grid: 3 columns, tiles sized so 5 fit on a 1440x900+ screen.
            tile_w, tile_h = 480, 360
            cols = 3
            default_win = vpu.get_active_viewport_window()
            if default_win is not None:
                default_win.visible = False
            for idx, (name, (prim_path, _)) in enumerate(sorted(cams.items())):
                col, row = idx % cols, idx // cols
                win = vpu.create_viewport_window(
                    name=name, width=tile_w, height=tile_h,
                    position_x=col * tile_w, position_y=row * tile_h,
                    camera_path=Sdf.Path(prim_path),
                )
                win.viewport_api.set_active_camera(Sdf.Path(prim_path))
            print(f"Multi-view: {len(cams)} cameras tiled in a {cols}-column grid")
        elif args.camera:
            if args.camera not in cams:
                print(f"Unknown camera '{args.camera}'. Available:")
                for n in sorted(cams.keys()):
                    print(f"  {n}")
            else:
                prim_path, _ = cams[args.camera]
                try:
                    vp = vpu.get_active_viewport()
                    vp.set_active_camera(prim_path)
                    print(f"Viewport set to camera: {args.camera}")
                except Exception as e:
                    print(f"Could not set viewport: {e}")
        while simulation_app.is_running():
            simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()

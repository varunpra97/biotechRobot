"""FROZEN SNAPSHOT for vision_pipeline's Isaac validation capture -- copied
from biotech-lab-robotics/sim/lab_scene.py so this work never reads from or
writes to that actively-developed repo. Only CAD_DIR/FRANKA_USD below were
changed (now point at the assets/ copy alongside this file instead of
biotech-lab-robotics/assets/); nothing else was touched. Do not edit this
copy to track upstream changes -- re-copy instead if something shared with
the live scene needs to change.

Biotech lab setup: open test-tube platform + Franka arm on a pedestal.

Room (10m x 6m x 3m), no enclosure. One open platform (1.4m x 0.7m, bench
height) holds a 5x4 test tube holder with 20 tubes, a 96-well plate, 2
pipettes and a centrifuge (lid off, fixed-angle rotor with six 8mm bores).
One fixed-base Franka arm sits on a pedestal facing the platform, base
height matched to it. Run directly, this script builds the scene with the
arm held at its home pose to view/capture it; the collect_data*.py scripts
build the same scene via build_scene() and move the arm.

Physics: 240 Hz (PHYSICS_HZ), static fixtures collide as their exact
triangle meshes, tubes as low-poly convex hulls -- see those constants and
load_mesh_prim for why each matters for 7mm tubes in 8mm bores.

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
    python sim\\lab_scene.py --capture        # save PNGs from all 6 cameras, exit
    python sim\\lab_scene.py --camera wrist   # GUI, viewport starts at this camera
    python sim\\lab_scene.py --multi-view     # GUI, all 6 cameras tiled at once
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

FRANKA_USD = str(Path(__file__).resolve().parent / "assets" / "panda" / "panda.usda")

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


def quat_to_matrix(q):
    """[qw, qx, qy, qz] -> 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def matrix_to_quat(R):
    """3x3 rotation matrix -> [qw, qx, qy, qz] (qw >= 0)."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q) / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def set_prim_world_pose(path, pos, quat=(1.0, 0.0, 0.0, 0.0)):
    """Set a parentless-in-practice prim's full pose (translate + orient,
    double precision). Full pose, not position only: a grasped object has
    to rotate with the hand -- e.g. a tube tilted 45 deg to go into a
    fixed-angle rotor bore -- or it visibly stops lining up with the
    fingers. Reuses existing translate/orient ops when present so PhysX's
    own pose write-back for a rigid body keeps using the same ops."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Gf, UsdGeom

    prim = stage_utils.get_current_stage().GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    if [op.GetOpType() for op in ops] != [UsdGeom.XformOp.TypeTranslate,
                                          UsdGeom.XformOp.TypeOrient]:
        xf.ClearXformOpOrder()
        ops = [xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble),
               xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)]
    t_op, o_op = ops
    t_op.Set(Gf.Vec3d(*(float(v) for v in pos)))
    w, x, y, z = (float(v) for v in quat)
    if o_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        o_op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    else:
        o_op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


LEFT_FINGER_PATH = HAND_PATH + "/left_finger"
RIGHT_FINGER_PATH = HAND_PATH + "/right_finger"

# Tool center point (fingertip) in the hand's local frame: on the gripper's
# center line, at the finger pads' lowest point. Measured from the real
# asset, not estimated (cad_source/probe_franka_kinematics.py and
# probe_gripper_colliders.py): the fingers open along hand-local +/-Y (the
# finger joint value is each pad's distance from the center line), the
# approach axis is hand-local +Z, and both the visual finger meshes and
# their PhysX convex-hull colliders end at z = 0.1123. The previous
# estimate (finger origin + 0.045 straight down in WORLD z) was 8.9mm short
# of the real tip, and wrong in direction whenever the hand isn't vertical.
TCP_LOCAL = np.array([0.0, 0.0, 0.1123])
# Finger pad contact face spans hand-local z 0.0945..0.1123.
FINGER_PAD_LENGTH = 0.0178


def get_tcp_pose():
    """(position, rotation matrix) of the TCP in world -- rotation is the
    hand's (z = approach axis, y = finger closing axis)."""
    h = get_prim_world_pose(HAND_PATH)
    R = quat_to_matrix(h[3:7])
    return h[:3] + R @ TCP_LOCAL, R


def get_fingertip_pos():
    """World position of the TCP (see TCP_LOCAL)."""
    return get_tcp_pose()[0]


MM_TO_M = 0.001

CAD_DIR = str(Path(__file__).resolve().parent / "assets" / "cad")
TUBE_HOLDER_OBJ = f"{CAD_DIR}/tube_holder.obj"      # 5x4 rack, 8.0mm blind bores
SAMPLE_TRAY_OBJ = f"{CAD_DIR}/sample_tray_5x2.obj"  # 10 wells (5x2, 9 mm pitch), gripper-width
SAMPLE_TRAY_COLLISION_OBJ = f"{CAD_DIR}/sample_tray_5x2_collision.obj"
PIPETTE_OBJ = f"{CAD_DIR}/pipette.obj"
PIPETTE_COLLISION_OBJ = f"{CAD_DIR}/pipette_collision.obj"   # 7 convex pieces from its real cross-sections
TIP_OBJ = f"{CAD_DIR}/pipette_tip.obj"              # P1000-style, sized to the pipette nozzle
TIP_COLLISION_OBJ = f"{CAD_DIR}/pipette_tip_collision.obj"
TIP_RACK_OBJ = f"{CAD_DIR}/tip_rack.obj"            # 10 x 10 tips, 9 mm pitch
WASTE_BIN_OBJ = f"{CAD_DIR}/waste_bin.obj"
CENTRIFUGE_OBJ = f"{CAD_DIR}/centrifuge.obj"
TEST_TUBE_OBJ = f"{CAD_DIR}/test_tube.obj"          # 7mm OD x 58mm, conical tip; local z=0 at the tip
TEST_TUBE_COLLISION_OBJ = f"{CAD_DIR}/test_tube_collision.obj"  # 40-vertex convex hull

# Holder design values (cad_source/make_tray_20holes.py, parametric CAD --
# exact, not measured): 5x4 grid at 61.5mm pitch centered on the holder's
# mesh origin, 8.0mm bores 28mm deep, so each bore's floor is 12mm above
# the holder's base. Tubes stand on that floor (they previously floated
# 30mm above it at a guessed insertion depth).
HOLDER_PITCH_MM = 61.5
HOLDER_HOLE_X_MM = [(i - 2) * HOLDER_PITCH_MM for i in range(5)]
HOLDER_HOLE_Y_MM = [(j - 1.5) * HOLDER_PITCH_MM for j in range(4)]
HOLDER_THICKNESS_MM = 40.0
HOLDER_BORE_FLOOR_MM = 12.0

# Physics rate. 240 Hz, not the 60 Hz default: at 60 Hz a released 7mm tube
# moves 1.4-2.7mm per step under gravity alone -- more than its 0.5mm bore
# clearance and more than half the rotor sleeve's 0.8mm wall -- so contacts
# flip to the far side of the wall and the tube is pushed out through it.
# At 240 Hz (and 480) it seats and stays (cad_source/probe_release_variants.py).
PHYSICS_HZ = 240

# Right side of the bench (all from cad_source/make_*.py design values):
SAMPLE_TRAY_XY = (0.30, 1.73)       # 52 x 24 mm, long side along x
SAMPLE_TRAY_PITCH = 0.009
# where the tray may be set down (bench xy); "right" is clear of the tip
# rack (x 0.27-0.37, y <= 1.58) and the waste bin (y <= 1.575)
TRAY_SPOTS_XY = {"home": SAMPLE_TRAY_XY, "right": (0.45, 1.70)}
TRAY_MASS_KG = 0.015                # 13.9 cm^3 polystyrene
TIP_RACK_XY = (0.32, 1.53)          # 100 x 100 mm, 55 mm tall
TIP_RACK_HEIGHT = 0.055
TIP_PITCH = 0.009
TIP_COLLAR_Z = 0.052                # tip local z of the collar's underside (rests on the rack)
TIP_LENGTH = 0.060
TIP_MASS_KG = 0.001
WASTE_BIN_XY = (0.49, 1.53)         # 90 x 90 x 70 mm open box
PIPETTE_XY = (0.21, 1.58)           # nozzle end; lies along +y. 6 cm clear of the holder:
                                    # grasping across it puts the 21 cm-wide hand over its side
PIPETTE_MASS_KG = 0.15              # electronic pipette
# initial liquid (uL, RGB) in holder tubes; everything else starts empty
# A 1000 uL tip's end can go at most ~36 mm below a 6 mm-bore tube's rim
# before its widening body reaches the wall, so a tube can be aspirated from
# only while it holds more than ~570 uL (surface 25 mm above the tube's base);
# source tubes start fuller than that.
INITIAL_LIQUIDS = {
    "tube_r0_c0": (900.0, (0.15, 0.75, 0.25)),
    "tube_r1_c0": (1200.0, (0.15, 0.35, 0.90)),
    "tube_r1_c2": (1000.0, (0.85, 0.15, 0.15)),
}
LIQUIDS = {}                        # container name -> liquids.LiquidBody (filled by build_platform)

TUBE_MASS_KG = 0.0015   # 580.6 mm^3 of borosilicate glass (2.23 g/cm^3) ~1.3 g, + residue

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
def _define_mesh(stage, path, mesh, color=None):
    from pxr import Gf, UsdGeom

    um = UsdGeom.Mesh.Define(stage, path)
    um.CreatePointsAttr([Gf.Vec3f(*v) for v in mesh.vertices])
    um.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    um.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist())
    if color:
        um.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return um


def _apply_collider(stage, path, approximation, material):
    from isaacsim.core.experimental.prims import GeomPrim
    from pxr import UsdPhysics

    geom = GeomPrim(paths=path, apply_collision_apis=True)
    UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath(path))         .GetApproximationAttr().Set(approximation)
    geom.apply_physics_materials(_get_physics_material(material))


def load_mesh_prim(prim_path, obj_path, pos, yaw_deg=0.0, pitch_deg=0.0,
                    roll_deg=0.0, color=None, collision=True, material="plastic",
                    approximation="convexDecomposition", collision_obj=None):
    """Load a triangle mesh from an OBJ file as a USD prim with a PhysX
    collider and a bound rigid-body physics material (friction/restitution)
    -- real CAD-derived geometry instead of a box/cylinder approximation.

    approximation: PhysX collider for the mesh --
      "none": the exact triangle mesh. Static fixtures only (PhysX can't
          simulate a dynamic triangle mesh), but it's the only option that
          keeps narrow bores (8mm holder/rotor bores) exactly as modeled;
          a convex decomposition approximates and can partially fill them.
      "convexHull": exact for a convex shape -- if it has <= 64 vertices.
          PhysX caps hulls at 64 (GPU pipeline) and EXPANDS a reduced hull
          to contain every input point, so a finely tessellated visual
          mesh gets an inflated collider. Pass a low-poly collision_obj.
      "convexDecomposition": general concave fallback.

    collision_obj: separate collision mesh. prim_path then becomes an
    Xform (the one to make a rigid body / move) holding a rendered
    "visual" mesh with no collider and an invisible ("guide" purpose)
    "collision" mesh.

    Pose is authored as translate + orient (double), the op layout
    set_prim_world_pose() and PhysX's rigid-body write-back both use.
    Returns (prim_path, visual mesh extents)."""
    import trimesh
    from pxr import UsdGeom
    import isaacsim.core.experimental.utils.stage as stage_utils

    mesh = trimesh.load(obj_path, force="mesh")
    stage = stage_utils.get_current_stage()
    if collision_obj is None:
        _define_mesh(stage, prim_path, mesh, color)
        collider_path = prim_path
    else:
        UsdGeom.Xform.Define(stage, prim_path)
        _define_mesh(stage, f"{prim_path}/visual", mesh, color)
        # one convex collider per connected piece of the collision mesh
        # process=False: welding coincident vertices would fuse touching
        # pieces (e.g. the pipette's bands share their boundary outlines)
        pieces = trimesh.load(collision_obj, force="mesh", process=False).split(only_watertight=False)
        collider_path = []
        for i, piece in enumerate(pieces):
            cp = f"{prim_path}/collision" if len(pieces) == 1 else f"{prim_path}/collision_{i}"
            cm = _define_mesh(stage, cp, piece)
            UsdGeom.Imageable(cm).CreatePurposeAttr(UsdGeom.Tokens.guide)
            collider_path.append(cp)

    def _rot(axis, deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
                "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
                "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]

    R = _rot("x", roll_deg) @ _rot("y", pitch_deg) @ _rot("z", yaw_deg)
    set_prim_world_pose(prim_path, pos, matrix_to_quat(R))

    if collision:
        for cp in ([collider_path] if isinstance(collider_path, str) else collider_path):
            _apply_collider(stage, cp, approximation, material)
    return prim_path, tuple(float(v) for v in mesh.extents)


def collider_paths(path):
    """Every collider prim at or under path."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Usd, UsdPhysics

    root = stage_utils.get_current_stage().GetPrimAtPath(path)
    return [str(p.GetPath()) for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.CollisionAPI)]


def make_rigid_body(path, mass_kg, kinematic=True, contact_offset=0.001, report_contacts=True):
    """Turn a prim (and the colliders under it) into a PhysX rigid body
    with a real mass. Starts kinematic (pose driven by set_prim_world_pose,
    e.g. while held rigidly by the gripper); set_kinematic(path, False)
    hands it to the physics solver (e.g. on release, so it settles under
    gravity and contact). Small contact offset, zero rest offset: a ~1.5 g,
    7mm tube working with ~0.5mm clearances, where the default
    (scene-scale) contact offset would be far coarser than the fit."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import PhysxSchema, UsdPhysics

    stage = stage_utils.get_current_stage()
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(bool(kinematic))
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass_kg))
    prb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    prb.CreateSolverPositionIterationCountAttr(16)
    # Never sleep: a sleeping body reports no contacts, and the task checks
    # a released object's resting contact (penetration depth) via the
    # contact report.
    prb.CreateSleepThresholdAttr(0.0)
    if report_contacts:
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    for cp in collider_paths(path):
        pc = PhysxSchema.PhysxCollisionAPI.Apply(stage.GetPrimAtPath(cp))
        pc.CreateContactOffsetAttr(float(contact_offset))
        pc.CreateRestOffsetAttr(0.0)


def contacts_of(body_path):
    """PhysX's contact report for the last simulation step, for one rigid
    body (which needs PhysxContactReportAPI -- make_rigid_body applies it):
    {other collider path: {"min_separation": m, "impulse": N*s, "n": count}}.
    Negative separation is penetration depth -- the meaningful check for a
    body RESTING on something, where an overlap query always reports a hit."""
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools

    headers, data = get_physx_simulation_interface().get_contact_report()
    out = {}
    for h in headers:
        a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
        a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
        if body_path not in (a0, a1):
            continue
        other = str(PhysicsSchemaTools.intToSdfPath(h.collider1 if a0 == body_path else h.collider0))
        pts = data[h.contact_data_offset:h.contact_data_offset + h.num_contact_data]
        e = out.setdefault(other, {"min_separation": float("inf"), "impulse": 0.0, "n": 0})
        for c in pts:
            e["min_separation"] = min(e["min_separation"], float(c.separation))
            e["impulse"] += float(np.linalg.norm([c.impulse.x, c.impulse.y, c.impulse.z]))
            e["n"] += 1
    return out


def set_gravity(path, enabled):
    """Switch gravity on/off for one rigid body (e.g. a held object being
    teleported along a planned path, so it stays exactly where placed)."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import PhysxSchema

    prim = stage_utils.get_current_stage().GetPrimAtPath(path)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(not enabled)


def zero_velocity(path):
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Gf, UsdPhysics

    rb = UsdPhysics.RigidBodyAPI(stage_utils.get_current_stage().GetPrimAtPath(path))
    rb.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    rb.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))


def _filter_tip_collisions(objects, others):
    """Tips never collide with tube or well interiors, the rack, or the
    pipette nozzle: those are hollow insides (or a press fit) that the
    tip's convex collider can't represent, so a tip inside them is checked
    analytically by the pipette model instead. Authored once, before the
    simulation starts (changing filters later makes PhysX rebuild bodies)."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Sdf, UsdPhysics

    stage = stage_utils.get_current_stage()
    paths = set()
    for p in others:  # the objects and their actual collider prims
        paths.add(p)
        paths.update(collider_paths(p))
    targets = [Sdf.Path(p) for p in sorted(paths)]
    for n, o in objects.items():
        if n.startswith("tip_"):
            UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(o["path"]))                 .CreateFilteredPairsRel().SetTargets(targets)


def set_collision_filter(path, other_paths):
    """Disable collisions between the body at path and the given bodies
    (empty list re-enables them all). Used while an object is rigidly held:
    it moves as one with the hand, so pad-object contact would only be the
    solver fighting a kinematic object it can't move."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Sdf, UsdPhysics

    prim = stage_utils.get_current_stage().GetPrimAtPath(path)
    rel = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
    rel.SetTargets([Sdf.Path(p) for p in other_paths])


def set_kinematic(path, kinematic):
    """Switch a rigid body between kinematic (pose-driven) and dynamic,
    zeroing its velocity so it doesn't inherit motion from the switch.
    CCD follows the mode: PhysX rejects CCD on kinematic bodies, and a
    small fast dynamic one needs it so it can't step through thin walls.
    Order matters -- CCD is only ever on while the body is dynamic, else
    PhysX logs "kinematic bodies with CCD enabled are not supported" and
    drops the CCD flag."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Gf, PhysxSchema, UsdPhysics

    prim = stage_utils.get_current_stage().GetPrimAtPath(path)
    rb = UsdPhysics.RigidBodyAPI(prim)
    ccd = PhysxSchema.PhysxRigidBodyAPI(prim).CreateEnableCCDAttr()
    if kinematic:
        ccd.Set(False)
        rb.GetKinematicEnabledAttr().Set(True)
    else:
        rb.GetKinematicEnabledAttr().Set(False)
        ccd.Set(True)
    rb.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    rb.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))


# The centrifuge mesh is one fused STEP/STL export, but trimesh.split() finds
# 26 connected components in it. Which are the lid (index 23, plus two small
# lid-top details at 0/1), the rotor hub (4) and the six fixed-angle rotor
# sleeves was confirmed from the geometry, not guessed: lid/body by a
# color-coded isolated render (cad_source/test_centrifuge_parts.py); hub and
# sleeves by fitting each part's axis -- the sleeves are hollow 8.0mm-ID
# bores with conical floors, tilted 44.9 deg from vertical, tops leaning in
# toward the hub (cad_source/raycast_rotor_bore.py). Indices are specific to
# this exact mesh export.
CENTRIFUGE_LID_PARTS = {23, 0, 1}
CENTRIFUGE_HUB_PART = 4
CENTRIFUGE_SLEEVE_PARTS = (5, 8, 11, 14, 17, 20)
CENTRIFUGE_ROLL_DEG = 90.0  # the export lies on its side; this stands it up


def centrifuge_parts_local(obj_path=None, roll_deg=CENTRIFUGE_ROLL_DEG):
    """The centrifuge's connected components as trimesh parts in the prim's
    local frame: rolled upright and leveled so its lowest point is z=0.
    World coordinates are these + centrifuge_pos(). One definition shared
    by the scene loader and the rotor-bore geometry used for task targets,
    so targets are computed from exactly what the simulator loads."""
    import trimesh

    mesh = trimesh.load(obj_path or CENTRIFUGE_OBJ, force="mesh")
    parts = mesh.split(only_watertight=False)
    rx = np.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    z_off = -(mesh.vertices @ Rx.T)[:, 2].min()
    out = []
    for p in parts:
        v = p.vertices @ Rx.T
        v[:, 2] += z_off
        out.append(trimesh.Trimesh(v, p.faces, process=False))
    return out


def load_centrifuge(prim_path, pos, color=None, material="metal"):
    """Load the centrifuge body (housing + rotor hub + six rotor sleeves),
    lid omitted -- several static "open lid" poses read as visually wrong
    in review, so it's open by simply having no lid; a real hinged lid
    would need to be an articulation anyway.

    Collision is the exact triangle mesh ("none"), not a convex
    decomposition: the body is static, and the rotor's 8mm bores must stay
    exactly as modeled for a tube to slide into them -- an automatic
    convex decomposition approximates (and can fill) bores that narrow.
    Degenerate parts (a few 1-2 vertex slivers in the export) are dropped."""
    import trimesh
    from pxr import Gf, UsdGeom, UsdPhysics
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.prims import GeomPrim

    parts = centrifuge_parts_local()
    body = trimesh.util.concatenate([p for i, p in enumerate(parts)
                                     if i not in CENTRIFUGE_LID_PARTS and len(p.faces) > 4])
    stage = stage_utils.get_current_stage()
    um = UsdGeom.Mesh.Define(stage, prim_path)
    um.CreatePointsAttr([Gf.Vec3f(*v) for v in body.vertices])
    um.CreateFaceVertexCountsAttr([3] * len(body.faces))
    um.CreateFaceVertexIndicesAttr(body.faces.flatten().tolist())
    if color:
        um.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    set_prim_world_pose(prim_path, pos)

    geom = GeomPrim(paths=prim_path, apply_collision_apis=True)
    UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath(prim_path))         .GetApproximationAttr().Set("none")
    geom.apply_physics_materials(_get_physics_material(material))
    return prim_path


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


CENTRIFUGE_EXTENTS = (0.143, 0.197, 0.189)  # upright, after the 90 deg roll


def holder_xy(cx=PLATFORM_CX, cy=PLATFORM_CY):
    """Tube holder origin (world XY). Left-of-center and toward the near
    (arm-facing) edge so the nearest rows are within the arm's reach."""
    return cx - 0.05, cy - 0.15


def centrifuge_pos(cx=PLATFORM_CX, cy=PLATFORM_CY):
    """Centrifuge mesh origin (world): left of the holder's edge
    (holder_x - 0.158) with an 8cm gap, standing on the bench top."""
    hx, hy = holder_xy(cx, cy)
    return (hx - 0.158 - 0.08 - CENTRIFUGE_EXTENTS[0] / 2, hy, BENCH_Z + 0.04)


def build_platform(cx, cy, rigid_objects=()):
    """Open test-tube platform: support stand, bench top, real CAD-derived
    props (tube holder + 20 test tubes, centrifuge, electronic pipette,
    10-well sample tray, 100-tip rack, waste bin, liquids), work light.

    rigid_objects: names of tubes to make (kinematic) rigid bodies, e.g.
    the one a task will pick up -- everything else stays a static collider.

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
    holder_x, holder_y = holder_xy(cx, cy)
    holder_path = f"{root}/TubeHolder"
    load_mesh_prim(holder_path, TUBE_HOLDER_OBJ, (holder_x, holder_y, pz),
                   color=(0.85, 0.85, 0.85), material="plastic", approximation="none")

    # Each tube stands on its bore's floor (tube local z=0 is its tip).
    tube_z = pz + HOLDER_BORE_FLOOR_MM * MM_TO_M
    for r, y_mm in enumerate(HOLDER_HOLE_Y_MM):
        for c, x_mm in enumerate(HOLDER_HOLE_X_MM):
            tx = holder_x + x_mm * MM_TO_M
            ty = holder_y + y_mm * MM_TO_M
            name = f"tube_r{r}_c{c}"
            path = f"{root}/Tube_r{r}_c{c}"
            _, tube_extents = load_mesh_prim(path, TEST_TUBE_OBJ, (tx, ty, tube_z),
                                             color=(0.75, 0.9, 0.95), material="glass",
                                             approximation="convexHull",
                                             collision_obj=TEST_TUBE_COLLISION_OBJ)
            if name in rigid_objects:
                make_rigid_body(path, TUBE_MASS_KG, kinematic=True)
            objects[name] = {"path": path, "size": tube_extents}

    # ---- electronic pipette, lying on the bench (settles physically) ----
    # Mesh origin isn't at its base (local Z spans -0.041 to +0.015), so it
    # starts lifted by that much. One pipette: an earlier layout had two
    # overlapping each other, harmless only while they were static meshes.
    path = f"{root}/Pipette0"
    load_mesh_prim(path, PIPETTE_OBJ, (PIPETTE_XY[0], PIPETTE_XY[1], pz + 0.041),
                   color=(0.45, 0.5, 0.55), material="plastic", approximation="convexHull",
                   collision_obj=PIPETTE_COLLISION_OBJ)
    if "pipette_0" in rigid_objects:
        make_rigid_body(path, PIPETTE_MASS_KG, kinematic=True)
    objects["pipette_0"] = {"path": path, "size": (0.0188, 0.181, 0.056)}

    # ---- 10-well sample tray (5 x 2), narrow enough to grip across ----
    tray_path = f"{root}/SampleTray"
    _, tray_ext = load_mesh_prim(tray_path, SAMPLE_TRAY_OBJ, (*SAMPLE_TRAY_XY, pz),
                                 color=(0.6, 0.6, 0.9), material="plastic",
                                 approximation="convexHull", collision_obj=SAMPLE_TRAY_COLLISION_OBJ)
    if "tray" in rigid_objects:
        make_rigid_body(tray_path, TRAY_MASS_KG, kinematic=True)
    objects["tray"] = {"path": tray_path, "size": tray_ext}

    # ---- 100-tip rack, tips hanging by their collars, and a waste bin ----
    rx, ry = TIP_RACK_XY
    load_mesh_prim(f"{root}/TipRack", TIP_RACK_OBJ, (rx, ry, pz), color=(0.25, 0.45, 0.8),
                   material="plastic", approximation="none")
    load_mesh_prim(f"{root}/WasteBin", WASTE_BIN_OBJ, (*WASTE_BIN_XY, pz), color=(0.5, 0.52, 0.55),
                   material="plastic", approximation="none")
    tip_z = pz + TIP_RACK_HEIGHT - TIP_COLLAR_Z
    tube_paths = [o["path"] for n, o in objects.items() if n.startswith("tube_")]
    for r in range(10):
        for c in range(10):
            name = f"tip_{'ABCDEFGHIJ'[r]}{c + 1}"
            path = f"{root}/Tips/{name}"
            load_mesh_prim(path, TIP_OBJ, (rx + (c - 4.5) * TIP_PITCH, ry + (r - 4.5) * TIP_PITCH, tip_z),
                           color=(0.93, 0.93, 0.9), material="plastic", approximation="convexHull",
                           collision_obj=TIP_COLLISION_OBJ)
            make_rigid_body(path, TIP_MASS_KG, kinematic=True, report_contacts=False)
            objects[name] = {"path": path, "size": (0.0082, 0.0082, TIP_LENGTH)}
    _filter_tip_collisions(objects, tube_paths + [tray_path, f"{root}/TipRack", objects["pipette_0"]["path"]])

    # ---- liquids ----
    import liquids

    LIQUIDS.clear()
    for name, (vol, color) in INITIAL_LIQUIDS.items():
        LIQUIDS[name] = liquids.LiquidBody(objects[name]["path"], liquids.TUBE, vol, color)
    for j in range(2):
        for i in range(5):
            wname = f"well_{'AB'[j]}{i + 1}"
            off = ((i - 2) * SAMPLE_TRAY_PITCH, (j - 0.5) * SAMPLE_TRAY_PITCH, 0.0)
            LIQUIDS[f"tray/{wname}"] = liquids.LiquidBody(tray_path, liquids.WELL, 0.0,
                                                          (0.2, 0.3, 0.9), offset=off, name=f"liquid_{wname}")

    # ---- centrifuge, to the LEFT of the test tube holder, upright, no lid ----
    centrifuge_body = load_centrifuge(f"{root}/CentrifugeBody", centrifuge_pos(cx, cy),
                                      color=(0.95, 0.95, 0.95))
    objects["centrifuge_body"] = {"path": centrifuge_body, "size": CENTRIFUGE_EXTENTS}

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


# The policy's exterior camera (platform_cam1): fixed, so it frames the WHOLE
# platform -- every task's objects stay in view, and a policy pipeline can
# crop per task later if it helps. Far side of the bench, in the original
# platform_cam1 viewing direction, 45 deg down, 1.6 m from the bench center
# (60 deg horizontal FOV -> ~1.85 m wide at the bench, which is 1.4 m).
EXTERIOR_CAM_TARGET = (PLATFORM_CX, PLATFORM_CY - 0.05, BENCH_Z + 0.05)
_az = np.array([1.2, 1.1]) / np.linalg.norm([1.2, 1.1])
_el = np.radians(45.0)
_dir = np.array([_az[0] * np.cos(_el), _az[1] * np.cos(_el), np.sin(_el)])
EXTERIOR_CAM_EYE = tuple(np.array(EXTERIOR_CAM_TARGET) + 1.6 * _dir)
EXTERIOR_CAM_FOCAL = 18.0

FINGER_JOINT_PATHS = (LEFT_FINGER_PATH + "/finger_joint1", RIGHT_FINGER_PATH + "/finger_joint2")
# Franka Hand: up to 70 N continuous grasp force, 80 mm stroke. Each finger
# drive is stiff with its force capped, so "close" squeezes with an ~constant
# force until the pads stop on the object -- a real friction grasp -- instead
# of parking the fingers at a pre-computed width. 4 N per finger (8 N
# grasp): at 35 N the solver crushed the pads straight through a 1.5 g tube
# (cad_source/probe_finger_contact.py), and at 8 N a held tube jittered up
# to 5 mm per step; at 4 N it holds to 0.06 mm (probe_grasp_jitter.py).
# (High-friction rubber pads made a held tube jitter again -- 2.5 mm.)
# The drive is a soft spring toward fully closed (k = 1000 N/m), so the grip
# scales with the object's width: ~3.4 N on a 7 mm tube, ~9 N on the 18 mm
# pipette body, ~12 N across the 24 mm sample tray (Franka allows up to 70 N).
# A stiff drive pinned at a force cap switched on/off as a held tube shifted
# (a lift out of its bore slipped 0.4 mm or dropped it; soft: 0.02 mm, held
# to 0.001 mm -- probe_grasp_jitter.py), and a 4 N cap let the 150 g pipette
# pivot between the pads when the wrist turned it nozzle-down.
FINGER_MAX_FORCE = 20.0     # N per finger
FINGER_STIFFNESS = 1.0e3    # N/m toward fully closed
FINGER_DAMPING = 40.0       # N*s/m (closes with a ~0.04 s time constant)
GRIPPER_STROKE = 0.04       # m per finger
ARM_SOLVER_POSITION_ITERATIONS = 64


def configure_franka_physics(gravity_compensation=True):
    """Physics settings the asset leaves out or gets wrong for grasping:
    - finger pads collide as the pads you see. The asset disables collision
      on the visible pad meshes and collides only with small hidden boxes,
      four of them 2.5 mm off the finger centerline -- on a 7 mm tube the
      fingers stopped 1 mm inside the visible tube surface. The pad mesh's
      convex hull is a flat face, which is what the camera shows.
    - both finger joints get a force-limited position drive (only one had
      any drive gains at all);
    - 64 solver position iterations for the arm: squeezing a 1.5 g tube
      between 15 g fingers needs them (at the default the pads slowly sank
      through it; at 64, 0.17 mm penetration, the tube held still);
    - gravity compensation: the real Franka's controller cancels the arm's
      own weight; here that's done by not applying gravity to the arm links
      (Isaac Lab's Franka setup does the same). Payloads still have weight."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = stage_utils.get_current_stage()
    for finger in (LEFT_FINGER_PATH, RIGHT_FINGER_PATH):
        stage.GetPrimAtPath(finger + "/finger_1").SetInstanceable(False)
        pad = finger + "/finger_1/finger_1"
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(pad)).GetCollisionEnabledAttr().Set(True)
    PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath("/World/Arm/Geometry/link0")) \
        .CreateSolverPositionIterationCountAttr(ARM_SOLVER_POSITION_ITERATIONS)
    for jp in FINGER_JOINT_PATHS:
        drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(jp), "linear")
        drive.CreateStiffnessAttr(FINGER_STIFFNESS)
        drive.CreateDampingAttr(FINGER_DAMPING)
        drive.CreateMaxForceAttr(FINGER_MAX_FORCE)
    if gravity_compensation:
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Arm")):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)


def tile_camera_views(cams, tile_w=480, tile_h=360, cols=3):
    """GUI: show every camera at once, one viewport window each, tiled in a
    grid (6 cameras at 480x360 in 3 columns fit a 1440x900 screen)."""
    import omni.kit.viewport.utility as vpu
    from pxr import Sdf

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


def define_physics_scene(path="/PhysicsScene", steps_per_second=60, gpu=True):
    """Author the physics scene explicitly (instead of letting omni.physx
    auto-create one at play) -- same solver settings as the auto-created
    default (TGS, GPU dynamics/broadphase, 60 Hz), plus one that matters
    for anything switched from kinematic to dynamic mid-simulation:

    reportKinematicStaticPairs. By default PhysX discards kinematic-vs-
    static pairs in the broadphase. A tube carried kinematically into the
    rotor bore (or resting kinematically in the holder) has its pairs
    with those static meshes discarded while their bounds overlap, and
    switching it to dynamic does NOT re-create them -- the overlap never
    "begins" again -- so on release it fell straight through the bore
    walls and floor (cad_source/probe_pickplace_physics.py). Keeping the
    pairs makes contact work from the moment it turns dynamic."""
    import isaacsim.core.experimental.utils.stage as stage_utils
    from pxr import Gf, PhysxSchema, UsdPhysics

    stage = stage_utils.get_current_stage()
    scene = UsdPhysics.Scene.Define(stage, path)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    px.CreateSolverTypeAttr("TGS")
    px.CreateEnableGPUDynamicsAttr(bool(gpu))
    px.CreateBroadphaseTypeAttr("GPU" if gpu else "MBP")
    px.CreateTimeStepsPerSecondAttr(int(steps_per_second))
    px.CreateReportKinematicStaticPairsAttr(True)
    px.CreateEnableCCDAttr(True)
    return path


def build_scene(simulation_app, rigid_objects=(), gpu_physics=True, physics_hz=PHYSICS_HZ):
    """Build room + platform + arm + cameras; settle the arm into its home
    pose. Returns (arm: Articulation, cams: {name: (prim_path, Camera)},
    objects: {name: {"path": prim_path, "size": (w, d, h)}}).
    rigid_objects: see build_platform.

    Shared by lab_scene.py's own CLI (viewing/capture) and sim/collect_data.py
    (data generation), so both always build the exact same scene.
    """
    from pxr import Gf, UsdGeom
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.experimental.objects import DomeLight
    from isaacsim.core.prims import Articulation

    define_physics_scene(steps_per_second=physics_hz, gpu=gpu_physics)

    # ---------------- room ----------------
    _box("/World/Floor", (ROOM_X, ROOM_Y, 0.1), (0, 0, -0.05), "lightgray")
    _box("/World/WallN", (ROOM_X, 0.1, ROOM_Z), (0, ROOM_Y / 2, ROOM_Z / 2), "white")
    _box("/World/WallS", (ROOM_X, 0.1, ROOM_Z), (0, -ROOM_Y / 2, ROOM_Z / 2), "white")
    _box("/World/WallE", (0.1, ROOM_Y, ROOM_Z), (ROOM_X / 2, 0, ROOM_Z / 2), "white")
    _box("/World/WallW", (0.1, ROOM_Y, ROOM_Z), (-ROOM_X / 2, 0, ROOM_Z / 2), "white")
    DomeLight("/World/DomeLight").set_intensities(1500)

    # ---------------- platform ----------------
    objects = build_platform(PLATFORM_CX, PLATFORM_CY, rigid_objects)

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
    configure_franka_physics()

    # ---------------- cameras ----------------
    import omni.replicator.core as rep
    from isaacsim.sensors.camera import Camera

    def aim(name, eye, target, res, focal_length=24.0):
        rep.create.camera(position=tuple(eye), look_at=tuple(target), name=name,
                          focal_length=focal_length)
        prim_path = f"/Replicator/{name}_Xform/{name}"
        # Replicator's default near clip is ~1 m: anything closer is cut away
        # (a camera 0.85 m from the work area showed the cut-open insides of
        # the bench and stand, rendered black).
        UsdGeom.Camera(stage_utils.get_current_stage().GetPrimAtPath(prim_path)) \
            .GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
        cam = Camera(prim_path, resolution=res)
        cam.initialize()
        return (prim_path, cam)

    # ---------------- physics: start, settle the arm into its home pose ------
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
    # 2 behind the tube platform, looking back at the arm. platform_cam1 is
    # the policy's exterior camera, so it's framed on the workspace instead
    # of the whole room: same viewing direction as the original wide shot,
    # but ~1 m from the work area and aimed between the holder and the
    # centrifuge (the wide version left the centrifuge ~20 px across and the
    # tube ~1 px once resized to the model's 224x224 input).
    eye = (PLATFORM_CX - 1.0, PLATFORM_CY + PLATFORM_D / 2 + 0.6, BENCH_Z + 0.9)
    target = (ARM_POS[0], ARM_POS[1], ARM_POS[2] + 0.3)
    cams["platform_cam0"] = aim("platform_cam0", eye, target, (640, 480), focal_length=18.0)
    cams["platform_cam1"] = aim("platform_cam1", EXTERIOR_CAM_EYE, EXTERIOR_CAM_TARGET,
                                (640, 480), focal_length=EXTERIOR_CAM_FOCAL)
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
            tile_camera_views(cams)
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

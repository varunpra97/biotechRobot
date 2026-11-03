"""Self-contained USD assets for the RL env, from the same CAD meshes and
collision settings as sim/lab_scene.py:

  assets/holder.usda  the 5x4 rack: exact triangle-mesh collider (static;
                      keeps the 8 mm blind bores exact), plastic friction
  assets/tube.usda    a test tube: a 1.5 g rigid body with the rendered mesh
                      and the 40-vertex convex-hull collider
                      (test_tube_collision.obj), glass friction, contact
                      offset 1 mm, rest offset 0; the env sets solver props
  assets/tray.usda    the sample tray: 15 g, one convex collider per piece
  assets/pipette.usda the electronic pipette: 150 g, its 7 convex pieces
                      (pipette_collision.obj), plastic; origin at the
                      nozzle end, lying along +y (--only pipette: just this)

Each file has one default prim holding the mesh(es) and its own physics
material, so an Isaac Lab UsdFileCfg reference carries everything.

Usage (Isaac venv, from the BiotechRobot folder):
    .venv-isaac\\Scripts\\python.exe biotech-lab-robotics\\simulation\\rl\\export_assets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ONLY = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else None
sys.path.insert(0, str(HERE.parent / "sim"))
OUT = HERE / "assets"

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402

import lab_scene  # noqa: E402

MATERIALS = {"glass": (0.2, 0.2, 0.05), "plastic": (0.3, 0.25, 0.1)}   # lab_scene.PHYSICS_MATERIALS


def define_mesh(stage, path, mesh, color=None):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in mesh.vertices]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.faces)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in mesh.faces.reshape(-1)]))
    m.CreateSubdivisionSchemeAttr("none")
    if color is not None:
        m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return m


def physics_material(stage, root, name):
    sf, df, rest = MATERIALS[name]
    mat = UsdShade.Material.Define(stage, f"{root}/PhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(sf)
    api.CreateDynamicFrictionAttr(df)
    api.CreateRestitutionAttr(rest)
    return mat


def collider(stage, path, approximation, mat):
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
    pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    pc.CreateContactOffsetAttr(0.001)
    pc.CreateRestOffsetAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def new_stage(path, root):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    r = UsdGeom.Xform.Define(stage, root)
    stage.SetDefaultPrim(r.GetPrim())
    return stage


def multi_piece_body(name, root, obj, collision_obj, mass, material, color):
    """A rigid body: the rendered mesh + one convex collider per collision piece (as lab_scene)."""
    stage = new_stage(OUT / f"{name}.usda", root)
    mat = physics_material(stage, root, material)
    define_mesh(stage, f"{root}/visual", trimesh.load(obj, force="mesh"), color)
    pieces = trimesh.load(collision_obj, force="mesh", process=False).split(only_watertight=False)
    for i, piece in enumerate(pieces):
        cp = f"{root}/collision" if len(pieces) == 1 else f"{root}/collision_{i}"
        cm = define_mesh(stage, cp, piece)
        UsdGeom.Imageable(cm).CreatePurposeAttr(UsdGeom.Tokens.guide)
        collider(stage, cp, "convexHull", mat)
    prim = stage.GetPrimAtPath(root)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    stage.GetRootLayer().Save()
    m = trimesh.load(obj, force="mesh")
    print(f"{name} ->", OUT / f"{name}.usda", "extents", np.round(m.extents, 4), "bounds", np.round(m.bounds, 4).tolist(),
          "collision pieces", len(pieces))


def centrifuge():
    """assets/centrifuge.usda: the body as loaded (housing + hub + six sleeves, no lid), exact triangle-mesh
    collider (static), already in world == env-local coordinates of the default layout (spawn at the origin);
    assets/centrifuge_bores.npz: the six bores' axes there (centrifuge_geometry.rotor_bores), labelled as
    prompts.bore_name ("1".."6")."""
    import centrifuge_geometry as cg
    import prompts
    body = cg.body_mesh_world()
    stage = new_stage(OUT / "centrifuge.usda", "/Centrifuge")
    mat = physics_material(stage, "/Centrifuge", "plastic")
    define_mesh(stage, "/Centrifuge/mesh", body, (0.8, 0.8, 0.82))
    collider(stage, "/Centrifuge/mesh", "none", mat)
    stage.GetRootLayer().Save()
    bores = cg.rotor_bores()
    parts = list(lab_scene.CENTRIFUGE_SLEEVE_PARTS)
    np.savez(OUT / "centrifuge_bores.npz", label=np.array([prompts.bore_name(i, parts) for i in parts]),
             point=np.array([bores[i].point for i in parts]), axis=np.array([bores[i].axis for i in parts]),
             opening_s=np.array([bores[i].opening_s for i in parts]), hub_center=bores[parts[0]].hub_center)
    print("centrifuge ->", OUT / "centrifuge.usda", "bounds", np.round(body.bounds, 3).tolist(), "faces", len(body.faces))
    for i in parts:
        b = bores[i]
        print(f"  bore {prompts.bore_name(i, parts)} (sleeve {i}): point {np.round(b.point, 4).tolist()} axis "
              f"{np.round(b.axis, 3).tolist()} tilt {b.tilt_deg:.1f} deg opening_s {b.opening_s:.4f}")


def main():
    OUT.mkdir(exist_ok=True)
    if ONLY:
        if "pipette" in ONLY:
            multi_piece_body("pipette", "/Pipette", lab_scene.PIPETTE_OBJ, lab_scene.PIPETTE_COLLISION_OBJ,
                             lab_scene.PIPETTE_MASS_KG, "plastic", (0.45, 0.5, 0.55))
        if "centrifuge" in ONLY:
            centrifuge()
        return
    # rack: exact triangle mesh, static
    stage = new_stage(OUT / "holder.usda", "/Holder")
    mat = physics_material(stage, "/Holder", "plastic")
    define_mesh(stage, "/Holder/mesh", trimesh.load(lab_scene.TUBE_HOLDER_OBJ, force="mesh"), (0.85, 0.85, 0.85))
    collider(stage, "/Holder/mesh", "none", mat)
    stage.GetRootLayer().Save()

    # tube: visual mesh (no collider) + convex-hull collider, glass
    stage = new_stage(OUT / "tube.usda", "/Tube")
    mat = physics_material(stage, "/Tube", "glass")
    define_mesh(stage, "/Tube/visual", trimesh.load(lab_scene.TEST_TUBE_OBJ, force="mesh"), (0.75, 0.9, 0.95))
    pieces = trimesh.load(lab_scene.TEST_TUBE_COLLISION_OBJ, force="mesh", process=False).split(only_watertight=False)
    assert len(pieces) == 1, len(pieces)
    cm = define_mesh(stage, "/Tube/collision", pieces[0])
    UsdGeom.Imageable(cm).CreatePurposeAttr(UsdGeom.Tokens.guide)
    collider(stage, "/Tube/collision", "convexHull", mat)
    # a dynamic rigid body (Isaac Lab's rigid_props/mass_props only edit an existing one)
    root = stage.GetPrimAtPath("/Tube")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.0015)            # lab_scene.TUBE_MASS_KG
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    stage.GetRootLayer().Save()
    # tray: 15 g rigid body, rendered mesh + one convex collider per collision piece (as lab_scene), plastic
    stage = new_stage(OUT / "tray.usda", "/Tray")
    mat = physics_material(stage, "/Tray", "plastic")
    define_mesh(stage, "/Tray/visual", trimesh.load(lab_scene.SAMPLE_TRAY_OBJ, force="mesh"), (0.6, 0.6, 0.9))
    pieces = trimesh.load(lab_scene.SAMPLE_TRAY_COLLISION_OBJ, force="mesh", process=False).split(only_watertight=False)
    for i, piece in enumerate(pieces):
        cp = "/Tray/collision" if len(pieces) == 1 else f"/Tray/collision_{i}"
        cm = define_mesh(stage, cp, piece)
        UsdGeom.Imageable(cm).CreatePurposeAttr(UsdGeom.Tokens.guide)
        collider(stage, cp, "convexHull", mat)
    root = stage.GetPrimAtPath("/Tray")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(lab_scene.TRAY_MASS_KG)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    stage.GetRootLayer().Save()
    tray = trimesh.load(lab_scene.SAMPLE_TRAY_OBJ, force="mesh")
    print("tray ->", OUT / "tray.usda", "extents", np.round(tray.extents, 4), "z range", np.round(tray.bounds[:, 2], 4),
          "collision pieces", len(pieces))
    tube = trimesh.load(lab_scene.TEST_TUBE_OBJ, force="mesh")
    print("holder ->", OUT / "holder.usda")
    print("tube ->", OUT / "tube.usda", "extents", np.round(tube.extents, 4), "z range", np.round(tube.bounds[:, 2], 4))


main()
app.close()

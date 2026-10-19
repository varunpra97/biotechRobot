"""Stage 6: simulation-ready assets from reconstructed objects.

Per object:
  - visual mesh
  - collision: convex decomposition (V-HACD), each piece capped at 64 vertices
  - rigid body with mass = envelope volume x solid fraction x density
  - physics material (static/dynamic friction, restitution, density)

Written two ways, so the other agent can use whichever fits their loader:
  - <name>.usda: self-contained USD (Z-up, meters, kg) with UsdPhysics
    rigid body, mass, collision APIs and a bound physics material
  - <name>.obj + <name>_collision.obj: the same split lab_scene.py's
    load_mesh_prim(..., collision_obj=...) already consumes -- one convex
    collider per connected piece of the collision OBJ.

Material, density and solid fraction come from LLM scene reasoning when
available; otherwise DEFAULT_MATERIAL (the sim's own "plastic" preset)
and solid_fraction 1.0 are used and the asset is flagged unverified.
"""
from __future__ import annotations

import re

import numpy as np
import trimesh

PHYSX_MAX_HULL_VERTICES = 64

DEFAULT_MATERIAL = {
    # lab_scene.PHYSICS_MATERIALS["plastic"], plus a typical labware-plastic density
    "name": "plastic", "density_kg_m3": 1050.0,
    "static_friction": 0.3, "dynamic_friction": 0.25, "restitution": 0.1,
}


def safe_name(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return ("obj_" + s if not s or s[0].isdigit() else s)[:48]


def _farthest_point_subset(points: np.ndarray, k: int) -> np.ndarray:
    chosen = [int(np.argmax(np.linalg.norm(points - points.mean(axis=0), axis=1)))]
    d = np.linalg.norm(points - points[chosen[0]], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(points - points[i], axis=1))
    return points[chosen]


def cap_hull_vertices(hull: trimesh.Trimesh, max_vertices: int = PHYSX_MAX_HULL_VERTICES) -> trimesh.Trimesh:
    """PhysX caps convex hulls at 64 vertices and, when it has to reduce
    one, EXPANDS it to contain every input point -- an inflated collider
    (lab_scene.py notes tubes ending up wider than their bores this way).
    Instead reduce it here to the hull of a well-spread subset of its own
    vertices, which can only sit inside the original surface."""
    if len(hull.vertices) <= max_vertices:
        return hull
    return trimesh.convex.convex_hull(_farthest_point_subset(hull.vertices, max_vertices))


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return mesh
    return mesh.simplify_quadric_decimation(face_count=target_faces)


def convex_decompose(mesh: trimesh.Trimesh, max_parts: int = 16, input_faces: int = 5000) -> list:
    """V-HACD approximate convex decomposition (via trimesh/vhacdx), each
    piece capped at PhysX's 64 vertices; falls back to one convex hull.

    CoACD was tried first and is too slow here: ~90s for a simple 5k-face
    box even at reduced settings; V-HACD takes 2-3s on a 240k-face mesh.
    Input is decimated first -- a collider needs nowhere near marching-cubes
    detail. Objects with a fitted template get colliders from the template
    instead; this is the generic fallback."""
    mesh = decimate(mesh, input_faces)
    try:
        parts = trimesh.decomposition.convex_decomposition(
            mesh, maxConvexHulls=max_parts, maxNumVerticesPerCH=PHYSX_MAX_HULL_VERTICES)
        parts = parts if isinstance(parts, list) else [parts]
        hulls = [p if isinstance(p, trimesh.Trimesh) else trimesh.Trimesh(p["vertices"], p["faces"])
                 for p in parts]
        hulls = [h for h in hulls if len(h.faces) >= 4]
    except Exception:
        hulls = []
    if not hulls:
        hulls = [mesh.convex_hull]
    return [cap_hull_vertices(h) for h in hulls]


def mass_properties(mesh: trimesh.Trimesh, density: float, solid_fraction: float,
                    envelope_volume_m3: float | None = None) -> tuple[float, np.ndarray]:
    """(mass kg, center of mass). Prefer the reconstruction's exact voxel
    volume: marching-cubes meshes are often not watertight (voxels touching
    only along an edge make non-manifold seams), and the convex-hull
    fallback would overstate the mass of anything concave."""
    if envelope_volume_m3 is None:
        envelope_volume_m3 = (mesh if mesh.is_watertight else mesh.convex_hull).volume
    com = mesh.center_mass if mesh.is_watertight else mesh.vertices.mean(axis=0)
    return float(envelope_volume_m3 * solid_fraction * density), np.asarray(com, float)


def write_object_usd(path, name: str, visual: trimesh.Trimesh, collision: list, mass_kg: float,
                     center_of_mass: np.ndarray, material: dict, metadata: dict,
                     color=(0.8, 0.8, 0.8), collision_approximation: str = "convexHull") -> None:
    """collision_approximation: "convexHull" (one convex collider per piece in
    `collision`) or "none" (exact triangle mesh -- static fixtures whose
    cavities must stay exact, e.g. bores a tube slides into, as lab_scene
    does for its tube holder)."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("vision_pipeline", metadata)

    def define_mesh(prim_path, m):
        um = UsdGeom.Mesh.Define(stage, prim_path)
        um.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in m.vertices])
        um.CreateFaceVertexCountsAttr([3] * len(m.faces))
        um.CreateFaceVertexIndicesAttr(m.faces.flatten().tolist())
        um.CreateExtentAttr([Gf.Vec3f(*map(float, m.bounds[0])), Gf.Vec3f(*map(float, m.bounds[1]))])
        return um

    vis = define_mesh(f"/{name}/visual", visual)
    vis.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    phys_mat = UsdShade.Material.Define(stage, f"/{name}/PhysicsMaterial")
    mat_api = UsdPhysics.MaterialAPI.Apply(phys_mat.GetPrim())
    mat_api.CreateStaticFrictionAttr(float(material["static_friction"]))
    mat_api.CreateDynamicFrictionAttr(float(material["dynamic_friction"]))
    mat_api.CreateRestitutionAttr(float(material["restitution"]))
    mat_api.CreateDensityAttr(float(material["density_kg_m3"]))

    for i, hull in enumerate(collision):
        cm = define_mesh(f"/{name}/collision_{i}", hull)
        UsdGeom.Imageable(cm).CreatePurposeAttr(UsdGeom.Tokens.guide)  # invisible, like lab_scene's colliders
        UsdPhysics.CollisionAPI.Apply(cm.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(cm.GetPrim()).CreateApproximationAttr(
            UsdPhysics.Tokens.none if collision_approximation == "none" else UsdPhysics.Tokens.convexHull)
        UsdShade.MaterialBindingAPI.Apply(cm.GetPrim()).Bind(
            phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(float(mass_kg))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, center_of_mass)))
    stage.GetRootLayer().Save()


def write_scene_usd(path, placements: list, frame_note: str) -> None:
    """Scene layer referencing each object's USD at its placement.
    placements: (prim name, relative usd path, xyz in the platform frame[, yaw rad])."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/Platform")
    stage.SetDefaultPrim(world.GetPrim())
    world.GetPrim().SetCustomDataByKey("frame", frame_note)
    for placement in placements:
        name, rel, xyz = placement[:3]
        yaw = placement[3] if len(placement) > 3 else 0.0
        prim = UsdGeom.Xform.Define(stage, f"/Platform/{name}")
        prim.GetPrim().GetReferences().AddReference(rel)
        prim.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, xyz)))
        if abs(yaw) > 1e-9:
            prim.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(float(np.degrees(yaw)))
    stage.GetRootLayer().Save()

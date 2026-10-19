"""Render a random-object test scene (scripts/make_random_scene.py) in Isaac
Sim: RGB + depth from the same 25-camera rig, on the same lab bench with the
same calibration markers as the lab-scene validation capture -- but with
every piece of labware hidden and the generated objects in its place.

  .venv-isaac/Scripts/python.exe biotech-lab-robotics/vision_pipeline/isaac_validation/render_rgbd_test_scene.py \
      --scene-dir biotech-lab-robotics/vision_pipeline/depth_tests/random_s1

Writes into --scene-dir: images/, depth/ (uint16 PNG, 0.1mm), manifest.json,
camera_intrinsics.json, marker_layout.json, ground_truth_camera_positions.json.
The ground-truth meshes are the generator's own (already in --scene-dir).

Same isolation as render_isaac_validation_capture.py: the scene comes from
sim_snapshot/, a frozen copy; nothing under biotech-lab-robotics/ is touched.
Check the GPU is free first (gpu_guard.sh).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_isaac_validation_capture import (CAPTURE_RESOLUTION, DEPTH_PNG_SCALE_M,  # noqa: E402
                                             SingleCaptureCamera, add_markers_to_stage, add_validation_cameras,
                                             camera_world_position, configure_renderer, cube_positions,
                                             marker_corners_world, marker_specs, save_depth_png)

import numpy as np  # noqa: E402

KEEP_PLATFORM_CHILDREN = {"Stand", "Bench", "WorkLight"}


def add_object(stage, path: str, points_world: np.ndarray, faces: np.ndarray, corner_normals: np.ndarray,
               material: dict) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points_world.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
    # Authored per-corner normals (smooth on curved surfaces, sharp across
    # creases) and no subdivision: renders the exact geometry being scored.
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(corner_normals.reshape(-1, 3).astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)

    mat = UsdShade.Material.Define(stage, f"{path}_Material")
    sh = UsdShade.Shader.Define(stage, f"{path}_Material/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*material["diffuse_rgb"]))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(material["roughness"]))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(material["metallic"]))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--only", nargs="+", help="render just these views (quick checks)")
    args = ap.parse_args()

    out_dir = Path(args.scene_dir)
    spec = json.loads((out_dir / "scene_spec.json").read_text())
    rm = np.load(out_dir / "render_meshes.npz")
    images_dir, depth_dir = out_dir / "images", out_dir / "depth"
    images_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    import isaacsim.core.experimental.utils.stage as stage_utils
    import lab_scene
    from PIL import Image
    from pxr import Usd, UsdGeom

    arm, cams, objects = lab_scene.build_scene(simulation_app)
    stage = stage_utils.get_current_stage()
    for path in ("/World/Arm", "/World/Pedestal"):
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
    # Deactivated, not just made invisible: the 100 tips are kinematic rigid
    # bodies, and with only their parent Xform invisible they still rendered
    # (first random-scene capture). An inactive prim isn't in the scene at all.
    removed = []
    for child in list(stage.GetPrimAtPath("/World/Platform").GetChildren()):
        if child.GetName() not in KEEP_PLATFORM_CHILDREN:
            removed.append(child.GetName())
            child.SetActive(False)
    print(f"removed the lab's labware: {', '.join(removed)}")
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        if prim.IsA(UsdGeom.Mesh):
            UsdGeom.Mesh(prim).GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    origin = np.array([lab_scene.PLATFORM_CX, lab_scene.PLATFORM_CY, lab_scene.BENCH_Z + 0.04])
    UsdGeom.Xform.Define(stage, "/World/RandomObjects")
    for o in spec["objects"]:
        name = o["name"]
        add_object(stage, f"/World/RandomObjects/{name}", rm[f"{name}/points"].astype(np.float64) + origin,
                   rm[f"{name}/faces"], rm[f"{name}/normals"], o["material"])
    print(f"added {len(spec['objects'])} random objects")

    add_markers_to_stage(lab_scene, stage, out_dir / "marker_textures")
    capture_cams, _ = add_validation_cameras(lab_scene, stage)
    for _ in range(4):
        simulation_app.update()
    for _, scene_cam in cams.values():
        scene_cam.destroy()
    configure_renderer()

    if args.only:
        capture_cams = {k: v for k, v in capture_cams.items() if k in args.only}
    camera = SingleCaptureCamera(simulation_app, stage, CAPTURE_RESOLUTION)
    cameras_json, shots = {}, []
    for name, prim_path in capture_cams.items():
        rgb, depth, intr = camera.capture(prim_path)
        Image.fromarray(rgb).save(str(images_dir / f"{name}.png"))
        save_depth_png(depth_dir / f"{name}.png", depth)
        cameras_json[name] = intr
        shots.append({"image": f"images/{name}.png", "depth": f"depth/{name}.png", "camera_id": name})
        ok = np.isfinite(depth) & (depth > 0)
        print(f"  captured {name}: depth {depth[ok].min():.3f}-{depth[ok].max():.3f} m", flush=True)

    (out_dir / "camera_intrinsics.json").write_text(json.dumps({"cameras": cameras_json}, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps({
        "shots": shots,
        "depth": {"format": "uint16 png", "scale_m": DEPTH_PNG_SCALE_M, "invalid": 0,
                  "kind": "z (distance to the image plane), registered to the colour image",
                  "noise": "none (rendered); see depth_sensor.py for sensor models",
                  # the depth annotator samples each pixel at its corner, half a
                  # pixel off K's pixel-center convention (measured; see reconstruct_depth.py)
                  "pixel_offset_px": 0.5}}, indent=2))
    specs = marker_specs(lab_scene)
    (out_dir / "marker_layout.json").write_text(json.dumps({
        "markers": {str(mid): {"corners_3d": (marker_corners_world(s) - origin).tolist()} for mid, s in specs.items()},
        "rigid_groups": [list(range(4 + 5 * i, 9 + 5 * i)) for i in range(len(cube_positions(lab_scene)))]},
        indent=2))
    (out_dir / "ground_truth_camera_positions.json").write_text(json.dumps(
        {"frame": "platform-local (origin at bench-top center, Z up, meters)",
         "positions": {n: (camera_world_position(stage, pp) - origin).tolist() for n, pp in capture_cams.items()}},
        indent=2))
    print(f"wrote {len(shots)} RGB-D views to {out_dir}")
    sys.stdout.flush()
    os._exit(0)   # see render_isaac_validation_capture.py: close() hangs after a path-traced capture


if __name__ == "__main__":
    main()

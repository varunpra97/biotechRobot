"""Render a scene of procedurally generated objects the pipeline has never
seen (random_objects.py in this folder) with the same bench, lighting,
markers, cubes, 25-camera rig and path-traced 12MP settings as the labware
validation capture -- RGB only, to test that camera-only reconstruction
generalizes beyond the lab scene. (The RGB-D variant is
render_rgbd_test_scene.py.)

The snapshot's lab_scene.build_scene() provides the room, bench and
lights; its labware is deactivated in this process's stage only, and the
random meshes are placed on the bench instead. Ground truth is exported
from the live stage exactly like render_isaac_validation_capture.py, so the
same scoring tools apply unchanged.

  .venv-isaac/Scripts/python.exe biotech-lab-robotics/vision_pipeline/isaac_validation/render_random_scene.py \
      --objects biotech-lab-robotics/vision_pipeline/isaac_validation/random_objects \
      --out biotech-lab-robotics/vision_pipeline/isaac_validation/out_random
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import render_isaac_validation_capture as rv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", required=True, help="random_objects.py output dir (objects.json + OBJs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="+", help="render just these views (quick quality checks)")
    args = ap.parse_args()

    obj_dir = Path(args.objects).resolve()
    manifest = json.loads((obj_dir / "objects.json").read_text())["objects"]
    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    import isaacsim.core.experimental.utils.stage as stage_utils
    import lab_scene
    from PIL import Image
    from pxr import Usd, UsdGeom

    arm, cams, _ = lab_scene.build_scene(simulation_app)
    stage = stage_utils.get_current_stage()
    for path in ("/World/Arm", "/World/Pedestal"):
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()

    # Keep the platform itself and its light; deactivate the labware (a
    # deactivated prim is neither rendered nor traversed by the GT export).
    keep = {"Stand", "Bench", "WorkLight"}
    for child in list(stage.GetPrimAtPath("/World/Platform").GetChildren()):
        if child.GetName() not in keep:
            child.SetActive(False)

    origin = np.array([lab_scene.PLATFORM_CX, lab_scene.PLATFORM_CY, lab_scene.BENCH_Z + 0.04])
    for o in manifest:
        x, y, z = o["placement_m"]
        prim = f"/World/Platform/{o['name'].title().replace('_', '')}"
        lab_scene.load_mesh_prim(prim, str(obj_dir / o["obj"]), tuple(origin + [x, y, z]),
                                 yaw_deg=o["yaw_deg"], color=tuple(o["rgb"]), collision=False)
        print(f"placed {o['name']} ({o['color']}) at {prim}")

    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        if prim.IsA(UsdGeom.Mesh):
            UsdGeom.Mesh(prim).GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    rv.add_markers_to_stage(lab_scene, stage, out_dir / "marker_textures")
    capture_cams, _ = rv.add_validation_cameras(lab_scene, stage)
    for _ in range(4):
        simulation_app.update()
    for _, scene_cam in cams.values():
        scene_cam.destroy()
    rv.configure_renderer()

    if args.only:
        capture_cams = {k: v for k, v in capture_cams.items() if k in args.only}
    cameras_json, shots = {}, []
    capture_camera = rv.SingleCaptureCamera(simulation_app, stage, rv.CAPTURE_RESOLUTION)
    for name, prim_path in capture_cams.items():
        frame = capture_camera.capture(prim_path)
        rgb, intr = frame[0], frame[-1]   # the capture may also return depth; RGB only here
        Image.fromarray(rgb).save(str(images_dir / f"{name}.png"))
        cameras_json[name] = intr
        shots.append({"image": f"images/{name}.png", "camera_id": name})
        print(f"  captured {name}: {rgb.shape[1]}x{rgb.shape[0]}", flush=True)

    (out_dir / "camera_intrinsics.json").write_text(json.dumps({"cameras": cameras_json}, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps({"shots": shots}, indent=2))
    specs = rv.marker_specs(lab_scene)
    (out_dir / "marker_layout.json").write_text(json.dumps(
        {"markers": {str(mid): {"corners_3d": (rv.marker_corners_world(s) - origin).tolist()} for mid, s in specs.items()},
         "rigid_groups": [list(range(4 + 5 * i, 9 + 5 * i)) for i in range(len(rv.cube_positions(lab_scene)))]},
        indent=2))
    (out_dir / "ground_truth_camera_positions.json").write_text(json.dumps(
        {"frame": "platform-local (origin at bench-top center, Z up, meters)",
         "positions": {n: (rv.camera_world_position(stage, pp) - origin).tolist() for n, pp in capture_cams.items()}},
        indent=2))
    (out_dir / "ground_truth_meshes.json").write_text(
        json.dumps(rv.export_ground_truth_meshes(lab_scene, stage, out_dir), indent=2))
    (out_dir / "random_objects.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "platform_dims.json").write_text(json.dumps(
        {"platform_w": lab_scene.PLATFORM_W, "platform_d": lab_scene.PLATFORM_D}, indent=2))
    print(f"\nwrote {len(shots)} images to {images_dir}")
    sys.stdout.flush()
    os._exit(0)   # simulation_app.close() hangs after a path-traced capture


if __name__ == "__main__":
    main()

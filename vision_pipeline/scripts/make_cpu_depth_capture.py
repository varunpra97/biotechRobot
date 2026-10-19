"""A depth-only capture of known meshes, ray-cast on the CPU -- for
developing the depth path while the GPU is busy, and as a fast unit test
of fusion. Same 25-camera rig as the Isaac renders (three rings at
15/35/55 deg elevation, 1.5m from the platform center, plus top-down), same
folder layout (depth/, manifest.json, camera_intrinsics.json,
poses/camera_poses.json, ground_truth_meshes.json), so the reconstruction
scripts run on it unchanged.

Poses are written EXACT (there are no colour images to solve them from
markers); --pose-noise-mm / --pose-noise-deg perturb them to mimic Stage 1.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/make_cpu_depth_capture.py \
      --gt-from biotech-lab-robotics/vision_pipeline/isaac_validation/out --out biotech-lab-robotics/vision_pipeline/depth_tests/lab_cpu
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import trimesh

from depth_sensor import save_depth
from geometry import camera_pose_to_rvec_tvec, look_at_rotation, rotation_matrix_to_quaternion
from synthetic_depth import DepthRaycaster, bench_meshes

# The Isaac capture camera: 18mm lens, 4032x3024, f = 3463.4px.
FULL_RES = (4032, 3024)
FULL_F = 3463.42


def ring25(target=(0.0, 0.0, 0.05)):
    target = np.asarray(target, float)
    specs = {}
    for elev_deg, az_offset in ((15, 0.0), (35, 22.5), (55, 0.0)):
        for i in range(8):
            az_deg = i * 45 + az_offset
            az, el = np.radians(az_deg), np.radians(elev_deg)
            eye = target + 1.5 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            specs[f"el{elev_deg:02d}_az{az_deg:05.1f}".replace(".", "p")] = (eye, target)
    specs["top_down"] = (np.array([0.0, 0.01, 1.9]), np.array([0.0, 0.0, 0.0]))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-from", help="capture dir whose ground_truth_meshes.json (platform frame) to render")
    ap.add_argument("--meshes-json", help="alternatively: a ground_truth_meshes.json-style file")
    ap.add_argument("--platform-w", type=float, default=1.4)
    ap.add_argument("--platform-d", type=float, default=0.7)
    ap.add_argument("--downscale", type=int, default=2, help="render at 4032x3024 / this")
    ap.add_argument("--pose-noise-mm", type=float, default=0.0)
    ap.add_argument("--pose-noise-deg", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rgb", action="store_true",
                    help="also write anti-aliased Lambert-shaded colour images (for silhouette refinement tests); "
                         "colours from scene_spec.json if present")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "poses").mkdir(exist_ok=True)
    src_json = Path(args.meshes_json) if args.meshes_json else Path(args.gt_from) / "ground_truth_meshes.json"
    gt = json.loads(src_json.read_text())
    (out / "ground_truth_meshes").mkdir(exist_ok=True)
    meshes = []
    for name, g in gt.items():
        src = src_json.parent / g["mesh"]
        dst = out / "ground_truth_meshes" / f"{name}.obj"
        if src.resolve() != dst.resolve():
            shutil.copy(src, dst)
        g["mesh"] = f"ground_truth_meshes/{name}.obj"
        meshes.append(trimesh.load(dst, force="mesh"))
    (out / "ground_truth_meshes.json").write_text(json.dumps(gt, indent=2))
    (out / "platform_dims.json").write_text(json.dumps({"platform_w": args.platform_w, "platform_d": args.platform_d}))

    caster = DepthRaycaster(meshes + bench_meshes(args.platform_w, args.platform_d))
    spec_file = src_json.parent / "scene_spec.json"
    mat = ({o["name"]: o["material"]["diffuse_rgb"] for o in json.loads(spec_file.read_text())["objects"]}
           if spec_file.exists() else {})
    colors = np.array([mat.get(n, [0.75, 0.75, 0.78]) for n in gt] + [[0.92, 0.92, 0.9], [0.5, 0.5, 0.5]])
    if args.rgb:
        (out / "images").mkdir(exist_ok=True)
    w, h = FULL_RES[0] // args.downscale, FULL_RES[1] // args.downscale
    K = np.array([[FULL_F / args.downscale, 0, (w - 1) / 2], [0, FULL_F / args.downscale, (h - 1) / 2], [0, 0, 1]])
    rng = np.random.default_rng(args.seed)
    cams, shots, poses = {}, [], []
    for name, (eye, target) in ring25().items():
        R_cw = look_at_rotation(eye, target)
        rvec, tvec = camera_pose_to_rvec_tvec(R_cw, eye)
        depth = caster.render(K, rvec, tvec, (w, h))
        save_depth(out / "depth" / f"{name}.png", depth)
        if args.rgb:
            rgb = caster.render_shaded(K, rvec, tvec, (w, h), colors)
            cv2.imwrite(str(out / "images" / f"{name}.png"), rgb[..., ::-1])
        # What Stage 1 would report: the true pose, optionally perturbed.
        dR, _ = cv2.Rodrigues(rng.normal(0, np.radians(args.pose_noise_deg), 3) if args.pose_noise_deg else np.zeros(3))
        pos = eye + (rng.normal(0, args.pose_noise_mm / 1000, 3) if args.pose_noise_mm else 0)
        poses.append({"image": f"images/{name}.png", "depth": f"depth/{name}.png", "camera_id": name,
                      "position_world": pos.tolist(),
                      "quaternion_wxyz": rotation_matrix_to_quaternion(R_cw @ dR).tolist(),
                      "marker_ids_used": [-1, -1, -1, -1]})
        cams[name] = {"K": K.tolist(), "dist": [0.0] * 5, "image_size": [w, h]}
        shots.append({"image": f"images/{name}.png", "depth": f"depth/{name}.png", "camera_id": name})
        print(f"  {name}: depth {depth[depth > 0].min():.3f}-{depth[depth > 0].max():.3f} m")
    (out / "camera_intrinsics.json").write_text(json.dumps({"cameras": cams}, indent=2))
    (out / "manifest.json").write_text(json.dumps({
        "shots": shots,
        "depth": {"format": "uint16 png", "scale_m": 1e-4, "invalid": 0, "kind": "z, registered to the colour image",
                  "noise": "none (ray-cast)"}}, indent=2))
    (out / "poses" / "camera_poses.json").write_text(json.dumps({
        "poses": poses, "note": f"exact rig poses + noise {args.pose_noise_mm}mm / {args.pose_noise_deg}deg"}, indent=2))
    print(f"wrote {len(shots)} depth views at {w}x{h} to {out}")


if __name__ == "__main__":
    main()

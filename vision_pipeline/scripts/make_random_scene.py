"""Generate a random test scene: N procedural complex objects (random_objects.py)
placed on the lab platform clear of the calibration markers, with random
plastic-like materials. Writes everything the renderers need, and the
ground truth the reconstruction is scored against:

  <out>/scene_spec.json            objects: recipe, parameters, pose, material
  <out>/ground_truth_meshes/*.obj  exact meshes, platform frame
  <out>/ground_truth_meshes.json   same index format as the Isaac captures
  <out>/render_meshes.npz          per object: points, triangles, and
                                   per-corner normals (smooth within 30 deg,
                                   sharp across creases -- without authored
                                   normals the renderer smooths every edge)
  <out>/platform_dims.json

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/make_random_scene.py --seed 1 --out biotech-lab-robotics/vision_pipeline/depth_tests/random_s1
Then render with isaac_validation/render_rgbd_test_scene.py (GPU), or on the
CPU (depth only) with make_cpu_depth_capture.py --gt-from <out>.
"""
import argparse
import colorsys
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh

from random_objects import make_scene

PLATFORM_W, PLATFORM_D = 1.4, 0.7
# Calibration fixtures, platform frame (render_isaac_validation_capture.py's
# marker_specs / cube_positions): flat 80mm markers and 120mm cubes.
FLAT_MARKERS_XY = [(-0.62, 0.0), (0.62, 0.0), (-0.2, 0.27), (0.2, 0.27)]
CUBES_XY = [(-0.55, 0.24), (0.55, 0.24)]


def keepouts(margin=0.03):
    boxes = [(np.array(c) - (0.04 + margin), np.array(c) + (0.04 + margin)) for c in FLAT_MARKERS_XY]
    boxes += [(np.array(c) - (0.06 + margin), np.array(c) + (0.06 + margin)) for c in CUBES_XY]
    return boxes


def corner_normals(mesh: trimesh.Trimesh, crease_deg: float = 30.0) -> np.ndarray:
    """(F, 3, 3) normals per triangle corner: the area-weighted mean of the
    adjacent face normals that are within crease_deg of this face's."""
    fn = mesh.face_normals
    out = np.repeat(fn[:, None, :], 3, axis=1).copy()
    cos_t = np.cos(np.radians(crease_deg))
    vf = mesh.vertex_faces                                  # (V, k), -1 padded
    area = mesh.area_faces
    for f in range(len(mesh.faces)):
        for c, v in enumerate(mesh.faces[f]):
            nb = vf[v][vf[v] >= 0]
            ok = nb[(fn[nb] @ fn[f]) > cos_t]
            n = (fn[ok] * area[ok, None]).sum(axis=0)
            nn = np.linalg.norm(n)
            if nn > 0:
                out[f, c] = n / nn
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-objects", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "ground_truth_meshes").mkdir(parents=True, exist_ok=True)
    scene = make_scene(args.seed, args.n_objects, (PLATFORM_W, PLATFORM_D), keepouts())
    rng = np.random.default_rng(args.seed + 1000)
    spec, gt, render = [], {}, {}
    for i, (name, mesh, info) in enumerate(scene):
        # Saturated, mid-brightness colours: distinct from the white bench
        # and from each other, like coloured lab plastics. One in four is
        # near-white -- white-on-white is the realistic hard case.
        if rng.random() < 0.25:
            rgb = [float(rng.uniform(0.82, 0.92))] * 3
        else:
            rgb = list(colorsys.hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.45, 0.85), rng.uniform(0.35, 0.8)))
        material = {"diffuse_rgb": rgb, "roughness": float(rng.uniform(0.35, 0.8)), "metallic": 0.0}
        mesh.export(out / "ground_truth_meshes" / f"{name}.obj")
        lo, hi = mesh.bounds
        gt[name] = {"mesh": f"ground_truth_meshes/{name}.obj", "bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
                    "extents": (hi - lo).tolist()}
        spec.append({"name": name, **info, "material": material})
        render[f"{name}/points"] = mesh.vertices.astype(np.float32)
        render[f"{name}/faces"] = mesh.faces.astype(np.int32)
        render[f"{name}/normals"] = corner_normals(mesh).astype(np.float32)
        e = (hi - lo) * 1000
        print(f"  {name:28s} {e[0]:5.0f} x {e[1]:5.0f} x {e[2]:5.0f} mm at ({info['center_xy_m'][0]:+.3f}, "
              f"{info['center_xy_m'][1]:+.3f})  genus {info['genus']}")
    np.savez_compressed(out / "render_meshes.npz", **render)
    (out / "scene_spec.json").write_text(json.dumps({"seed": args.seed, "frame": "platform-local, meters",
                                                     "objects": spec}, indent=2))
    (out / "ground_truth_meshes.json").write_text(json.dumps(gt, indent=2))
    (out / "platform_dims.json").write_text(json.dumps({"platform_w": PLATFORM_W, "platform_d": PLATFORM_D}))
    trimesh.util.concatenate([m for _, m, _ in scene]).export(out / "ground_truth_scene.obj")
    print(f"{len(scene)} objects -> {out}")


if __name__ == "__main__":
    main()

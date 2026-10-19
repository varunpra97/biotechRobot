"""Generate a test scene of COMPOSITE objects (composite_objects.py): racks
with tubes, a mini centrifuge with a loaded rotor, a tray with metal weights,
a capped bottle, a lidded box, a pegboard -- each part a separate mesh with
its own material. Same output layout as make_random_scene.py (so
isaac_validation/render_rgbd_test_scene.py and make_cpu_depth_capture.py
render it unchanged), plus the per-part truth the decomposition, material
and mass estimates are scored against:

  <out>/ground_truth_composites.json   per composite: parts, roles,
                                       material, density, volume, mass

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/make_composite_scene.py --seed 1 \
      --out biotech-lab-robotics/vision_pipeline/depth_tests/composite_s1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import trimesh

from composite_objects import MATERIALS, make_composite_scene, material_name
from make_random_scene import PLATFORM_D, PLATFORM_W, corner_normals, keepouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "ground_truth_meshes").mkdir(parents=True, exist_ok=True)
    scene = make_composite_scene(args.seed, (PLATFORM_W, PLATFORM_D), keepouts())
    spec, gt, render, comps = [], {}, {}, {}
    for group, parts, info in scene:
        comp_parts = []
        for pname, mesh, mat_key, role in parts:
            name = f"{group}__{pname}"
            m = MATERIALS[mat_key]
            mesh.export(out / "ground_truth_meshes" / f"{name}.obj")
            lo, hi = mesh.bounds
            gt[name] = {"mesh": f"ground_truth_meshes/{name}.obj", "bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
                        "extents": (hi - lo).tolist(), "group": group}
            spec.append({"name": name, "group": group, "part": pname, "role": role,
                         "material": {"diffuse_rgb": m["rgb"], "roughness": m["roughness"], "metallic": m["metallic"]},
                         "material_name": material_name(mat_key)})
            render[f"{name}/points"] = mesh.vertices.astype(np.float32)
            render[f"{name}/faces"] = mesh.faces.astype(np.int32)
            render[f"{name}/normals"] = corner_normals(mesh).astype(np.float32)
            vol = float(mesh.volume)
            comp_parts.append({"name": name, "part": pname, "role": role, "material": material_name(mat_key),
                               "density_kg_m3": m["density"], "volume_m3": vol,
                               "mass_kg": vol * m["density"], "extents_m": (hi - lo).tolist(),
                               "center_m": ((lo + hi) / 2).tolist()})
        allm = trimesh.util.concatenate([p[1] for p in parts])
        comps[group] = {"recipe": info["recipe"], "info": info, "n_parts": len(parts),
                        "total_mass_kg": sum(p["mass_kg"] for p in comp_parts),
                        "bbox_min": allm.bounds[0].tolist(), "bbox_max": allm.bounds[1].tolist(),
                        "parts": comp_parts}
        e = allm.extents * 1000
        print(f"  {group:28s} {len(parts):2d} parts  {e[0]:5.0f} x {e[1]:5.0f} x {e[2]:5.0f} mm  "
              f"{comps[group]['total_mass_kg'] * 1000:7.1f} g  ({', '.join(sorted({p['material'] for p in comp_parts}))})")
    np.savez_compressed(out / "render_meshes.npz", **render)
    (out / "scene_spec.json").write_text(json.dumps({"seed": args.seed, "frame": "platform-local, meters",
                                                     "objects": spec}, indent=2))
    (out / "ground_truth_meshes.json").write_text(json.dumps(gt, indent=2))
    (out / "ground_truth_composites.json").write_text(json.dumps(comps, indent=2))
    (out / "platform_dims.json").write_text(json.dumps({"platform_w": PLATFORM_W, "platform_d": PLATFORM_D}))
    print(f"{len(comps)} composites, {len(spec)} parts -> {out}")


if __name__ == "__main__":
    main()

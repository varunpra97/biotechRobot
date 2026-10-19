"""Score the pipeline's reconstruction of the random-object scene, object by
object, against the exact generated meshes.

For every reconstructed object: run freeform silhouette refinement
(scripts/refine_freeform.py, unchanged), match it to a ground-truth object
by bounding-box overlap, and report size error plus surface accuracy for
the visual hull and the refined mesh.

Completeness is split in two, because a camera-only method has a hard
limit there. The IDEAL visual hull -- the true mesh carved by its own
exact silhouettes from the same camera positions -- is the best any
outline-based method can do with this rig. A ground-truth surface point
more than CONCAVE_MM inside it is "concave" (inside of the mug, funnel
bore): no silhouette shows it. The rest is "convex" (outline-observable).
Fixable error shows up in the convex number; the concave number and the
ideal hull's own score show the method's limit.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/isaac_validation/score_random_scene.py \
      --capture-dir biotech-lab-robotics/vision_pipeline/isaac_validation/out_random [--skip-refine]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
CONCAVE_MM = 2.0


def ideal_visual_hull(gt_mesh: trimesh.Trimesh, cam_positions, voxel_m: float = 0.001,
                      px_m: float = 0.00025) -> trimesh.Trimesh:
    """Carve a voxel grid with the true mesh's exact perspective silhouettes
    from each camera center (rasterized at px_m per pixel at the object)."""
    import cv2
    from skimage.measure import marching_cubes
    lo, hi = gt_mesh.bounds[0] - 3 * voxel_m, gt_mesh.bounds[1] + 3 * voxel_m
    axes = [np.arange(a, b + voxel_m, voxel_m) for a, b in zip(lo, hi)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    inside = np.ones(len(grid), bool)
    center = (lo + hi) / 2
    for C in np.asarray(cam_positions):
        f = center - C
        dist = np.linalg.norm(f)
        f /= dist
        r = np.cross(f, [0, 0, 1.0]) if abs(f[2]) < 0.99 else np.cross(f, [0, 1.0, 0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        scale = dist / px_m

        def proj(p):
            d = p - C
            return np.c_[d @ r, d @ u] / (d @ f)[:, None] * scale

        pv, pg = proj(gt_mesh.vertices), proj(grid[inside])
        off = np.minimum(pv.min(0), pg.min(0)) - 2
        size = (np.maximum(pv.max(0), pg.max(0)) - off + 3).astype(int)
        mask = np.zeros((size[1], size[0]), np.uint8)
        # One call per triangle: a single fillPoly over all faces uses the
        # even-odd rule, so overlapping front/back faces cancel (a closed
        # sphere filled ~60% of its disk).
        for tri in np.round((pv - off)[gt_mesh.faces] * 4).astype(np.int32):
            cv2.fillConvexPoly(mask, tri, 1, shift=2)
        q = np.round(pg - off).astype(int)
        idx = np.where(inside)[0]
        inside[idx[mask[q[:, 1], q[:, 0]] == 0]] = False
    vol = inside.reshape([len(a) for a in axes]).astype(np.float32)
    v, fcs, _, _ = marching_cubes(vol, 0.5, spacing=(voxel_m,) * 3)
    return trimesh.Trimesh(v + lo, fcs)


def split_completeness(gt_mesh: trimesh.Trimesh, rec: trimesh.Trimesh, ideal: trimesh.Trimesh,
                       n: int = 20000) -> dict:
    pts = trimesh.sample.sample_surface(gt_mesh, n, seed=0)[0]
    concave = trimesh.proximity.closest_point(ideal, pts)[1] * 1000 > CONCAVE_MM
    d = trimesh.proximity.closest_point(rec, pts)[1] * 1000
    out = {"concave_fraction": round(float(concave.mean()), 3)}
    for name, sel in (("convex", ~concave), ("concave", concave)):
        if sel.any():
            out[f"{name}_compl_median"] = round(float(np.median(d[sel])), 2)
            out[f"{name}_compl_p90"] = round(float(np.percentile(d[sel], 90)), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--skip-refine", action="store_true", help="reuse existing reconstruction/refined/*.json")
    args = ap.parse_args()
    cap = Path(args.capture_dir)
    scene = json.loads((cap / "reconstruction" / "scene.json").read_text())["objects"]
    gt = json.loads((cap / "ground_truth_meshes.json").read_text())
    listed = json.loads((cap / "random_objects.json").read_text())
    # both naming schemes: GT prim names equal to the manifest's, or Title-cased
    types = {k: o for o in listed for k in (o["name"], o["name"].title().replace("_", ""))}

    cams = list(json.loads((cap / "ground_truth_camera_positions.json").read_text())["positions"].values())
    ideal_cache = {}

    def ideal_for(name):
        if name not in ideal_cache:
            gm = trimesh.load(cap / gt[name]["mesh"], force="mesh")
            ih = ideal_visual_hull(gm, cams)
            a = trimesh.proximity.closest_point(gm, trimesh.sample.sample_surface(ih, 20000, seed=0)[0])[1] * 1000
            ideal_cache[name] = (ih, {"acc_median": round(float(np.median(a)), 2),
                                      "acc_p90": round(float(np.percentile(a, 90)), 2)}
                                 | split_completeness(gm, ih, ih))
        return ideal_cache[name]

    rows = []
    for obj in scene:
        rep_path = cap / "reconstruction" / "refined" / f"{obj['id']}.json"
        if not (args.skip_refine and rep_path.exists()):
            print(f"refining {obj['id']} ...", flush=True)
            subprocess.run([sys.executable, str(ROOT / "vision_pipeline/scripts/refine_freeform.py"),
                            "--capture-dir", str(cap), "--object", obj["id"], "--ground-truth"],
                           check=True, capture_output=True)
        rep = json.loads(rep_path.read_text())
        name = rep["score_refined"]["ground_truth"] or rep["score_hull"]["ground_truth"]
        if name is None:
            rows.append({"object": obj["id"], "ground_truth": None, "label": obj["label"]})
            continue
        gm = trimesh.load(cap / gt[name]["mesh"], force="mesh")
        hull = trimesh.load(cap / "reconstruction" / obj["mesh"], force="mesh")
        hull.apply_translation(obj["placement_m"])
        refined = trimesh.load(cap / "reconstruction" / "refined" / f"{obj['id']}.obj", force="mesh")
        ih, ideal_score = ideal_for(name)
        rows.append({"object": obj["id"], "ground_truth": name, "type": types.get(name, {}).get("type"),
                     "color": types.get(name, {}).get("color"), "label": obj["label"], "ideal_hull": ideal_score,
                     "hull": rep["score_hull"] | split_completeness(gm, hull, ih),
                     "refined": rep["score_refined"] | split_completeness(gm, refined, ih)})

    # One row per ground-truth object: its best-overlapping reconstruction;
    # extra reconstructions matched to the same object are fragments.
    best = {}
    for r in rows:
        if r["ground_truth"] and (r["ground_truth"] not in best or
                                  r["hull"]["box_iou"] > best[r["ground_truth"]]["hull"]["box_iou"]):
            best[r["ground_truth"]] = r
    fragments = [r["object"] for r in rows if r["ground_truth"] and best[r["ground_truth"]] is not r]
    phantoms = [r["object"] for r in rows if not r["ground_truth"]]
    missed = sorted(set(gt) - set(best))

    f = lambda v: "/".join(f"{x:+.1f}" for x in v)
    print(f"\n{'object':18s} {'color':9s} {'IoU':>5s} {'dims err hull (mm)':>20s} {'dims err refined':>20s} "
          f"{'acc med/p90':>12s} {'convex compl':>13s} {'concave compl':>14s} {'conc%':>5s} {'IDEAL acc':>11s}  CLIP label")
    for name in sorted(best):
        r = best[name]
        h, s, i = r["hull"], r["refined"], r["ideal_hull"]
        print(f"{(r['type'] or name):18s} {r['color'] or '':9s} {s['box_iou']:5.2f} {f(h['dim_err_mm']):>20s} "
              f"{f(s['dim_err_mm']):>20s} {s['surface_mm']['acc_median']:5.2f}/{s['surface_mm']['acc_p90']:<6.2f} "
              f"{s.get('convex_compl_median', float('nan')):5.2f}/{s.get('convex_compl_p90', float('nan')):<7.2f} "
              f"{s.get('concave_compl_median', float('nan')):6.2f}/{s.get('concave_compl_p90', float('nan')):<7.2f} "
              f"{100 * s['concave_fraction']:4.0f}% {i['acc_median']:5.2f}/{i['acc_p90']:<5.2f}  {r['label']}")
    print(f"\nmatched {len(best)}/{len(gt)}; missed: {missed or 'none'}; fragments: {fragments or 'none'}; "
          f"phantoms: {phantoms or 'none'}")
    (cap / "random_scene_report.json").write_text(json.dumps(
        {"per_object": [best[n] for n in sorted(best)], "missed": missed, "fragments": fragments,
         "phantoms": phantoms, "all_reconstructions": rows}, indent=2))


if __name__ == "__main__":
    main()

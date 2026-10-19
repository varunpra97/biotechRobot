"""Score the reconstruction against the Isaac scene's actual meshes.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/isaac_validation/compare_to_ground_truth.py \
      --capture-dir biotech-lab-robotics/vision_pipeline/isaac_validation/out

Ground-truth objects that physically touch (a holder with tubes standing
in its bores, a rack with tips hanging in it) are merged into one group
first: from silhouettes alone, objects in contact are one connected
volume, so scoring them separately would measure the method against a
question it can't answer. Each group is then matched one-to-one to the
reconstructed object with the highest 3D bounding-box IoU.

Per match: size error per axis, footprint-center error, box IoU, and
surface distance both ways --
  accuracy     reconstructed surface -> nearest true surface
  completeness true surface -> nearest reconstructed surface
Completeness is expected to be poor wherever the true object has
interior surfaces no silhouette sees (open bin, blind bores, rotor
wells); that's a visual-hull limitation, reported rather than hidden.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

TOUCH_GAP_M = 0.003
SAMPLES = 20000


def box_iou(amin, amax, bmin, bmax) -> float:
    inter = np.clip(np.minimum(amax, bmax) - np.maximum(amin, bmin), 0, None).prod()
    va, vb = (amax - amin).prod(), (bmax - bmin).prod()
    return float(inter / (va + vb - inter)) if inter > 0 else 0.0


def group_touching(gt: dict) -> list[dict]:
    names = list(gt)
    parent = {n: n for n in names}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            amin, amax = np.array(gt[a]["bbox_min"]), np.array(gt[a]["bbox_max"])
            bmin, bmax = np.array(gt[b]["bbox_min"]), np.array(gt[b]["bbox_max"])
            if np.all(np.minimum(amax, bmax) - np.maximum(amin, bmin) > -TOUCH_GAP_M):
                parent[find(a)] = find(b)

    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    out = []
    for members in groups.values():
        lo = np.min([gt[m]["bbox_min"] for m in members], axis=0)
        hi = np.max([gt[m]["bbox_max"] for m in members], axis=0)
        main = max(members, key=lambda m: np.prod(np.array(gt[m]["extents"]) + 1e-9))
        extra = len(members) - 1
        out.append({"name": main + (f" (+{extra} touching)" if extra else ""),
                    "members": members, "bbox_min": lo, "bbox_max": hi})
    return out


def surface_distances(a: trimesh.Trimesh, b: trimesh.Trimesh) -> np.ndarray:
    # Exact point-to-surface distance, not nearest sample (which adds the
    # sample spacing, ~1mm on large meshes, to every value).
    pa = trimesh.sample.sample_surface(a, SAMPLES, seed=0)[0]
    return trimesh.proximity.closest_point(b, pa)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--min-iou", type=float, default=0.1)
    args = ap.parse_args()
    cap = Path(args.capture_dir)

    gt = json.loads((cap / "ground_truth_meshes.json").read_text())
    groups = group_touching(gt)
    scene = json.loads((cap / "reconstruction" / "scene.json").read_text())
    recon = scene["objects"]

    pairs = []
    for gi, g in enumerate(groups):
        for ri, r in enumerate(recon):
            iou = box_iou(g["bbox_min"], g["bbox_max"], np.array(r["bbox_min_m"]), np.array(r["bbox_max_m"]))
            if iou >= args.min_iou:
                pairs.append((iou, gi, ri))
    pairs.sort(reverse=True)
    used_g, used_r, matches = set(), set(), []
    for iou, gi, ri in pairs:
        if gi in used_g or ri in used_r:
            continue
        used_g.add(gi)
        used_r.add(ri)
        matches.append((gi, ri, iou))

    rows = []
    for gi, ri, iou in sorted(matches, key=lambda m: groups[m[0]]["name"]):
        g, r = groups[gi], recon[ri]
        gt_mesh = trimesh.util.concatenate([trimesh.load(cap / gt[m]["mesh"], force="mesh") for m in g["members"]])
        rc_mesh = trimesh.load(cap / "reconstruction" / r["mesh"], force="mesh")
        rc_mesh.apply_translation(r["placement_m"])
        acc = surface_distances(rc_mesh, gt_mesh) * 1000
        comp = surface_distances(gt_mesh, rc_mesh) * 1000
        gdim = (g["bbox_max"] - g["bbox_min"]) * 1000
        rdim = (np.array(r["bbox_max_m"]) - np.array(r["bbox_min_m"])) * 1000
        gctr = (g["bbox_max"][:2] + g["bbox_min"][:2]) / 2
        rctr = (np.array(r["bbox_max_m"][:2]) + np.array(r["bbox_min_m"][:2])) / 2
        rows.append({
            "ground_truth": g["name"], "reconstructed": r["id"], "label": r["label"],
            "true_dims_mm": gdim.round(1).tolist(), "measured_dims_mm": rdim.round(1).tolist(),
            "dim_error_mm": (rdim - gdim).round(1).tolist(),
            "center_error_mm": round(float(np.linalg.norm(rctr - gctr) * 1000), 1),
            "box_iou": round(iou, 3),
            "accuracy_mm": {"median": round(float(np.median(acc)), 1), "p90": round(float(np.percentile(acc, 90)), 1)},
            "completeness_mm": {"median": round(float(np.median(comp)), 1), "p90": round(float(np.percentile(comp, 90)), 1)},
        })

    missed = [groups[i]["name"] for i in range(len(groups)) if i not in used_g]
    phantoms = [{"id": recon[i]["id"], "dims_mm": [round(v * 1000, 1) for v in recon[i]["dims_m"].values()],
                 "placement_m": [round(v, 3) for v in recon[i]["placement_m"]]}
                for i in range(len(recon)) if i not in used_r]

    out = cap / "comparison"
    out.mkdir(exist_ok=True)
    (out / "report.json").write_text(json.dumps(
        {"method": scene["method"], "matches": rows, "missed_ground_truth": missed, "unmatched_reconstructions": phantoms}, indent=2))

    print(f"{'ground truth':28s} {'true dims (mm)':>22s} {'measured (mm)':>22s} {'dim err (mm)':>20s} {'ctr':>6s} {'IoU':>5s} {'acc med/p90':>12s} {'compl med/p90':>14s}")
    for r in rows:
        f = lambda v: "x".join(f"{x:.0f}" for x in v)
        print(f"{r['ground_truth'][:28]:28s} {f(r['true_dims_mm']):>22s} {f(r['measured_dims_mm']):>22s} "
              f"{', '.join(f'{x:+.0f}' for x in r['dim_error_mm']):>20s} {r['center_error_mm']:6.1f} {r['box_iou']:5.2f} "
              f"{r['accuracy_mm']['median']:5.1f}/{r['accuracy_mm']['p90']:<5.1f} {r['completeness_mm']['median']:6.1f}/{r['completeness_mm']['p90']:<6.1f}")
    print(f"\nmatched {len(rows)}/{len(groups)} ground-truth objects; missed: {missed or 'none'}")
    print(f"{len(phantoms)} reconstructed objects with no ground-truth match (phantoms/fragments)")


if __name__ == "__main__":
    main()

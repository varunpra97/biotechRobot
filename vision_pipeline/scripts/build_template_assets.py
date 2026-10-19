"""Fitted templates -> simulation assets (+ optional scoring vs ground truth).

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/build_template_assets.py \
      --capture-dir CAPTURE --templates-subdir templates [--ground-truth]

Reads <capture>/<templates-subdir>/<object>.json (fit_templates.py output)
and, when present, <capture>/reasoning/<object>.json for material and
hidden-dimension hints. Writes <capture>/template_assets/: one USD + OBJ
per piece (a rack and each of its tubes are separate pieces), scene.usda,
assets.json. --ground-truth also scores every piece against the Isaac
scene's meshes (validation captures only).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh

from asset_export import DEFAULT_MATERIAL, cap_hull_vertices, safe_name, write_object_usd, write_scene_usd
from template_fitting import _rot_z
from template_meshes import box_mesh, open_box_mesh, plate_with_wells_mesh, rack_with_tubes_meshes

FRAME_NOTE = ("platform-local: origin at bench-top center, Z up, meters; "
              "world = (PLATFORM_CX + x, PLATFORM_CY + y, bench_top_z + z)")


def pieces_for(report: dict, hints: dict):
    """[(piece name, local mesh, placement xyz, yaw, collision mode, provenance)]"""
    body = report["body"]
    p = body["params_m"]
    fam = body["family"]
    base = np.array([p["cx"], p["cy"], 0.0])
    yaw = p.get("yaw", 0.0)
    if fam == "box":
        m, prov = box_mesh(p)
        return [("body", m, base, yaw, "convexHull", prov)]
    if fam == "open_box":
        m, prov = open_box_mesh(p, hints)
        return [("body", m, base, yaw, "none", prov)]
    if fam == "plate_with_well_grid":
        m, prov = plate_with_wells_mesh(p, body["rows"], body["cols"], hints)
        return [("body", m, base, yaw, "none", prov)]
    if fam == "rack_with_tube_grid":
        rack, tubes, prov = rack_with_tubes_meshes(p, body["rows"], body["cols"], hints)
        out = [("rack", rack, base, yaw, "none", prov)]
        R = _rot_z(yaw)
        for i, (tube, local) in enumerate(tubes):
            out.append((f"tube_{i:02d}", tube, base + R @ local, yaw, "convexHull", prov))
        return out
    raise ValueError(f"no mesh builder for family {fam}")


def placed(mesh: trimesh.Trimesh, xyz, yaw) -> trimesh.Trimesh:
    m = mesh.copy()
    T = np.eye(4)
    T[:3, :3] = _rot_z(yaw)
    T[:3, 3] = xyz
    m.apply_transform(T)
    return m


def score_against(gt: dict, cap: Path, mesh_world: trimesh.Trimesh) -> dict:
    """Best-overlapping ground-truth object by bounding box, then size,
    center and two-way surface distance."""
    lo, hi = mesh_world.bounds
    best, best_iou = None, 0.0
    for name, g in gt.items():
        glo, ghi = np.array(g["bbox_min"]), np.array(g["bbox_max"])
        inter = np.clip(np.minimum(hi, ghi) - np.maximum(lo, glo), 0, None).prod()
        iou = inter / ((hi - lo).prod() + (ghi - glo).prod() - inter + 1e-12)
        if iou > best_iou:
            best, best_iou = name, iou
    if best is None:
        return {"ground_truth": None}
    g = gt[best]
    gm = trimesh.load(cap / g["mesh"], force="mesh")
    # Exact point-to-SURFACE distance. Nearest-sample distance (an earlier
    # version) adds the sample spacing (~1mm on a large mesh) to every value.
    a = trimesh.sample.sample_surface(mesh_world, 20000, seed=0)[0]
    b = trimesh.sample.sample_surface(gm, 20000, seed=0)[0]
    acc = trimesh.proximity.closest_point(gm, a)[1] * 1000
    comp = trimesh.proximity.closest_point(mesh_world, b)[1] * 1000
    gdim = (np.array(g["bbox_max"]) - np.array(g["bbox_min"])) * 1000
    mdim = (hi - lo) * 1000
    return {"ground_truth": best, "box_iou": round(float(best_iou), 3),
            "true_dims_mm": gdim.round(2).tolist(), "dims_mm": mdim.round(2).tolist(),
            "dim_err_mm": (mdim - gdim).round(2).tolist(),
            "center_err_mm": round(float(np.linalg.norm(((hi + lo) / 2 - (np.array(g["bbox_max"]) + np.array(g["bbox_min"])) / 2)[:2]) * 1000), 2),
            "surface_mm": {"acc_median": round(float(np.median(acc)), 2), "acc_p90": round(float(np.percentile(acc, 90)), 2),
                           "compl_median": round(float(np.median(comp)), 2), "compl_p90": round(float(np.percentile(comp, 90)), 2)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--templates-subdir", default="templates")
    ap.add_argument("--ground-truth", action="store_true")
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    out = cap / "template_assets"
    out.mkdir(exist_ok=True)
    gt = json.loads((cap / "ground_truth_meshes.json").read_text()) if args.ground_truth else None

    summary, placements = [], []
    for f in sorted((cap / args.templates_subdir).glob("*.json")):
        report = json.loads(f.read_text())
        oid = report["object"]
        analysis, hints = None, {}
        rpath = cap / "reasoning" / f"{oid}.json"
        if rpath.exists():
            r = json.loads(rpath.read_text())
            analysis = r.get("analysis") if r.get("status") == "ok" else None
        material = dict(analysis["material"]) if analysis else dict(DEFAULT_MATERIAL)
        mat_src = "llm_scene_reasoning" if analysis else "default_unverified"
        label = safe_name(analysis["identity"]["name"]) if analysis else oid

        for piece, mesh, xyz, yaw, collision_mode, prov in pieces_for(report, hints):
            name = f"{label}_{piece}" if piece != "body" else label
            hulls = [mesh] if collision_mode == "none" else [cap_hull_vertices(mesh.convex_hull)]
            mass = float(mesh.volume * material["density_kg_m3"])
            meta = {"source_object": oid, "piece": piece, "family": report["body"]["family"],
                    "fit_rms_px": round(report["body"]["rms_px"], 3), "provenance": prov,
                    "material_source": mat_src, "geometry_method": "template fit to image edges"}
            write_object_usd(out / f"{name}.usda", name, mesh, hulls, mass, mesh.center_mass, material, meta,
                             collision_approximation=collision_mode)
            mesh.export(out / f"{name}.obj")
            trimesh.util.concatenate(hulls).export(out / f"{name}_collision.obj")
            placements.append((name, f"./{name}.usda", xyz, yaw))
            entry = {"name": name, **meta, "mass_kg": round(mass, 5), "material": material,
                     "placement_m": [float(v) for v in xyz], "yaw_deg": round(float(np.degrees(yaw)), 3),
                     "local_extents_mm": (mesh.extents * 1000).round(2).tolist()}
            if gt is not None:
                entry["score"] = score_against(gt, cap, placed(mesh, xyz, yaw))
            summary.append(entry)

    write_scene_usd(out / "scene.usda", placements, FRAME_NOTE)
    (out / "assets.json").write_text(json.dumps({"frame": FRAME_NOTE, "objects": summary}, indent=2))
    print(f"wrote {len(summary)} pieces to {out}")
    if gt is not None:
        print(f"\n{'piece':22s} {'ground truth':14s} {'true dims':>22s} {'dim err (mm)':>22s} {'ctr':>6s} {'surf acc med/p90':>17s}")
        for e in summary:
            s = e["score"]
            if not s.get("ground_truth"):
                print(f"{e['name']:22s} (no ground-truth match)")
                continue
            f2 = lambda v: "x".join(f"{x:.1f}" for x in v)
            print(f"{e['name']:22s} {s['ground_truth']:14s} {f2(s['true_dims_mm']):>22s} "
                  f"{', '.join(f'{x:+.2f}' for x in s['dim_err_mm']):>22s} {s['center_err_mm']:6.2f} "
                  f"{s['surface_mm']['acc_median']:6.2f}/{s['surface_mm']['acc_p90']:<6.2f}")


if __name__ == "__main__":
    main()

"""Stage 6 CLI: reconstructed objects -> simulation assets.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/export_assets.py --capture-dir CAPTURE

Reads <capture>/reconstruction/scene.json (+ objects/*.obj) and, when
present, <capture>/reasoning/<id>.json. Writes <capture>/assets/:
  <name>.usda                  rigid body, mass, collision, physics material
  <name>.obj, <name>_collision.obj   the same geometry for lab_scene.py's
                               load_mesh_prim(..., collision_obj=...)
  scene.usda                   every object placed in the platform frame
  assets.json                  per object: dims, mass, material, and where
                               each value came from (LLM vs. default)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh

from asset_export import (DEFAULT_MATERIAL, convex_decompose, mass_properties, safe_name,
                          write_object_usd, write_scene_usd)

FRAME_NOTE = ("platform-local: origin at bench-top center, Z up, meters; "
              "world = (PLATFORM_CX + x, PLATFORM_CY + y, bench_top_z + z)")


def export_parts(rec: Path, scene: dict, out: Path) -> None:
    """One asset per part (decompose_objects.py): a rack and each of its
    tubes are separate rigid bodies, each with its own material and mass;
    removable parts are flagged so the sim can treat them as loose."""
    parts = json.loads((rec / "parts.json").read_text())["parts"]
    summary, placements, used = [], [], set()
    for p in parts:
        base = safe_name(f"{p['object']}_{p['part']}")
        name, k = base, 2
        while name in used:
            name, k = f"{base}_{k}", k + 1
        used.add(name)
        visual = trimesh.load(rec / p["mesh"], force="mesh")
        lo, hi = np.array(p["bbox_min_m"]), np.array(p["bbox_max_m"])
        placement = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        visual.apply_translation(-placement)
        hulls = convex_decompose(visual)
        com = visual.center_mass if visual.is_watertight else visual.vertices.mean(axis=0)
        material = dict(DEFAULT_MATERIAL, **p["material"])
        meta = {"source_object": p["object"], "part": p["part"], "kind": p["kind"], "relation": p["relation"],
                "parent": p["parent"], "removable": p["removable"],
                "dims_mm": dict(zip("xyz", ((hi - lo) * 1000).round(1).tolist())),
                "material_source": p["description_source"], "material_evidence": p.get("material_evidence", ""),
                "solid_fraction": p["solid_fraction"], "filled_envelope_m3": p["filled_envelope_m3"],
                "hidden_geometry_estimated": p["hidden_voxels_estimated"] > 0,
                "geometry_method": scene["method"] + "; decomposed (decompose.py)"}
        write_object_usd(out / f"{name}.usda", name, visual, hulls, p["mass_kg"], np.asarray(com, float), material, meta)
        visual.export(out / f"{name}.obj")
        trimesh.util.concatenate(hulls).export(out / f"{name}_collision.obj")
        placements.append((name, f"./{name}.usda", placement))
        summary.append({**meta, "name": name, "usd": f"{name}.usda", "visual_obj": f"{name}.obj",
                        "collision_obj": f"{name}_collision.obj", "collision_pieces": len(hulls),
                        "mass_kg": round(p["mass_kg"], 5), "material": material, "placement_m": placement.tolist()})
        print(f"{name:34s} {p['relation']:15s} {len(hulls):2d} convex pieces  {p['mass_kg'] * 1000:8.1f} g  "
              f"{material['name']}")
    write_scene_usd(out / "scene.usda", placements, FRAME_NOTE)
    (out / "assets.json").write_text(json.dumps({"frame": FRAME_NOTE, "objects": summary}, indent=2))
    print(f"\nwrote {len(summary)} part assets + scene.usda to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--recon-subdir", default="reconstruction",
                    help="reconstruction to use: reconstruction (visual hull) or reconstruction_depth")
    ap.add_argument("--whole-objects", action="store_true",
                    help="ignore parts.json (decompose_objects.py) and export each reconstructed object whole")
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    rec = cap / args.recon_subdir
    scene = json.loads((rec / "scene.json").read_text())
    out = cap / ("assets" if args.recon_subdir == "reconstruction" else "assets_" + args.recon_subdir.replace("reconstruction_", ""))
    out.mkdir(exist_ok=True)

    summary, placements, used = [], [], set()
    if (rec / "parts.json").exists() and not args.whole_objects:
        export_parts(rec, scene, out)
        return
    for obj in scene["objects"]:
        reasoning_path = ((cap / "reasoning") if args.recon_subdir == "reconstruction"
                          else rec / "reasoning") / f"{obj['id']}.json"
        analysis = None
        if reasoning_path.exists():
            r = json.loads(reasoning_path.read_text())
            analysis = r.get("analysis") if r.get("status") == "ok" else None

        base = safe_name(analysis["identity"]["name"]) if analysis else obj["id"]
        name = base
        k = 2
        while name in used:
            name, k = f"{base}_{k}", k + 1
        used.add(name)

        material = dict(analysis["material"]) if analysis else dict(DEFAULT_MATERIAL)
        solid_fraction = float(analysis["solid_fraction"]) if analysis else 1.0
        provenance = "llm_scene_reasoning" if analysis else "default_unverified"

        refined = rec / "refined" / f"{obj['id']}.obj"
        if refined.exists():       # silhouette-refined, platform frame -> the object's local frame
            visual = trimesh.load(refined, force="mesh")
            visual.apply_translation(-np.array(obj["placement_m"]))
        else:
            visual = trimesh.load(rec / obj["mesh"], force="mesh")
        hulls = convex_decompose(visual)
        mass, com = mass_properties(visual, material["density_kg_m3"], solid_fraction, obj.get("volume_m3"))

        meta = {"source_object": obj["id"], "label": analysis["identity"]["name"] if analysis else obj["label"],
                "dims_mm": {k2: round(v * 1000, 1) for k2, v in obj["dims_m"].items()},
                "material_source": provenance, "solid_fraction": solid_fraction,
                "geometry_method": scene["method"] + ("; silhouette-refined" if refined.exists() else "")}
        write_object_usd(out / f"{name}.usda", name, visual, hulls, mass, com, material, meta)
        visual.export(out / f"{name}.obj")
        trimesh.util.concatenate(hulls).export(out / f"{name}_collision.obj")
        placements.append((name, f"./{name}.usda", np.array(obj["placement_m"])))

        summary.append({**meta, "name": name, "usd": f"{name}.usda", "visual_obj": f"{name}.obj",
                        "collision_obj": f"{name}_collision.obj", "collision_pieces": len(hulls),
                        "max_hull_vertices": max(len(h.vertices) for h in hulls),
                        "mass_kg": round(mass, 4), "material": material,
                        "placement_m": obj["placement_m"]})
        print(f"{name:28s} {len(hulls):2d} convex pieces  mass {mass*1000:8.1f} g  "
              f"material {material['name']} ({provenance})")

    write_scene_usd(out / "scene.usda", placements, FRAME_NOTE)
    (out / "assets.json").write_text(json.dumps({"frame": FRAME_NOTE, "objects": summary}, indent=2))
    print(f"\nwrote {len(summary)} assets + scene.usda to {out}")


if __name__ == "__main__":
    main()

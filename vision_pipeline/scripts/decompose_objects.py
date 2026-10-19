"""Split every reconstructed object into its parts and weigh each part.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/decompose_objects.py \
      --capture-dir CAPTURE [--recon-subdir reconstruction_depth] [--hints FILE]

Per object, the parts description comes from LLM scene reasoning
(<recon>/reasoning/<id>.json, reason_about_objects.py) or, for testing,
--hints: a JSON {object id: {"decomposition": {"parts": [...]}}} in the
same schema (a stand-in must say so in its "source" field). Objects with
neither are kept whole with the default material.

Geometry: decompose.py on the object's fine TSDF. Mass per part:

    density x filled-envelope volume x solid_fraction

The filled envelope closes every cavity slice by slice (a tube's bore, a
tray's interior, a box sealed by its lid), whether or not a camera saw
into it, so the description's solid_fraction always applies to the same,
computable volume. Material and solid fraction are the description's.

Writes <recon>/parts/<object>__<part>.obj (platform frame) and
<recon>/parts.json.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh
from scipy import ndimage

from asset_export import DEFAULT_MATERIAL
from decompose import decompose, load_field, part_mesh


def filled_envelope_m3(mask: np.ndarray, voxel: float, z0: float) -> float:
    """Voxel volume with every slice's holes filled, plus the slab between
    the bench and the grid's first layer under the part's lowest slice."""
    filled = np.zeros_like(mask)
    for k in range(mask.shape[2]):
        if mask[:, :, k].any():
            filled[:, :, k] = ndimage.binary_fill_holes(mask[:, :, k])
    v = float(filled.sum()) * voxel ** 3
    ks = np.nonzero(filled.any(axis=(0, 1)))[0]
    if len(ks) and ks.min() == 0:
        v += float(filled[:, :, 0].sum()) * voxel ** 2 * max(z0 - voxel / 2, 0.0)
    return v


class ColourSampler:
    """Mean colour of a part's camera-visible surface, from the photos.
    Visibility is checked against the fused depth (depth_used/), so points
    hidden behind another part don't vote."""

    def __init__(self, cap: Path, rec: Path, downscale: int = 4, max_views: int = 12):
        import cv2
        from capture import resolve_shot_image
        from depth_sensor import load_depth
        from geometry import camera_pose_to_rvec_tvec, quaternion_to_rotation_matrix
        self.views = []
        poses_file = rec / "poses" / "camera_poses.json"
        if not poses_file.exists():
            poses_file = cap / "poses" / "camera_poses.json"
        dk = json.loads((rec / "depth_used" / "intrinsics.json").read_text()) if (rec / "depth_used").exists() else {}
        cams = json.loads((cap / "camera_intrinsics.json").read_text())["cameras"]
        for p in json.loads(poses_file.read_text())["poses"][::max(1, 25 // max_views)]:
            cid = p["camera_id"]
            img_path = resolve_shot_image(cap, p["image"])
            if cid not in dk or not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, None, fx=1 / downscale, fy=1 / downscale, interpolation=cv2.INTER_AREA)
            K = np.array(cams[cid]["K"], float)
            K[:2] /= downscale
            rvec, tvec = camera_pose_to_rvec_tvec(quaternion_to_rotation_matrix(np.array(p["quaternion_wxyz"])),
                                                  np.array(p["position_world"]))
            self.views.append((img[..., ::-1], K, rvec, tvec, load_depth(rec / "depth_used" / f"{cid}.png"),
                               np.array(dk[cid]["K"], float)))

    def __call__(self, mesh: trimesh.Trimesh, n: int = 400):
        import cv2
        pts = trimesh.sample.sample_surface(mesh, n, seed=0)[0]
        cols = []
        for img, K, rvec, tvec, depth, Kd in self.views:
            R, _ = cv2.Rodrigues(rvec)
            cam = pts @ R.T + np.asarray(tvec).reshape(1, 3)
            z = cam[:, 2]
            ud = np.round(Kd[0, 0] * cam[:, 0] / z + Kd[0, 2]).astype(int)
            vd = np.round(Kd[1, 1] * cam[:, 1] / z + Kd[1, 2]).astype(int)
            ok = (z > 0) & (ud >= 0) & (ud < depth.shape[1]) & (vd >= 0) & (vd < depth.shape[0])
            d = np.zeros(len(pts))
            d[ok] = depth[vd[ok], ud[ok]]
            ok &= (d > 0) & (np.abs(d - z) < 0.003)
            u = np.round(K[0, 0] * cam[:, 0] / z + K[0, 2]).astype(int)
            v = np.round(K[1, 1] * cam[:, 1] / z + K[1, 2]).astype(int)
            ok &= (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
            if ok.any():
                cols.append(img[v[ok], u[ok]].astype(float))
        return np.median(np.concatenate(cols), axis=0) if cols else None


def assign_kinds(pieces, colours, kinds_desc):
    """Assign split pieces to part kinds of one group: nearest colour, each
    kind taking at most its count (overflow goes to the nearest kind)."""
    from scipy.optimize import linear_sum_assignment

    def hex_rgb(h):
        h = (h or "").lstrip("#")
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) if len(h) == 6 else None
    targets = [hex_rgb(k.get("color_hex")) for k in kinds_desc]
    slots = [i for i, k in enumerate(kinds_desc) for _ in range(max(1, int(k.get("count", 1))))]
    while len(slots) < len(pieces):
        slots += list(range(len(kinds_desc)))
    if any(c is None for c in colours) or any(t is None for t in targets):
        return [kinds_desc[slots[i]]["name"] for i in range(len(pieces))]       # no colours: in order
    # brightness-normalized RGB: shading changes intensity more than hue
    def norm(c):
        return c / max(np.linalg.norm(c), 1e-6)
    cost = np.array([[np.linalg.norm(norm(colours[i]) - norm(targets[s])) + 0.002 * abs(
        np.linalg.norm(colours[i]) - np.linalg.norm(targets[s])) for s in slots] for i in range(len(pieces))])
    ri, ci = linear_sum_assignment(cost)
    out = [None] * len(pieces)
    for i, j in zip(ri, ci):
        out[i] = kinds_desc[slots[j]]["name"]
    return out


def describe(analysis: dict | None) -> list[dict]:
    """Normalized parts list: exactly one base, parents defaulting to it."""
    if not analysis:
        return [{"name": "object", "relation": "base", "parent": "", "count": 1,
                 "material": dict(DEFAULT_MATERIAL, confidence=0.0, evidence="default"), "solid_fraction": 1.0}]
    parts = [dict(p) for p in analysis["decomposition"]["parts"]]
    bases = [p for p in parts if p["relation"] == "base"]
    if not bases:
        parts.insert(0, {"name": analysis["identity"]["name"], "relation": "base", "parent": "", "count": 1,
                         "material": analysis["material"], "solid_fraction": analysis["solid_fraction"]})
        bases = parts[:1]
    base = bases[0]
    for p in parts:
        if p is not base and p["relation"] == "base":
            p["relation"] = "resting_on_top"            # a second "base" can only sit on the first
        p["name"] = re.sub(r"[^a-z0-9_]+", "_", p["name"].lower()).strip("_") or "part"
    names = [p["name"] for p in parts]
    for p in parts:
        if p is not base and p.get("parent", "") not in names:
            p["parent"] = base["name"]
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--recon-subdir", default="reconstruction_depth")
    ap.add_argument("--hints", help="stand-in parts descriptions (testing without the LLM)")
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    rec = cap / args.recon_subdir
    scene = json.loads((rec / "scene.json").read_text())
    hints = json.loads(Path(args.hints).read_text()) if args.hints else {}
    out_dir = rec / "parts"
    out_dir.mkdir(exist_ok=True)
    for stale in out_dir.glob("*.obj"):
        stale.unlink()

    records = []
    sampler = None
    for o in scene["objects"]:
        oid = o["id"]
        analysis, source = None, "none (kept whole, default material)"
        rpath = rec / "reasoning" / f"{oid}.json"
        if rpath.exists():
            r = json.loads(rpath.read_text())
            if r.get("status") == "ok":
                analysis, source = r["analysis"], f"llm ({r.get('model', '?')})"
        if analysis is None and oid in hints:
            analysis, source = hints[oid], hints[oid].get("source", "hints file")
        desc = describe(analysis)
        by_name = {p["name"]: p for p in desc}
        field_file = rec / "objects" / f"{oid}_field.npz"
        if not field_file.exists():
            print(f"{oid}: no fine field saved -- skipped")
            continue
        grid = load_field(field_file)
        occ = grid.field < 0
        parts, log = decompose(grid, desc)
        print(f"{oid} ({source}): {len(parts)} part(s)")
        for line in log:
            print(f"   {line}")
        meshes = [part_mesh(grid, part.mask, occ, part.forced_inside) for part in parts]
        # pieces whose group has several kinds (a brass and a steel weight): match by colour
        groups = {}
        for i, part in enumerate(parts):
            if len(part.kinds) > 1:
                groups.setdefault(tuple(part.kinds), []).append(i)
        kind_of = {}
        for kinds, idx in groups.items():
            if sampler is None:
                sampler = ColourSampler(cap, rec)
            cols = [sampler(meshes[i]) if meshes[i] is not None else None for i in idx]
            names = assign_kinds(idx, cols, [by_name[k] for k in kinds])
            counter = {}
            for i, nm, c in zip(idx, names, cols):
                kind_of[i] = nm
                counter[nm] = counter.get(nm, -1) + 1
                parts[i].name = f"{nm}_{counter[nm]:02d}"
                parts[i].notes.append(f"colour {np.round(c).astype(int).tolist() if c is not None else '?'} -> {nm}")
        for i, (part, mesh) in enumerate(zip(parts, meshes)):
            if mesh is None:
                continue
            kind = (by_name.get(kind_of.get(i, "")) or by_name.get(part.name)
                    or by_name.get(re.sub(r"_\d+$", "", part.name)) or desc[0])
            mat = kind.get("material") or DEFAULT_MATERIAL
            sf = float(kind.get("solid_fraction", 1.0))
            env = filled_envelope_m3(part.mask, grid.voxel, grid.origin[2])
            mass = float(mat["density_kg_m3"]) * env * sf
            fname = f"{oid}__{part.name}.obj"
            mesh.export(out_dir / fname)
            lo, hi = mesh.bounds
            records.append({
                "object": oid, "part": part.name, "kind": kind["name"], "relation": part.relation,
                "parent": part.parent, "removable": part.relation != "base",
                "material": {k: mat[k] for k in ("name", "density_kg_m3", "static_friction", "dynamic_friction",
                                                 "restitution") if k in mat},
                "material_evidence": mat.get("evidence", ""), "solid_fraction": sf,
                "filled_envelope_m3": env, "mesh_volume_m3": float(mesh.volume) if mesh.is_volume else None,
                "hidden_voxels_estimated": int(part.forced_inside.sum()) if part.forced_inside is not None else 0,
                "mass_kg": mass, "bbox_min_m": lo.tolist(), "bbox_max_m": hi.tolist(),
                "mesh": f"parts/{fname}", "description_source": source, "notes": part.notes})
            e = (hi - lo) * 1000
            print(f"   {part.name:14s} {part.relation:15s} {e[0]:6.1f} x {e[1]:6.1f} x {e[2]:6.1f} mm  "
                  f"{mat['name']:18s} sf {sf:.2f}  {mass * 1000:8.1f} g")
    (rec / "parts.json").write_text(json.dumps({"frame": scene.get("frame"), "parts": records}, indent=2))
    print(f"{len(records)} parts -> {rec / 'parts.json'}")


if __name__ == "__main__":
    main()

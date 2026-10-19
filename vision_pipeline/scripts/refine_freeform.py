"""Freeform silhouette refinement of one reconstructed object.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/refine_freeform.py \
      --capture-dir CAPTURE --object obj_03 [--ground-truth]

Starts from <capture>/reconstruction/objects/<object>.obj (visual hull),
writes <capture>/reconstruction/refined/<object>.obj (platform frame) and
<object>.json (per-iteration stats; with --ground-truth, before/after scores).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import trimesh

from build_template_assets import score_against
from calibration import load_intrinsics
from capture import resolve_shot_image
from freeform_refine import prepare_mesh, refine
from scipy import ndimage


def rim_height(mesh, views, max_elev_deg: float = 25.0):
    """Height of an open container's rim, read off its WALLS: the highest
    outline vertex whose surface faces sideways (|normal z| < 0.3), median
    over the low views. The plain topmost outline point is ambiguous -- a
    phantom dome over the mouth can project onto the same image line as the
    far rim (measured 128.9mm for a 121.9mm rim) -- but the dome's surface
    faces up, the walls' doesn't."""
    from freeform_refine import _silhouette
    zs = []
    for v in views:
        R, _ = cv2.Rodrigues(v.rvec)
        C = -R.T @ v.tvec.reshape(3)
        centre = mesh.bounds.mean(axis=0)
        elev = np.degrees(np.arctan2(C[2] - centre[2], np.linalg.norm(C[:2] - centre[:2])))
        if elev > max_elev_deg:
            continue
        idx, p2, _ = _silhouette(mesh, v, 1.0)
        walls = idx[np.abs(mesh.vertex_normals[idx, 2]) < 0.3]
        if len(walls):
            zs.append(mesh.vertices[walls, 2].max())
    return float(np.median(zs)) if zs else None


def hull_inside_fn(cap: Path, dilate_voxels: int = 1):
    """Exact 'inside the visual hull' test from the saved occupancy grid
    (dilated by a voxel for segmentation error). Carving starts a couple of
    voxels above the bench; below that, a column counts as inside if its
    lowest carved voxel is occupied."""
    d = np.load(cap / "reconstruction" / "occupancy.npz")
    occ = ndimage.binary_dilation(d["occupancy"], iterations=dilate_voxels)
    origin, vox = d["origin"], float(d["voxel_m"])

    def inside(pts):
        ijk = np.floor((pts - origin) / vox + 0.5).astype(int)
        ijk[:, 2] = np.maximum(ijk[:, 2], 0)
        ok = np.all((ijk >= 0) & (ijk < occ.shape), axis=1)
        res = np.zeros(len(pts), bool)
        res[ok] = occ[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        return res
    return inside


def field_inside_fn(field_file: Path, max_out_m: float):
    """Depth-fused start: 'inside' = within max_out_m outside the fused
    surface (trilinear in the object's saved fine TSDF). Vertices may move
    outward only as far as the depth measurement plausibly allows."""
    from scipy.ndimage import map_coordinates
    d = np.load(field_file)
    F = d["field_q"].astype(np.float32) / 127.0
    origin, vox, trunc = d["origin"], float(d["voxel_m"]), float(d["trunc_m"])
    thr = min(max_out_m / trunc, 0.99)

    def inside(pts):
        ijk = ((pts - origin) / vox).T
        f = map_coordinates(F, ijk, order=1, mode="nearest")
        below = pts[:, 2] < origin[2]                       # under the grid: the bench-cut extension
        return (f < thr) | below
    return inside


def depth_boundary_map(rec: Path, view_name: str, roi, full_shape, min_jump_m: float = 0.008):
    """Distance (ROI px, full resolution) to the depth map's discontinuities
    -- where an object's outline really is, with no segmentation needed.
    Pixels on either side of a jump > min_jump_m are edge pixels, mapped to
    full resolution and dilated by the resolution ratio (a 640px depth map
    pins an edge to ~6px of a 4032px photo). Returns (map, ratio) or None."""
    from depth_sensor import load_depth
    f = rec / "depth_used" / f"{view_name}.png"
    if not f.exists():
        return None
    d = load_depth(f)
    valid = d > 0
    h, w = d.shape
    p = np.pad(d, 1)
    pv = np.pad(valid, 1)
    edge = np.zeros_like(valid)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                nb, nbv = p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w], pv[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                edge |= valid & nbv & (np.abs(d - nb) > min_jump_m)
    H, W = full_shape
    ratio = W / d.shape[1]
    ys, xs = np.nonzero(edge)
    u = np.round((xs + 0.5) * ratio - 0.5).astype(int)
    v = np.round((ys + 0.5) * H / d.shape[0] - 0.5).astype(int)
    x0, y0, x1, y1 = roi
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    k = (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
    m[v[k] - y0, u[k] - x0] = 1
    if not m.any():
        return None
    r = max(1, int(np.ceil(ratio / 2)))
    m = cv2.dilate(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    return cv2.distanceTransform((m == 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32), ratio


def boundary_map(cap: Path, view_name: str, roi, full_shape):
    """Distance (ROI px) to the segmentation foreground boundary, at full
    resolution (masks are stored downscaled)."""
    f = cap / "segmentation" / f"{view_name}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    fg = cv2.resize(d["foreground"].astype(np.uint8), (full_shape[1], full_shape[0]), interpolation=cv2.INTER_NEAREST)
    x0, y0, x1, y1 = roi
    crop = fg[y0:y1, x0:x1]
    edge = cv2.morphologyEx(crop, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return cv2.distanceTransform((edge == 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
from geometry import camera_pose_to_rvec_tvec, quaternion_to_rotation_matrix
from template_fitting import build_edge_view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--object", required=True)
    ap.add_argument("--poses-subdir", default="poses")
    ap.add_argument("--min-markers", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=16)
    ap.add_argument("--ground-truth", action="store_true")
    ap.add_argument("--recon-subdir", default="reconstruction",
                    help="reconstruction (visual hull start) or reconstruction_depth (depth-fused start)")
    ap.add_argument("--anchor-weight", type=float, default=None,
                    help="pull non-outline vertices to zero displacement (default: 1 for a depth start, 0 for a hull)")
    ap.add_argument("--combine", choices=["mean", "lsq"], default=None,
                    help="per-vertex combination of the views' moves (default: lsq for a depth start, mean for a hull)")
    ap.add_argument("--no-depth-gate", action="store_true",
                    help="don't gate image edges on the depth map's discontinuities when there's no segmentation")
    ap.add_argument("--smooth-iters", type=int, default=3,
                    help="Taubin smoothing of the start mesh (volume-preserving); more for noisy depth")
    ap.add_argument("--max-out-mm", type=float, default=None,
                    help="depth start: how far outside the fused surface a vertex may move (default: 3 sigma of "
                         "the fused surface, at least 0.5mm; a fixed 3mm let corner spikes run to the bound)")
    ap.add_argument("--open-top", action="store_true",
                    help="object is an open-topped container (an LLM shape hint): cut it at the rim plane "
                         "measured in the low views. A visual hull can't remove the phantom cap over an open "
                         "mouth -- it is consistent with every silhouette (the centrifuge's was +8mm)")
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    rec = cap / args.recon_subdir
    obj = next(o for o in json.loads((rec / "scene.json").read_text())["objects"] if o["id"] == args.object)
    hull = trimesh.load(rec / obj["mesh"], force="mesh")
    hull.apply_translation(obj["placement_m"])          # local -> platform frame

    lo, hi = np.array(obj["bbox_min_m"]) - 0.02, np.array(obj["bbox_max_m"]) + 0.02
    lo[2] = 0.0
    roi = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    intrinsics = load_intrinsics(cap / "camera_intrinsics.json")
    views, bmaps = [], {}
    depth_gate = (rec / "depth_used").exists() and not args.no_depth_gate
    gate_px = 8.0
    # a depth reconstruction's own poses (refined by depth ICP) when it has them
    poses_file = (rec / "poses" / "camera_poses.json") if (rec / "poses" / "camera_poses.json").exists()         and args.poses_subdir == "poses" else cap / args.poses_subdir / "camera_poses.json"
    for p in json.loads(poses_file.read_text())["poses"]:
        if len(p["marker_ids_used"]) < args.min_markers:
            continue
        cam = intrinsics[p["camera_id"]]
        rvec, tvec = camera_pose_to_rvec_tvec(quaternion_to_rotation_matrix(np.array(p["quaternion_wxyz"])),
                                              np.array(p["position_world"]))
        gray = cv2.imread(str(resolve_shot_image(cap, p["image"])), cv2.IMREAD_GRAYSCALE)
        v = build_edge_view(Path(p["image"]).stem, gray, cam.K, cam.dist, rvec, tvec, roi)
        if v is not None:
            views.append(v)
            bm = boundary_map(cap, v.name, v.roi, gray.shape)
            if bm is None and depth_gate:
                got = depth_boundary_map(rec, v.name, v.roi, gray.shape)
                if got is not None:
                    bm, ratio = got
                    gate_px = max(gate_px, 1.5 * ratio)
            if bm is not None:
                bmaps[v.name] = bm

    start = prepare_mesh(hull, smooth_iters=args.smooth_iters)
    field_file = rec / "objects" / f"{args.object}_field.npz"
    depth_start = field_file.exists()
    if depth_start:
        sig = json.loads((rec / "scene.json").read_text()).get("fused_sigma_m", 0.001)
        max_out = args.max_out_mm / 1000 if args.max_out_mm is not None else max(0.0005, 3 * sig)
        inside = field_inside_fn(field_file, max_out)
        print(f"depth start: vertices may move up to {max_out * 1000:.2f}mm outside the fused surface")
    else:
        inside = hull_inside_fn(cap) if (rec / "occupancy.npz").exists() else None
    anchor = args.anchor_weight if args.anchor_weight is not None else (1.0 if depth_start else 0.0)
    print(f"{len(views)} views, {len(bmaps)} with boundary gating ({'segmentation or depth edges' if depth_gate else 'segmentation'}, "
          f"{gate_px:.1f}px), bound: {inside is not None}")
    refined, stats = refine(start, views, iterations=args.iterations, inside_fn=inside,
                            boundary_maps=bmaps or None, boundary_gate_px=gate_px, anchor_weight=anchor,
                            combine=args.combine or ("lsq" if depth_start else "mean"))
    for s in stats:
        print(f"  iter {s.iteration:2d} radius {s.radius_px:4.1f}px  outline vertices {s.observed_vertices:6d}  "
              f"mean |offset| {s.mean_abs_offset_px:.2f}px")

    if args.open_top:
        z_rim = rim_height(refined, views)
        if z_rim is not None:
            refined = refined.slice_plane([0, 0, z_rim], [0, 0, -1], cap=True)
            print(f"open top: rim plane measured at z = {z_rim * 1000:.2f} mm in low views; cut above it")

    out = rec / "refined"
    out.mkdir(parents=True, exist_ok=True)
    refined.export(out / f"{args.object}.obj")
    report = {"object": args.object, "views": len(views), "faces": len(refined.faces),
              "iterations": [s.__dict__ for s in stats]}
    report["smooth_iters"] = args.smooth_iters
    if args.ground_truth:
        gt = json.loads((cap / "ground_truth_meshes.json").read_text())
        report["score_hull"] = score_against(gt, cap, hull)
        report["score_prepared"] = score_against(gt, cap, start)
        report["score_refined"] = score_against(gt, cap, refined)
        for label in ("score_hull", "score_prepared", "score_refined"):
            s = report[label]
            print(f"{label:15s} vs {s['ground_truth']}: dims err {s['dim_err_mm']} mm, "
                  f"surface acc med/p90 {s['surface_mm']['acc_median']}/{s['surface_mm']['acc_p90']} mm, "
                  f"completeness med/p90 {s['surface_mm']['compl_median']}/{s['surface_mm']['compl_p90']} mm")
    (out / f"{args.object}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

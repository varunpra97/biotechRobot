"""Stage 4 CLI: fit a shape template to one reconstructed object.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/fit_templates.py \
      --capture-dir CAPTURE --object obj_06 --family box

Initializes from the visual-hull object (scene.json), builds edge maps from
the full-resolution images over that object's region in every view whose
pose rests on --min-markers markers, fits, and writes
<capture>/templates/<object>.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from calibration import load_intrinsics
from capture import resolve_shot_image
from geometry import camera_pose_to_rvec_tvec, quaternion_to_rotation_matrix
from template_fitting import (EDGE_MODELS, TEMPLATES, _param_bounds, _rot_z, box_height_edges,
                              build_edge_view, edge_score, fit_template, make_plate_with_well_grid, make_rack_with_tube_grid,
                              scan_parameter)


def scan_box_side(p: np.ndarray, axis: int, sign: int, offsets, views) -> np.ndarray:
    """Move one side of the box (axis 0 = width / x, 1 = depth / y in the
    box frame; sign -1 / +1 = which side) by each offset, keeping the
    opposite side fixed, and keep the best-supported position. Scanning
    sides independently matters when the hull's center is off: a symmetric
    size scan about a wrong center can't place both sides (holder depth
    came out 15mm short)."""
    best, best_score = p, -np.inf
    for off in offsets:
        q = p.copy()
        q[3 + axis] = p[3 + axis] + off
        shift = np.zeros(3)
        shift[axis] = sign * off / 2
        q[:2] = p[:2] + (_rot_z(p[2]) @ shift)[:2]
        sc = edge_score(EDGE_MODELS["box"], q, views)
        if sc > best_score:
            best, best_score = q, sc
    return best


def hull_bounded(family: str, p0: np.ndarray, obj: dict):
    """Bounds from the visual hull: it is a trustworthy OUTER envelope (a few
    to ~30mm too big, never smaller), so the footprint may only shrink to
    80% of it. With 50%, crisp interior features -- the holder's bores and
    tubes, the tray's wells -- out-pulled the outer outline and boxes
    collapsed onto them (holder 316 -> 239mm, tray 52 -> 24mm). Height
    stays loose: it is the hull's least reliable number."""
    lo, hi = _param_bounds(family, p0)
    size = np.array(obj["bbox_max_m"]) - np.array(obj["bbox_min_m"])
    if family in ("box", "open_box", "rack_with_tube_grid", "plate_with_well_grid"):
        lo[3:5], hi[3:5] = 0.8 * size[:2], 1.05 * size[:2]
        lo[5], hi[5] = 0.1 * size[2], 1.05 * size[2]
    return lo, hi


def initial_params(family: str, obj: dict, shrink_m: float) -> np.ndarray:
    lo, hi = np.array(obj["bbox_min_m"]), np.array(obj["bbox_max_m"])
    c = (lo + hi) / 2
    # Shrink proportionally for small objects: a fixed 10mm turned a 28mm
    # hull depth into an 18mm start the fit never left.
    size = hi - lo
    size = np.maximum(size - np.minimum(shrink_m, 0.15 * size), 0.002)
    if family in ("box", "open_box"):
        return np.array([c[0], c[1], 0.0, size[0], size[1], size[2]])
    if family == "cylinder":
        return np.array([c[0], c[1], min(size[0], size[1]) / 2, size[2]])
    raise ValueError(family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--object", required=True)
    ap.add_argument("--family", required=True, choices=sorted(TEMPLATES))
    ap.add_argument("--poses-subdir", default="poses")
    ap.add_argument("--min-markers", type=int, default=3)
    ap.add_argument("--init-scene", help="scene.json to take the initial object box from "
                                          "(default: <capture>/reconstruction/scene.json)")
    ap.add_argument("--out-subdir", default="templates")
    ap.add_argument("--height-starts", type=float, nargs="+", default=[1.0],
                    help="box only: also start from these fractions of the hull height and keep the "
                         "best fit -- a wide, flat object's hull can be several times too tall")
    ap.add_argument("--features-spec", help="JSON hint for fixed (non-removable) features, e.g. "
                                            '{"well_grid": {"rows": 2, "cols": 5, "pitch_mm": 10, "radius_hint_mm": 3}}')
    ap.add_argument("--parts-spec", help="JSON decomposition hint (scene_reasoning's 'decomposition' "
                                          "shape): after fitting the body, fit each grid part on top of it")
    ap.add_argument("--init-shrink-mm", type=float, default=10.0,
                    help="visual hulls run oversized; start the template this much smaller per axis")
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    init_scene = Path(args.init_scene) if args.init_scene else cap / "reconstruction" / "scene.json"
    obj = next(o for o in json.loads(init_scene.read_text())["objects"] if o["id"] == args.object)
    intrinsics = load_intrinsics(cap / "camera_intrinsics.json")
    poses = json.loads((cap / args.poses_subdir / "camera_poses.json").read_text())["poses"]

    lo, hi = np.array(obj["bbox_min_m"]), np.array(obj["bbox_max_m"])
    roi = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    views = []
    for p in poses:
        if len(p["marker_ids_used"]) < args.min_markers:
            continue
        cam = intrinsics[p["camera_id"]]
        R_cw = quaternion_to_rotation_matrix(np.array(p["quaternion_wxyz"]))
        rvec, tvec = camera_pose_to_rvec_tvec(R_cw, np.array(p["position_world"]))
        gray = cv2.imread(str(resolve_shot_image(cap, p["image"])), cv2.IMREAD_GRAYSCALE)
        v = build_edge_view(Path(p["image"]).stem, gray, cam.K, cam.dist, rvec, tvec, roi)
        if v is not None:
            views.append(v)

    p0 = initial_params(args.family, obj, args.init_shrink_mm / 1000)
    skip_coarse = False
    if args.family == "open_box":
        p0 = np.r_[initial_params("box", obj, args.init_shrink_mm / 1000), 0.002]
    if args.family in ("box", "open_box"):
        # Coordinate scan by edge evidence instead of the coarse distance-map
        # stage, which drifted onto interior features (bores, tubes, wells)
        # whenever the object's outer outline was faint (white on white).
        # Height first: it is the hull's least trustworthy number (the tube
        # holder's hull was 172mm for a 40mm body).
        size = np.array(obj["bbox_max_m"]) - np.array(obj["bbox_min_m"])
        p0[3:5] = size[:2]  # start from the hull footprint; sides only move inward from there
        for _ in range(2):
            p0 = scan_parameter(box_height_edges, p0, 5, np.linspace(0.1 * size[2], size[2], 40), views)
            for axis in (0, 1):
                for sign in (-1, 1):
                    p0 = scan_box_side(p0, axis, sign, np.linspace(-0.12 * size[axis], 0.02 * size[axis], 29), views)
            if args.family == "open_box":
                p0 = scan_parameter(EDGE_MODELS["open_box"], p0, 6,
                                    np.linspace(0.0005, 0.15 * min(p0[3], p0[4]), 40), views)
        print(f"edge scan: w {p0[3] * 1000:.1f}  d {p0[4] * 1000:.1f}  h {p0[5] * 1000:.1f} mm "
              f"(hull {size[0] * 1000:.0f} x {size[1] * 1000:.0f} x {size[2] * 1000:.0f})")
        skip_coarse = True
    res = fit_template(args.family, views, p0, bounds=hull_bounded(args.family, p0, obj), skip_coarse=skip_coarse)
    report = {"object": args.object, "body": _result_dict(res, len(views)), "parts": []}
    _print(args.object, args.family, res, len(views))

    if args.features_spec:
        wg = json.loads(Path(args.features_spec).read_text()).get("well_grid")
        if wg:
            b = res.params
            gen, names, edges, well_centers = make_plate_with_well_grid(wg["rows"], wg["cols"])
            pitch = wg["pitch_mm"] / 1000
            pw = np.array([b["cx"], b["cy"], b["yaw"], b["w"], b["d"], b["h"], 0.0, 0.0, pitch, pitch,
                           wg.get("radius_hint_mm", 3.0) / 1000])
            size = np.array(obj["bbox_max_m"]) - np.array(obj["bbox_min_m"])
            # Well openings sit on the top face: scanning height now aligns them
            # with the dark openings. A filled dark disc has edge support only
            # at its rim, so (unlike a thin tube) scanning its radius is safe.
            for _ in range(2):
                pw = scan_parameter(edges, pw, 5, np.linspace(0.1 * size[2], size[2], 40), views)
                pw = scan_parameter(edges, pw, 10, np.linspace(0.001, 0.006, 26), views)
                for idx in (8, 9):
                    pw = scan_parameter(edges, pw, idx, np.linspace(0.7 * pitch, 1.4 * pitch, 29), views)
            print(f"  well scans: h {pw[5] * 1000:.1f}mm, well r {pw[10] * 1000:.2f}mm, "
                  f"pitch {pw[8] * 1000:.1f} x {pw[9] * 1000:.1f}mm")
            wres = fit_template("plate_with_well_grid", views, pw, template=(gen, names), edges_fn=edges,
                                skip_coarse=True, bounds=hull_bounded("plate_with_well_grid", pw, obj))
            report["body"] = {**_result_dict(wres, len(views)), "features": "well_grid",
                              "rows": wg["rows"], "cols": wg["cols"],
                              "well_centers_m": [list(map(float, c)) for c in well_centers(
                                  np.array([wres.params[n] for n in names]))]}
            _print(f"  plate + {wg['rows']}x{wg['cols']} wells", "plate_with_well_grid", wres, len(views))

    if args.parts_spec:
        spec = json.loads(Path(args.parts_spec).read_text())
        b = res.params
        for part in spec["parts"]:
            if part["grid_rows"] <= 0 or part["shape_family"] not in ("tube_with_conical_tip", "cylinder"):
                print(f"  (part '{part['name']}': no template for {part['shape_family']} yet, skipped)")
                continue
            rows, cols = part["grid_rows"], part["grid_cols"]
            gen, names, edges, tube_centers = make_rack_with_tube_grid(rows, cols)
            pitch = part["pitch_mm"] / 1000
            pc = np.array([b["cx"], b["cy"], b["yaw"], b["w"], b["d"], b["h"], 0.0, 0.0, pitch, pitch,
                           part.get("radius_hint_mm", 5.0) / 1000, b["h"] + 0.03,
                           0.8 * part.get("radius_hint_mm", 5.0) / 1000])
            # tube top and radius: whatever the image edges support best
            pc = scan_parameter(edges, pc, 11, np.linspace(b["h"] + 0.005, b["h"] + 0.2, 80), views)
            # No scan for radius / wall: the scan score is a per-sample mean, so
            # a tiny ring sitting on strong edges near a tube's axis wins (it
            # picked 1.5mm for a 3.5mm tube). Sizes start from the hint and are
            # left to the joint fit.
            print(f"  tube top scan: {pc[11] * 1000:.1f}mm (radius starts at hint {pc[10] * 1000:.1f}mm)")
            clo, chi = hull_bounded("rack_with_tube_grid", pc, obj)
            # Freeze the rack at its own fit: fitted jointly, the tubes' edges
            # nudged it off by up to ~0.9mm; alone it was within ~0.4mm.
            clo[:6], chi[:6] = pc[:6] - 1e-7, pc[:6] + 1e-7
            cres = fit_template("rack_with_tube_grid", views, pc, template=(gen, names), edges_fn=edges,
                                skip_coarse=True, bounds=(clo, chi))
            report["body"] = {**_result_dict(cres, len(views)), "rack_frozen_from_box_fit": True,
                              "rows": rows, "cols": cols}
            centers = tube_centers(np.array([cres.params[n] for n in names]))
            report["parts"].append({"name": part["name"], "rows": rows, "cols": cols, "family": "cylinder",
                                    "radius_m": cres.params["r"], "z_bottom_m": cres.params["h"],
                                    "z_top_m": cres.params["z_top"],
                                    "centers_m": [list(map(float, c)) for c in centers]})
            _print(f"  joint rack + '{part['name']}' {rows}x{cols}", "rack_with_tube_grid", cres, len(views))

    out = cap / args.out_subdir
    out.mkdir(exist_ok=True)
    (out / f"{args.object}.json").write_text(json.dumps(report, indent=2))


def _result_dict(res, n_views):
    return {"family": res.family, "params_m": res.params, "rms_px": res.rms_px, "n_points": res.n_points,
            "views": n_views, "per_view_rms_px": res.per_view_rms_px}


def _print(label, family, res, n_views):
    print(f"{label} [{family}] {n_views} views, {res.n_points} outline points, rms {res.rms_px:.2f}px")
    for k, v in res.params.items():
        print(f"    {k:8s} = {np.degrees(v):9.2f} deg" if k == "yaw" else f"    {k:8s} = {v * 1000:9.2f} mm")


if __name__ == "__main__":
    main()

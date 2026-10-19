"""Depth-based reconstruction: fuse every view's depth map into a TSDF over
the platform, split it into objects, and re-fuse each object finely.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/reconstruct_depth.py \
      --capture-dir CAPTURE [--simulate-sensor tof] [--ground-truth]

Needs <capture>/manifest.json with a "depth" entry per shot,
camera_intrinsics.json and poses/camera_poses.json (Stage 1, from the
markers in the colour images; depth is registered to the colour camera,
so it shares that pose). Writes <capture>/<out-subdir>/ (default
reconstruction_depth/) in the same layout as reconstruct_scene.py:
  scene.json, objects/<id>.obj (local frame: origin at the object's
  bounding-box bottom center), scene_platform_frame.obj, fusion.npz
  (the coarse field, for refinement bounds), and with --ground-truth
  per-object scores.

--simulate-sensor applies a depth_sensor model to rendered (perfect)
depth first; leave it off for real sensor data.

Two passes. Coarse (4mm) over the whole platform finds the objects: the
fused volume, minus the bench (the grid starts above it), minus the
calibration fixtures (known positions), as 3D connected components.
Fine (1mm default) per object, restricted to a dilation of its coarse
component so neighbors don't leak in; its mesh is closed at the bench.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh
from scipy import ndimage

from calibration import load_intrinsics
from depth_fusion import DepthView, field_normals, fuse, grid_mesh, merge_nested, split_components
from depth_sensor import PRESETS, load_depth, load_sensor, remove_flying_pixels, sensor_to_dict, simulate
from evaluation import format_score, score_mesh
from geometry import camera_pose_to_rvec_tvec, quaternion_to_rotation_matrix


def load_depth_views(cap: Path, sensor_name: str | None, min_markers: int, poses_subdir: str = "poses",
                     flying_filter: bool = True):
    manifest = json.loads((cap / "manifest.json").read_text())
    scale = manifest.get("depth", {}).get("scale_m", 1e-4)
    # Where depth pixel (i, j) was sampled, minus (i, j), in depth pixels.
    # 0 = at the pixel center (OpenCV convention, what K assumes). Isaac's distance_to_image_plane annotator samples at
    # +0.5px (the pixel corner): left uncorrected, every surface shifted
    # ~0.2mm at 1.5m and fused 0.2mm median off instead of 0.04mm.
    pix_off = float(manifest.get("depth", {}).get("pixel_offset_px", 0.0))
    depth_of = {s["camera_id"]: s.get("depth") for s in manifest["shots"]}
    intrinsics = load_intrinsics(cap / "camera_intrinsics.json")
    model = load_sensor(sensor_name) if sensor_name else None
    sigma_fn = model.sigma if model is not None and model.name != "perfect" else None
    views, skipped = [], []
    for p in json.loads((cap / poses_subdir / "camera_poses.json").read_text())["poses"]:
        cid = p["camera_id"]
        if not depth_of.get(cid):
            continue
        if len(p["marker_ids_used"]) < min_markers:
            skipped.append(cid)
            continue
        cam = intrinsics[cid]
        rvec, tvec = camera_pose_to_rvec_tvec(quaternion_to_rotation_matrix(np.array(p["quaternion_wxyz"])),
                                              np.array(p["position_world"]))
        depth = load_depth(cap / depth_of[cid], scale)
        K = cam.K.copy()
        K[0, 2] -= pix_off * cam.image_size[0] / depth.shape[1]     # K is still at colour resolution here
        K[1, 2] -= pix_off * cam.image_size[1] / depth.shape[0]
        if depth.shape[::-1] != tuple(cam.image_size):          # depth at its own resolution
            from depth_sensor import scale_intrinsics
            K = scale_intrinsics(K, cam.image_size, depth.shape[::-1])
        if model is not None:
            depth, K = simulate(depth, K, model, seed=cid)
        if flying_filter:
            depth = remove_flying_pixels(depth, K, sigma_fn=sigma_fn)
        views.append(DepthView(cid, K, rvec, tvec, depth, sigma_fn))
    return views, skipped, model


def write_refined_poses(cap: Path, out: Path, poses_subdir: str, refined: dict) -> None:
    """<out>/poses/camera_poses.json: Stage 1's file with the depth-refined
    poses substituted, for the later stages (refinement, labels)."""
    from geometry import rotation_matrix_to_quaternion, world_to_camera_to_pose
    data = json.loads((cap / poses_subdir / "camera_poses.json").read_text())
    for p in data["poses"]:
        if p["camera_id"] in refined:
            R_cw, pos = world_to_camera_to_pose(*refined[p["camera_id"]])
            p["quaternion_wxyz"] = rotation_matrix_to_quaternion(R_cw).tolist()
            p["position_world"] = pos.tolist()
            p["refined_by"] = "depth ICP"
    (out / "poses").mkdir(exist_ok=True)
    (out / "poses" / "camera_poses.json").write_text(json.dumps(data, indent=2))


def load_label_views(cap: Path, min_markers: int, poses_subdir: str = "poses"):
    """Segmented colour views (segment_capture.py) for CLIP-vote labels, as
    reconstruct_scene.py builds them; [] when there is no segmentation."""
    from reconstruction import View
    seg = cap / "segmentation"
    if not seg.exists():
        return []
    intrinsics = load_intrinsics(cap / "camera_intrinsics.json")
    views = []
    for pose in json.loads((cap / poses_subdir / "camera_poses.json").read_text())["poses"]:
        name = Path(pose["image"]).stem
        if not (seg / f"{name}.npz").exists() or len(pose["marker_ids_used"]) < min_markers:
            continue
        cam = intrinsics[pose["camera_id"]]
        rvec, tvec = camera_pose_to_rvec_tvec(quaternion_to_rotation_matrix(np.array(pose["quaternion_wxyz"])),
                                              np.array(pose["position_world"]))
        data = np.load(seg / f"{name}.npz")
        K = cam.K.copy()
        K[:2] *= float(data["scale"]) if "scale" in data else 1.0
        views.append(View(name, K, cam.dist, rvec, tvec, data["foreground"], data["instance_masks"],
                          [i["labels"] for i in json.loads((seg / f"{name}.json").read_text())]))
    return views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--out-subdir", default="reconstruction_depth")
    ap.add_argument("--poses-subdir", default="poses")
    ap.add_argument("--simulate-sensor", choices=sorted(PRESETS), default=None,
                    help="apply this sensor model to rendered perfect depth (or a JSON model file path via --sensor-json)")
    ap.add_argument("--sensor-json", default=None)
    ap.add_argument("--platform-w", type=float, default=None)
    ap.add_argument("--platform-d", type=float, default=None)
    ap.add_argument("--max-height", type=float, default=0.35)
    ap.add_argument("--coarse-voxel-mm", type=float, default=4.0)
    ap.add_argument("--fine-voxel-mm", type=float, default=1.0)
    ap.add_argument("--trunc-mm", type=float, default=None,
                    help="TSDF truncation; default max(3 fine voxels, 4 sigma at 1.5m)")
    ap.add_argument("--bench-cut-mm", type=float, default=None,
                    help="fine grids start this far above the bench (the bench itself is not an object); objects "
                         "are closed there and extended straight down to it. Default: half a fine voxel + 3 sigma "
                         "of the fused bench")
    ap.add_argument("--min-markers", type=int, default=3)
    ap.add_argument("--max-support-gap-mm", type=float, default=10.0)
    ap.add_argument("--min-object-volume-cm3", type=float, default=2.0)
    ap.add_argument("--marker-margin-m", type=float, default=0.03)
    ap.add_argument("--ground-truth", action="store_true")
    ap.add_argument("--icp-no-anchor", action="store_true",
                    help="ICP without the known-geometry anchor (bench plane + marker quads)")
    ap.add_argument("--icp-rounds", type=int, default=4,
                    help="refine camera poses by aligning the views' depth to each other (0 = marker poses as is)")
    ap.add_argument("--single-pass", action="store_true",
                    help="skip the facing-aware second fine pass (half the time; thin walls thicken under noise: "
                         "a 2mm bin wall fused 0.82mm median off with ToF noise vs 0.27mm with it)")
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    out = cap / args.out_subdir
    (out / "objects").mkdir(parents=True, exist_ok=True)
    for stale in list((out / "objects").glob("*.obj")) + list((out / "objects").glob("*_field.npz")):
        stale.unlink()
    dims = json.loads((cap / "platform_dims.json").read_text()) if (cap / "platform_dims.json").exists() else {}
    pw = args.platform_w or dims["platform_w"]
    pd = args.platform_d or dims["platform_d"]

    t0 = time.time()
    sensor = args.sensor_json or args.simulate_sensor
    views, skipped, model = load_depth_views(cap, sensor, args.min_markers, args.poses_subdir)
    print(f"{len(views)} depth views" + (f" (sensor model: {model.name})" if model else " (depth used as given)")
          + (f", {len(skipped)} skipped (< {args.min_markers} markers)" if skipped else ""))
    if not views:
        raise SystemExit("no depth views")
    # The depth actually fused (sensor model applied, flying pixels removed),
    # with its K: refinement gates image edges on its discontinuities.
    from depth_sensor import save_depth
    (out / "depth_used").mkdir(exist_ok=True)
    for v in views:
        save_depth(out / "depth_used" / f"{v.name}.png", v.depth)
    (out / "depth_used" / "intrinsics.json").write_text(json.dumps(
        {v.name: {"K": np.asarray(v.K).tolist(), "size": list(v.depth.shape[::-1])} for v in views}, indent=2))

    sig15 = float(model.sigma(np.array(1.5))) if model is not None else 0.0
    hw, hd = pw / 2, pd / 2
    refined_poses = {}
    if args.icp_rounds > 0:
        from depth_registration import known_geometry_cloud, refine_poses
        anchor = None
        if (cap / "marker_layout.json").exists() and not args.icp_no_anchor:
            from markers import load_marker_layout
            anchor = known_geometry_cloud((pw, pd), list(load_marker_layout(cap / "marker_layout.json").corners_3d.values()))
        print(f"refining camera poses by depth ICP ({args.icp_rounds} rounds, anchored to "
              f"{'the bench plane + marker quads' if anchor is not None else 'the markers frame (mean removed)'})")
        refine_poses(views, (-hw - 0.05, -hd - 0.05, -0.02), (hw + 0.05, hd + 0.05, args.max_height),
                     rounds=args.icp_rounds, anchor=anchor)
        refined_poses = {v.name: (v.rvec, v.tvec) for v in views}
        write_refined_poses(cap, out, args.poses_subdir, refined_poses)
    fine = args.fine_voxel_mm / 1000
    trunc = args.trunc_mm / 1000 if args.trunc_mm else max(3 * fine, 4 * sig15)
    coarse = args.coarse_voxel_mm / 1000
    coarse_trunc = max(trunc, 3 * coarse)
    behind = max(1.5 * fine, 3 * sig15)
    coarse_behind = max(1.5 * coarse, 3 * sig15)

    # ---- coarse pass over the platform
    # Start just above the bench's fused noise. (Starting 2 voxels up, like
    # the visual hull, missed a bracket's 5mm-thick leg lying flat -- and
    # would miss any object thinner than 8mm.)
    z0 = 0.5 * coarse + 3 * sig15 / np.sqrt(max(len(views) / 2, 1))
    bounds = ((-hw, -hd, z0), (hw, hd, args.max_height))
    g = fuse(views, *bounds, coarse, coarse_trunc, coarse_behind)
    occ = g.field < 0
    print(f"coarse fusion {g.field.shape} at {args.coarse_voxel_mm}mm: {int(occ.sum())} occupied voxels "
          f"({time.time() - t0:.0f}s)")
    if (cap / "marker_layout.json").exists():
        from markers import load_marker_layout
        from reconstruction import clear_marker_columns
        markers = list(load_marker_layout(cap / "marker_layout.json").corners_3d.values())
        print(f"cleared {clear_marker_columns(occ, g.origin, coarse, markers, args.marker_margin_m)} voxels over "
              f"{len(markers)} calibration-marker footprints")
    labels, comps = split_components(occ, coarse, args.min_object_volume_cm3 * 1e-6)
    n_before = len(comps)
    comps = merge_nested(labels, comps)
    if len(comps) < n_before:
        print(f"merged {n_before - len(comps)} component(s) lying inside a bigger one's footprint")
    np.savez_compressed(out / "fusion.npz", field=g.field, origin=g.origin, voxel_m=coarse, labels=labels)

    # As low as the fused bench's noise allows: a cut above an object's
    # floor erases it (a 2mm bin floor vanished with a 1.5mm cut). The fused
    # surface averages roughly half the views.
    bench_cut = (args.bench_cut_mm / 1000 if args.bench_cut_mm is not None
                 else 0.5 * fine + 3 * sig15 / np.sqrt(max(len(views) / 2, 1)))
    gt = json.loads((cap / "ground_truth_meshes.json").read_text()) if args.ground_truth else None
    vis_views = None
    if gt is not None:   # synthetic capture: the rendered depth is the truth, for observability
        raw, _, _ = load_depth_views(cap, None, args.min_markers, args.poses_subdir, flying_filter=False)
        vis_views = [(v.K, *refined_poses.get(v.name, (v.rvec, v.tvec)), v.depth) for v in raw]
    objects, scene_meshes, floating = [], [], 0
    label_views = load_label_views(cap, args.min_markers, args.poses_subdir)
    for lv in label_views:
        if lv.name in refined_poses:
            lv.rvec, lv.tvec = refined_poses[lv.name]
    if label_views:
        from reconstruction import label_component
        print(f"labeling from {len(label_views)} segmented views")
    for cid in comps:
        ijk = np.argwhere(labels == cid)
        lo = g.origin + ijk.min(axis=0) * coarse - coarse / 2
        hi = g.origin + ijk.max(axis=0) * coarse + coarse / 2
        if lo[2] > z0 + coarse + args.max_support_gap_mm / 1000:
            floating += 1
            continue
        margin = 2 * coarse + trunc
        flo = np.array([lo[0] - margin, lo[1] - margin, bench_cut])
        fhi = np.array([hi[0] + margin, hi[1] + margin, hi[2] + margin])
        # keep = fine voxels whose coarse voxel is (near) this component
        own = ndimage.binary_dilation(labels == cid, iterations=2)
        shape = tuple(int(s) for s in np.ceil((fhi - flo) / fine - 1e-9).astype(int))
        fi = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), axis=-1).reshape(-1, 3)
        pts = flo + fine / 2 + fi * fine
        cijk = np.floor((pts - g.origin) / coarse + 0.5).astype(int)
        below = cijk[:, 2] < 1
        cijk[:, 2] = np.maximum(cijk[:, 2], 0)
        inb = np.all((cijk >= 0) & (cijk < labels.shape), axis=1)
        keep = np.zeros(len(pts), bool)
        keep[inb] = own[cijk[inb, 0], cijk[inb, 1], cijk[inb, 2]]
        # Under the coarse grid's first layers: the component's FILLED
        # footprint. A bin's floor lies below the coarse grid and its
        # interior is free space in it, so "nearest coarse voxel" dropped
        # the floor entirely.
        foot = ndimage.binary_fill_holes(own.any(axis=2))
        sel = below & inb
        keep[sel] = foot[cijk[sel, 0], cijk[sel, 1]]
        keep = keep.reshape(shape)
        fg = fuse(views, flo, fhi, fine, trunc, behind, keep=keep)
        if not args.single_pass:
            # second pass: behind-surface evidence only from views the
            # surface faces (thin walls under noise; see depth_fusion.fuse)
            fg = fuse(views, flo, fhi, fine, trunc, behind, keep=keep, normals=field_normals(fg.field, 1.5))
        mesh = grid_mesh(fg, bottom_z=0.0)
        if mesh is None:
            continue
        parts = mesh.split(only_watertight=False)
        biggest = max(len(m.faces) for m in parts)
        mesh = trimesh.util.concatenate([m for m in parts if len(m.faces) >= 0.02 * biggest])  # drops specks
        oid = f"obj_{len(objects):02d}"
        mlo, mhi = mesh.bounds
        placement = np.array([(mlo[0] + mhi[0]) / 2, (mlo[1] + mhi[1]) / 2, 0.0])
        local = mesh.copy()
        local.apply_translation(-placement)
        local.export(out / "objects" / f"{oid}.obj")
        # The fine field, for refinement's "stay near the measured surface"
        # bound; int8 (field x 127) is ample for that and ~10x smaller.
        np.savez_compressed(out / "objects" / f"{oid}_field.npz", field_q=np.round(fg.field * 127).astype(np.int8),
                            origin=fg.origin, voxel_m=fine, trunc_m=trunc)
        scene_meshes.append(mesh)
        label, conf, votes = "unlabeled", 0.0, {}
        if label_views:
            label, conf, votes = label_component(trimesh.sample.sample_surface(mesh, 4000, seed=0)[0], label_views)
        entry = {"id": oid, "label": label, "label_confidence": round(float(conf), 3), "label_votes": votes,
                 "dims_m": dict(zip("xyz", (mhi - mlo).tolist())),
                 "placement_m": placement.tolist(), "bbox_min_m": mlo.tolist(), "bbox_max_m": mhi.tolist(),
                 "volume_m3": float(mesh.volume) if mesh.is_volume else None,
                 "watertight": bool(mesh.is_watertight), "faces": len(mesh.faces),
                 "mesh": f"objects/{oid}.obj"}
        if gt is not None:
            entry["score"] = score_mesh(mesh, gt, cap, vis_views)
        objects.append(entry)
        d = (mhi - mlo) * 1000
        line = (f"  {oid}  {d[0]:6.1f} x {d[1]:6.1f} x {d[2]:6.1f} mm  at ({placement[0]:+.3f}, {placement[1]:+.3f})"
                + (f"  '{label}' ({conf:.2f})" if label_views else ""))
        if gt is not None:
            line += "\n      " + format_score(entry["score"])
        print(line, flush=True)

    if scene_meshes:
        trimesh.util.concatenate(scene_meshes).export(out / "scene_platform_frame.obj")
    (out / "scene.json").write_text(json.dumps({
        "frame": "platform-local (origin at bench-top center, Z up, meters)",
        "method": f"TSDF depth fusion: coarse {args.coarse_voxel_mm}mm, fine {args.fine_voxel_mm}mm, "
                  f"{'1' if args.single_pass else '2'}-pass, trunc {trunc * 1000:.2f}mm / behind {behind * 1000:.2f}mm, bench cut {bench_cut * 1000:.2f}mm, {len(views)} views",
        "sensor_model": sensor_to_dict(model) if model else "as captured",
        # noise of the fused surface (~half the views see any surface point): how far
        # refinement may move a vertex off it
        "fused_sigma_m": float(sig15 / np.sqrt(max(len(views) / 2, 1))),
        "objects": objects}, indent=2))
    print(f"{len(objects)} objects ({floating} floating components dropped) in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()

"""Stages 3-5: carve the platform volume from every solved view's
silhouettes, split it into objects, label and measure them, and export
meshes + placements.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/reconstruct_scene.py \
      --capture-dir biotech-lab-robotics/vision_pipeline/isaac_validation/out --platform-w 1.4 --platform-d 0.7

Needs <capture-dir>/poses (solve_camera_poses.py) and
<capture-dir>/segmentation (segment_capture.py). Writes
<capture-dir>/reconstruction/:
  scene.json                one entry per object: label, dims, placement
  objects/<id>.obj          mesh in the object's own frame (origin at its
                            bounding-box bottom center) -- the same
                            "local mesh + placement" split as assets/cad
  scene_platform_frame.obj  every object placed, for a quick visual check
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from calibration import load_intrinsics
from geometry import camera_pose_to_rvec_tvec, quaternion_to_rotation_matrix
from markers import load_marker_layout
from reconstruction import (View, carve, clear_marker_columns, component_mesh,
                            label_component, split_objects, view_consistency)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--platform-w", type=float, required=True)
    ap.add_argument("--platform-d", type=float, required=True)
    ap.add_argument("--max-height", type=float, default=0.30)
    ap.add_argument("--voxel-mm", type=float, default=2.5)
    ap.add_argument("--min-agree", type=float, default=0.75,
                     help="loose first pass only: fraction of views that must agree")
    ap.add_argument("--min-view-consistency", type=float, default=0.5,
                     help="drop views whose silhouettes cover less of the loose hull than this. Only for "
                          "catastrophic segmentation failures (top-down scored 0.09): the loose hull is "
                          "inflated, so a high bar drops exactly the side views that would carve it")
    ap.add_argument("--mask-dilate-px", type=int, default=0,
                     help="grow every silhouette by this many px to absorb residual camera-pose error. 0 by "
                          "default: on the Isaac capture, 2px inflated every object by ~5-10mm per axis "
                          "and bought nothing (poses from 3+ markers are already within ~1-5mm)")
    ap.add_argument("--max-support-gap-mm", type=float, default=10.0,
                     help="drop components whose bottom is higher than this above the bench: "
                          "objects rest on the platform (something stacked is connected to what's "
                          "below it), so a floating component is a phantom from ray intersections")
    ap.add_argument("--max-misses", type=int, default=1,
                     help="strict pass: views allowed to disagree per voxel")
    ap.add_argument("--min-markers", type=int, default=3,
                     help="skip views whose Stage 1 pose used fewer markers: on the Isaac "
                          "validation capture, 1-2 marker poses were 10-64mm off vs 1-5mm for 3+, "
                          "and one mis-posed view carves real geometry away")
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    intrinsics = load_intrinsics(cap / "camera_intrinsics.json")
    poses = json.loads((cap / "poses" / "camera_poses.json").read_text())["poses"]
    seg = cap / "segmentation"

    views, skipped = [], []
    for pose in poses:
        name = Path(pose["image"]).stem
        if not (seg / f"{name}.npz").exists():
            continue
        if len(pose["marker_ids_used"]) < args.min_markers:
            skipped.append(name)
            continue
        cam = intrinsics[pose["camera_id"]]
        R_cw = quaternion_to_rotation_matrix(np.array(pose["quaternion_wxyz"]))
        rvec, tvec = camera_pose_to_rvec_tvec(R_cw, np.array(pose["position_world"]))
        data = np.load(seg / f"{name}.npz")
        instance_labels = [i["labels"] for i in json.loads((seg / f"{name}.json").read_text())]
        K = cam.K.copy()
        K[:2] *= float(data["scale"]) if "scale" in data else 1.0  # masks may be at reduced resolution
        views.append(View(name, K, cam.dist, rvec, tvec, data["foreground"],
                          data["instance_masks"], instance_labels))
    print(f"{len(views)} calibrated, segmented views"
          + (f" ({len(skipped)} skipped, < {args.min_markers} markers: {', '.join(skipped)})" if skipped else ""))

    voxel = args.voxel_mm / 1000.0
    hw, hd = args.platform_w / 2, args.platform_d / 2
    # Start 2 voxels above the bench: the bench plane and the 1mm flat
    # markers on it aren't objects.
    bounds = ((-hw, -hd, 2 * voxel), (hw, hd, args.max_height))
    loose, origin = carve(views, *bounds, voxel, min_agree_fraction=args.min_agree,
                          mask_dilate_px=args.mask_dilate_px)
    consistency = view_consistency(loose, origin, voxel, views, mask_dilate_px=args.mask_dilate_px)
    dropped = [v.name for v in views if consistency[v.name] < args.min_view_consistency]
    print("view consistency with loose hull: " + ", ".join(
        f"{n} {c:.2f}" for n, c in sorted(consistency.items(), key=lambda kv: kv[1])[:6]) + " ...")
    if dropped:
        print(f"dropping {len(dropped)} inconsistent views: {', '.join(dropped)}")
    views = [v for v in views if v.name not in dropped]
    occ, origin = carve(views, *bounds, voxel, max_misses=args.max_misses,
                        mask_dilate_px=args.mask_dilate_px)
    print(f"strict carve over {len(views)} views (<= {args.max_misses} miss): "
          f"{int(occ.sum())} occupied voxels at {args.voxel_mm}mm")
    markers = list(load_marker_layout(cap / "marker_layout.json").corners_3d.values())
    # 3cm: the raised markers' stands are 1cm wider than the marker on each side.
    cleared = clear_marker_columns(occ, origin, voxel, markers, margin_m=0.03)
    print(f"cleared {cleared} voxels over {len(markers)} calibration-marker footprints")

    labels, comp_ids = split_objects(occ, voxel)
    (cap / "reconstruction").mkdir(parents=True, exist_ok=True)
    # Saved for freeform refinement: an exact "inside the visual hull" test
    # (a mesh-based inside/outside test leaked on the non-watertight hull).
    np.savez_compressed(cap / "reconstruction" / "occupancy.npz", occupancy=occ, origin=origin, voxel_m=voxel)

    out = cap / "reconstruction"
    (out / "objects").mkdir(parents=True, exist_ok=True)
    for stale in (out / "objects").glob("*.obj"):
        stale.unlink()  # scene.json lists only this run's objects; old meshes would just mislead
    import trimesh
    scene_meshes, objects, floating = [], [], 0
    for cid in comp_ids:
        ijk = np.argwhere(labels == cid)
        pts = origin + ijk * voxel
        bmin, bmax = pts.min(axis=0) - voxel / 2, pts.max(axis=0) + voxel / 2
        if bmin[2] > args.max_support_gap_mm / 1000.0:
            floating += 1
            continue
        bmin[2] = 0.0  # carving starts 2 voxels above the bench; supported objects sit on it
        label, score, votes = label_component(pts, views)
        mesh = component_mesh(labels, cid, origin, voxel)
        placement = np.array([(bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, bmin[2]])
        oid = f"obj_{len(objects):02d}"
        local = mesh.copy()
        local.apply_translation(-placement)
        local.export(out / "objects" / f"{oid}.obj")
        scene_meshes.append(mesh)
        objects.append({
            "id": oid, "label": label, "label_confidence": round(score, 3), "label_votes": votes,
            "dims_m": {"x": float(bmax[0] - bmin[0]), "y": float(bmax[1] - bmin[1]), "z": float(bmax[2] - bmin[2])},
            "placement_m": placement.tolist(),
            "bbox_min_m": bmin.tolist(), "bbox_max_m": bmax.tolist(),
            "volume_m3": float(len(ijk) * voxel ** 3),
            "mesh": f"objects/{oid}.obj",
        })

    if scene_meshes:
        trimesh.util.concatenate(scene_meshes).export(out / "scene_platform_frame.obj")
    (out / "scene.json").write_text(json.dumps({
        "frame": "platform-local (origin at bench-top center, Z up, meters); "
                 "world = (PLATFORM_CX + x, PLATFORM_CY + y, bench_top_z + z)",
        "method": f"visual hull, {args.voxel_mm}mm voxels, {len(views)} views "
                  f"(after consistency gate), <= {args.max_misses} disagreeing view per voxel",
        "objects": objects}, indent=2))

    print(f"{len(objects)} objects ({floating} unsupported floating components dropped):")
    for o in objects:
        d = o["dims_m"]
        print(f"  {o['id']}  {d['x']*1000:6.1f} x {d['y']*1000:6.1f} x {d['z']*1000:6.1f} mm  "
              f"at ({o['placement_m'][0]:+.3f}, {o['placement_m'][1]:+.3f})  "
              f"'{o['label']}' ({o['label_confidence']:.2f})")


if __name__ == "__main__":
    main()

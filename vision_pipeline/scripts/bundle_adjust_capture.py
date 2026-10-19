"""Refine every camera pose and the marker layout jointly (see
bundle_adjust.py), starting from Stage 1's per-view PnP poses.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/bundle_adjust_capture.py \
      --capture-dir CAPTURE [--marker-layout CAPTURE/marker_layout.json]

Reads <capture>/manifest.json, camera_intrinsics.json, poses/camera_poses.json.
Writes <capture>/poses_ba/camera_poses.json (same schema as Stage 1, so
every later stage can use it) and poses_ba/marker_layout_refined.json.

marker_layout.json may list "rigid_groups": [[ids...], ...] -- markers on
one rigid object (a calibration cube) whose relative geometry is trusted;
every other marker is its own body with a free pose.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from bundle_adjust import bundle_adjust
from calibration import load_intrinsics
from capture import CaptureManifest
from geometry import (camera_pose_to_rvec_tvec, quaternion_to_rotation_matrix,
                      rotation_matrix_to_quaternion, world_to_camera_to_pose)
from markers import detect_markers, load_marker_layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--marker-layout", help="defaults to <capture>/marker_layout.json")
    ap.add_argument("--poses-subdir", default="poses", help="initial poses (Stage 1 output)")
    ap.add_argument("--out-subdir", default="poses_ba")
    ap.add_argument("--layout-sigma-mm", type=float, default=3.0,
                     help="how far each marker body may be from its measured position (1 sigma)")
    ap.add_argument("--layout-sigma-deg", type=float, default=1.0)
    ap.add_argument("--free-layout", action="store_true",
                     help="no layout prior (worse: scale then rests on marker size alone)")
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    layout_path = Path(args.marker_layout) if args.marker_layout else cap / "marker_layout.json"
    layout = load_marker_layout(layout_path)
    layout_json = json.loads(layout_path.read_text())
    groups = layout_json.get("rigid_groups", [])
    # optional per-body uncertainty: [{"markers": [...], "pos_sigma_mm": .., "rot_sigma_deg": ..}]
    body_priors = {m: (bp["pos_sigma_mm"] / 1000, np.radians(bp["rot_sigma_deg"]))
                   for bp in layout_json.get("body_priors", []) for m in bp["markers"]}
    intrinsics = load_intrinsics(cap / "camera_intrinsics.json")
    manifest = CaptureManifest.load(cap / "manifest.json")
    init = {Path(p["image"]).stem: p for p in
            json.loads((cap / args.poses_subdir / "camera_poses.json").read_text())["poses"]}

    shots, detections, Ks, dists, rvecs, tvecs = [], [], [], [], [], []
    for shot in manifest.shots:
        p = init.get(shot.image_path.stem)
        if p is None:
            continue  # Stage 1 couldn't pose it; nothing to initialize from
        cam = intrinsics[shot.camera_id]
        det = {m: c for m, c in detect_markers(cv2.imread(str(shot.image_path))).items() if m in layout.corners_3d}
        R_cw = quaternion_to_rotation_matrix(np.array(p["quaternion_wxyz"]))
        rv, tv = camera_pose_to_rvec_tvec(R_cw, np.array(p["position_world"]))
        shots.append(shot); detections.append(det); Ks.append(cam.K); dists.append(cam.dist)
        rvecs.append(rv.ravel()); tvecs.append(tv.ravel())

    res = bundle_adjust(detections, Ks, dists, np.array(rvecs), np.array(tvecs), layout.corners_3d, groups,
                        prior_pos_sigma_m=None if args.free_layout else args.layout_sigma_mm / 1000,
                        prior_rot_sigma_rad=np.radians(args.layout_sigma_deg), body_priors=body_priors)
    print(f"{len(shots)} views, {sum(len(d) for d in detections)} marker sightings; "
          f"reprojection RMS {res.rms_before_px:.2f}px -> {res.rms_after_px:.2f}px")

    out = cap / args.out_subdir
    out.mkdir(parents=True, exist_ok=True)
    poses = []
    for i, shot in enumerate(shots):
        R_cw, pos = world_to_camera_to_pose(res.rvecs[i], res.tvecs[i])
        poses.append({"image": str(shot.image_path), "camera_id": shot.camera_id,
                      "position_world": pos.tolist(),
                      "quaternion_wxyz": rotation_matrix_to_quaternion(R_cw).tolist(),
                      "marker_ids_used": sorted(detections[i]),
                      "mean_reproj_error_px": float(res.per_view_rms_px[i]),
                      "max_reproj_error_px": float(res.per_view_rms_px[i]),
                      "method": "bundle_adjusted"})
    (out / "camera_poses.json").write_text(json.dumps({"poses": poses}, indent=2))

    from bundle_adjust import build_bodies
    bodies, _ = build_bodies(layout.corners_3d, groups)
    refined = {}
    for ids, corners in zip(bodies, res.body_corners):
        for k, m in enumerate(ids):
            refined[str(m)] = {"corners_3d": corners[4 * k:4 * k + 4].tolist()}
    (out / "marker_layout_refined.json").write_text(json.dumps(
        {"markers": refined, "rigid_groups": groups}, indent=2))
    print(f"wrote {out / 'camera_poses.json'} and {out / 'marker_layout_refined.json'}")


if __name__ == "__main__":
    main()

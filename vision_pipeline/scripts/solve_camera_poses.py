"""Stage 1 entry point: solve every camera's pose in the platform frame
from a multi-view capture and a marker layout, report reprojection QA,
and write annotated overlay images.

Usage:
  python scripts/solve_camera_poses.py \
      --captures path/to/photos_dir \
      --camera-id phone_main \
      --intrinsics config/camera_intrinsics.json \
      --marker-layout config/marker_layout.json \
      --out-dir out/poses

Or, for a multi-camera rig with a manifest mapping each shot to its own
camera_id:
  python scripts/solve_camera_poses.py \
      --manifest path/to/manifest.json \
      --intrinsics config/camera_intrinsics.json \
      --marker-layout config/marker_layout.json \
      --out-dir out/poses
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration import load_intrinsics
from capture import CaptureManifest
from markers import load_marker_layout
from pose_solver import solve_all_camera_poses
from qa import draw_overlay, summarize


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--captures", help="directory of photos, single camera")
    src.add_argument("--manifest", help="JSON manifest for a multi-camera rig")
    ap.add_argument("--camera-id", default="default",
                     help="used with --captures; must match an entry in "
                          "--intrinsics")
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--marker-layout", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    manifest = (CaptureManifest.from_directory(args.captures, args.camera_id)
                if args.captures else CaptureManifest.load(args.manifest))
    intrinsics = load_intrinsics(args.intrinsics)
    marker_layout = load_marker_layout(args.marker_layout)

    results = solve_all_camera_poses(manifest, intrinsics, marker_layout)

    print(summarize(results))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    poses_out = []
    for r in results:
        if r.success:
            poses_out.append({
                "image": str(r.image_path),
                "camera_id": r.camera_id,
                "position_world": r.position_world.tolist(),
                "quaternion_wxyz": r.quaternion_wxyz.tolist(),
                "marker_ids_used": r.marker_ids_used,
                "mean_reproj_error_px": r.mean_reproj_error_px,
                "max_reproj_error_px": r.max_reproj_error_px,
            })
        draw_overlay(r.image_path, intrinsics[r.camera_id], marker_layout, r,
                      out_dir / "overlays" / f"{r.image_path.stem}.jpg")

    (out_dir / "camera_poses.json").write_text(
        json.dumps({"poses": poses_out}, indent=2))
    print(f"\n{len(poses_out)}/{len(results)} views solved. "
          f"Wrote {out_dir / 'camera_poses.json'} and overlays to "
          f"{out_dir / 'overlays'}")


if __name__ == "__main__":
    main()

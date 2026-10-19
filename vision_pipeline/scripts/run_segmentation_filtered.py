"""Full Stage 1 -> Stage 2 chain on one already-pose-solved image: load
the camera pose solve_camera_poses.py produced, use it to restrict SAM's
candidate masks to the platform footprint, then label what's left.

Usage (after running solve_camera_poses.py into <out-dir>):
  .venv-vision/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/run_segmentation_filtered.py \
      --image biotech-lab-robotics/vision_pipeline/out/synthetic_capture/images/ring_05_az225.png \
      --camera-poses biotech-lab-robotics/vision_pipeline/out/synthetic_capture/poses/camera_poses.json \
      --intrinsics biotech-lab-robotics/vision_pipeline/out/synthetic_capture/camera_intrinsics.json \
      --platform-w 1.4 --platform-d 0.7 \
      --out biotech-lab-robotics/vision_pipeline/out/seg_filtered_demo
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from calibration import load_intrinsics
from geometry import camera_pose_to_rvec_tvec, quaternion_to_rotation_matrix
from labeling import label_mask
from platform_filter import filter_masks_to_platform, platform_silhouette_polygon
from segmentation import generate_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--camera-poses", required=True,
                     help="camera_poses.json from solve_camera_poses.py")
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--platform-w", type=float, required=True)
    ap.add_argument("--platform-d", type=float, required=True)
    ap.add_argument("--height-clearance", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    image_path = Path(args.image).resolve()
    poses = json.loads(Path(args.camera_poses).read_text())["poses"]
    pose = next((p for p in poses if Path(p["image"]).resolve() == image_path), None)
    if pose is None:
        raise SystemExit(f"{image_path} not found in {args.camera_poses} "
                          "(did it fail to solve in Stage 1?)")

    intrinsics = load_intrinsics(args.intrinsics)
    cam = intrinsics[pose["camera_id"]]

    R_cw = quaternion_to_rotation_matrix(np.array(pose["quaternion_wxyz"]))
    position_world = np.array(pose["position_world"])
    rvec, tvec = camera_pose_to_rvec_tvec(R_cw, position_world)

    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]

    polygon = platform_silhouette_polygon(
        rvec, tvec, cam.K, cam.dist, args.platform_w, args.platform_d,
        height_clearance=args.height_clearance, image_size=(w, h))
    print(f"platform silhouette polygon: {len(polygon)} hull points")

    print("running SAM automatic mask generation...")
    all_masks = generate_masks(image_rgb)
    print(f"{len(all_masks)} candidates before platform filtering")

    on_platform = filter_masks_to_platform(all_masks, polygon, (h, w))
    print(f"{len(on_platform)} candidates after platform filtering")

    print("labeling...")
    results = []
    for inst in on_platform:
        labels = label_mask(image_rgb, inst.mask, inst.bbox_xyxy)
        top = labels[0] if labels else None
        print(f"  bbox={inst.bbox_xyxy} area={inst.area_px:6d} "
              f"-> {top.text if top else '?'} ({top.score:.2f})" if top else "  no label")
        results.append({
            "bbox_xyxy": inst.bbox_xyxy, "area_px": inst.area_px,
            "labels": [{"text": l.text, "score": l.score} for l in labels],
        })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay = image_bgr.copy()
    cv2.polylines(overlay, [polygon.astype(np.int32)], True, (0, 255, 255), 2)
    rng = np.random.default_rng(0)
    for inst in on_platform:
        color = tuple(int(c) for c in rng.integers(60, 255, 3))
        colored = np.zeros_like(image_bgr)
        colored[inst.mask] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.45, 0)
        x0, y0, x1, y1 = inst.bbox_xyxy
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)
    (out_dir / "instances.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir / 'overlay.png'} (yellow = platform silhouette used for filtering)")


if __name__ == "__main__":
    main()

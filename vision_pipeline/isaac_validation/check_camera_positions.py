"""Compare Stage 1's recovered camera centers against the exact eye
positions render_isaac_validation_capture.py placed its cameras at.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/isaac_validation/check_camera_positions.py \
      --capture-dir biotech-lab-robotics/vision_pipeline/isaac_validation/out
(run solve_camera_poses.py into <capture-dir>/poses first)
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--poses-subdir", default="poses")
    args = ap.parse_args()
    d = Path(args.capture_dir)
    gt = json.loads((d / "ground_truth_camera_positions.json").read_text())["positions"]
    solved = {Path(p["image"]).stem: p for p in
              json.loads((d / args.poses_subdir / "camera_poses.json").read_text())["poses"]}
    errs = []
    for name, true_pos in gt.items():
        p = solved.get(name)
        if p is None:
            print(f"{name:26s} not solved")
            continue
        err = 1000 * np.linalg.norm(np.array(p["position_world"]) - np.array(true_pos))
        errs.append(err)
        print(f"{name:26s} markers={sorted(p['marker_ids_used'])}  "
              f"reproj={p['mean_reproj_error_px']:.2f}px  position error={err:6.1f} mm")
    if errs:
        print(f"\nsolved {len(errs)}/{len(gt)}; position error median {np.median(errs):.1f} mm, max {max(errs):.1f} mm")


if __name__ == "__main__":
    main()

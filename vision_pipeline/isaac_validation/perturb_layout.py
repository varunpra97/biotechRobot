"""Simulate a hand-measured marker layout: perturb each rigid marker body
(a flat marker, or a whole calibration cube) by a random rigid motion of
roughly tape-measure size, so bundle adjustment can be tested against the
realistic case where the layout is only approximately known.

  python biotech-lab-robotics/vision_pipeline/isaac_validation/perturb_layout.py \
      --layout CAPTURE/marker_layout.json --out CAPTURE/marker_layout_measured.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bundle_adjust import build_bodies  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pos-sigma-mm", type=float, default=3.0, help="per-axis in-plane position error")
    ap.add_argument("--height-sigma-mm", type=float, default=1.0)
    ap.add_argument("--yaw-sigma-deg", type=float, default=1.0)
    ap.add_argument("--tilt-sigma-deg", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only-markers", type=int, nargs="+",
                    help="perturb only the bodies containing these markers (e.g. roughly placed cubes)")
    args = ap.parse_args()

    data = json.loads(Path(args.layout).read_text())
    corners = {int(k): np.array(v["corners_3d"]) for k, v in data["markers"].items()}
    groups = data.get("rigid_groups", [])
    bodies, _ = build_bodies(corners, groups)
    rng = np.random.default_rng(args.seed)

    out = {}
    for ids in bodies:
        pts = np.concatenate([corners[m] for m in ids])
        if args.only_markers and not set(ids) & set(args.only_markers):
            for k, m in enumerate(ids):
                out[str(m)] = {"corners_3d": pts[4 * k:4 * k + 4].tolist()}
            continue
        c = pts.mean(axis=0)
        R = Rotation.from_euler("xyz", [rng.normal(0, args.tilt_sigma_deg), rng.normal(0, args.tilt_sigma_deg),
                                        rng.normal(0, args.yaw_sigma_deg)], degrees=True)
        t = np.array([rng.normal(0, args.pos_sigma_mm), rng.normal(0, args.pos_sigma_mm),
                      rng.normal(0, args.height_sigma_mm)]) / 1000
        moved = R.apply(pts - c) + c + t
        for k, m in enumerate(ids):
            out[str(m)] = {"corners_3d": moved[4 * k:4 * k + 4].tolist()}
    Path(args.out).write_text(json.dumps({**data, "markers": out}, indent=2))
    print(f"perturbed {len(bodies)} bodies -> {args.out}")


if __name__ == "__main__":
    main()

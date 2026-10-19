"""One-time per-camera intrinsics calibration from a folder of chessboard
photos. Produces a camera_intrinsics.json consumable by calibration.py.

Usage:
  python scripts/calibrate_camera.py \
      --images path/to/chessboard_photos/*.jpg \
      --pattern-cols 9 --pattern-rows 6 --square-size-m 0.025 \
      --camera-id phone_main \
      --out config/camera_intrinsics.json
"""
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration import calibrate_chessboard, load_intrinsics, save_intrinsics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True,
                     help="chessboard image paths or glob patterns")
    ap.add_argument("--pattern-cols", type=int, required=True,
                     help="inner corners per row")
    ap.add_argument("--pattern-rows", type=int, required=True,
                     help="inner corners per column")
    ap.add_argument("--square-size-m", type=float, required=True)
    ap.add_argument("--camera-id", required=True)
    ap.add_argument("--out", required=True,
                     help="output JSON; merged with existing entries if present")
    args = ap.parse_args()

    paths = [p for pattern in args.images for p in glob.glob(pattern)]
    if not paths:
        raise SystemExit("no images matched --images")

    intrinsics, rms = calibrate_chessboard(
        paths, (args.pattern_cols, args.pattern_rows),
        args.square_size_m, args.camera_id)

    print(f"calibrated {args.camera_id} from {len(paths)} images, "
          f"rms reprojection error = {rms:.4f}px")
    if rms > 1.0:
        print("warning: rms > 1px is high for a chessboard calibration; "
              "check focus, board flatness, and view coverage")

    out_path = Path(args.out)
    cameras = load_intrinsics(out_path) if out_path.exists() else {}
    cameras[args.camera_id] = intrinsics
    save_intrinsics(out_path, cameras)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

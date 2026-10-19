"""Stage 2 over a whole capture: for every view Stage 1 solved, propose
SAM masks, keep the ones over the platform, drop the bench surface itself,
label the rest with CLIP, and save per-view silhouettes for Stage 4.

  .venv-vision/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/segment_capture.py \
      --capture-dir biotech-lab-robotics/vision_pipeline/isaac_validation/out --platform-w 1.4 --platform-d 0.7

Writes <capture-dir>/segmentation/<view>.npz (foreground bool mask +
instance masks) and <view>.json (instance labels), plus an overlay PNG.
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
from labeling import label_mask
from platform_filter import filter_masks_to_platform, platform_silhouette_polygon
from segmentation import dedup_masks, drop_tile_clipped, generate_masks

# A mask covering more than this fraction of the projected bench-top quad
# is the bench surface (or a merge of most of it), not an object on it.
# Coverage, not raw area: at a 15 deg view the bench top projects to a thin
# strip, and a large object's mask can exceed a fraction of the strip's
# AREA while still covering only a small part of the bench.
BENCH_COVER_FRACTION = 0.4


def bench_top_mask(rvec, tvec, K, dist, w, d, image_shape) -> np.ndarray:
    hw, hd = w / 2, d / 2
    quad = np.array([[-hw, -hd, 0], [hw, -hd, 0], [hw, hd, 0], [-hw, hd, 0]], dtype=np.float64)
    proj, _ = cv2.projectPoints(quad, rvec, tvec, K, dist)
    m = np.zeros(image_shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(proj.reshape(-1, 2).astype(np.float32)).astype(np.int32), 1)
    return m.astype(bool)


def cached_masks(cache: Path, image_rgb: np.ndarray, tiles: int):
    """SAM is the slow step (~5 passes per view with tiling); cache its
    output per view so changing any downstream filter doesn't re-run it.
    Caches every proposal, before deduplication (see main())."""
    from segmentation import _to_instance
    if cache.exists():
        d = np.load(cache)
        return [_to_instance(m, s) for m, s in zip(d["masks"], d["scores"])]
    masks = generate_masks(image_rgb, tiles=tiles, dedup=False)
    np.savez_compressed(cache, masks=np.array([m.mask for m in masks]) if masks
                        else np.zeros((0, *image_rgb.shape[:2]), bool),
                        scores=np.array([m.score for m in masks]))
    return masks


GPU_BUSY_EXIT = 75   # EX_TEMPFAIL: the caller re-runs later; finished views are cached


def defer_if_gpu_busy(guard: str) -> None:
    """Shared GPU: never compete with another job. If it's busy, EXIT (and
    free the GPU) rather than sleep: a paused run still holds SAM and the
    CUDA context -- ~9GB measured, on a 12GB card, while the other agent's
    Isaac job was running. Re-run after the guard clears; the SAM cache
    makes finished views free."""
    import os
    import shutil
    import subprocess
    # shutil.which, not "bash": on Windows CreateProcess searches System32
    # first and finds WSL's bash.exe, where the guard can't see Windows
    # processes. The guard must not count this job itself: pass our PID.
    r = subprocess.run([shutil.which("bash") or "bash", guard], capture_output=True, text=True,
                       env={**os.environ, "GPU_GUARD_SELF_PID": str(os.getpid())})
    if r.returncode == 0:
        print(f"GPU busy, deferring (exit {GPU_BUSY_EXIT}): {r.stdout.strip()}", flush=True)
        sys.exit(GPU_BUSY_EXIT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--platform-w", type=float, required=True)
    ap.add_argument("--platform-d", type=float, required=True)
    ap.add_argument("--height-clearance", type=float, default=0.3)
    ap.add_argument("--seg-max-side", type=int, default=1600,
                     help="segment a copy downscaled to this long side. Finding objects doesn't need "
                          "12MP (SAM resizes to 1024 anyway) and full-res masks cost ~12MB each; "
                          "sub-pixel measurement uses the full-res images later, not these masks")
    ap.add_argument("--gpu-guard", help="script that exits 0 while someone else is using the GPU "
                                        "(isaac_validation/gpu_guard.sh); checked before every view")
    ap.add_argument("--tiles", type=int, default=2,
                     help="also run SAM on a tiles x tiles grid of crops (small objects)")
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    intrinsics = load_intrinsics(cap / "camera_intrinsics.json")
    poses = json.loads((cap / "poses" / "camera_poses.json").read_text())["poses"]
    out = cap / "segmentation"
    # sam_raw, not sam_cache: older caches hold already-deduplicated masks
    (out / "sam_raw").mkdir(parents=True, exist_ok=True)

    for pose in poses:
        if args.gpu_guard:
            defer_if_gpu_busy(args.gpu_guard)
        name = Path(pose["image"]).stem
        cam = intrinsics[pose["camera_id"]]
        R_cw = quaternion_to_rotation_matrix(np.array(pose["quaternion_wxyz"]))
        rvec, tvec = camera_pose_to_rvec_tvec(R_cw, np.array(pose["position_world"]))

        image_bgr = cv2.imread(str(resolve_shot_image(cap, pose["image"])))
        scale = min(1.0, args.seg_max_side / max(image_bgr.shape[:2]))
        if scale < 1.0:
            image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_bgr.shape[:2]
        K = cam.K.copy()
        K[:2] *= scale  # intrinsics of the downscaled image

        polygon = platform_silhouette_polygon(rvec, tvec, K, cam.dist, args.platform_w,
                                              args.platform_d, args.height_clearance, (w, h))
        sam = cached_masks(out / "sam_raw" / f"{name}.npz", image_rgb, args.tiles)
        sam = drop_tile_clipped(sam, (h, w), args.tiles)
        masks = filter_masks_to_platform(sam, polygon, (h, w))
        bench = bench_top_mask(rvec, tvec, K, cam.dist, args.platform_w, args.platform_d, (h, w))
        bench_px = max(int(bench.sum()), 1)
        masks = [m for m in masks if np.logical_and(m.mask, bench).sum() / bench_px < BENCH_COVER_FRACTION]
        # Deduplicate only now: a bench-surface mask contains every object on
        # the bench, so deduplicating before it was removed dropped them all.
        masks = dedup_masks(masks)

        foreground = np.zeros((h, w), bool)
        instances = []
        for m in masks:
            foreground |= m.mask
            labels = label_mask(image_rgb, m.mask, m.bbox_xyxy)
            instances.append({"bbox_xyxy": m.bbox_xyxy, "area_px": m.area_px,
                              "labels": [{"text": l.text, "score": l.score} for l in labels]})

        np.savez_compressed(out / f"{name}.npz", foreground=foreground, scale=scale,
                            instance_masks=np.array([m.mask for m in masks]) if masks else np.zeros((0, h, w), bool))
        (out / f"{name}.json").write_text(json.dumps(instances, indent=2))

        overlay = image_bgr.copy()
        overlay[foreground] = (0.5 * overlay[foreground] + 0.5 * np.array([0, 160, 255])).astype(np.uint8)
        cv2.polylines(overlay, [polygon.astype(np.int32)], True, (0, 255, 255), 2)
        cv2.imwrite(str(out / f"{name}_overlay.jpg"), overlay)
        print(f"{name:26s} {len(masks):3d} object masks, foreground {foreground.mean() * 100:.1f}% of image", flush=True)


if __name__ == "__main__":
    main()

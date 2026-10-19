"""Stage 1: recover every camera's pose in the platform frame from
fiducial markers. Ties together calibration.py (intrinsics),
markers.py (layout + detection) and capture.py (which image goes with
which camera).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from calibration import CameraIntrinsics
from capture import CaptureManifest
from geometry import rotation_matrix_to_quaternion, world_to_camera_to_pose
from markers import MarkerLayout, detect_markers

MIN_POINTS_FOR_PNP = 4
RANSAC_REPROJ_ERROR_PX = 4.0


@dataclass
class CameraPoseResult:
    image_path: Path
    camera_id: str
    success: bool
    reason: str = ""
    marker_ids_used: list[int] = None
    num_points: int = 0
    R_cw: np.ndarray | None = None          # camera-to-world rotation
    position_world: np.ndarray | None = None  # camera center, platform frame
    quaternion_wxyz: np.ndarray | None = None
    rvec: np.ndarray | None = None          # raw world->camera (for QA/overlay)
    tvec: np.ndarray | None = None
    mean_reproj_error_px: float | None = None
    max_reproj_error_px: float | None = None


def solve_camera_pose(
    obj_pts: np.ndarray, img_pts: np.ndarray,
    K: np.ndarray, dist: np.ndarray,
) -> tuple[bool, str, np.ndarray | None, np.ndarray | None, float, float]:
    """RANSAC PnP + LM refinement on inliers. Returns
    (success, reason_if_failed, rvec, tvec, mean_err_px, max_err_px).
    """
    if len(obj_pts) < MIN_POINTS_FOR_PNP:
        return False, f"only {len(obj_pts)} correspondences (need >= {MIN_POINTS_FOR_PNP})", None, None, 0.0, 0.0

    # SQPnP as the minimal solver: ITERATIVE on RANSAC's small subsets of
    # (often coplanar) marker corners failed outright on ~30-40% of noisy
    # inputs in testing; SQPnP is globally optimal and handles planar
    # configurations.
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, K, dist,
        flags=cv2.SOLVEPNP_SQPNP,
        reprojectionError=RANSAC_REPROJ_ERROR_PX,
        iterationsCount=200,
    )
    if not ok or inliers is None or len(inliers) < MIN_POINTS_FOR_PNP:
        return False, "PnP RANSAC found too few inliers", None, None, 0.0, 0.0

    inlier_idx = inliers.reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(
        obj_pts[inlier_idx], img_pts[inlier_idx], K, dist, rvec, tvec)

    reproj, _ = cv2.projectPoints(obj_pts[inlier_idx], rvec, tvec, K, dist)
    err = np.linalg.norm(reproj.reshape(-1, 2) - img_pts[inlier_idx], axis=1)
    return True, "", rvec, tvec, float(err.mean()), float(err.max())


def solve_all_camera_poses(
    manifest: CaptureManifest,
    intrinsics: dict[str, CameraIntrinsics],
    marker_layout: MarkerLayout,
) -> list[CameraPoseResult]:
    results = []
    for shot in manifest.shots:
        if shot.camera_id not in intrinsics:
            results.append(CameraPoseResult(
                image_path=shot.image_path, camera_id=shot.camera_id,
                success=False,
                reason=f"no intrinsics loaded for camera_id={shot.camera_id!r}",
                marker_ids_used=[]))
            continue

        image = cv2.imread(str(shot.image_path))
        if image is None:
            results.append(CameraPoseResult(
                image_path=shot.image_path, camera_id=shot.camera_id,
                success=False, reason="could not read image",
                marker_ids_used=[]))
            continue

        cam = intrinsics[shot.camera_id]
        detected = detect_markers(image)
        obj_pts, img_pts, used = marker_layout.point_correspondences(detected)

        ok, reason, rvec, tvec, mean_err, max_err = solve_camera_pose(
            obj_pts, img_pts, cam.K, cam.dist)

        if not ok:
            results.append(CameraPoseResult(
                image_path=shot.image_path, camera_id=shot.camera_id,
                success=False, reason=reason, marker_ids_used=used,
                num_points=len(obj_pts)))
            continue

        R_cw, position_world = world_to_camera_to_pose(rvec, tvec)
        quat = rotation_matrix_to_quaternion(R_cw)

        results.append(CameraPoseResult(
            image_path=shot.image_path, camera_id=shot.camera_id,
            success=True, marker_ids_used=used, num_points=len(obj_pts),
            R_cw=R_cw, position_world=position_world,
            quaternion_wxyz=quat, rvec=rvec, tvec=tvec,
            mean_reproj_error_px=mean_err, max_reproj_error_px=max_err,
        ))
    return results

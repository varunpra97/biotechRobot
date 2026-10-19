"""QA diagnostics for the camera-pose stage: a reprojection-error summary
and per-image annotated overlays, so a bad camera pose is visible before
it poisons every later reconstruction stage.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from calibration import CameraIntrinsics
from markers import MarkerLayout, detect_markers
from pose_solver import CameraPoseResult

# Reject a view outright above this mean reprojection error (pixels).
# 1-2px is typical for a well-detected ArUco marker at this scale; above
# ~3px usually means a wrong correspondence or a bad marker measurement.
REPROJ_ERROR_WARN_PX = 1.5
REPROJ_ERROR_FAIL_PX = 3.0


def summarize(results: list[CameraPoseResult]) -> str:
    lines = [
        f"{'image':40s} {'camera':12s} {'ok':4s} {'markers':8s} "
        f"{'pts':5s} {'mean_err':9s} {'max_err':9s} note"
    ]
    for r in results:
        if not r.success:
            lines.append(
                f"{r.image_path.name:40s} {r.camera_id:12s} {'NO':4s} "
                f"{'-':8s} {'-':5s} {'-':9s} {'-':9s} {r.reason}")
            continue
        flag = ""
        if r.mean_reproj_error_px > REPROJ_ERROR_FAIL_PX:
            flag = "FAIL: reprojection error too high"
        elif r.mean_reproj_error_px > REPROJ_ERROR_WARN_PX:
            flag = "warn: reprojection error elevated"
        lines.append(
            f"{r.image_path.name:40s} {r.camera_id:12s} {'yes':4s} "
            f"{len(r.marker_ids_used):8d} {r.num_points:5d} "
            f"{r.mean_reproj_error_px:9.3f} {r.max_reproj_error_px:9.3f} {flag}")
    return "\n".join(lines)


def draw_overlay(
    image_path: Path, camera: CameraIntrinsics, marker_layout: MarkerLayout,
    result: CameraPoseResult, out_path: Path,
) -> None:
    """Draw detected marker corners, their reprojection from the solved
    pose, and a world-axis gizmo at the platform origin, onto a copy of
    the image. Reprojected corners drawn right on top of detected ones is
    the fastest visual check that a pose is actually correct.
    """
    image = cv2.imread(str(image_path))
    detected = detect_markers(image)

    for marker_id, px in detected.items():
        pts = px.astype(int)
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
        cv2.putText(image, str(marker_id), tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if result.success:
        obj_pts, _, used = marker_layout.point_correspondences(detected)
        if len(obj_pts):
            reproj, _ = cv2.projectPoints(
                obj_pts, result.rvec, result.tvec, camera.K, camera.dist)
            for pt in reproj.reshape(-1, 2).astype(int):
                cv2.drawMarker(image, tuple(pt), (0, 0, 255),
                                markerType=cv2.MARKER_CROSS, markerSize=12,
                                thickness=2)
        cv2.drawFrameAxes(image, camera.K, camera.dist,
                           result.rvec, result.tvec, 0.1)

    cv2.putText(
        image,
        f"mean_err={result.mean_reproj_error_px:.2f}px" if result.success
        else f"FAILED: {result.reason}",
        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 0) if result.success else (0, 0, 255), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)

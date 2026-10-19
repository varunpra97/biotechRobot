"""Synthetic validation of the camera-pose math (correspondence pooling,
PnP, world/camera frame conversion) without needing real photos or
OpenCV's marker detector: corners are projected analytically with
cv2.projectPoints from a known ground-truth camera pose, so the only
thing under test is our own code, not ArUco detection.

Run directly: python tests/test_pose_solver_synthetic.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry import (
    camera_pose_to_rvec_tvec,
    look_at_rotation,
    rotation_matrix_to_quaternion,
    world_to_camera_to_pose,
)
from markers import MarkerLayout, corners_from_center_axes
from pose_solver import solve_camera_pose

RNG = np.random.default_rng(0)


def _make_layout() -> MarkerLayout:
    # Three markers spanning different positions/orientations so the
    # correspondences aren't all coplanar (a degenerate case for PnP).
    corners_3d = {
        0: corners_from_center_axes(
            center=[-0.6, -0.3, 0.0], x_axis=[1, 0, 0], y_axis=[0, 1, 0],
            side_length_m=0.05),
        1: corners_from_center_axes(
            center=[0.6, -0.3, 0.0], x_axis=[1, 0, 0], y_axis=[0, 1, 0],
            side_length_m=0.05),
        2: corners_from_center_axes(
            center=[0.0, 0.3, 0.2], x_axis=[1, 0, 0], y_axis=[0, 0, 1],
            side_length_m=0.05),
    }
    return MarkerLayout(corners_3d=corners_3d)


def _project_layout(layout, rvec, tvec, K, dist):
    detected = {}
    for marker_id, pts3d in layout.corners_3d.items():
        proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
        detected[marker_id] = proj.reshape(4, 2)
    return detected


def test_recovers_known_camera_pose():
    layout = _make_layout()
    K = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1.0]])
    dist = np.zeros(5)

    # Ground truth: camera above and in front of the platform, looking
    # down at the marker cluster near the world origin.
    true_position_world = np.array([0.0, -1.2, 0.8])
    R_cw_true = look_at_rotation(true_position_world, target=(0.0, 0.0, 0.1))
    rvec_true, tvec_true = camera_pose_to_rvec_tvec(R_cw_true, true_position_world)

    detected = _project_layout(layout, rvec_true, tvec_true, K, dist)
    obj_pts, img_pts, used = layout.point_correspondences(detected)
    assert len(used) == 3
    assert obj_pts.shape == (12, 3)

    # Sanity-check the test fixture itself: every marker point must be in
    # front of the camera (positive depth), otherwise projectPoints
    # silently "projects" a point behind the camera and the fixture isn't
    # testing what it claims to.
    R_wc_true, _ = cv2.Rodrigues(rvec_true)
    depths = (R_wc_true @ obj_pts.T + tvec_true.reshape(3, 1))[2, :]
    assert np.all(depths > 0), f"fixture places points behind the camera: {depths}"

    ok, reason, rvec, tvec, mean_err, max_err = solve_camera_pose(
        obj_pts, img_pts, K, dist)
    assert ok, reason
    assert mean_err < 1e-3, f"mean reprojection error too high: {mean_err}"

    R_cw, position_world = world_to_camera_to_pose(rvec, tvec)
    assert np.allclose(position_world, true_position_world, atol=1e-3), (
        position_world, true_position_world)
    assert np.allclose(R_cw, R_cw_true, atol=1e-3)

    quat = rotation_matrix_to_quaternion(R_cw)
    assert np.isclose(np.linalg.norm(quat), 1.0, atol=1e-6)


def test_rejects_insufficient_correspondences():
    layout = _make_layout()
    K = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1.0]])
    # Only 2 points (less than the 4 minimum for PnP).
    obj_pts = layout.corners_3d[0][:2]
    img_pts = np.array([[640.0, 360.0], [650.0, 360.0]])
    ok, reason, *_ = solve_camera_pose(obj_pts, img_pts, K, np.zeros(5))
    assert not ok
    assert "correspondences" in reason


if __name__ == "__main__":
    test_recovers_known_camera_pose()
    test_rejects_insufficient_correspondences()
    print("PASS")

"""Small shared geometry helpers (rotation conventions) used across the
camera-pose and (later) object-pose stages, kept in one place so every
stage agrees on the same (w, x, y, z) quaternion convention that
`biotech-lab-robotics/sim/lab_scene.py` uses for USD orient ops.
"""
from __future__ import annotations

import numpy as np


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (w, x, y, z), unit quaternion."""
    m = np.asarray(R, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quaternion_to_rotation_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """(w, x, y, z) unit quaternion -> 3x3 rotation matrix. Inverse of
    rotation_matrix_to_quaternion.
    """
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def look_at_rotation(
    camera_pos, target, world_up=(0.0, 0.0, 1.0)
) -> np.ndarray:
    """Camera-to-world rotation (R_cw) for a camera at `camera_pos` looking
    toward `target`, OpenCV axes (X right, Y down, Z forward). Built from
    cross products rather than hand-picked Euler angles so it's correct
    by construction for any camera placement.
    """
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)

    world_up = np.asarray(world_up, dtype=np.float64)
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # forward is (anti-)parallel to world_up -- a straight-down or
        # straight-up shot. Any up reference not parallel to forward works;
        # world X is never parallel to world_up=(0,0,1) by construction.
        world_up = np.array([1.0, 0.0, 0.0])
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
    right /= right_norm
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)  # columns = camera axes in world


def camera_pose_to_rvec_tvec(
    R_cw: np.ndarray, position_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of world_to_camera_to_pose: given a camera's own pose in
    the world/platform frame, return the (rvec, tvec) solvePnP/
    projectPoints convention expects for the world->camera transform.
    """
    import cv2
    R_wc = np.asarray(R_cw, dtype=np.float64).T
    t_wc = -R_wc @ np.asarray(position_world, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R_wc)
    return rvec, t_wc


def world_to_camera_to_pose(
    rvec: np.ndarray, tvec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """solvePnP gives (rvec, tvec) for the world->camera transform:
    p_cam = R_wc @ p_world + tvec. Returns (R_cw, position_world): the
    camera's own orientation (camera-to-world rotation) and its position,
    both expressed in the world/platform frame.
    """
    import cv2
    R_wc, _ = cv2.Rodrigues(rvec)
    R_cw = R_wc.T
    position_world = -R_cw @ tvec.reshape(3)
    return R_cw, position_world

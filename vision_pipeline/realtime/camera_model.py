"""Wrist cameras as pinhole models posed by the arm, not by markers.

A wrist camera is bolted to the hand: its pose is the hand's pose (from the
arm's joint encoders through forward kinematics, or the simulator's hand
prim) times a fixed hand-to-camera transform. The transform and the lens
come from the simulation's own definition (simulation/sim/lab_scene.py
WRIST_CAMERAS, read here, never copied), so this model follows any change
to the mount. For the real robot the same transform comes from the mount
CAD and a hand-eye calibration.

Conventions: OpenCV camera (x right, y down, z forward). USD cameras look
down their local -Z with +Y up; converting is a flip of the y and z axes.
Poses are 4x4 world-from-frame matrices.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parents[2] / "simulation" / "sim"
USD_DEFAULT_HORIZONTAL_APERTURE_MM = 20.955     # lab_scene sets focal length only
USD_TO_CV = np.diag([1.0, -1.0, -1.0])


def _lab_scene():
    if str(SIM_DIR) not in sys.path:
        sys.path.insert(0, str(SIM_DIR))
    import lab_scene                       # numpy-only at import time; Isaac is imported lazily inside it
    return lab_scene


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_to_matrix(pose7) -> np.ndarray:
    """(x, y, z, qw, qx, qy, qz) -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_matrix(pose7[3:7])
    T[:3, 3] = pose7[:3]
    return T


def interpolate_pose7(p0, p1, a):
    """Pose (x, y, z, qw, qx, qy, qz) a of the way from p0 to p1 (lerp + nlerp).
    Images are taken at their own time, not the controller's: pose each one
    by interpolating the hand pose to its timestamp."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    q1 = p1[3:] * np.sign(p0[3:] @ p1[3:])
    q = (1 - a) * p0[3:] + a * q1
    return np.r_[(1 - a) * p0[:3] + a * p1[:3], q / np.linalg.norm(q)]


@dataclass
class PinholeCamera:
    name: str
    K: np.ndarray                 # 3x3
    size: tuple                   # (w, h)
    T_hand_cam: np.ndarray        # 4x4, OpenCV camera frame in the hand frame (world frame if fixed)
    fixed: bool = False           # a static camera: T_hand_cam is its world pose

    def world_from_cam(self, T_world_hand: np.ndarray = None) -> np.ndarray:
        return self.T_hand_cam if self.fixed else T_world_hand @ self.T_hand_cam

    def project(self, T_world_hand: np.ndarray, pts_world: np.ndarray):
        """(pixels (N,2), depth z (N,)) of world points."""
        T = np.linalg.inv(self.world_from_cam(T_world_hand))
        pc = pts_world @ T[:3, :3].T + T[:3, 3]
        z = pc[:, 2]
        zs = np.where(z > 1e-6, z, 1e-6)
        uv = np.stack([self.K[0, 0] * pc[:, 0] / zs + self.K[0, 2], self.K[1, 1] * pc[:, 1] / zs + self.K[1, 2]], 1)
        return uv, z


def wrist_cameras(resolution=(640, 480)) -> dict[str, PinholeCamera]:
    """The simulated wrist cameras, from lab_scene.WRIST_CAMERAS."""
    ls = _lab_scene()
    w, h = resolution
    out = {}
    for name, cfg in ls.WRIST_CAMERAS.items():
        R_usd = ls.wrist_camera_frame(cfg["eye"], cfg["target"], cfg.get("right"))   # USD camera axes in hand frame
        T = np.eye(4)
        T[:3, :3] = R_usd @ USD_TO_CV
        T[:3, 3] = cfg["eye"]
        f = w * float(cfg["focal"]) / USD_DEFAULT_HORIZONTAL_APERTURE_MM          # square pixels
        K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
        out[name] = PinholeCamera(name, K, (w, h), T)
    return out


def exterior_camera(resolution=(640, 480)) -> PinholeCamera:
    """The simulator's fixed exterior camera (platform_cam1): Replicator's
    look-at camera with world +Z up, at lab_scene.EXTERIOR_CAM_EYE."""
    ls = _lab_scene()
    w, h = resolution
    eye, tgt = np.asarray(ls.EXTERIOR_CAM_EYE, float), np.asarray(ls.EXTERIOR_CAM_TARGET, float)
    z = (tgt - eye) / np.linalg.norm(tgt - eye)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    T[:3, 3] = eye
    f = w * float(ls.EXTERIOR_CAM_FOCAL) / USD_DEFAULT_HORIZONTAL_APERTURE_MM
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    return PinholeCamera("platform_cam1", K, (w, h), T, fixed=True)

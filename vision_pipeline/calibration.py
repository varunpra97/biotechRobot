"""Per-camera intrinsics: load from a one-time OpenCV calibration, or
produce that calibration file from a set of chessboard images.

Intrinsics are assumed known ahead of time (see README) -- this module
does not estimate them from the scene photos themselves.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    camera_id: str
    K: np.ndarray          # 3x3
    dist: np.ndarray       # (N,) distortion coefficients, OpenCV convention
    image_size: tuple[int, int]  # (width, height)

    def to_dict(self) -> dict:
        return {
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "image_size": list(self.image_size),
        }

    @staticmethod
    def from_dict(camera_id: str, d: dict) -> "CameraIntrinsics":
        return CameraIntrinsics(
            camera_id=camera_id,
            K=np.asarray(d["K"], dtype=np.float64).reshape(3, 3),
            dist=np.asarray(d["dist"], dtype=np.float64).reshape(-1),
            image_size=(int(d["image_size"][0]), int(d["image_size"][1])),
        )


def load_intrinsics(path: str | Path) -> dict[str, CameraIntrinsics]:
    """Load {camera_id: CameraIntrinsics} from a JSON file shaped
    {"cameras": {camera_id: {K, dist, image_size}}}.
    """
    data = json.loads(Path(path).read_text())
    return {cid: CameraIntrinsics.from_dict(cid, d)
            for cid, d in data["cameras"].items()}


def save_intrinsics(path: str | Path, cameras: dict[str, CameraIntrinsics]) -> None:
    data = {"cameras": {cid: c.to_dict() for cid, c in cameras.items()}}
    Path(path).write_text(json.dumps(data, indent=2))


def calibrate_chessboard(
    image_paths: list[str | Path],
    pattern_size: tuple[int, int],
    square_size_m: float,
    camera_id: str,
) -> tuple[CameraIntrinsics, float]:
    """Standard OpenCV chessboard calibration from several images of a
    flat checkerboard held at different angles in front of ONE camera.

    pattern_size: (cols, rows) of INNER corners (not squares).
    square_size_m: physical side length of one checkerboard square, meters.

    Returns (intrinsics, rms_reprojection_error_px). Raises if fewer than
    6 images yield a detected board (calibration is poorly conditioned
    below that).
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_m

    obj_points, img_points = [], []
    image_size = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)

    if len(obj_points) < 6:
        raise ValueError(
            f"chessboard found in only {len(obj_points)}/{len(image_paths)} "
            "images; need at least 6 well-spread views for a conditioned "
            "calibration")

    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None)

    return CameraIntrinsics(camera_id, K, dist.reshape(-1), image_size), rms

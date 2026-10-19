"""Fiducial marker layout and detection.

The reference object is a set of printed ArUco markers at fixed, hand-
measured 3D positions in the platform frame (see README's coordinate
convention). Pooling every visible marker's corners in an image gives
enough 3D-2D correspondences for a single, well-conditioned solvePnP per
image -- camera pose comes out directly in the platform frame, with no
separate SfM scale-ambiguity step.

Marker corner order follows OpenCV's ArUco convention: top-left,
top-right, bottom-right, bottom-left, as seen when the marker is held
upright per its own encoded orientation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DICTIONARY = cv2.aruco.DICT_4X4_50


@dataclass
class MarkerLayout:
    """marker_id -> 4x3 array of corner positions in the platform frame,
    in OpenCV's top-left/top-right/bottom-right/bottom-left order."""
    corners_3d: dict[int, np.ndarray]

    def point_correspondences(
        self, detected: dict[int, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Pool corners from every detected marker that's also in the
        layout. detected: {marker_id: 4x2 pixel corners}.

        Returns (object_points Nx3, image_points Nx2, marker_ids_used).
        """
        obj_pts, img_pts, used = [], [], []
        for marker_id, px in detected.items():
            if marker_id not in self.corners_3d:
                continue
            obj_pts.append(self.corners_3d[marker_id])
            img_pts.append(np.asarray(px, dtype=np.float64).reshape(4, 2))
            used.append(marker_id)
        if not obj_pts:
            return np.zeros((0, 3)), np.zeros((0, 2)), []
        return (np.concatenate(obj_pts, axis=0),
                np.concatenate(img_pts, axis=0),
                used)


def corners_from_center_axes(
    center: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray,
    side_length_m: float,
) -> np.ndarray:
    """Build the 4 corner positions (top-left, top-right, bottom-right,
    bottom-left) of a square marker lying flat in the plane spanned by
    x_axis/y_axis, centered at `center`. x_axis/y_axis need not be unit
    length or exactly orthogonal; they're orthonormalized here (x_axis is
    kept, y_axis is re-derived) -- convenient for a human-specified "facing
    direction" rather than a hand-built rotation matrix.
    """
    x = np.asarray(x_axis, dtype=np.float64)
    x = x / np.linalg.norm(x)
    y = np.asarray(y_axis, dtype=np.float64)
    y = y - np.dot(y, x) * x
    y = y / np.linalg.norm(y)
    h = side_length_m / 2.0
    c = np.asarray(center, dtype=np.float64)
    top_left = c - h * x + h * y
    top_right = c + h * x + h * y
    bottom_right = c + h * x - h * y
    bottom_left = c - h * x - h * y
    return np.stack([top_left, top_right, bottom_right, bottom_left])


def load_marker_layout(path: str | Path) -> MarkerLayout:
    """Load a marker layout from JSON. Each entry under "markers" may
    give either explicit "corners_3d" (4x3, authoritative) or the more
    human-friendly "center"/"x_axis"/"y_axis"/"side_length_m".
    """
    data = json.loads(Path(path).read_text())
    corners_3d: dict[int, np.ndarray] = {}
    for marker_id_str, spec in data["markers"].items():
        marker_id = int(marker_id_str)
        if "corners_3d" in spec:
            pts = np.asarray(spec["corners_3d"], dtype=np.float64).reshape(4, 3)
        else:
            pts = corners_from_center_axes(
                center=spec["center"],
                x_axis=spec["x_axis"],
                y_axis=spec["y_axis"],
                side_length_m=spec["side_length_m"],
            )
        corners_3d[marker_id] = pts
    return MarkerLayout(corners_3d=corners_3d)


def detect_markers(
    image: np.ndarray, dictionary_id: int = DEFAULT_DICTIONARY
) -> dict[int, np.ndarray]:
    """Detect ArUco markers in a BGR or grayscale image.

    Returns {marker_id: 4x2 pixel corners} in OpenCV's corner order.
    """
    gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3 else image)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    params = cv2.aruco.DetectorParameters()
    # OpenCV's default is no corner refinement (integer-ish corners).
    # Pose sensitivity to corner noise is high for a sparse marker set
    # (~50-100mm camera error per 1px at 1.5m with 4 coplanar markers,
    # measured on the Isaac validation capture), so sub-pixel corners are
    # not optional here.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    ids = np.asarray(ids).reshape(-1)
    return {int(marker_id): corner.reshape(4, 2)
            for marker_id, corner in zip(ids, corners)}

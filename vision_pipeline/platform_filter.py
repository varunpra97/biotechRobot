"""Stage 2c: keep only the segmentation masks that actually sit over the
platform, using Stage 1's solved camera pose -- discards background,
walls, the arm/pedestal, anything off to the side. SAM's automatic mode
(segmentation.py) has no notion of "the platform"; this is where that
geometric knowledge gets applied.
"""
from __future__ import annotations

import cv2
import numpy as np

from segmentation import InstanceMask


def platform_silhouette_polygon(
    rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, dist: np.ndarray,
    platform_w: float, platform_d: float, height_clearance: float = 0.6,
    image_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """2D convex hull (in pixels) of the platform's footprint extruded
    up to `height_clearance` (to cover objects standing on it, not just
    the bench surface itself), as seen from this camera pose.

    Platform frame per README: origin at bench-top center, Z up, so the
    box spans x in [-w/2, w/2], y in [-d/2, d/2], z in [0, height_clearance].
    """
    hw, hd = platform_w / 2.0, platform_d / 2.0
    corners = np.array([
        [x, y, z]
        for x in (-hw, hw) for y in (-hd, hd) for z in (0.0, height_clearance)
    ], dtype=np.float64)

    R_wc, _ = cv2.Rodrigues(rvec)
    depths = (R_wc @ corners.T + tvec.reshape(3, 1))[2, :]
    if np.all(depths <= 0):
        return np.zeros((0, 2))

    proj, _ = cv2.projectPoints(corners, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    # Corners behind the camera would project nonsensically; this only
    # matters for a pathological camera placement (inside the platform
    # volume), which a real capture shouldn't produce, so just drop them.
    proj = proj[depths > 0]
    if len(proj) < 3:
        return np.zeros((0, 2))

    hull = cv2.convexHull(proj.astype(np.float32)).reshape(-1, 2)
    if image_size is not None:
        w, h = image_size
        hull[:, 0] = np.clip(hull[:, 0], 0, w - 1)
        hull[:, 1] = np.clip(hull[:, 1], 0, h - 1)
    return hull


def filter_masks_to_platform(
    masks: list[InstanceMask], polygon: np.ndarray, image_shape: tuple[int, int],
    min_overlap_fraction: float = 0.5,
) -> list[InstanceMask]:
    """Keep masks with at least `min_overlap_fraction` of their pixels
    inside `polygon` (from platform_silhouette_polygon). A fraction
    rather than a centroid-in-polygon test, so an object that's mostly
    on the platform but partly occluded/cropped at the polygon's edge
    isn't dropped outright.
    """
    if len(polygon) < 3:
        return []

    h, w = image_shape
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(poly_mask, polygon.astype(np.int32), 1)
    poly_mask = poly_mask.astype(bool)

    kept = []
    for m in masks:
        overlap = np.logical_and(m.mask, poly_mask).sum()
        if overlap / max(m.area_px, 1) >= min_overlap_fraction:
            kept.append(m)
    return kept

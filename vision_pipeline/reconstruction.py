"""Stages 3+4: multi-view visual hull over the platform volume, split into
3D objects, measured, labeled, and meshed.

Volumetric carving instead of per-object primitive fitting as the general
method: the scene's objects (bored racks, a rotor with tilted sleeves, a
pipette) are not cylinders and boxes, and a visual hull makes no shape
assumption. Doing it over the whole platform at once also solves cross-
view association for free -- an object is a 3D connected component of
the carved volume, not a guess about which 2D mask matches which.

Known limitation, inherent to visual hulls: concavities no silhouette can
see into (an open bin's interior, blind bores, a rotor's wells) are filled.
Outer dimensions and placement are what this measures; interior cavities
would need a different cue (depth, or a shape prior from the label).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class View:
    name: str
    K: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    foreground: np.ndarray            # HxW bool
    instance_masks: np.ndarray        # NxHxW bool
    instance_labels: list             # per instance: [{"text", "score"}, ...]


@dataclass
class ReconstructedObject:
    object_id: int
    voxel_count: int
    bbox_min: np.ndarray              # platform frame, meters
    bbox_max: np.ndarray
    label: str
    label_score: float
    label_votes: dict = field(default_factory=dict)
    mesh: object = None               # trimesh.Trimesh, platform frame


def _project(points: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coords (N,2) and a validity mask (in front of camera)."""
    R, _ = cv2.Rodrigues(view.rvec)
    cam = points @ R.T + view.tvec.reshape(1, 3)
    in_front = cam[:, 2] > 1e-3
    if np.any(view.dist):
        px, _ = cv2.projectPoints(points, view.rvec, view.tvec, view.K, view.dist)
        return px.reshape(-1, 2), in_front
    z = np.where(in_front, cam[:, 2], 1.0)
    u = view.K[0, 0] * cam[:, 0] / z + view.K[0, 2]
    v = view.K[1, 1] * cam[:, 1] / z + view.K[1, 2]
    return np.stack([u, v], axis=1), in_front


def carve(views: list[View], bounds_min, bounds_max, voxel_m: float,
          min_agree_fraction: float = 0.75, min_views: int = 4,
          mask_dilate_px: int = 2, chunk: int = 2_000_000, max_misses: int | None = None):
    """Vote-based visual hull. A voxel survives if, among the views it
    projects into, at least `min_agree_fraction` see foreground there (and
    at least `min_views` see it at all). A strict all-views hull lets a
    single view's missed silhouette delete an entire object; segmentation
    on white-on-white labware does miss objects in some views.

    Masks are dilated by `mask_dilate_px` so ~1-2px of residual camera-pose
    error doesn't shave real surface off.

    max_misses, if given, replaces the fraction rule with "at most this many
    views may disagree". The fraction rule inflates the hull: with 20 views,
    75% lets 5 views see background and the voxel still survives, so space
    just outside every object stays (+25-50mm on every dimension on the
    Isaac capture). Use the fraction rule only for a first, loose pass,
    then drop unreliable views (view_consistency) and carve strictly.

    Returns (occupancy bool grid [nx,ny,nz], grid origin (voxel 0 center)).
    """
    bmin, bmax = np.asarray(bounds_min, float), np.asarray(bounds_max, float)
    shape = np.ceil((bmax - bmin) / voxel_m).astype(int)
    origin = bmin + voxel_m / 2
    n = int(np.prod(shape))

    kernel = np.ones((2 * mask_dilate_px + 1,) * 2, np.uint8)
    fgs = [cv2.dilate(v.foreground.astype(np.uint8), kernel).astype(bool) for v in views]

    occ = np.zeros(n, bool)
    for start in range(0, n, chunk):
        idx = np.arange(start, min(start + chunk, n))
        ijk = np.stack(np.unravel_index(idx, shape), axis=1)
        pts = origin + ijk * voxel_m
        seen = np.zeros(len(idx), np.uint8)
        agree = np.zeros(len(idx), np.uint8)
        for view, fg in zip(views, fgs):
            px, ok = _project(pts, view)
            h, w = fg.shape
            u, v = np.round(px[:, 0]).astype(int), np.round(px[:, 1]).astype(int)
            inside = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            seen += inside
            agree[inside] += fg[v[inside], u[inside]]
        if max_misses is None:
            occ[idx] = (seen >= min_views) & (agree >= np.ceil(min_agree_fraction * seen))
        else:
            occ[idx] = (seen >= min_views) & (seen - agree <= max_misses)
    return occ.reshape(shape), origin


def view_consistency(occ: np.ndarray, origin: np.ndarray, voxel_m: float, views: list[View],
                     mask_dilate_px: int = 2, max_points: int = 200_000) -> dict[str, float]:
    """For each view, the fraction of the carved volume that projects onto
    that view's foreground. A view whose segmentation missed most objects
    (e.g. white labware from straight overhead, where SAM finds almost
    nothing) scores low, and in a strict carve it would veto real geometry,
    so it gets dropped instead of loosening the rule for every view.
    """
    pts = origin + np.argwhere(occ) * voxel_m
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    kernel = np.ones((2 * mask_dilate_px + 1,) * 2, np.uint8)
    scores = {}
    for view in views:
        fg = cv2.dilate(view.foreground.astype(np.uint8), kernel).astype(bool)
        px, ok = _project(pts, view)
        h, w = fg.shape
        u, v = np.round(px[:, 0]).astype(int), np.round(px[:, 1]).astype(int)
        inside = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        scores[view.name] = float(fg[v[inside], u[inside]].mean()) if inside.any() else 0.0
    return scores


def split_objects(occ: np.ndarray, voxel_m: float, min_volume_m3: float = 2e-6):
    """26-connected components of the carved volume, dropping specks."""
    labels, count = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    min_vox = max(1, int(min_volume_m3 / voxel_m ** 3))
    sizes = ndimage.sum_labels(occ, labels, index=np.arange(1, count + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_vox]
    return labels, keep


def component_mesh(labels: np.ndarray, comp_id: int, origin: np.ndarray, voxel_m: float):
    import trimesh
    from skimage import measure

    sl = ndimage.find_objects((labels == comp_id).astype(int))[0]
    sub = np.pad((labels[sl] == comp_id).astype(np.float32), 1)
    verts, faces, _, _ = measure.marching_cubes(sub, level=0.5)
    offset = np.array([s.start for s in sl]) - 1
    verts = origin + (verts + offset) * voxel_m
    return trimesh.Trimesh(verts, faces, process=True)


def label_component(points: np.ndarray, views: list[View], max_points: int = 4000):
    """Vote over views: project the component's voxels, find the 2D
    instance mask they overlap most, and weight that mask's CLIP labels by
    the overlap. A label is a hint (see labeling.py), so it's aggregated,
    not trusted from any single view.
    """
    if len(points) > max_points:
        points = points[np.random.default_rng(0).choice(len(points), max_points, replace=False)]
    votes: dict[str, float] = {}
    for view in views:
        if len(view.instance_masks) == 0:
            continue
        px, ok = _project(points, view)
        _, h, w = view.instance_masks.shape
        u, v = np.round(px[:, 0]).astype(int), np.round(px[:, 1]).astype(int)
        inside = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if inside.sum() < 10:
            continue
        hits = view.instance_masks[:, v[inside], u[inside]].mean(axis=1)
        best = int(np.argmax(hits))
        if hits[best] < 0.3:
            continue
        for lab in view.instance_labels[best]:
            votes[lab["text"]] = votes.get(lab["text"], 0.0) + hits[best] * lab["score"]
    if not votes:
        return "unlabeled", 0.0, {}
    total = sum(votes.values())
    top = max(votes, key=votes.get)
    return top, votes[top] / total, {k: round(v / total, 3) for k, v in sorted(votes.items(), key=lambda kv: -kv[1])}


def clear_marker_columns(occ: np.ndarray, origin: np.ndarray, voxel_m: float,
                         marker_corners: list[np.ndarray], margin_m: float = 0.02) -> int:
    """Empty every voxel column over a calibration marker's footprint
    (+ margin, which also covers a raised marker's stand). Markers are
    placed in clear bench space by design (CAPTURE_GUIDE), so this can't
    remove a real object -- and it doesn't depend on reconstructed heights
    being right, unlike matching a component's top against a marker's z.
    Returns the number of voxels cleared.
    """
    cleared = 0
    for corners in marker_corners:
        lo = corners[:, :2].min(axis=0) - margin_m
        hi = corners[:, :2].max(axis=0) + margin_m
        i0, j0 = np.floor((lo - origin[:2]) / voxel_m).astype(int)
        i1, j1 = np.ceil((hi - origin[:2]) / voxel_m).astype(int) + 1
        i0, j0 = max(i0, 0), max(j0, 0)
        cleared += int(occ[i0:i1, j0:j1, :].sum())
        occ[i0:i1, j0:j1, :] = False
    return cleared

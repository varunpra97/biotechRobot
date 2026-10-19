"""Multi-view depth fusion: a truncated signed distance field (TSDF) with
free-space carving, on a dense voxel grid in the platform frame.

Why this replaces the visual hull as the base geometry when depth exists:
a silhouette only says "the object is somewhere along these rays", so a
visual hull fills every concavity no silhouette can see into (the inside
of the centrifuge bowl came out as a solid dome). A depth pixel says
where the surface IS along its ray, and that everything in front of it is
empty -- so a bowl seen from above is carved hollow and its floor located.

Per voxel x and view (pose from the fiducial markers, like everything
else), with d the measured depth at x's pixel and z x's own depth:

  sdf = (d - z) along the ray       > 0: x is in front of the surface
  sdf >  trunc        free space: field +1 (counted as a free observation)
  -behind <= sdf <= trunc
                      near a surface: field sdf / trunc
  sdf < -behind       behind a surface: occluded, no information

The fused field is the weighted mean over the views that said something.
A voxel no view said anything about (inside a solid, under an object) is
-1: unobserved space is treated as occupied, so the result is always
watertight and the surface is exact wherever any view saw it, and falls
back to the free-space boundary (a depth-carved hull) where none did.
Marching cubes at 0 gives the mesh, interpolated well below the voxel size.

Weights are 1 / sigma(z)^2 of the sensor model: close, low-noise views
dominate. Depth is sampled bilinearly where the 4 neighbors agree and by
nearest neighbor across a depth edge (never blended over an edge).

The band behind a surface ("behind") is much shorter than in front.
With a symmetric band, a wall thinner than trunc is averaged away: the
views looking at its far side put their behind-band values in the air
just outside its near side (a 2mm bin wall fused to ~0 and vanished at
6mm trunc). Behind only needs to cover the noise (about 3 sigma) and a
voxel or so; free space in front can and should use the full trunc. A
wall thinner than the noise still can't be resolved from depth alone --
that is where silhouette refinement and templates take over.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class DepthView:
    name: str
    K: np.ndarray                 # intrinsics of the depth map
    rvec: np.ndarray              # platform -> camera
    tvec: np.ndarray
    depth: np.ndarray             # HxW meters, 0 = no return
    sigma_fn: object = None       # z -> noise std (m); None = perfect depth


@dataclass
class FusedGrid:
    field: np.ndarray             # [nx, ny, nz] float32 in [-1, 1], < 0 inside
    weight: np.ndarray            # summed weights of the near-surface + free observations
    free_views: np.ndarray        # uint8: views that saw the voxel as free space
    origin: np.ndarray            # center of voxel (0, 0, 0)
    voxel: float
    trunc: float

    def points(self, ijk: np.ndarray) -> np.ndarray:
        return self.origin + ijk * self.voxel


def _sample_depth(depth: np.ndarray, u: np.ndarray, v: np.ndarray, jump_m: np.ndarray) -> np.ndarray:
    """Bilinear where the 4 neighbors are all valid and within jump_m of
    each other, nearest otherwise; 0 outside the image or with no return."""
    h, w = depth.shape
    out = np.zeros(len(u), np.float32)
    inside = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    if not inside.any():
        return out
    ui, vi = u[inside], v[inside]
    x0 = np.clip(np.floor(ui).astype(np.int64), 0, w - 2)
    y0 = np.clip(np.floor(vi).astype(np.int64), 0, h - 2)
    fx, fy = (ui - x0).astype(np.float32), (vi - y0).astype(np.float32)
    d00, d10 = depth[y0, x0], depth[y0, x0 + 1]
    d01, d11 = depth[y0 + 1, x0], depth[y0 + 1, x0 + 1]
    stack = np.stack([d00, d10, d01, d11])
    lo, hi = stack.min(axis=0), stack.max(axis=0)
    smooth = (lo > 0) & (hi - lo <= jump_m[inside])
    bil = (d00 * (1 - fx) * (1 - fy) + d10 * fx * (1 - fy) + d01 * (1 - fx) * fy + d11 * fx * fy)
    nn = depth[np.clip(np.round(vi).astype(np.int64), 0, h - 1), np.clip(np.round(ui).astype(np.int64), 0, w - 1)]
    out[inside] = np.where(smooth, bil, nn)
    return out


def fuse(views: list[DepthView], bounds_min, bounds_max, voxel: float, trunc: float,
         behind: float | None = None, chunk: int = 2_000_000, keep: np.ndarray | None = None,
         normals: np.ndarray | None = None, facing_cos: float = -0.05) -> FusedGrid:
    """Dense TSDF over the axis-aligned box. behind: depth of the band
    behind each observed surface (default 1.5 voxels). keep (optional bool
    grid of the same shape): only these voxels are integrated, the rest
    stay +1. normals (optional [*shape, 3] outward unit normals from a first
    pass, see field_normals): behind-the-surface evidence is only taken from
    views the local surface FACES. A view looking at one side of a wall
    then says nothing about the air on its other side -- which is what
    thickens thin walls when the noise forces a deep behind band."""
    behind = 1.5 * voxel if behind is None else behind
    nflat = None if normals is None else normals.reshape(-1, 3)
    bmin, bmax = np.asarray(bounds_min, float), np.asarray(bounds_max, float)
    shape = tuple(int(s) for s in np.ceil((bmax - bmin) / voxel - 1e-9).astype(int))
    origin = bmin + voxel / 2
    n = int(np.prod(shape))
    fsum = np.zeros(n, np.float32)
    wsum = np.zeros(n, np.float32)
    free = np.zeros(n, np.uint8)
    keep_flat = None if keep is None else keep.reshape(-1)

    poses = [(cv2.Rodrigues(np.asarray(v.rvec, float))[0], np.asarray(v.tvec, float).reshape(3)) for v in views]
    centers = [-R.T @ t for R, t in poses]
    for start in range(0, n, chunk):
        idx = np.arange(start, min(start + chunk, n))
        if keep_flat is not None:
            idx = idx[keep_flat[idx]]
            if len(idx) == 0:
                continue
        pts = origin + np.stack(np.unravel_index(idx, shape), axis=1) * voxel
        nrm = None if nflat is None else nflat[idx].astype(np.float32)
        for view, (R, t), C in zip(views, poses, centers):
            cam = pts @ R.T + t
            z = cam[:, 2]
            front = z > 1e-3
            zs = np.where(front, z, 1.0)
            xn, yn = cam[:, 0] / zs, cam[:, 1] / zs
            u = view.K[0, 0] * xn + view.K[0, 2]
            v = view.K[1, 1] * yn + view.K[1, 2]
            sig = view.sigma_fn(zs) if view.sigma_fn is not None else np.zeros_like(zs)
            footprint = zs / view.K[0, 0]
            jump = 3.0 * footprint + 4.0 * sig
            d = _sample_depth(view.depth, u, v, jump)
            ok = front & (d > 0)
            if not ok.any():
                continue
            sdf = (d - z) * np.sqrt(1.0 + xn * xn + yn * yn)        # along the ray, meters
            near = ok & (sdf >= -behind) & (sdf <= trunc)
            if nrm is not None:
                ray = pts - C
                facing = np.einsum("ij,ij->i", nrm, ray) < facing_cos * np.linalg.norm(ray, axis=1)
                has_n = np.any(nrm != 0, axis=1)
                near &= ~((sdf < 0) & has_n & ~facing)
            far = ok & (sdf > trunc)
            # weight: inverse noise variance, scaled so sigma = 0.1mm weighs 1
            w = 1e-8 / np.maximum(sig, 1e-4) ** 2 if view.sigma_fn is not None else np.ones_like(zs)
            obs = near | far
            fsum[idx[obs]] += (np.clip(sdf[obs] / trunc, -1.0, 1.0) * w[obs]).astype(np.float32)
            wsum[idx[obs]] += w[obs].astype(np.float32)
            free[idx[far]] += 1
    field = np.where(wsum > 0, fsum / np.maximum(wsum, 1e-12), -1.0).astype(np.float32)
    if keep_flat is not None:
        field[~keep_flat] = 1.0
    return FusedGrid(field.reshape(shape), wsum.reshape(shape), free.reshape(shape), origin, voxel, trunc)


def field_normals(field: np.ndarray, smooth_vox: float = 1.0, min_grad: float = 0.02) -> np.ndarray:
    """Outward unit normals (gradient of the field: it rises from inside to
    outside), zero where the field is flat (deep inside, far outside)."""
    F = ndimage.gaussian_filter(field.astype(np.float32), smooth_vox) if smooth_vox else field
    g = np.stack(np.gradient(F), axis=-1)
    n = np.linalg.norm(g, axis=-1, keepdims=True)
    return np.where(n > min_grad, g / np.maximum(n, 1e-12), 0.0).astype(np.float32)


def occupied(grid: FusedGrid) -> np.ndarray:
    return grid.field < 0


def grid_mesh(grid: FusedGrid, bottom_z: float | None = None):
    """Marching cubes at 0, padded with an 'outside' layer so the surface is
    always closed. bottom_z: vertices on the closing face under the grid
    (an object cut at the bench) are snapped down to it."""
    import trimesh
    from skimage.measure import marching_cubes
    F = np.pad(grid.field, 1, mode="constant", constant_values=1.0)
    if (F < 0).sum() == 0:
        return None
    verts, faces, _, _ = marching_cubes(F, level=0.0, spacing=(grid.voxel,) * 3)
    verts = verts + grid.origin - grid.voxel                      # undo the pad
    mesh = trimesh.Trimesh(verts, faces, process=True)
    if mesh.volume < 0:
        mesh.invert()
    if bottom_z is not None:
        z0 = grid.origin[2]
        low = mesh.vertices[:, 2] < z0
        mesh.vertices[low, 2] = bottom_z
    return mesh


def split_components(occ: np.ndarray, voxel: float, min_volume_m3: float = 2e-6):
    """26-connected components of the occupied grid, dropping specks."""
    labels, n = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    if n == 0:
        return labels, []
    sizes = ndimage.sum_labels(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s * voxel ** 3 >= min_volume_m3]
    return labels, keep


def merge_nested(labels: np.ndarray, comps: list[int], min_inside: float = 0.8) -> list[int]:
    """Fold a component into a bigger one when its footprint lies (mostly)
    inside the bigger one's: a piece under or inside an object's outline
    that fusion separated from it (a foot under a centrifuge housing,
    seen through the gap beneath it, came out as its own 'object')."""
    boxes, sizes = {}, {}
    for c in comps:
        ijk = np.argwhere(labels == c)
        boxes[c] = (ijk[:, :2].min(axis=0), ijk[:, :2].max(axis=0) + 1)
        sizes[c] = len(ijk)
    order = sorted(comps, key=lambda c: sizes[c])
    alive = set(comps)
    for a in order:
        alo, ahi = boxes[a]
        area = float(np.prod(ahi - alo))
        for b in sorted(alive - {a}, key=lambda c: -sizes[c]):
            if sizes[b] <= sizes[a]:
                continue
            blo, bhi = boxes[b]
            inter = float(np.prod(np.clip(np.minimum(ahi, bhi) - np.maximum(alo, blo), 0, None)))
            if inter >= min_inside * area:
                labels[labels == a] = b
                sizes[b] += sizes[a]
                alive.discard(a)
                break
    return [c for c in comps if c in alive]

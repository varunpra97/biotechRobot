"""Freeform refinement: pull a mesh's silhouettes onto the image edges in
every calibrated view -- for objects no template family describes (a
centrifuge housing, a pipette).

Start from the visual-hull mesh (it already contains the object; its
surface is only as good as the SAM masks and the voxels). Each iteration:

  1. per view, rasterize the mesh and take the vertices on its projected
     outline (within ~1px of the silhouette boundary of a signed distance
     field, whose gradient also gives each vertex's outward 2D normal);
  2. find the real edge along that normal -- the same sub-pixel,
     orientation-checked search the templates use;
  3. turn the pixel offset into a 3D move perpendicular to that view
     (offset x depth / focal length);
  4. average the moves per vertex over the views that see it, diffuse the
     DISPLACEMENT field over the mesh (spreads corrections to neighbors
     without the shrinkage that smoothing positions causes), apply, keep
     the object on or above the bench.

Limit, and it is fundamental: a concavity is never on any silhouette (the
inside of a centrifuge bowl, the gap inside a pipette's finger hook), so
those surfaces stay where the visual hull left them. Report silhouette-
observable accuracy separately from whole-surface accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import trimesh
from scipy import sparse

from template_fitting import EdgeView, _bilinear, _normal_search, _project_roi


@dataclass
class RefineStats:
    iteration: int
    radius_px: float
    observed_vertices: int
    mean_abs_offset_px: float


def prepare_mesh(mesh: trimesh.Trimesh, smooth_iters: int = 3) -> trimesh.Trimesh:
    """The visual hull's largest component with its voxel staircase lightly
    smoothed (Taubin: smooths without shrinking). No resampling: quadric
    decimation spiked on the non-manifold hull (143mm centrifuge -> 327mm),
    and a voxel round-trip grew the surface by about a voxel; the hull is
    already at reconstruction resolution, which is fine to refine."""
    m = max(mesh.split(only_watertight=False), key=lambda p: len(p.faces)).copy()
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=smooth_iters)
    return m


def _silhouette(mesh: trimesh.Trimesh, view: EdgeView, band_px: float):
    """(vertex indices on the projected outline, their 2D positions in ROI
    pixels, outward unit normals)."""
    pts = _project_roi(view, mesh.vertices)
    h, w = view.gx.shape
    mask = np.zeros((h, w), np.uint8)
    # One fill per triangle (fast: ~1.5us each). A single fillPoly over all
    # faces fills even-odd, so front and back faces of a closed mesh cancel:
    # a sphere's disk came out 4375 of ~5027 px, with holes, and the outline
    # band and normals came from that. 2 fractional bits = quarter-pixel.
    for tri in np.round(pts[mesh.faces] * 4).astype(np.int32):
        cv2.fillConvexPoly(mask, tri, 1, shift=2)
    outside = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    sdf = (outside - inside).astype(np.float32)          # > 0 outside the silhouette
    d = _bilinear(sdf, pts[:, 0], pts[:, 1], 1e9)
    on = np.where(np.abs(d) <= band_px)[0]
    if len(on) == 0:
        return on, pts[on], pts[on]
    gy, gx = np.gradient(sdf)
    n = np.stack([_bilinear(gx, pts[on, 0], pts[on, 1], 0.0), _bilinear(gy, pts[on, 0], pts[on, 1], 0.0)], axis=1)
    norm = np.linalg.norm(n, axis=1)
    keep = norm > 1e-6
    return on[keep], pts[on[keep]], n[keep] / norm[keep, None]


def _adjacency(mesh: trimesh.Trimesh) -> sparse.csr_matrix:
    e = mesh.edges_unique
    n = len(mesh.vertices)
    a = sparse.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    return a.tocsr()


def refine(mesh: trimesh.Trimesh, views: list[EdgeView], iterations: int = 16, band_px: float = 1.0,
           min_grad: float = 4.0, diffusion_iters: int = 15, data_weight: float = 4.0,
           step: float = 0.8, inside_fn=None, boundary_maps: dict | None = None,
           boundary_gate_px: float = 8.0, anchor_weight: float = 0.0,
           combine: str = "mean") -> tuple[trimesh.Trimesh, list[RefineStats]]:
    """inside_fn(points) -> bool: the visual hull as a hard bound (every true
    surface point projects inside every silhouette, so the hull contains the
    object up to segmentation error); vertices leaving it are put back.
    boundary_maps[view name]: distance (ROI px) to that view's segmentation
    boundary; a matched edge farther than boundary_gate_px from it is
    rejected. Without these, low views matched background edges BEHIND the
    centrifuge's top rim and grew it 13mm upward.
    anchor_weight > 0: vertices on no outline are pulled toward ZERO
    displacement instead of following their neighbors (a membrane). For a
    depth-fused start, whose surface is already measured everywhere depth
    saw it: silhouette corrections fix the outline without dragging a
    bowl's measured interior along with it.
    combine: how one vertex's per-view moves become one 3D move. "mean"
    averages them (the original; validated on visual-hull starts). "lsq"
    treats each view's move as what it is -- a constraint along ONE
    direction (the image-plane edge normal, back-projected) -- and solves
    them jointly per vertex; directions no view constrains stay put.
    Averaging turned "the far top edge is too close" seen from elevated
    views into "the top is too low": +1-2mm heights on depth-fused starts."""
    m = mesh.copy()
    A = _adjacency(m)
    deg = np.asarray(A.sum(axis=1)).ravel()
    stats = []
    for it in range(iterations):
        radius = max(2.5, 8.0 - 0.5 * it)
        disp = np.zeros((len(m.vertices), 3))
        count = np.zeros(len(m.vertices))
        MtM = np.zeros((len(m.vertices), 3, 3))
        offsets = []
        for v in views:
            idx, p2, n2 = _silhouette(m, v, band_px)
            if len(idx) == 0:
                continue
            off = _normal_search(v, p2, n2, radius, min_grad)
            ok = ~np.isnan(off) & (np.abs(off) < 0.9 * radius)   # a match at the search limit is dubious
            if boundary_maps is not None and v.name in boundary_maps:
                hit = p2 + np.nan_to_num(off)[:, None] * n2
                ok &= _bilinear(boundary_maps[v.name], hit[:, 0], hit[:, 1], 1e9) <= boundary_gate_px
            if not ok.any():
                continue
            idx, n2, off = idx[ok], n2[ok], off[ok]
            R, _ = cv2.Rodrigues(v.rvec)
            depth = (m.vertices[idx] @ R.T + v.tvec.reshape(1, 3))[:, 2]
            if combine == "lsq":
                mdir = np.c_[n2, np.zeros(len(idx))] @ R            # unit constraint direction, world
                sval = off * depth / v.K[0, 0]                       # signed move along it, meters
                np.add.at(disp, idx, mdir * sval[:, None])
                np.add.at(MtM, idx, mdir[:, :, None] * mdir[:, None, :])
            else:
                cam_dir = np.c_[n2[:, 0] * off * depth / v.K[0, 0], n2[:, 1] * off * depth / v.K[1, 1], np.zeros(len(idx))]
                np.add.at(disp, idx, cam_dir @ R)        # camera -> world: R^T applied as row-vector @ R
            np.add.at(count, idx, 1.0)
            offsets.append(np.abs(off))
        observed = count > 0
        D = np.zeros_like(disp)
        if combine == "lsq":
            # (sum m m^T + lam I) d = sum m s; lam keeps unconstrained directions at ~0
            A3 = MtM[observed] + 0.05 * np.eye(3)[None]
            D[observed] = np.linalg.solve(A3, disp[observed][:, :, None])[:, :, 0]
        else:
            D[observed] = disp[observed] / count[observed, None]
        W = data_weight * observed.astype(float) + anchor_weight * (~observed).astype(float)
        target = D.copy()
        for _ in range(diffusion_iters):   # data term at observed vertices, membrane elsewhere
            D = (W[:, None] * target + A @ D) / (W + deg)[:, None]
        prev = m.vertices.copy()
        m.vertices = m.vertices + step * D
        m.vertices[:, 2] = np.maximum(m.vertices[:, 2], 0.0)   # rests on the bench
        if inside_fn is not None:
            out = ~inside_fn(m.vertices)
            m.vertices[out] = prev[out]
        stats.append(RefineStats(it, radius, int(observed.sum()),
                                 float(np.mean(np.concatenate(offsets))) if offsets else float("nan")))
    return m, stats

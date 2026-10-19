"""Stage 4: fit a parametric shape template to an object by aligning its
projected silhouette outline with real image edges in every calibrated view.

Why: a visual hull is only as sharp as the segmentation masks (SAM, at
reduced resolution) and can't be tighter than its voxels -- it was
+2..+25mm on outer dimensions. A template has a handful of parameters
(position, yaw, width, depth, height, radius, ...), and every view
contributes hundreds of edge observations at full resolution. Each edge
point is only pixel-accurate, but a few parameters fitted to thousands of
them across ~20 views land well below a pixel -- which is what a 0.5mm
target needs (~0.4mm per pixel at 12MP from 1.5m).

Objective: for each view, Canny edges of the full-resolution image ->
distance transform; residual = distance from each projected outline point
to the nearest edge (bilinear, so continuous). Outline points that fall
where the object is hidden behind something else just find the wrong
edge or none; a robust loss (soft L1) keeps them from dragging the fit.

Templates sit on the bench (z=0 at their base), matching the support
assumption used in reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares


MAX_EDGE_DIST_PX = 30.0


@dataclass
class EdgeView:
    name: str
    K: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    roi: tuple[int, int, int, int]       # x0, y0, x1, y1 in full-res pixels
    dist_map: np.ndarray                 # distance (px) to nearest edge, ROI-sized
    gx: np.ndarray = None                # image gradient (ROI), for sub-pixel normal search
    gy: np.ndarray = None


@dataclass
class FitResult:
    family: str
    params: dict
    rms_px: float
    n_points: int
    per_view_rms_px: dict = field(default_factory=dict)


# ---------------------------------------------------------------- templates

def _rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def box_points(p: np.ndarray) -> np.ndarray:
    """p = [cx, cy, yaw, w, d, h] -> 8 corners, base on the bench."""
    cx, cy, yaw, w, d, h = p[:6]
    local = np.array([[sx * w / 2, sy * d / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0, h)])
    return local @ _rot_z(yaw).T + [cx, cy, 0]


def cylinder_points(p: np.ndarray, n: int = 72) -> np.ndarray:
    """p = [cx, cy, r, h] -> rim points of an upright cylinder on the bench."""
    cx, cy, r, h = p
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)
    return np.vstack([np.c_[ring, np.zeros(n)], np.c_[ring, np.full(n, h)]])


def make_cylinder_grid(rows: int, cols: int, z_bottom: float, n_ring: int = 36):
    """rows x cols upright cylinders on a regular grid sharing one radius and
    top height -- e.g. tubes standing in a rack's bores, visible from the
    rack's top surface (z_bottom) up. 7 parameters for any number of tubes,
    so every tube's edges inform the shared ones.
    p = [gx, gy, yaw, pitch_x, pitch_y, r, z_top]; (gx, gy) = grid center."""
    def gen(p):
        gx, gy, yaw, px, py, r, z_top = p
        R = _rot_z(yaw)[:2, :2]
        a = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
        parts = []
        for i in range(rows):
            for j in range(cols):
                c = R @ np.array([(j - (cols - 1) / 2) * px, (i - (rows - 1) / 2) * py]) + [gx, gy]
                ring = np.stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)], axis=1)
                parts.append(np.vstack([np.c_[ring, np.full(n_ring, z_bottom)], np.c_[ring, np.full(n_ring, z_top)]]))
        return parts
    return gen, ["gx", "gy", "yaw", "pitch_x", "pitch_y", "r", "z_top"]


TEMPLATES = {
    # family: (point generator -> point set or list of parts, parameter names)
    "box": (box_points, ["cx", "cy", "yaw", "w", "d", "h"]),
    "open_box": (lambda p: box_points(p[:6]), ["cx", "cy", "yaw", "w", "d", "h", "t"]),
    "cylinder": (cylinder_points, ["cx", "cy", "r", "h"]),
}


# ---------------------------------------------------------------- edges

def build_edge_view(name, image_gray: np.ndarray, K, dist, rvec, tvec, roi_corners_3d: np.ndarray,
                    margin_px: int = 40, canny=(30, 90)) -> EdgeView | None:
    """Edge distance map over the object's projected region (+ margin),
    from the full-resolution image."""
    px, _ = cv2.projectPoints(roi_corners_3d, rvec, tvec, K, dist)
    px = px.reshape(-1, 2)
    h, w = image_gray.shape
    x0, y0 = np.floor(px.min(axis=0)).astype(int) - margin_px
    x1, y1 = np.ceil(px.max(axis=0)).astype(int) + margin_px
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    crop = cv2.GaussianBlur(image_gray[y0:y1, x0:x1], (3, 3), 0)
    edges = cv2.Canny(crop, *canny, L2gradient=True)
    if not edges.any():
        return None  # distanceTransform would return ~FLT_MAX everywhere
    dmap = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    # Beyond a few tens of px "nearest edge" means "no matching edge"; cap
    # it so one empty region can't dominate the cost.
    smooth = cv2.GaussianBlur(image_gray[y0:y1, x0:x1].astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return EdgeView(name, K, dist, rvec, tvec, (x0, y0, x1, y1), np.minimum(dmap, MAX_EDGE_DIST_PX).astype(np.float32),
                    gx, gy)


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray, outside: float) -> np.ndarray:
    h, w = img.shape
    ok = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
    out = np.full(len(x), outside, np.float64)
    xi, yi = np.floor(x[ok]).astype(int), np.floor(y[ok]).astype(int)
    fx, fy = x[ok] - xi, y[ok] - yi
    out[ok] = (img[yi, xi] * (1 - fx) * (1 - fy) + img[yi, xi + 1] * fx * (1 - fy)
               + img[yi + 1, xi] * (1 - fx) * fy + img[yi + 1, xi + 1] * fx * fy)
    return out


def outline_samples(points_3d, view: EdgeView, n: int = 240, with_normals: bool = False):
    """Points (and outward normals) along each part's projected outline;
    n points for a single part, n // 6 (min 24) per part for a multi-part
    template such as a tube grid."""
    if isinstance(points_3d, list):
        per = max(24, n // 6)
        outs = [_outline_one(p, view, per, with_normals) for p in points_3d]
        if not with_normals:
            return np.vstack(outs)
        return np.vstack([o[0] for o in outs]), np.vstack([o[1] for o in outs])
    return _outline_one(points_3d, view, n, with_normals)


def _outline_one(points_3d: np.ndarray, view: EdgeView, n: int, with_normals: bool):
    """n points evenly spaced by arc length along the projected silhouette
    outline (convex hull of the projected template points -- exact for
    convex templates like box and cylinder), in ROI pixel coordinates.
    A fixed n, not a fixed spacing: the optimizer needs the residual count
    to stay constant while the template changes size."""
    px, _ = cv2.projectPoints(points_3d, view.rvec, view.tvec, view.K, view.dist)
    hull = cv2.convexHull(px.reshape(-1, 2).astype(np.float32)).reshape(-1, 2).astype(np.float64)
    hull -= [view.roi[0], view.roi[1]]
    closed = np.vstack([hull, hull[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n, endpoint=False)
    i = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(seg) - 1)
    frac = ((t - cum[i]) / np.maximum(seg[i], 1e-9))[:, None]
    pts = closed[i] + frac * (closed[i + 1] - closed[i])
    if not with_normals:
        return pts
    d = closed[i + 1] - closed[i]
    nrm = np.stack([d[:, 1], -d[:, 0]], axis=1) / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    # orient outward: away from the outline's centroid
    flip = np.sum((pts - hull.mean(axis=0)) * nrm, axis=1) < 0
    nrm[flip] *= -1
    return pts, nrm


# ---------------------------------------------------------------- visible-edge models
# The outline alone is a weak cue for objects whose top face is as bright as
# the bench behind them: the tube holder's top silhouette edge has almost no
# contrast, and its fitted height wandered between 6 and 150mm (true 40mm).
# What IS visible is every crease where differently shaded faces meet, and
# where tubes enter the top surface. These models return ALL visible edges
# of a template -- silhouette and creases -- as a fixed number of samples
# with a per-view visibility mask, for the precise stage.

def _cam_center(view: EdgeView) -> np.ndarray:
    R, _ = cv2.Rodrigues(view.rvec)
    return -R.T @ view.tvec.reshape(3)


def _project_roi(view: EdgeView, X: np.ndarray) -> np.ndarray:
    px, _ = cv2.projectPoints(np.asarray(X, np.float64), view.rvec, view.tvec, view.K, view.dist)
    return px.reshape(-1, 2) - [view.roi[0], view.roi[1]]


def _line_normals(pts: np.ndarray, groups: int) -> np.ndarray:
    """Normals for samples laid out as consecutive straight runs."""
    out = np.empty_like(pts)
    for g in np.split(np.arange(len(pts)), groups):
        d = pts[g[-1]] - pts[g[0]]
        out[g] = np.array([d[1], -d[0]]) / max(np.linalg.norm(d), 1e-9)
    return out


def _rim_normals(ring_pts: np.ndarray) -> np.ndarray:
    tang = np.roll(ring_pts, -1, axis=0) - np.roll(ring_pts, 1, axis=0)
    return np.stack([tang[:, 1], -tang[:, 0]], axis=1) / np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)


# corner index = (sx_i * 2 + sy_i) * 2 + z_i, matching box_points()
_BOX_FACES = [((0, 1, 2, 3), (-1, 0, 0)), ((4, 5, 6, 7), (1, 0, 0)), ((0, 1, 4, 5), (0, -1, 0)),
              ((2, 3, 6, 7), (0, 1, 0)), ((0, 2, 4, 6), (0, 0, -1)), ((1, 3, 5, 7), (0, 0, 1))]
_BOX_EDGES = [(0, 1, 0, 2), (2, 3, 0, 3), (4, 5, 1, 2), (6, 7, 1, 3), (0, 2, 0, 4), (1, 3, 0, 5),
              (4, 6, 1, 4), (5, 7, 1, 5), (0, 4, 2, 4), (1, 5, 2, 5), (2, 6, 3, 4), (3, 7, 3, 5)]


_BOX_EDGE_GROUPS = {"vertical": (0, 1, 2, 3), "base": (4, 6, 8, 10), "top": (5, 7, 9, 11)}


def box_height_edges(p: np.ndarray, view: EdgeView, base_clearance_px: float = 4.0):
    """Only the edges that move with height (top + vertical), minus samples
    landing within a few px of the base outline. Scoring height with every
    edge has a degenerate optimum: a nearly flat box lays its top edges on
    its base edges and double-counts the strong base contact line (a 14.4mm
    tray scanned to 2.5mm)."""
    pts, nrm, vis = box_edges(p, view)
    k = len(pts) // len(_BOX_EDGES)
    keep = np.zeros(len(pts), bool)
    for e in _BOX_EDGE_GROUPS["top"] + _BOX_EDGE_GROUPS["vertical"]:
        keep[e * k:(e + 1) * k] = True
    base = np.concatenate([np.arange(e * k, (e + 1) * k) for e in _BOX_EDGE_GROUPS["base"]])
    base_pts = pts[base][vis[base]]
    nrm = nrm.copy()
    if len(base_pts):
        d = np.min(np.linalg.norm(pts[:, None, :] - base_pts[None, :, :], axis=2), axis=1)
        # Zero evidence, not removed: dropping these samples left a flat box
        # scored on its few surviving samples, which still won.
        nrm[keep & (d <= base_clearance_px)] = 0.0
    return pts, nrm, vis & keep


def box_edges(p: np.ndarray, view: EdgeView, k: int = 24, inset: float = 0.06):
    """All 12 edges; an edge is visible if either adjoining face faces the
    camera (exact for a convex box). Samples stay off the corners, where
    several edges meet and the nearest edge is ambiguous."""
    corners = box_points(p)
    R = _rot_z(p[2])
    C = _cam_center(view)
    front = [np.dot(R @ np.array(n, float), C - corners[list(idx)].mean(axis=0)) > 0 for idx, n in _BOX_FACES]
    t = np.linspace(inset, 1 - inset, k)[:, None]
    P3 = np.vstack([corners[i] + t * (corners[j] - corners[i]) for i, j, _, _ in _BOX_EDGES])
    vis = np.concatenate([np.full(k, front[a] or front[b]) for _, _, a, b in _BOX_EDGES])
    pts = _project_roi(view, P3)
    return pts, _line_normals(pts, len(_BOX_EDGES)), vis


def cylinder_edges(cx, cy, r, z0, z1, view: EdgeView, m: int = 36, q: int = 12, r_inner: float | None = None):
    """Upright cylinder from z0 to z1: top rim (visible when the camera is
    above it), the half of the bottom rim facing the camera, and the two
    side silhouette lines (tangent generators). With r_inner, an open hollow
    tube: its top also shows the inner rim -- without modelling it, the
    inner rim's edge pulled a 3.5mm tube radius down to 2.7mm."""
    C = _cam_center(view)
    th = np.linspace(0, 2 * np.pi, m, endpoint=False)
    ring = np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], axis=1)
    top3 = np.c_[ring, np.full(m, z1)]
    bot3 = np.c_[ring, np.full(m, z0)]
    vis_top = np.full(m, C[2] > z1)
    vis_bot = np.cos(th) * (C[0] - ring[:, 0]) + np.sin(th) * (C[1] - ring[:, 1]) > 0
    phi = np.arctan2(C[1] - cy, C[0] - cx)
    a = np.arccos(r / max(np.hypot(C[0] - cx, C[1] - cy), r * 1.0001))
    zs = np.linspace(z0, z1, q + 2)[1:-1]
    lines3 = np.vstack([np.c_[np.full(q, cx + r * np.cos(phi + s * a)), np.full(q, cy + r * np.sin(phi + s * a)), zs]
                        for s in (-1, 1)])
    pts = _project_roi(view, np.vstack([top3, bot3, lines3]))
    nrm = np.vstack([_rim_normals(pts[:m]), _rim_normals(pts[m:2 * m]), _line_normals(pts[2 * m:], 2)])
    vis = np.concatenate([vis_top, vis_bot, np.full(2 * q, True)])
    if r_inner is not None:
        inner3 = np.c_[cx + r_inner * np.cos(th), cy + r_inner * np.sin(th), np.full(m, z1)]
        ipts = _project_roi(view, inner3)
        pts, nrm, vis = np.vstack([pts, ipts]), np.vstack([nrm, _rim_normals(ipts)]), np.concatenate([vis, vis_top])
    return pts, nrm, vis


def make_rack_with_tube_grid(rows: int, cols: int):
    """A box-shaped rack with a rows x cols grid of upright tubes standing in
    it, fitted jointly. Tubes are visible from the rack's top surface up, so
    where each tube enters the rack is an edge at z = rack height: the tubes
    pin down the rack's height, and the rack the tubes' base.
    p = [cx, cy, yaw, w, d, h, ox, oy, pitch_x, pitch_y, r, z_top, r_rim]
    (ox, oy: grid center offset from the rack center, in the rack frame).

    Tube radius r comes only from the side silhouette lines. The top rim gets
    its own nuisance radius r_rim: on an open 7mm tube the outer rim, the
    0.5mm wall and the dark opening blur into one edge inside the true
    radius (at ~0.4mm/px), and fitting the rim at r pulled 3.5mm to 2.8mm;
    with r_rim the rim still measures height and position, and r came out
    3.39mm. The base ring is left out -- the tube/bore gap makes it ambiguous.
    Returns (gen, names, edges_fn, tube_centers_fn)."""
    names = ["cx", "cy", "yaw", "w", "d", "h", "ox", "oy", "pitch_x", "pitch_y", "r", "z_top", "r_rim"]

    def tube_centers(p):
        cx, cy, yaw, _, _, _, ox, oy, px, py = p[:10]
        R = _rot_z(yaw)[:2, :2]
        return [R @ np.array([ox + (j - (cols - 1) / 2) * px, oy + (i - (rows - 1) / 2) * py]) + [cx, cy]
                for i in range(rows) for j in range(cols)]

    def gen(p):  # outline: the rack body only (tubes come in through the edge model)
        return box_points(p[:6])

    def edges(p, view):
        out = [box_edges_unambiguous(p[:6], view)]
        m, q = 36, 12  # cylinder_edges layout: top rim m, base rim m, side lines 2q
        for c in tube_centers(p):
            lines = cylinder_edges(c[0], c[1], p[10], p[5], p[11], view, m=m, q=q)
            rim = cylinder_edges(c[0], c[1], p[12], p[5], p[11], view, m=m, q=q)
            out.append((np.vstack([lines[0][2 * m:], rim[0][:m]]), np.vstack([lines[1][2 * m:], rim[1][:m]]),
                        np.concatenate([lines[2][2 * m:], rim[2][:m]])))
        return np.vstack([o[0] for o in out]), np.vstack([o[1] for o in out]), np.concatenate([o[2] for o in out])

    return gen, names, edges, tube_centers


def make_plate_with_well_grid(rows: int, cols: int, m: int = 32):
    """A box-shaped plate whose top face has a rows x cols grid of circular
    well openings (sample tray, microplate). Modelling the wells matters:
    their dark openings are the strongest edges on the plate, and as
    unmodelled clutter they pulled a box template's top edges onto a row of
    wells (a 14.4mm tray fitted to 3.7mm). Modelled, they sit at z = plate
    height and measure it -- independent of whether the plate's top edges
    are sharp or rounded.
    p = [cx, cy, yaw, w, d, h, ox, oy, pitch_x, pitch_y, rw]"""
    names = ["cx", "cy", "yaw", "w", "d", "h", "ox", "oy", "pitch_x", "pitch_y", "rw"]

    def well_centers(p):
        cx, cy, yaw, _, _, _, ox, oy, px, py = p[:10]
        R = _rot_z(yaw)[:2, :2]
        return [R @ np.array([ox + (j - (cols - 1) / 2) * px, oy + (i - (rows - 1) / 2) * py]) + [cx, cy]
                for i in range(rows) for j in range(cols)]

    def gen(p):
        return box_points(p[:6])

    def edges(p, view):
        pts, nrm, vis = box_edges_unambiguous(p[:6], view)
        C = _cam_center(view)
        th = np.linspace(0, 2 * np.pi, m, endpoint=False)
        rims = []
        for c in well_centers(p):
            ring = np.c_[c[0] + p[10] * np.cos(th), c[1] + p[10] * np.sin(th), np.full(m, p[5])]
            rp = _project_roi(view, ring)
            rims.append((rp, _rim_normals(rp), np.full(m, C[2] > p[5])))
        return (np.vstack([pts] + [r[0] for r in rims]), np.vstack([nrm] + [r[1] for r in rims]),
                np.concatenate([vis] + [r[2] for r in rims]))

    return gen, names, edges, well_centers


def box_edges_unambiguous(p: np.ndarray, view: EdgeView, base_clearance_px: float = 4.0):
    """box_edges for refinement: drops top/vertical samples that project onto
    the base outline, where "nearest edge" can't tell them apart -- otherwise
    each refinement step lets the top edges grab the strong base line and
    walks a 14.4mm tray down to its 2mm bound."""
    pts, nrm, vis = box_edges(p, view)
    k = len(pts) // len(_BOX_EDGES)
    base = np.concatenate([np.arange(e * k, (e + 1) * k) for e in _BOX_EDGE_GROUPS["base"]])
    base_pts = pts[base][vis[base]]
    if len(base_pts):
        movers = np.concatenate([np.arange(e * k, (e + 1) * k)
                                 for e in _BOX_EDGE_GROUPS["top"] + _BOX_EDGE_GROUPS["vertical"]])
        d = np.min(np.linalg.norm(pts[movers][:, None, :] - base_pts[None, :, :], axis=2), axis=1)
        vis = vis.copy()
        vis[movers[d <= base_clearance_px]] = False
    return pts, nrm, vis


def open_box_edges(p: np.ndarray, view: EdgeView, k: int = 24, inset: float = 0.06):
    """Open-topped box (bin, tray-like container), p = [cx, cy, yaw, w, d, h, t]:
    the outer box's edges plus the inner top rim, inset by wall thickness t.
    The rim -- dark interior against the lit wall top -- is what measures t;
    as a solid box, the bin's interior surface was off by ~11mm (p90)."""
    pts, nrm, vis = box_edges_unambiguous(p[:6], view)
    cx, cy, yaw, w, d, h, t = p
    R = _rot_z(yaw)
    iw, idd = w / 2 - t, d / 2 - t
    corners = np.array([[-iw, -idd, h], [iw, -idd, h], [iw, idd, h], [-iw, idd, h]]) @ R.T + [cx, cy, 0]
    s = np.linspace(inset, 1 - inset, k)[:, None]
    P3 = np.vstack([corners[i] + s * (corners[(i + 1) % 4] - corners[i]) for i in range(4)])
    rp = _project_roi(view, P3)
    above = _cam_center(view)[2] > h
    return (np.vstack([pts, rp]), np.vstack([nrm, _line_normals(rp, 4)]),
            np.concatenate([vis, np.full(len(rp), above)]))


EDGE_MODELS = {
    "box": lambda p, v: box_edges_unambiguous(p, v),
    "open_box": open_box_edges,
    "cylinder": lambda p, v: cylinder_edges(p[0], p[1], p[2], 0.0, p[3], v),
}


def edge_score(edges_fn, p: np.ndarray, views: list[EdgeView]) -> float:
    """Mean |gradient . normal| over all visible template edges in all views:
    how strongly the images support this geometry. Used to initialize
    parameters the visual hull can't be trusted for (heights)."""
    total, count = 0.0, 0
    for v in views:
        pts, nrm, vis = edges_fn(p, v)
        pts, nrm = pts[vis], nrm[vis]
        g = np.abs(_bilinear(v.gx, pts[:, 0], pts[:, 1], 0.0) * nrm[:, 0]
                   + _bilinear(v.gy, pts[:, 0], pts[:, 1], 0.0) * nrm[:, 1])
        total, count = total + g.sum(), count + len(g)
    return total / max(count, 1)


def scan_parameter(edges_fn, p: np.ndarray, index: int, values, views: list[EdgeView]) -> np.ndarray:
    """Set p[index] to whichever of `values` the images support best."""
    scores = []
    for val in values:
        q = p.copy()
        q[index] = val
        scores.append(edge_score(edges_fn, q, views))
    best = p.copy()
    best[index] = values[int(np.argmax(scores))]
    return best


# ---------------------------------------------------------------- fitting

def _param_bounds(family: str, p0: np.ndarray):
    """Keep the optimizer physical: positive sizes within a factor of the
    visual-hull start, center within a few cm, yaw within +-45 deg of the
    start (a box is 90-deg symmetric; unbounded, yaw wrapped to -5581 deg
    and a small tray collapsed to zero width)."""
    lo, hi = np.full(len(p0), -np.inf), np.full(len(p0), np.inf)
    lo[:2], hi[:2] = p0[:2] - 0.03, p0[:2] + 0.03
    if family == "box":
        lo[2], hi[2] = p0[2] - np.pi / 4, p0[2] + np.pi / 4
        lo[3:], hi[3:] = p0[3:] * 0.5, p0[3:] * 1.6
        lo[5] = p0[5] * 0.15  # height: visual hulls overshoot height most (phantom caps on wide, flat objects)
    elif family == "cylinder":
        lo[2:], hi[2:] = p0[2:] * 0.5, p0[2:] * 1.6
    elif family == "rack_with_tube_grid":
        lo[2], hi[2] = p0[2] - np.pi / 4, p0[2] + np.pi / 4
        lo[3:6], hi[3:6] = p0[3:6] * 0.5, p0[3:6] * 1.6
        lo[6:8], hi[6:8] = -0.03, 0.03
        lo[8:10], hi[8:10] = p0[8:10] * 0.7, p0[8:10] * 1.4   # pitch: a hint, not a given
        lo[10], hi[10] = p0[10] * 0.4, p0[10] * 2.5            # tube radius
        lo[11], hi[11] = p0[5] * 0.5 + 0.002, p0[11] * 1.6     # tube top, above the rack
        lo[12], hi[12] = 0.2 * p0[10], 1.6 * p0[10]            # apparent rim radius (nuisance)
    elif family == "open_box":
        lo[2], hi[2] = p0[2] - np.pi / 4, p0[2] + np.pi / 4
        lo[3:6], hi[3:6] = p0[3:6] * 0.5, p0[3:6] * 1.6
        lo[6], hi[6] = 0.0003, 0.25 * min(p0[3], p0[4])
    elif family == "plate_with_well_grid":
        lo[2], hi[2] = p0[2] - np.pi / 4, p0[2] + np.pi / 4
        lo[3:6], hi[3:6] = p0[3:6] * 0.5, p0[3:6] * 1.6
        lo[6:8], hi[6:8] = -0.02, 0.02
        lo[8:10], hi[8:10] = p0[8:10] * 0.7, p0[8:10] * 1.4   # pitch: a hint, not a given
        lo[10], hi[10] = p0[10] * 0.4, p0[10] * 2.0            # well radius
    elif family == "cylinder_grid":
        lo[2], hi[2] = p0[2] - np.pi / 4, p0[2] + np.pi / 4
        lo[3:5], hi[3:5] = p0[3:5] * 0.7, p0[3:5] * 1.4   # pitch: a hint, not a given
        lo[5], hi[5] = p0[5] * 0.4, p0[5] * 2.5           # radius
        lo[6], hi[6] = p0[6] * 0.8, p0[6] * 1.3           # top height
    return lo, hi


def _normal_search(view: EdgeView, pts: np.ndarray, nrm: np.ndarray, radius_px: float,
                   min_grad: float, step: float = 0.5) -> np.ndarray:
    """Sub-pixel edge offset along each outline normal: the nearest local
    maximum of |grad . n| (gradient aligned with the normal -- edges that
    cross the outline don't count) above min_grad, refined by a parabola
    through the peak. NaN where nothing qualifies."""
    ts = np.arange(-radius_px, radius_px + step / 2, step)
    xs = pts[:, :1] + ts[None, :] * nrm[:, :1]
    ys = pts[:, 1:] + ts[None, :] * nrm[:, 1:]
    flat_x, flat_y = xs.ravel(), ys.ravel()
    gx = _bilinear(view.gx, flat_x, flat_y, 0.0).reshape(xs.shape)
    gy = _bilinear(view.gy, flat_x, flat_y, 0.0).reshape(xs.shape)
    score = np.abs(gx * nrm[:, :1] + gy * nrm[:, 1:])
    out = np.full(len(pts), np.nan)
    center = len(ts) // 2
    for k in range(len(pts)):
        sk = score[k]
        peaks = np.where((sk[1:-1] >= sk[:-2]) & (sk[1:-1] > sk[2:]) & (sk[1:-1] >= min_grad))[0] + 1
        if len(peaks) == 0:
            continue
        j = peaks[np.argmin(np.abs(peaks - center))]
        a, b, c = sk[j - 1], sk[j], sk[j + 1]
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0
        out[k] = ts[j] + np.clip(delta, -0.5, 0.5) * step
    return out


def fit_template(family: str, views: list[EdgeView], p0: np.ndarray, outside_px: float = 20.0,
                 robust_scale_px: float = 1.5, refine_iters: int = 6, min_grad: float = 4.0,
                 template=None, edges_fn=None, skip_coarse: bool = False, bounds=None) -> FitResult:
    """Coarse: distance-to-nearest-Canny-edge objective from the visual-hull
    start. Refine: ICP-style -- match each outline point to a sub-pixel,
    orientation-consistent edge along its normal, then minimize point-to-
    line distances; repeat with a shrinking search radius."""
    gen, names = template or TEMPLATES[family]
    p0 = np.asarray(p0, float)
    lo, hi = bounds if bounds is not None else _param_bounds(family, p0)
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

    def coarse_residuals(p):
        pts = gen(p)
        return np.concatenate([_bilinear(v.dist_map, *outline_samples(pts, v).T, outside_px) for v in views])

    p = p0 if skip_coarse else least_squares(
        coarse_residuals, p0, bounds=(lo, hi), loss="soft_l1", f_scale=robust_scale_px,
        x_scale="jac", diff_step=1e-4, max_nfev=400).x
    edges_fn = edges_fn or EDGE_MODELS.get(family)

    def samples(q, v):
        if edges_fn is not None:
            return edges_fn(q, v)
        pts, nrm = outline_samples(gen(q), v, with_normals=True)
        return pts, nrm, np.ones(len(pts), bool)

    for it in range(refine_iters):
        radius = max(3.0, 10.0 - 1.5 * it)
        matches = []
        for v in views:
            pts, nrm, vis = samples(p, v)
            off = np.full(len(pts), np.nan)
            off[vis] = _normal_search(v, pts[vis], nrm[vis], radius, min_grad)
            ok = ~np.isnan(off)
            matches.append((v, ok, pts[ok] + off[ok, None] * nrm[ok], nrm[ok]))

        def refine_residuals(q):
            r = []
            for v, ok, target, n_ in matches:
                cur = samples(q, v)[0][ok]
                r.append(np.sum((cur - target) * n_, axis=1))
            return np.concatenate(r)

        p = least_squares(refine_residuals, p, bounds=(lo, hi), loss="soft_l1", f_scale=1.0,
                          x_scale="jac", diff_step=1e-5, max_nfev=200).x

    r = refine_residuals(p)
    per_view, i = {}, 0
    for v, ok, _, _ in matches:
        n = int(ok.sum())
        per_view[v.name] = float(np.sqrt(np.mean(r[i:i + n] ** 2))) if n else float("nan")
        i += n
    return FitResult(family, dict(zip(names, map(float, p))), float(np.sqrt(np.mean(r ** 2))), len(r), per_view)

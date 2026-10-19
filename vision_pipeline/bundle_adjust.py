"""Joint refinement of every camera pose AND every marker body's pose from
all views' marker corners at once (bundle adjustment).

Why: per-view PnP trusts the marker layout as given. A real layout is
tape-measured -- millimetres off -- which caps camera accuracy at that
level, far from a 0.5mm measurement target. Here the layout is only an
initial guess: each rigid marker body (a printed flat marker, or a whole
calibration cube) gets a free 6-DoF pose, and its *internal* geometry is
trusted instead -- the printed square's side (measured with calipers) and
a cube's machined faces. Those fix absolute scale; every marker seen from
several cameras ties the cameras together.

The measured layout is a PRIOR, not discarded: each body may move from
its measured pose, penalised beyond the measurement uncertainty
(prior_pos_sigma_m / prior_rot_sigma_rad). Leaving bodies completely free
was tried first and was worse than plain PnP (camera median 4.0 -> 12.9mm
from a correct layout): absolute scale then rests only on each small
marker's own size, a few dozen pixels across, instead of the 1.3m spread
between markers. The priors also fix the gauge (the platform frame).

Without priors (prior_pos_sigma_m=None) the old behavior applies: the
most-observed body is held fixed and the result is rigidly re-aligned to
the nominal layout.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


@dataclass
class Observation:
    view: int
    body: int
    corner_index: int      # into that body's stacked nominal corners (n*4)
    pixel: np.ndarray      # (2,) undistorted pixel coordinates


@dataclass
class BAResult:
    rvecs: np.ndarray          # (V,3) world->camera
    tvecs: np.ndarray          # (V,3)
    body_corners: list         # per body: (n*4, 3) refined world corners
    rms_before_px: float
    rms_after_px: float
    per_view_rms_px: np.ndarray


def _undistort(px: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    if not np.any(dist):
        return px
    return cv2.undistortPoints(px.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)


def build_bodies(marker_corners: dict[int, np.ndarray], rigid_groups: list[list[int]]):
    """Bodies = each rigid group, plus every marker not in a group alone.
    Returns (bodies: list of marker-id lists, lookup: marker_id -> (body, offset))."""
    grouped = {m for g in rigid_groups for m in g}
    bodies = [list(g) for g in rigid_groups] + [[m] for m in sorted(marker_corners) if m not in grouped]
    lookup = {}
    for b, ids in enumerate(bodies):
        for k, m in enumerate(ids):
            lookup[m] = (b, 4 * k)
    return bodies, lookup


def bundle_adjust(
    detections: list[dict[int, np.ndarray]],     # per view: marker_id -> (4,2) px
    Ks: list[np.ndarray], dists: list[np.ndarray],
    init_rvecs: np.ndarray, init_tvecs: np.ndarray,
    marker_corners: dict[int, np.ndarray],       # nominal world corners (4,3)
    rigid_groups: list[list[int]],
    huber_px: float = 1.0,
    prior_pos_sigma_m: float | None = 0.003,
    prior_rot_sigma_rad: float = np.radians(1.0),
    body_priors: dict[int, tuple[float, float]] | None = None,
) -> BAResult:
    """body_priors: marker_id -> (pos_sigma_m, rot_sigma_rad) overriding the
    default prior for the body containing that marker -- e.g. a printed
    board known to ~0.1mm, next to cubes placed only roughly."""
    bodies, lookup = build_bodies(marker_corners, rigid_groups)
    nominal = [np.concatenate([marker_corners[m] for m in ids]) for ids in bodies]
    centroids = [c.mean(axis=0) for c in nominal]

    obs: list[Observation] = []
    for v, det in enumerate(detections):
        for m, px in det.items():
            if m not in lookup:
                continue
            b, off = lookup[m]
            und = _undistort(np.asarray(px, float), Ks[v], dists[v])
            for k in range(4):
                obs.append(Observation(v, b, off + k, und[k]))
    if not obs:
        raise ValueError("no marker observations")

    counts = np.bincount([o.body for o in obs], minlength=len(bodies))
    use_priors = prior_pos_sigma_m is not None
    fixed = None if use_priors else int(np.argmax(counts))
    free = [b for b in range(len(bodies)) if b != fixed and counts[b] > 0]
    free_index = {b: i for i, b in enumerate(free)}
    V = len(detections)

    ov = np.array([o.view for o in obs])
    ob = np.array([o.body for o in obs])
    oc = np.array([o.corner_index for o in obs])
    opx = np.array([o.pixel for o in obs])
    fx = np.array([Ks[v][0, 0] for v in range(V)])[ov]
    fy = np.array([Ks[v][1, 1] for v in range(V)])[ov]
    cx = np.array([Ks[v][0, 2] for v in range(V)])[ov]
    cy = np.array([Ks[v][1, 2] for v in range(V)])[ov]
    X_nom = np.array([nominal[b][c] for b, c in zip(ob, oc)])
    X_ctr = np.array([centroids[b] for b in ob])

    def unpack(x):
        cams = x[:6 * V].reshape(V, 6)
        bod = np.zeros((len(bodies), 6))
        if free:
            bod[free] = x[6 * V:].reshape(len(free), 6)
        return cams, bod

    def world_points(bod):
        R = Rotation.from_rotvec(bod[ob, :3])
        return R.apply(X_nom - X_ctr) + X_ctr + bod[ob, 3:]

    def residuals(x):
        cams, bod = unpack(x)
        Xw = world_points(bod)
        Xc = Rotation.from_rotvec(cams[ov, :3]).apply(Xw) + cams[ov, 3:]
        z = Xc[:, 2]
        u = fx * Xc[:, 0] / z + cx
        w = fy * Xc[:, 1] / z + cy
        r = np.stack([u - opx[:, 0], w - opx[:, 1]], axis=1).ravel()
        if use_priors and free:
            # Deviation from the measured layout, in units of its uncertainty.
            # Pixel residuals are ~1px-sigma (Huber scale), so these weigh in
            # comparably.
            dev = bod[free]
            r = np.concatenate([r, (dev[:, :3] / rot_sig[:, None]).ravel(),
                                (dev[:, 3:] / pos_sig[:, None]).ravel()])
        return r

    x0 = np.concatenate([np.hstack([init_rvecs.reshape(V, 3), init_tvecs.reshape(V, 3)]).ravel(),
                         np.zeros(6 * len(free))])
    pos_sig = np.full(len(free), prior_pos_sigma_m or 1.0)
    rot_sig = np.full(len(free), prior_rot_sigma_rad)
    for m, (ps, rs) in (body_priors or {}).items():
        if m in lookup and lookup[m][0] in free_index:
            i = free_index[lookup[m][0]]
            pos_sig[i], rot_sig[i] = ps, rs

    n_prior = 6 * len(free) if use_priors else 0
    sparsity = lil_matrix((2 * len(obs) + n_prior, len(x0)), dtype=int)
    rows = np.arange(len(obs))
    for k in range(6):
        sparsity[2 * rows, 6 * ov + k] = 1
        sparsity[2 * rows + 1, 6 * ov + k] = 1
    for i, b in enumerate(ob):
        if b in free_index:
            col = 6 * V + 6 * free_index[b]
            sparsity[2 * i, col:col + 6] = 1
            sparsity[2 * i + 1, col:col + 6] = 1
    if use_priors:
        for i in range(len(free)):
            for k in range(6):
                # rows: rot priors first (3 per body), then pos priors
                row = 2 * len(obs) + (3 * i + k if k < 3 else 3 * len(free) + 3 * i + (k - 3))
                sparsity[row, 6 * V + 6 * i + k] = 1

    n_px = 2 * len(obs)
    r0 = residuals(x0)[:n_px]
    sol = least_squares(residuals, x0, jac_sparsity=sparsity, loss="huber", f_scale=huber_px,
                        x_scale="jac", method="trf", max_nfev=200)
    r1 = sol.fun[:n_px].reshape(-1, 2)

    cams, bod = unpack(sol.x)
    refined = []
    for b in range(len(bodies)):
        R = Rotation.from_rotvec(bod[b, :3])
        refined.append(R.apply(nominal[b] - centroids[b]) + centroids[b] + bod[b, 3:])

    # Re-anchor (no-prior mode only -- with priors the layout already fixes
    # the frame): best rigid fit of all refined corners onto the nominal
    # layout, applied to bodies and cameras alike.
    if use_priors:
        R_al, t_al = np.eye(3), np.zeros(3)
    else:
        R_al, t_al = _procrustes(np.concatenate(refined), np.concatenate(nominal))
    refined = [c @ R_al.T + t_al for c in refined]
    rvecs, tvecs = [], []
    for v in range(V):
        R_wc = Rotation.from_rotvec(cams[v, :3]).as_matrix()
        # x_cam = R_wc x_old + t; x_new = R_al x_old + t_al  =>  x_old = R_al^T (x_new - t_al)
        R_new = R_wc @ R_al.T
        t_new = cams[v, 3:] - R_new @ t_al
        rvecs.append(Rotation.from_matrix(R_new).as_rotvec())
        tvecs.append(t_new)

    per_view = np.array([np.sqrt(np.mean(np.sum(r1[ov == v] ** 2, axis=1))) if np.any(ov == v) else np.nan
                         for v in range(V)])
    return BAResult(np.array(rvecs), np.array(tvecs), refined,
                    float(np.sqrt(np.mean(r0 ** 2) * 2)), float(np.sqrt(np.mean(r1 ** 2) * 2)), per_view)


def _procrustes(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimising ||R A + t - B||, no scale."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca

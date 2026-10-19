"""Scoring a reconstructed object against ground-truth meshes.

Matched against the UNION of the ground-truth meshes whose boxes lie
mostly inside the reconstruction's box: a rack fused with its tubes is one
reconstructed object, and splitting it is the decomposition step's job.

Two-way surface distances (exact point-to-surface, not point-to-sample):
  accuracy      reconstructed surface -> ground truth  (is what we built there?)
  completeness  ground truth -> reconstructed surface  (did we build what's there?)

Each is also reported for the OBSERVABLE part only (points that some
camera actually sees, judged against the ground truth's own depth): the
inside of a closed housing, the shafts of pipette tips sunk in a rack or
a rotor under its lid are seen by no camera, and no method can measure
them -- they're reported separately, not hidden.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import trimesh


def gt_members(gt: dict, lo: np.ndarray, hi: np.ndarray, min_inside: float = 0.5) -> list[str]:
    names = []
    for name, g in gt.items():
        glo, ghi = np.array(g["bbox_min"]), np.array(g["bbox_max"])
        vol = max(float((ghi - glo).prod()), 1e-12)
        inter = np.clip(np.minimum(hi, ghi) - np.maximum(lo, glo), 0, None).prod()
        if inter > min_inside * vol:
            names.append(name)
    return names


def visible_mask(points: np.ndarray, vis_views, tol_m: float = 0.002) -> np.ndarray:
    """vis_views: [(K, rvec, tvec, depth)] with depth of the TRUE scene.
    A point is observable if, in some view, it is the first surface along
    its ray (projected depth within tol of the depth map there)."""
    seen = np.zeros(len(points), bool)
    for K, rvec, tvec, depth in vis_views:
        R, _ = cv2.Rodrigues(np.asarray(rvec, float))
        cam = points @ R.T + np.asarray(tvec, float).reshape(1, 3)
        z = cam[:, 2]
        ok = z > 1e-3
        zs = np.where(ok, z, 1.0)
        u = np.round(K[0, 0] * cam[:, 0] / zs + K[0, 2]).astype(int)
        v = np.round(K[1, 1] * cam[:, 1] / zs + K[1, 2]).astype(int)
        h, w = depth.shape
        ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d = np.zeros(len(points), np.float32)
        d[ok] = depth[v[ok], u[ok]]
        # tolerance grows with the pixel footprint (a 1px offset on a slope)
        tol = np.maximum(tol_m, 3.0 * zs / K[0, 0])
        seen |= ok & (d > 0) & (np.abs(d - z) < tol)
    return seen


def _stats(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"median": None, "p90": None}
    return {"median": round(float(np.median(x)), 3), "p90": round(float(np.percentile(x, 90)), 3)}


def score_mesh(mesh: trimesh.Trimesh, gt: dict, gt_root: Path, vis_views=None, n: int = 30000) -> dict:
    lo, hi = mesh.bounds
    names = gt_members(gt, lo, hi)
    if not names:
        return {"ground_truth": None}
    gm = trimesh.util.concatenate([trimesh.load(Path(gt_root) / gt[nm]["mesh"], force="mesh") for nm in names])
    a = trimesh.sample.sample_surface(mesh, n, seed=0)[0]
    b = trimesh.sample.sample_surface(gm, n, seed=0)[0]
    acc = trimesh.proximity.closest_point(gm, a)[1] * 1000
    comp = trimesh.proximity.closest_point(mesh, b)[1] * 1000
    glo, ghi = gm.bounds
    out = {"ground_truth": names if len(names) > 1 else names[0],
           "true_dims_mm": ((ghi - glo) * 1000).round(2).tolist(),
           "dims_mm": ((hi - lo) * 1000).round(2).tolist(),
           "dim_err_mm": (((hi - lo) - (ghi - glo)) * 1000).round(2).tolist(),
           "center_err_mm": round(float(np.linalg.norm(((hi + lo) - (ghi + glo))[:2] / 2) * 1000), 2),
           "accuracy_mm": _stats(acc), "completeness_mm": _stats(comp)}
    if vis_views:
        va, vb = visible_mask(a, vis_views), visible_mask(b, vis_views)
        out["observable"] = {"accuracy_mm": _stats(acc[va]), "completeness_mm": _stats(comp[vb]),
                             "fraction_of_recon_observable": round(float(va.mean()), 3),
                             "fraction_of_truth_observable": round(float(vb.mean()), 3)}
    return out


def format_score(s: dict) -> str:
    if not s.get("ground_truth"):
        return "no ground-truth match"
    g = s["ground_truth"] if isinstance(s["ground_truth"], str) else f"{len(s['ground_truth'])} parts"
    txt = (f"vs {g}: dims err {s['dim_err_mm']} mm, acc {s['accuracy_mm']['median']}/{s['accuracy_mm']['p90']}, "
           f"compl {s['completeness_mm']['median']}/{s['completeness_mm']['p90']}")
    if "observable" in s:
        o = s["observable"]
        txt += (f" | observable: acc {o['accuracy_mm']['median']}/{o['accuracy_mm']['p90']}, "
                f"compl {o['completeness_mm']['median']}/{o['completeness_mm']['p90']}"
                f" ({o['fraction_of_truth_observable'] * 100:.0f}% of truth seen)")
    return txt + " mm (med/p90)"

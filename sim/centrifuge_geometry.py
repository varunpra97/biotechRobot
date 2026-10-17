"""Rotor bore geometry of the scene's centrifuge, in world coordinates,
derived from the same mesh parts the simulator loads
(lab_scene.centrifuge_parts_local + centrifuge_pos) -- so task targets
come from the real geometry, not hand-entered numbers.

The rotor is fixed-angle: each of its six sleeves is an 8.0mm-ID bore
with a conical closed floor, tilted ~45 deg from vertical with its top
leaning in toward the hub. A bore is described by an axis line
(point + unit direction pointing UP, out of the bore) and axial
coordinates s along it (s = 0 at the fitted axis point).

Measured once with cad_source/raycast_rotor_bore.py (bore 8 shown; all six
are copies rotated about the hub):
  - cylindrical bore radius 3.93-4.07mm (tessellation), s -14 .. +17mm
  - conical floor from s = -14mm down to its apex at s = -16.3mm
  - tightest clear radius ~3.65mm where the bore passes the hub's top face
  - opening (axis meets the hub's top face) at s ~ +16mm
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import lab_scene

BORE_RADIUS = 0.0040           # m, cylindrical part of the bore
BORE_CLEAR_RADIUS = 0.00365    # m, tightest point (hub top face)


@dataclass
class Bore:
    sleeve_part: int
    point: np.ndarray       # world point on the axis (s = 0)
    axis: np.ndarray        # unit, pointing up/out of the bore
    opening_s: float        # s where the axis crosses the hub's top face
    hub_center: np.ndarray  # world

    def at(self, s):
        return self.point + s * self.axis

    @property
    def tilt_deg(self):
        return float(np.degrees(np.arccos(self.axis[2])))

    def s_of(self, p):
        return float((np.asarray(p) - self.point) @ self.axis)

    def radial_offset(self, p):
        d = np.asarray(p) - self.point
        return float(np.linalg.norm(d - (d @ self.axis) * self.axis))


def _fit_axis(v):
    """PCA direction, refined by a line through the centroids of thin
    axial slices (the closed conical end would otherwise bias the plain
    vertex mean along the axis). Direction returned pointing up."""
    c = v.mean(0)
    _, vecs = np.linalg.eigh(np.cov((v - c).T))
    a = vecs[:, -1]
    if a[2] < 0:
        a = -a
    for _ in range(3):
        s = (v - c) @ a
        edges = np.linspace(s.min(), s.max(), 25)
        cents = np.array([v[(s >= lo) & (s < hi)].mean(0)
                          for lo, hi in zip(edges[:-1], edges[1:])
                          if ((s >= lo) & (s < hi)).sum() >= 8])
        c = cents.mean(0)
        _, _, vt = np.linalg.svd(cents - c)
        a = vt[0] if vt[0][2] > 0 else -vt[0]
    return c, a / np.linalg.norm(a)


def rotor_bores():
    """All six bores, keyed by sleeve part index."""
    parts = lab_scene.centrifuge_parts_local()
    pos = np.array(lab_scene.centrifuge_pos())
    hub = parts[lab_scene.CENTRIFUGE_HUB_PART].vertices + pos
    hub_center = hub.mean(0)
    hub_top = hub[:, 2].max()
    out = {}
    for idx in lab_scene.CENTRIFUGE_SLEEVE_PARTS:
        c, a = _fit_axis(parts[idx].vertices + pos)
        out[idx] = Bore(idx, c, a, float((hub_top - c[2]) / a[2]), hub_center)
    return out


def body_mesh_world():
    """The loaded centrifuge body (no lid, no degenerate slivers) as one
    trimesh in world coordinates -- for geometric clearance checks."""
    import trimesh

    parts = lab_scene.centrifuge_parts_local()
    body = trimesh.util.concatenate([p for i, p in enumerate(parts)
                                     if i not in lab_scene.CENTRIFUGE_LID_PARTS
                                     and len(p.faces) > 4])
    return trimesh.Trimesh(body.vertices + np.array(lab_scene.centrifuge_pos()),
                           body.faces, process=True)

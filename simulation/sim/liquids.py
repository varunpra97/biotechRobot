"""Liquid in tubes, wells and pipette tips: a tracked volume per container
plus a rendered liquid body that fills the container's real inner profile
up to the matching height. (No fluid simulation -- volumes move only when
the pipette aspirates or dispenses; see pipette.py.)

Profiles are (z, inner radius) knots in the container's local frame, from
the CAD that built each container:
  - test tube (cad_source/make_test_tube.py): inner cone r 0.5 -> 3.0 mm
    from z 0.8 -> 7.5 mm, then a 3.0 mm-radius bore up to the 58 mm rim
  - tray well (cad_source/make_sample_tray_5x2.py): 3.45 mm radius from the
    well floor (3.7 mm) to the top (14.4 mm)
  - pipette tip (cad_source/make_lab_consumables.py): inner cone r 0.25 ->
    3.5 mm from z 0.6 -> 51 mm, then the 3.5 mm collar bore to 60 mm
Units: meters for geometry, microliters for volume (1 uL = 1e-9 m^3).
"""
from __future__ import annotations

import math

import numpy as np

UL = 1e-9  # m^3 per microliter

TUBE_PROFILE = [(0.0008, 0.0005), (0.0075, 0.0030), (0.0580, 0.0030)]
WELL_PROFILE = [(0.0037, 0.00345), (0.0144, 0.00345)]
TIP_PROFILE = [(0.0006, 0.00025), (0.0510, 0.0035), (0.0600, 0.0035)]
WELL_TOP_Z = 0.0144


class Profile:
    def __init__(self, knots):
        self.z = np.array([k[0] for k in knots])
        self.r = np.array([k[1] for k in knots])

    def radius(self, z):
        return float(np.interp(z, self.z, self.r))

    def volume_ul(self, h):
        """Liquid volume (uL) filled up to local height h."""
        v = 0.0
        for z0, z1, r0, r1 in zip(self.z[:-1], self.z[1:], self.r[:-1], self.r[1:]):
            if h <= z0:
                break
            zt = min(h, z1)
            rt = r0 + (r1 - r0) * (zt - z0) / (z1 - z0)
            v += math.pi * (zt - z0) / 3.0 * (r0 * r0 + r0 * rt + rt * rt)
        return v / UL

    def height(self, vol_ul):
        if vol_ul <= 0:
            return float(self.z[0])
        lo, hi = float(self.z[0]), float(self.z[-1])
        for _ in range(50):
            mid = (lo + hi) / 2
            if self.volume_ul(mid) < vol_ul:
                lo = mid
            else:
                hi = mid
        return hi

    @property
    def capacity_ul(self):
        return self.volume_ul(float(self.z[-1]))

    def mesh(self, h, sections=24, inset=5e-5):
        """Closed liquid body (vertices, triangles) from the bottom of the
        profile up to height h, radius inset slightly from the wall."""
        zs = [z for z in self.z if z < h] + [h]
        ring_r = [max(self.radius(z) - inset, 1e-5) for z in zs]
        th = np.linspace(0, 2 * np.pi, sections, endpoint=False)
        verts = [(0.0, 0.0, zs[0])]
        for z, r in zip(zs, ring_r):
            verts += [(r * math.cos(t), r * math.sin(t), z) for t in th]
        verts.append((0.0, 0.0, zs[-1]))
        n, tris = sections, []
        for i in range(n):  # bottom cap
            tris.append((0, 1 + (i + 1) % n, 1 + i))
        for k in range(len(zs) - 1):  # sides
            a, b = 1 + k * n, 1 + (k + 1) * n
            for i in range(n):
                j = (i + 1) % n
                tris += [(a + i, a + j, b + j), (a + i, b + j, b + i)]
        top, last = len(verts) - 1, 1 + (len(zs) - 1) * n
        for i in range(n):  # top surface
            tris.append((top, last + i, last + (i + 1) % n))
        return np.array(verts), np.array(tris)


TUBE = Profile(TUBE_PROFILE)
WELL = Profile(WELL_PROFILE)
TIP = Profile(TIP_PROFILE)


class LiquidBody:
    """A rendered liquid inside one container prim (a child mesh, so it
    moves with the container)."""

    def __init__(self, parent_path, profile, volume_ul, color, offset=(0.0, 0.0, 0.0), name="liquid"):
        self.path = f"{parent_path}/{name}"
        self.profile = profile
        self.color = color
        self.offset = np.asarray(offset, float)
        self.volume_ul = float(volume_ul)
        self._define()
        self.set_volume(volume_ul)

    def _define(self):
        import isaacsim.core.experimental.utils.stage as stage_utils
        from pxr import Gf, UsdGeom

        stage = stage_utils.get_current_stage()
        self.mesh = UsdGeom.Mesh.Define(stage, self.path)
        self.mesh.CreateDisplayColorAttr([Gf.Vec3f(*self.color)])
        xf = UsdGeom.Xformable(self.mesh)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*self.offset))

    def set_volume(self, vol_ul):
        from pxr import Gf, UsdGeom

        self.volume_ul = max(0.0, float(vol_ul))
        vis = UsdGeom.Imageable(self.mesh)
        if self.volume_ul < 0.05:
            vis.MakeInvisible()
            return
        v, t = self.profile.mesh(self.profile.height(self.volume_ul))
        self.mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in v])
        self.mesh.CreateFaceVertexCountsAttr([3] * len(t))
        self.mesh.CreateFaceVertexIndicesAttr(t.flatten().tolist())
        vis.MakeVisible()

    def set_color(self, color):
        from pxr import Gf

        self.color = tuple(color)
        self.mesh.CreateDisplayColorAttr([Gf.Vec3f(*self.color)])

    def surface_z_local(self):
        return self.profile.height(self.volume_ul) + self.offset[2]

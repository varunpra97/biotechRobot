"""Meshes from fitted templates (template_fitting.py results).

A fitted template is a handful of measured numbers; this turns them into
clean, watertight geometry for the simulation -- exact planar faces and
cylinders instead of a voxel surface, with cavities (wells, bores) the
visual hull could never see.

Some dimensions can't be measured from outside photos at all: how deep a
well or bore goes, how far a tube reaches down inside its bore. Those come
from `hints` (LLM scene reasoning: a standard, a product spec, or a
category-typical value) and otherwise from conservative defaults. Every
such value is returned in `provenance` as "assumed" or "hint", so an
asset never presents a guess as a measurement.

All meshes are built in the object's own frame: origin at the center of
its footprint on the bench, z up, unrotated. The fitted center and yaw go
into the placement instead (same split as assets/cad + lab_scene.py).
"""
from __future__ import annotations

import numpy as np
import trimesh

DEFAULTS = {
    "well_depth_frac": 0.7,       # of plate height
    "bore_depth_frac": 0.7,       # of rack height
    "bore_clearance_m": 0.0005,   # bore radius = tube radius + clearance
    "tube_wall_m": 0.0005,
}


def _cyl(radius, height, z0, xy=(0.0, 0.0), sections=48):
    c = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    c.apply_translation([xy[0], xy[1], z0 + height / 2])
    return c


def _local_grid(rows, cols, px, py, ox, oy):
    return [np.array([ox + (j - (cols - 1) / 2) * px, oy + (i - (rows - 1) / 2) * py])
            for i in range(rows) for j in range(cols)]


def box_mesh(p: dict) -> tuple[trimesh.Trimesh, dict]:
    m = trimesh.creation.box([p["w"], p["d"], p["h"]])
    m.apply_translation([0, 0, p["h"] / 2])
    return m, {"w": "measured", "d": "measured", "h": "measured"}


def open_box_mesh(p: dict, hints: dict | None = None):
    """Box with its interior removed down to the floor. Wall thickness t is
    measured (inner top rim); floor thickness can't be seen from outside."""
    hints = hints or {}
    box, prov = box_mesh(p)
    if "floor_thickness_m" in hints:
        floor, prov["floor_thickness"] = hints["floor_thickness_m"], "hint"
    else:
        floor, prov["floor_thickness"] = p["t"], "assumed (= measured wall)"
    prov["wall"] = "measured"
    inner = trimesh.creation.box([p["w"] - 2 * p["t"], p["d"] - 2 * p["t"], p["h"]])
    inner.apply_translation([0, 0, floor + p["h"] / 2])
    return box.difference(inner, engine="manifold"), prov


def plate_with_wells_mesh(p: dict, rows: int, cols: int, hints: dict | None = None):
    hints = hints or {}
    plate, prov = box_mesh(p)
    if "well_depth_m" in hints:
        depth, prov["well_depth"] = hints["well_depth_m"], "hint"
    else:
        depth, prov["well_depth"] = DEFAULTS["well_depth_frac"] * p["h"], "assumed"
    wells = [_cyl(p["rw"], depth + 0.001, p["h"] - depth, c)
             for c in _local_grid(rows, cols, p["pitch_x"], p["pitch_y"], p["ox"], p["oy"])]
    prov.update(well_radius="measured", well_pitch="measured")
    return plate.difference(trimesh.util.concatenate(wells), engine="manifold"), prov


def rack_with_tubes_meshes(p: dict, rows: int, cols: int, hints: dict | None = None):
    """(rack mesh, [(tube mesh, tube center xy in the rack frame)], provenance).
    Each tube is its own mesh: tubes in a rack are separate, removable objects."""
    hints = hints or {}
    rack, prov = box_mesh(p)
    if "bore_depth_m" in hints:
        bore_depth, prov["bore_depth"] = hints["bore_depth_m"], "hint"
    else:
        bore_depth, prov["bore_depth"] = DEFAULTS["bore_depth_frac"] * p["h"], "assumed"
    bore_r = p["r"] + hints.get("bore_clearance_m", DEFAULTS["bore_clearance_m"])
    prov["bore_radius"] = "measured tube radius + " + ("hint" if "bore_clearance_m" in hints else "assumed") + " clearance"
    centers = _local_grid(rows, cols, p["pitch_x"], p["pitch_y"], p["ox"], p["oy"])
    bores = [_cyl(bore_r, bore_depth + 0.001, p["h"] - bore_depth, c) for c in centers]
    rack = rack.difference(trimesh.util.concatenate(bores), engine="manifold")

    # A tube rests on its bore floor (as lab_scene stands its tubes), so its
    # length = measured top height - bore floor height.
    floor = p["h"] - bore_depth
    length = p["z_top"] - floor
    wall = p.get("wall", 0.0) if p.get("wall", 0.0) > 1e-4 else DEFAULTS["tube_wall_m"]
    prov["tube_length"] = "measured top - " + prov["bore_depth"] + " bore floor"
    prov["tube_wall"] = "measured" if p.get("wall", 0.0) > 1e-4 else "assumed"
    outer = _cyl(p["r"], length, 0.0)
    inner = _cyl(p["r"] - wall, length, wall)   # open top, closed bottom
    tube = outer.difference(inner, engine="manifold")
    tubes = [(tube.copy(), np.array([c[0], c[1], floor])) for c in centers]
    return rack, tubes, prov

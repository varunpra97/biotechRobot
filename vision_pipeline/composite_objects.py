"""Procedural COMPOSITE test objects: several separate parts in contact, each
with its own material -- the cases the pipeline must split and weigh:

  tube_rack         a block with a grid of blind bores, tubes standing in a
                    random subset of them (test tubes in a holder)
  mini_centrifuge   a housing with an open bowl; inside, a rotor on a hub
                    with tubes in some of its holes (a centrifuge interior,
                    visible only from above)
  tray_with_weights an open tray with metal cylinders standing inside it
  bottle_with_cap   a bottle with a screw cap of a different material
  box_with_lid      a box with a loose lid resting on it
  pegboard          a board with metal pins in a random subset of holes

Every part is a separate watertight mesh with a material from MATERIALS,
so per-part geometry, material, density and mass are exact ground truth
(mass = mesh volume x density). Parts touch but never interpenetrate.
Object-local frame: base on z = 0, centered in xy, meters.
"""
from __future__ import annotations

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold

SEG = 64

# Rendering appearance per material (UsdPreviewSurface), and the physical
# truth the pipeline is scored against. Densities are typical handbook
# values. No glass: transparent parts defeat both depth sensors and
# silhouettes, which is a different problem from the one tested here.
MATERIALS = {
    "polypropylene": {"density": 905.0, "rgb": [0.92, 0.92, 0.88], "roughness": 0.55, "metallic": 0.0},
    "polypropylene_blue": {"density": 905.0, "rgb": [0.20, 0.40, 0.80], "roughness": 0.5, "metallic": 0.0,
                           "name": "polypropylene"},
    "polystyrene": {"density": 1050.0, "rgb": [0.85, 0.30, 0.25], "roughness": 0.4, "metallic": 0.0},
    "abs": {"density": 1040.0, "rgb": [0.25, 0.25, 0.28], "roughness": 0.45, "metallic": 0.0},
    "hdpe": {"density": 950.0, "rgb": [0.95, 0.95, 0.95], "roughness": 0.6, "metallic": 0.0},
    "aluminium": {"density": 2700.0, "rgb": [0.80, 0.81, 0.83], "roughness": 0.35, "metallic": 1.0},
    "stainless_steel": {"density": 8000.0, "rgb": [0.70, 0.70, 0.72], "roughness": 0.2, "metallic": 1.0},
    "brass": {"density": 8500.0, "rgb": [0.80, 0.62, 0.30], "roughness": 0.3, "metallic": 1.0},
    "wood_beech": {"density": 720.0, "rgb": [0.72, 0.52, 0.32], "roughness": 0.75, "metallic": 0.0},
    "rubber": {"density": 1200.0, "rgb": [0.08, 0.08, 0.08], "roughness": 0.9, "metallic": 0.0},
    "ceramic": {"density": 2400.0, "rgb": [0.96, 0.95, 0.92], "roughness": 0.15, "metallic": 0.0},
}


def material_name(key: str) -> str:
    return MATERIALS[key].get("name", key)


def _cyl(h, r, r_top=None, seg=SEG):
    return Manifold.cylinder(h, r, r if r_top is None else r_top, seg)


def _box(x, y, z):
    return Manifold.cube((x, y, z)).translate((-x / 2, -y / 2, 0))


def _mesh(m: Manifold) -> trimesh.Trimesh:
    me = m.to_mesh()
    return trimesh.Trimesh(np.asarray(me.vert_properties)[:, :3], np.asarray(me.tri_verts), process=True)


def _tube(r: float, length: float, wall: float, seg: int = 40) -> Manifold:
    """Round-bottomed tube, open top, bottom of the hemisphere at z = 0."""
    outer = _cyl(length - r, r, seg=seg).translate((0, 0, r)) + Manifold.sphere(r, seg).translate((0, 0, r))
    inner = (_cyl(length, r - wall, seg=seg).translate((0, 0, r))
             + Manifold.sphere(r - wall, seg).translate((0, 0, r)))
    return outer - inner


def _grid(rows, cols, pitch):
    return [((c - (cols - 1) / 2) * pitch, (r - (rows - 1) / 2) * pitch) for r in range(rows) for c in range(cols)]


def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


# ------------------------------------------------------------------ recipes
# Each returns (parts, info): parts = [(part_name, Manifold, material_key, role)],
# role: "base" (the holder / housing / container) or "insert" (removable).

def tube_rack(rng):
    rows, cols = int(rng.integers(2, 5)), int(rng.integers(3, 7))
    r_tube = rng.uniform(0.005, 0.008)
    pitch = 2 * r_tube + rng.uniform(0.008, 0.016)
    H = rng.uniform(0.03, 0.05)
    bore_depth = H * rng.uniform(0.55, 0.8)
    W, D = cols * pitch + rng.uniform(0.01, 0.02), rows * pitch + rng.uniform(0.01, 0.02)
    r_bore = r_tube + 0.0004
    rack = _box(W, D, H)
    holes = _grid(rows, cols, pitch)
    for x, y in holes:
        rack -= _cyl(bore_depth + 0.01, r_bore).translate((x, y, H - bore_depth))
    rack_mat = _pick(rng, ["polypropylene_blue", "aluminium", "abs"])
    tube_mat = _pick(rng, ["polypropylene", "polystyrene"])
    length = bore_depth + rng.uniform(0.025, 0.06)
    filled = [h for h in holes if rng.random() < 0.7] or holes[:1]
    parts = [("rack", rack, rack_mat, "base")]
    for i, (x, y) in enumerate(filled):
        parts.append((f"tube_{i:02d}", _tube(r_tube, length, rng.uniform(0.0007, 0.0012))
                      .translate((x, y, H - bore_depth)), tube_mat, "insert"))
    return parts, {"recipe": "tube_rack", "grid": [rows, cols], "pitch_m": pitch, "filled": len(filled),
                   "tube_length_m": length, "bore_depth_m": bore_depth}


def mini_centrifuge(rng):
    R = rng.uniform(0.07, 0.10)
    H = rng.uniform(0.09, 0.13)
    wall = rng.uniform(0.004, 0.007)
    floor = H * rng.uniform(0.25, 0.4)
    housing = _cyl(H, R, R * rng.uniform(0.9, 1.0)) - _cyl(H, R - wall).translate((0, 0, floor))
    hub_h = rng.uniform(0.01, 0.02)
    r_rotor = R - wall - rng.uniform(0.006, 0.012)
    t_rotor = rng.uniform(0.02, 0.03)
    z_rotor = floor + hub_h
    n = int(rng.integers(6, 13))
    r_tube = rng.uniform(0.004, 0.0065)
    ring = r_rotor - r_tube - rng.uniform(0.006, 0.012)
    hole_depth = t_rotor * 0.8
    rotor = _cyl(t_rotor, r_rotor).translate((0, 0, z_rotor)) + _cyl(hub_h, rng.uniform(0.008, 0.015)) \
        .translate((0, 0, floor))
    holes = [(ring * np.cos(2 * np.pi * k / n), ring * np.sin(2 * np.pi * k / n)) for k in range(n)]
    for x, y in holes:
        rotor -= _cyl(hole_depth + 0.01, r_tube + 0.0004).translate((x, y, z_rotor + t_rotor - hole_depth))
    # tubes stay below the housing rim: the interior is seen only from above
    top_room = H - (z_rotor + t_rotor) - 0.003
    length = hole_depth + rng.uniform(0.3, 0.9) * top_room
    parts = [("housing", housing, _pick(rng, ["abs", "hdpe"]), "base"),
             ("rotor", rotor, "aluminium", "insert")]
    tube_mat = _pick(rng, ["polypropylene", "polystyrene"])
    filled = [h for h in holes if rng.random() < 0.6] or holes[:2]
    for i, (x, y) in enumerate(filled):
        parts.append((f"tube_{i:02d}", _tube(r_tube, length, 0.0008)
                      .translate((x, y, z_rotor + t_rotor - hole_depth)), tube_mat, "insert"))
    return parts, {"recipe": "mini_centrifuge", "rotor_holes": n, "filled": len(filled), "bowl_floor_m": floor}


def tray_with_weights(rng):
    W, D, H = rng.uniform(0.10, 0.16), rng.uniform(0.07, 0.12), rng.uniform(0.025, 0.045)
    wall = rng.uniform(0.002, 0.004)
    tray = _box(W, D, H) - _box(W - 2 * wall, D - 2 * wall, H).translate((0, 0, wall))
    parts = [("tray", tray, _pick(rng, ["polypropylene_blue", "polystyrene", "hdpe"]), "base")]
    k = int(rng.integers(2, 5))
    placed = []
    for i in range(k):
        r, h = rng.uniform(0.008, 0.016), rng.uniform(0.015, 0.05)
        for _ in range(200):
            c = np.array([rng.uniform(-W / 2 + wall + r + 0.002, W / 2 - wall - r - 0.002),
                          rng.uniform(-D / 2 + wall + r + 0.002, D / 2 - wall - r - 0.002)])
            if all(np.linalg.norm(c - pc) > r + pr + 0.004 for pc, pr in placed):
                break
        else:
            continue
        placed.append((c, r))
        parts.append((f"weight_{i}", _cyl(h, r).translate((c[0], c[1], wall)),
                      _pick(rng, ["stainless_steel", "brass", "aluminium"]), "insert"))
    return parts, {"recipe": "tray_with_weights", "weights": len(parts) - 1}


def bottle_with_cap(rng):
    r, h = rng.uniform(0.025, 0.04), rng.uniform(0.08, 0.14)
    r_neck = r * rng.uniform(0.35, 0.5)
    neck_h = rng.uniform(0.015, 0.025)
    shoulder = rng.uniform(0.015, 0.03)
    wall = 0.0025
    prof = [(0.0, 0.0), (r, 0.0), (r, h - shoulder), (r_neck, h), (r_neck, h + neck_h), (0.0, h + neck_h)]
    body = CrossSection([prof]).revolve(SEG)
    inner = [(0.0, wall), (r - wall, wall), (r - wall, h - shoulder), (r_neck - wall, h),
             (r_neck - wall, h + neck_h + 0.01), (0.0, h + neck_h + 0.01)]
    body -= CrossSection([inner]).revolve(SEG)
    cap_h = neck_h * rng.uniform(0.8, 1.1)
    cap_r = r_neck + rng.uniform(0.002, 0.004)
    cap = _cyl(cap_h, cap_r) - _cyl(cap_h - 0.002, r_neck).translate((0, 0, -0.0001))
    # cap's inner roof rests on the neck rim
    cap = cap.translate((0, 0, h + neck_h - (cap_h - 0.002)))
    return [("bottle", body, _pick(rng, ["hdpe", "ceramic"]), "base"),
            ("cap", cap, _pick(rng, ["polystyrene", "polypropylene_blue", "rubber"]), "insert")], \
        {"recipe": "bottle_with_cap"}


def box_with_lid(rng):
    W, D, H = rng.uniform(0.08, 0.15), rng.uniform(0.06, 0.12), rng.uniform(0.04, 0.08)
    wall = rng.uniform(0.003, 0.006)
    box = _box(W, D, H) - _box(W - 2 * wall, D - 2 * wall, H).translate((0, 0, wall))
    lip = rng.uniform(0.003, 0.006)
    t = rng.uniform(0.004, 0.008)
    lid = _box(W + 2 * lip, D + 2 * lip, t) + \
        (_box(W + 2 * lip, D + 2 * lip, lip * 2) - _box(W + 0.0004, D + 0.0004, lip * 2)).translate((0, 0, -lip * 2))
    lid = lid.translate((0, 0, H))
    mat = _pick(rng, ["wood_beech", "polystyrene", "abs"])
    return [("box", box, mat, "base"), ("lid", lid, _pick(rng, ["wood_beech", "polypropylene_blue", "aluminium"]),
                                        "insert")], {"recipe": "box_with_lid"}


def pegboard(rng):
    rows, cols = int(rng.integers(2, 4)), int(rng.integers(3, 6))
    pitch = rng.uniform(0.02, 0.03)
    W, D = cols * pitch + 0.02, rows * pitch + 0.02
    T = rng.uniform(0.015, 0.025)
    r = rng.uniform(0.002, 0.0035)
    depth = T * 0.7
    board = _box(W, D, T)
    holes = _grid(rows, cols, pitch)
    for x, y in holes:
        board -= _cyl(depth + 0.01, r + 0.0003).translate((x, y, T - depth))
    pin_len = depth + rng.uniform(0.02, 0.05)
    parts = [("board", board, "wood_beech", "base")]
    for i, (x, y) in enumerate([h for h in holes if rng.random() < 0.6] or holes[:1]):
        parts.append((f"pin_{i:02d}", _cyl(pin_len, r).translate((x, y, T - depth)),
                      _pick(rng, ["stainless_steel", "brass"]), "insert"))
    return parts, {"recipe": "pegboard", "grid": [rows, cols]}


RECIPES = [tube_rack, mini_centrifuge, tray_with_weights, bottle_with_cap, box_with_lid, pegboard]


def make_composite(rng, recipe):
    parts, info = recipe(rng)
    meshes = [(n, _mesh(m), mat, role) for n, m, mat, role in parts]
    allm = trimesh.util.concatenate([m for _, m, _, _ in meshes])
    lo, hi = allm.bounds
    off = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    for _, m, _, _ in meshes:
        m.apply_translation(off)
    info["extents_m"] = (hi - lo).tolist()
    return meshes, info


def make_composite_scene(seed: int, platform_wd, keepout_xy, recipes=None, gap: float = 0.04):
    """One of each recipe (or the given list), placed without overlap with a
    random yaw. Returns [(group_name, [(part_name, mesh, material_key, role)], info)]."""
    rng = np.random.default_rng(seed)
    hw, hd = platform_wd[0] / 2 - 0.02, platform_wd[1] / 2 - 0.02
    placed = []
    out = []
    for k, recipe in enumerate(recipes or RECIPES):
        parts, info = make_composite(rng, recipe)
        yaw = rng.uniform(0, 2 * np.pi)
        Rz = trimesh.transformations.rotation_matrix(yaw, (0, 0, 1))
        for _, m, _, _ in parts:
            m.apply_transform(Rz)
        ext = trimesh.util.concatenate([m for _, m, _, _ in parts]).extents
        r = np.linalg.norm(ext[:2]) / 2
        for _ in range(5000):
            c = np.array([rng.uniform(-hw + r, hw - r), rng.uniform(-hd + r, hd - r)])
            if any(np.linalg.norm(c - pc) < r + pr + gap for pc, pr in placed):
                continue
            if any(np.all(c + r > lo) and np.all(c - r < hi) for lo, hi in keepout_xy):
                continue
            break
        else:
            raise RuntimeError(f"could not place composite {k}")
        placed.append((c, r))
        for _, m, _, _ in parts:
            m.apply_translation((c[0], c[1], 0.0))
        info.update({"yaw_rad": float(yaw), "center_xy_m": c.tolist()})
        out.append((f"Comp{k:02d}_{info['recipe']}", parts, info))
    return out

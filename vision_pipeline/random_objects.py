"""Procedural random "complex" objects for testing reconstruction against
known geometry: solids built by exact booleans (manifold3d) from
primitives, with the features that break simple methods --

  through holes and tunnels   (a visual hull fills them unless seen end-on)
  blind pockets, open cavities (no silhouette ever sees into them)
  thin walls                   (2-4mm: fusion's hardest case)
  overhangs and undercuts      (surfaces facing down, partly hidden)
  handles                      (genus > 0, thin curved parts)
  revolved profiles            (bottles, flasks, funnels: smooth curvature)

Every object rests on a flat base at z = 0, centered in xy, in meters.
Deterministic for a given seed.
"""
from __future__ import annotations

import numpy as np
import trimesh
from manifold3d import CrossSection, Manifold

SEGMENTS = 96          # circle resolution: ~3mm chords at 100mm diameter


def _cyl(h, r, r_top=None, seg=SEGMENTS):
    return Manifold.cylinder(h, r, r if r_top is None else r_top, seg)


def _box(x, y, z, centered_xy=True):
    m = Manifold.cube((x, y, z))
    return m.translate((-x / 2, -y / 2, 0)) if centered_xy else m


def _to_trimesh(m: Manifold) -> trimesh.Trimesh:
    mesh = m.to_mesh()
    t = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=True)
    return t


# ------------------------------------------------------------------ recipes
# Each takes an rng, returns (Manifold in meters, base on z=0, description dict).

def lathe(rng):
    """A revolved profile -- bottle, flask, vase or funnel -- optionally
    hollow with an open top (a deep cavity only top views see into)."""
    h = rng.uniform(0.06, 0.16)
    r0 = rng.uniform(0.02, 0.05)
    n = 7
    zs = np.linspace(0, h, n)
    # smooth random radius profile: a base radius with a bulge/neck
    bulge = rng.uniform(-0.4, 0.5)
    neck = rng.uniform(0.3, 0.8)
    t = zs / h
    rs = r0 * (1 + bulge * np.sin(np.pi * t) ** 2) * (1 - (1 - neck) * np.clip((t - 0.65) / 0.35, 0, 1))
    rs = np.maximum(rs, 0.006)
    pts = [(0.0, 0.0)] + [(float(r), float(z)) for r, z in zip(rs, zs)] + [(0.0, float(h))]
    solid = CrossSection([pts]).revolve(SEGMENTS)
    hollow = bool(rng.random() < 0.6)
    wall = rng.uniform(0.002, 0.004)
    if hollow:
        inner = [(0.0, wall)] + [(float(max(r - wall, 0.002)), float(max(z, wall))) for r, z in zip(rs, zs)] + \
                [(float(max(rs[-1] - wall, 0.002)), float(h + 0.01)), (0.0, float(h + 0.01))]
        solid = solid - CrossSection([inner]).revolve(SEGMENTS)
    return solid, {"recipe": "lathe", "height": h, "hollow": hollow, "wall": wall if hollow else None}


def drilled_block(rng):
    """A block with through holes on two axes, a blind pocket from the top
    and a chamfer-free step: holes a silhouette sees only end-on."""
    x, y, z = rng.uniform(0.05, 0.12), rng.uniform(0.04, 0.10), rng.uniform(0.03, 0.08)
    m = _box(x, y, z)
    feats = []
    # through hole along y
    r = rng.uniform(0.004, min(z, x) * 0.25)
    m -= _cyl(y + 0.02, r).rotate((90, 0, 0)).translate((rng.uniform(-x / 4, x / 4), y / 2 + 0.01, z / 2))
    feats.append(f"through hole y r={r * 1000:.1f}mm")
    # through hole along x
    r2 = rng.uniform(0.003, min(z, y) * 0.2)
    m -= _cyl(x + 0.02, r2).rotate((0, 90, 0)).translate((-x / 2 - 0.01, rng.uniform(-y / 4, y / 4), z * 0.3))
    feats.append(f"through hole x r={r2 * 1000:.1f}mm")
    # blind pocket from top
    px, py, pd = x * rng.uniform(0.2, 0.4), y * rng.uniform(0.2, 0.4), z * rng.uniform(0.3, 0.6)
    m -= _box(px, py, pd + 0.01).translate((x / 4, -y / 5, z - pd))
    feats.append(f"pocket {px * 1000:.0f}x{py * 1000:.0f}x{pd * 1000:.0f}mm")
    # step cut on one end
    if rng.random() < 0.6:
        sx, sz = x * rng.uniform(0.15, 0.3), z * rng.uniform(0.3, 0.6)
        m -= _box(sx + 0.01, y + 0.02, sz + 0.01).translate((-x / 2 + sx / 2 - 0.005, 0, z - sz))
        feats.append("step")
    return m, {"recipe": "drilled_block", "size": [x, y, z], "features": feats}


def mushroom(rng):
    """A stem with a wider cap: the cap's underside is an overhang only
    low views see, and the stem hides behind it."""
    hs, rs = rng.uniform(0.03, 0.08), rng.uniform(0.008, 0.02)
    base = _cyl(0.006, rs * 2.2)                                  # a foot so it stands
    stem = _cyl(hs, rs)
    if rng.random() < 0.5:
        cap = _cyl(rng.uniform(0.01, 0.025), rng.uniform(0.03, 0.05)).translate((0, 0, hs))
        kind = "disc cap"
    else:
        cx, cy = rng.uniform(0.05, 0.10), rng.uniform(0.04, 0.08)
        cap = _box(cx, cy, rng.uniform(0.008, 0.02)).translate((rng.uniform(-0.01, 0.01), 0, hs))
        kind = "plate cap"
    return base + stem + cap, {"recipe": "mushroom", "stem_h": hs, "stem_r": rs, "cap": kind}


def bracket(rng):
    """An L bracket with holes in both legs and a gusset."""
    t = rng.uniform(0.004, 0.008)
    lx, ly, hz = rng.uniform(0.06, 0.12), rng.uniform(0.03, 0.06), rng.uniform(0.04, 0.09)
    m = _box(lx, ly, t) + _box(t, ly, hz).translate((-lx / 2 + t / 2, 0, 0))
    gusset = CrossSection([[(0, 0), (lx * 0.4, 0), (0, hz * 0.6)]]).extrude(t)
    m += gusset.rotate((90, 0, 0)).translate((-lx / 2 + t, t / 2, t))
    r = rng.uniform(0.003, 0.006)
    m -= _cyl(t + 0.02, r).translate((lx / 4, 0, -0.01))
    m -= _cyl(t + 0.02, r).rotate((0, 90, 0)).translate((-lx / 2 - 0.01, 0, hz * 0.7))
    return m, {"recipe": "bracket", "thickness": t, "legs": [lx, hz], "width": ly}


def open_container(rng):
    """A thin-walled open box with an internal divider -- like a bin or a
    tray: cavities seen only from above, walls 2-4mm."""
    x, y, z = rng.uniform(0.06, 0.14), rng.uniform(0.05, 0.12), rng.uniform(0.03, 0.08)
    w = rng.uniform(0.002, 0.004)
    m = _box(x, y, z) - _box(x - 2 * w, y - 2 * w, z).translate((0, 0, w))
    if rng.random() < 0.7:
        m += _box(w, y - 2 * w, z * rng.uniform(0.5, 0.95)).translate((rng.uniform(-x / 4, x / 4), 0, 0))
    return m, {"recipe": "open_container", "size": [x, y, z], "wall": w}


def mug(rng):
    """A cylindrical cup with a handle: a through-loop and a cavity."""
    r, h = rng.uniform(0.025, 0.045), rng.uniform(0.05, 0.11)
    w = rng.uniform(0.0025, 0.004)
    m = _cyl(h, r) - _cyl(h, r - w).translate((0, 0, w))
    hr = rng.uniform(0.012, 0.02)
    ht = rng.uniform(0.003, 0.005)
    handle = Manifold.revolve(CrossSection.circle(ht, 32).translate((hr, 0)), 48)   # torus in xy
    handle = handle.rotate((90, 0, 0)).translate((r, 0, h * rng.uniform(0.45, 0.6)))
    m += handle - _cyl(h, r - w * 0.5)                         # don't fill the cup
    return m, {"recipe": "mug", "radius": r, "height": h, "wall": w}


def tunnel_arch(rng):
    """A bridge: a block with a tunnel through its base (sees daylight
    only end-on) and a rounded top."""
    x, y, z = rng.uniform(0.07, 0.14), rng.uniform(0.03, 0.06), rng.uniform(0.04, 0.08)
    m = _box(x, y, z)
    r = min(x * 0.3, z * 0.55)
    m -= _cyl(y + 0.02, r).rotate((90, 0, 0)).translate((0, y / 2 + 0.01, 0))
    if rng.random() < 0.5:
        m = m ^ (_cyl(y, x * 0.75).rotate((90, 0, 0)).translate((0, y / 2, z - x * 0.75)) + _box(x, y, z - x * 0.4 * 0.75))
    return m, {"recipe": "tunnel_arch", "size": [x, y, z], "tunnel_r": r}


def star_extrusion(rng):
    """A star/gear prism with a bore: sharp convex and concave corners."""
    n = int(rng.integers(5, 10))
    ro, ri = rng.uniform(0.03, 0.06), None
    ri = ro * rng.uniform(0.55, 0.8)
    ang = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False)
    rad = np.where(np.arange(2 * n) % 2 == 0, ro, ri)
    poly = [(float(r * np.cos(a)), float(r * np.sin(a))) for r, a in zip(rad, ang)]
    h = rng.uniform(0.02, 0.07)
    m = CrossSection([poly]).extrude(h)
    bore = rng.uniform(0.005, ri * 0.5)
    m -= _cyl(h + 0.02, bore).translate((0, 0, -0.01))
    return m, {"recipe": "star_extrusion", "points": n, "r_out": ro, "r_in": ri, "height": h, "bore": bore}


RECIPES = [lathe, drilled_block, mushroom, bracket, open_container, mug, tunnel_arch, star_extrusion]


def make_object(rng, recipe=None) -> tuple[trimesh.Trimesh, dict]:
    fn = recipe or RECIPES[int(rng.integers(len(RECIPES)))]
    m, info = fn(rng)
    mesh = _to_trimesh(m)
    lo, hi = mesh.bounds
    mesh.apply_translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    info["extents_m"] = mesh.extents.tolist()
    info["watertight"] = bool(mesh.is_watertight)
    info["volume_m3"] = float(mesh.volume)
    info["genus"] = int(m.genus())
    return mesh, info


def make_scene(seed: int, n_objects: int, platform_wd: tuple[float, float], keepout_xy: list[tuple],
               gap: float = 0.035, max_tries: int = 4000):
    """n random objects placed on the platform without overlap (footprint
    circles + gap), with random yaw, outside every keep-out box
    (calibration markers). Returns [(name, mesh in platform frame, info)]."""
    rng = np.random.default_rng(seed)
    hw, hd = platform_wd[0] / 2 - 0.02, platform_wd[1] / 2 - 0.02
    placed = []
    order = list(range(len(RECIPES)))
    rng.shuffle(order)
    for k in range(n_objects):
        mesh, info = make_object(rng, RECIPES[order[k % len(RECIPES)]])
        yaw = rng.uniform(0, 2 * np.pi)
        mesh.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0, 0, 1)))
        r = np.linalg.norm(mesh.extents[:2]) / 2
        for _ in range(max_tries):
            c = np.array([rng.uniform(-hw + r, hw - r), rng.uniform(-hd + r, hd - r)])
            if any(np.linalg.norm(c - pc) < r + pr + gap for _, pc, pr, _, _ in placed):
                continue
            if any(np.all(c + r > lo) and np.all(c - r < hi) for lo, hi in keepout_xy):
                continue
            break
        else:
            raise RuntimeError(f"could not place object {k} (r={r:.3f}m); fewer objects or a bigger platform")
        mesh.apply_translation((c[0], c[1], 0.0))
        info.update({"yaw_rad": float(yaw), "center_xy_m": c.tolist()})
        placed.append((f"Rand{k:02d}_{info['recipe']}", c, r, mesh, info))
    return [(name, mesh, info) for name, _, _, mesh, info in placed]

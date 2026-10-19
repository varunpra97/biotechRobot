"""Procedurally generated test objects the pipeline has never seen, with
exact geometry -- to test generalization beyond the lab scene's labware.

Ten shape types, each chosen to stress something different:
  mug       hollow interior + handle with a through-hole
  bottle    lathe profile, shoulder and neck
  bracket   L-extrusion with drilled holes, sharp edges
  gear      toothed extrusion with a bore
  elbow     thin bent tube lying on its side
  blob      organic, no primitives at all
  funnel    thin-walled hollow lathe shape
  wrench    flat (6-8mm) outline extrusion
  torus     through-hole
  stepped   stacked offset blocks with a boss

Every mesh is watertight, in meters, in its own frame (footprint centered
on the origin, resting on z=0). Dimensions are randomized per seed.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/isaac_validation/random_objects.py \
      --out biotech-lab-robotics/vision_pipeline/isaac_validation/random_objects --seed 1 --count 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Point, Polygon, box as sbox
from shapely.ops import unary_union

MM = 0.001


def _cyl(r, h, z0=0.0, sections=96):
    c = trimesh.creation.cylinder(radius=r, height=h, sections=sections)
    c.apply_translation([0, 0, z0 + h / 2])
    return c


def _revolve(profile_rz, sections=128):
    """profile: (r, z) points, closed via the axis."""
    return trimesh.creation.revolve(np.asarray(profile_rz), sections=sections)


def mug(rng):
    R, H, wall, floor = rng.uniform(30, 45), rng.uniform(70, 110), 3.0, 4.0
    body = _cyl(R * MM, H * MM).difference(_cyl((R - wall) * MM, H * MM, floor * MM), engine="manifold")
    major, minor = 0.32 * H, 5.0
    handle = trimesh.creation.torus(major_radius=major * MM, minor_radius=minor * MM, major_sections=96, minor_sections=24)
    handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    handle.apply_translation([(R + major - 0.6 * minor) * MM, 0, H / 2 * MM])
    keep = trimesh.creation.box([3 * major * MM, 3 * major * MM, 3 * major * MM])
    keep.apply_translation([(R - 1 + 1.5 * major) * MM, 0, H / 2 * MM])
    handle = handle.intersection(keep, engine="manifold")
    return body.union(handle, engine="manifold"), dict(R=R, H=H, wall=wall)


def bottle(rng):
    R, H, rn = rng.uniform(30, 45), rng.uniform(150, 210), rng.uniform(10, 15)
    h1, sh = 0.6 * H, 0.18 * H
    prof = [(0, 0), (R, 0), (R, h1), (rn, h1 + sh), (rn, H), (0, H)]
    return _revolve(np.array(prof) * MM), dict(R=R, H=H, neck_r=rn)


def bracket(rng):
    a, b, t, w = rng.uniform(60, 90), rng.uniform(45, 70), rng.uniform(6, 10), rng.uniform(30, 45)
    L = Polygon([(0, 0), (a, 0), (a, t), (t, t), (t, b), (0, b)])          # x-z profile
    m = trimesh.creation.extrude_polygon(L, w)                            # along z -> rotate to y
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    m.apply_translation([0, w, 0])
    for zc in (0.45 * b, 0.8 * b):                                        # holes through the vertical leg
        hole = trimesh.creation.cylinder(radius=4.5, height=3 * t, sections=48)
        hole.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        hole.apply_translation([t / 2, w / 2, zc])
        m = m.difference(hole, engine="manifold")
    m.apply_scale(MM)
    return m, dict(leg_a=a, leg_b=b, thickness=t, width=w)


def gear(rng):
    n, Ro, th, rb = int(rng.integers(10, 17)), rng.uniform(35, 50), rng.uniform(15, 25), rng.uniform(8, 12)
    Rr = Ro - 6
    pts = []
    for i in range(n * 4):
        ang = 2 * np.pi * i / (n * 4)
        r = Ro if (i % 4) in (1, 2) else Rr
        pts.append((r * np.cos(ang), r * np.sin(ang)))
    poly = Polygon(pts).difference(Point(0, 0).buffer(rb, 64))
    return trimesh.creation.extrude_polygon(poly, th).apply_scale(MM), dict(teeth=n, R_outer=Ro, thickness=th, bore_r=rb)


def elbow(rng):
    rt, Rb = rng.uniform(8, 11), rng.uniform(50, 70)
    ang = np.linspace(0, np.pi / 2, 48)
    path = np.c_[Rb * np.cos(ang), Rb * np.sin(ang), np.full(len(ang), rt)]
    m = trimesh.creation.sweep_polygon(Point(0, 0).buffer(rt, 32), path)
    return m.apply_scale(MM), dict(tube_r=rt, bend_R=Rb)


def blob(rng):
    m = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    v = m.vertices
    disp = np.zeros(len(v))
    for _ in range(5):
        k = rng.normal(size=3) * 2.0
        disp += rng.uniform(0.04, 0.12) * np.sin(v @ k + rng.uniform(0, 2 * np.pi))
    R = rng.uniform(40, 55)
    m.vertices = v * (1 + disp)[:, None] * [R, R * rng.uniform(0.8, 1.2), R * 0.75]
    return m.apply_scale(MM), dict(R=R)


def funnel(rng):
    Rt, rs, Hc, Hs, t = rng.uniform(50, 65), rng.uniform(7, 10), rng.uniform(50, 70), rng.uniform(35, 50), 2.5
    outer = [(0, 0), (rs, 0), (rs, Hs), (Rt, Hs + Hc), (0, Hs + Hc)]
    inner = [(0, -1), (rs - t, -1), (rs - t, Hs), (Rt - t, Hs + Hc + 1), (0, Hs + Hc + 1)]
    m = _revolve(np.array(outer) * MM).difference(_revolve(np.array(inner) * MM), engine="manifold")
    return m, dict(R_top=Rt, spout_r=rs, height=Hs + Hc, wall=t)


def wrench(rng):
    L, w, t, jaw = rng.uniform(140, 190), rng.uniform(14, 20), rng.uniform(6, 8), rng.uniform(16, 22)
    handle = sbox(0, -w / 2, L, w / 2).buffer(2, 16)
    head = Point(L, 0).buffer(jaw, 64).difference(sbox(L, -jaw * 0.35, L + jaw + 5, jaw * 0.35))
    ring = Point(0, 0).buffer(w * 0.85, 64).difference(Point(0, 0).buffer(w * 0.45, 64))
    poly = unary_union([handle, head, ring])
    m = trimesh.creation.extrude_polygon(poly, t)
    m.apply_translation([-L / 2, 0, 0])
    return m.apply_scale(MM), dict(length=L, width=w, thickness=t)


def torus(rng):
    Rm, rm = rng.uniform(35, 45), rng.uniform(10, 14)
    m = trimesh.creation.torus(major_radius=Rm * MM, minor_radius=rm * MM, major_sections=128, minor_sections=32)
    m.apply_translation([0, 0, rm * MM])
    return m, dict(R_major=Rm, r_minor=rm)


def stepped(rng):
    parts, z = [], 0.0
    sx, sy = rng.uniform(70, 100), rng.uniform(50, 80)
    for _ in range(3):
        h = rng.uniform(12, 22)
        b = trimesh.creation.box([sx, sy, h])
        b.apply_translation([rng.uniform(-4, 4), rng.uniform(-4, 4), z + h / 2])
        parts.append(b)
        z += h
        sx, sy = sx * 0.7, sy * 0.7
    boss = _cyl(min(sx, sy) * 0.4, rng.uniform(10, 18), z)
    m = parts[0]
    for p in parts[1:] + [boss]:
        m = m.union(p, engine="manifold")
    return m.apply_scale(MM), dict(height=float(m.extents[2] / MM))


BUILDERS = {f.__name__: f for f in (mug, bottle, bracket, gear, elbow, blob, funnel, wrench, torus, stepped)}
PALETTE = {"white": (0.92, 0.92, 0.92), "light_gray": (0.75, 0.75, 0.75), "dark_gray": (0.25, 0.25, 0.27),
           "blue": (0.2, 0.4, 0.8), "red": (0.8, 0.22, 0.2), "green": (0.25, 0.6, 0.3),
           "yellow": (0.9, 0.8, 0.25), "orange": (0.9, 0.5, 0.2)}

# Clear bench area in the platform frame, away from the markers and the
# calibration cubes (see render_isaac_validation_capture.marker_specs).
REGION = (-0.42, 0.42, -0.32, 0.13)


def centered(m: trimesh.Trimesh) -> trimesh.Trimesh:
    lo, hi = m.bounds
    m.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return m


def place(objs, rng, gap=0.02, tries=4000):
    placed = []
    for o in objs:
        rad = 0.5 * np.hypot(*o["mesh"].extents[:2]) + gap
        for _ in range(tries):
            x, y = rng.uniform(REGION[0] + rad, REGION[1] - rad), rng.uniform(REGION[2] + rad, REGION[3] - rad)
            if all(np.hypot(x - p["xy"][0], y - p["xy"][1]) > rad + p["rad"] for p in placed):
                placed.append({"xy": (x, y), "rad": rad})
                o["placement_m"] = [float(x), float(y), 0.0]
                o["yaw_deg"] = float(rng.uniform(0, 360))
                break
        else:
            raise RuntimeError(f"could not place {o['type']}; reduce --count")
    return objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--types", nargs="+", choices=sorted(BUILDERS), help="exact shape types (overrides --count)")
    ap.add_argument("--gap-mm", type=float, default=20.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    types = args.types if args.types else rng.choice(sorted(BUILDERS), size=args.count, replace=False)
    args.count = len(types)
    colors = rng.choice(sorted(PALETTE), size=args.count, replace=True)
    objs = []
    for i, (t, c) in enumerate(zip(types, colors)):
        mesh, params = BUILDERS[t](rng)
        mesh = centered(mesh)
        assert mesh.is_watertight, f"{t} is not watertight"
        objs.append({"name": f"rand_{i:02d}_{t}", "type": str(t), "params_mm": params, "color": str(c),
                     "rgb": PALETTE[c], "mesh": mesh})
    objs = place(sorted(objs, key=lambda o: -np.hypot(*o["mesh"].extents[:2])), rng, gap=args.gap_mm / 1000)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for o in objs:
        o["mesh"].export(out / f"{o['name']}.obj")
        manifest.append({k: v for k, v in o.items() if k != "mesh"} |
                        {"obj": f"{o['name']}.obj", "extents_mm": (o["mesh"].extents / MM).round(1).tolist()})
        print(f"{o['name']:22s} {o['color']:10s} extents {manifest[-1]['extents_mm']} mm at "
              f"({o['placement_m'][0]:+.3f}, {o['placement_m'][1]:+.3f}) yaw {o['yaw_deg']:.0f}")
    (out / "objects.json").write_text(json.dumps({"seed": args.seed, "objects": manifest}, indent=2))


if __name__ == "__main__":
    main()

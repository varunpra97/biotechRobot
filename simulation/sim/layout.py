"""Randomized platform layouts: where the tube holder, centrifuge, tip rack,
waste bin, pipette and sample tray stand on the bench, sampled per chain
(collect_primitives.py) so that a policy sees the objects at many positions
and in many arrangements, not one.

Free arrangement over the whole platform: the objects are placed one by one,
in a random order, each anywhere on the platform -- the holder may stand
leftmost or rightmost, the centrifuge in the middle, the bin and the pipette
in either corner. The arm's base moves too (ARM_RANGE), which shifts the
reachable part of the platform from chain to chain.

Orientation: the centrifuge and the waste bin turn freely. The tube holder
and the tip rack carry labelled grids (slot "B3", tip "E10") and the pipette
and the tray are grasped in one way, so these keep their default orientation
*relative to the arm* -- turned with the bearing from the arm's base to them,
plus YAW_JITTER: row A of the holder stays toward the arm and columns run
left to right as seen from it (the prompts' slot names assume this), and the
pipette and tray grasps stay the ones the experts were built for.

A placement is kept only if
  - its footprint (mesh bounds, plus CLEARANCE, plus room for the hand where
    it needs it) stays on the platform and clear of everything placed so far,
  - the points the arm has to reach (the holder's corner slots, the rotor
    hub, the tray, the bin, the rack, the pipette grip) lie between REACH_MIN
    and REACH_M from the arm's base,
  - the keep-outs of the moves around it stay clear: the wrist_side camera
    module beside the pipette grasp (the grasp is always the one that puts
    the nozzle toward that camera, so the module comes down on the pipette's
    local +x side) and the pipette's nozzle-down ready spot beside the tip
    rack (lab_scene.PIPETTE_READY_FROM_RACK; only the centrifuge is tall
    enough to be in its way).
Fine reach (IK with joint-limit margin per bore, slot and tray spot) is
checked after the scene is built.

    sample(rng)          a layout dict and the number of attempts it took
    apply(layout)        set lab_scene's placement globals from a layout
    tray_spots(rng, n)   tray destinations beside another object, for the
                         current layout, with their prompt keys
"""
from __future__ import annotations

import math

import numpy as np

import lab_scene

NAMES = ("holder", "centrifuge", "tip_rack", "waste_bin", "pipette", "tray")
FREE_YAW = ("centrifuge", "waste_bin")
# deg about the arm-facing orientation, for the rest
YAW_JITTER = {"holder": 20.0, "tip_rack": 20.0, "pipette": 15.0, "tray": 20.0}
# the arm's base (on its pedestal, facing the platform): offset from
# lab_scene's default ARM_POS. Toward the bench at most +5 cm, so the 50 cm
# pedestal (front face 25 cm ahead of the base) stays clear of its edge.
ARM_RANGE = dict(x=(-0.20, 0.20), y=(-0.06, 0.05))
ARM_DEFAULT = tuple(lab_scene.ARM_POS)
CLEARANCE = 0.02                 # m between footprints
# extra room around an object, for the hand that works it (m, local x / y)
# (the hand is ~21 cm wide along its closing axis, which runs across the
# pipette: it passes over the neighbours' tops -- tube tops stand 7 cm high)
# (tray: the hand spans +-10.4 cm along the closing axis, below the tube
# tops at the grasp -- 7 cm let it knock a tube beside a placed tray)
HAND_ROOM = {"pipette": (0.05, 0.0), "tray": (0.0, 0.10)}
# wrist_side camera module + bracket while the pipette is grasped: pipette
# local box (x0, x1, y0, y1) -- local y runs from the nozzle end, the grip
# at 0.11; the module sits 11 cm out along the closing axis (+x) and 6.5 cm
# toward the nozzle, ~5 cm above the bench, so anything taller is hit
PIPETTE_CAMERA_BOX = (0.09, 0.14, 0.015, 0.075)
READY_BOX_HALF = 0.05            # m around the pipette's ready spot (vs the centrifuge)
PLATFORM_MARGIN = 0.03
REACH_M = 0.78                   # horizontal distance from the arm base (m)
# ... tighter for the rotor: tilted inserts into its off-centre bores at
# 0.75 m ran out of joint range (the old near-default layouts kept it
# within ~0.71 m)
REACH_M_OBJ = {"centrifuge": 0.70}
REACH_MIN = 0.40                 # ... and not closer than this (the arm folds onto itself)
TRIES = 3000                     # whole layouts
OBJ_TRIES = 300                  # positions per object within one layout
MIN_TRAY_SPOTS = 3               # free spots beside another object, for the tray block

_LOCAL = {}
_CURRENT = None                  # the layout last apply()'d


def _local_box(name):
    """Local xy bounding box (xmin, xmax, ymin, ymax) of an object's mesh."""
    if name not in _LOCAL:
        import trimesh
        if name == "centrifuge":
            v = np.concatenate([p.vertices for i, p in enumerate(lab_scene.centrifuge_parts_local())
                                if i not in lab_scene.CENTRIFUGE_LID_PARTS])
        else:
            obj = {"holder": lab_scene.TUBE_HOLDER_OBJ, "tip_rack": lab_scene.TIP_RACK_OBJ,
                   "waste_bin": lab_scene.WASTE_BIN_OBJ, "pipette": lab_scene.PIPETTE_OBJ,
                   "tray": lab_scene.SAMPLE_TRAY_OBJ}[name]
            v = trimesh.load(obj, force="mesh").vertices
        _LOCAL[name] = (v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max())
    return _LOCAL[name]


def defaults():
    """Default (x, y, yaw_deg) of each object's mesh origin."""
    return {
        "holder": (*_default_holder(), 0.0),
        "centrifuge": (*_default_centrifuge(), 0.0),
        "tip_rack": (*lab_scene.TIP_RACK_XY, 0.0),
        "waste_bin": (*lab_scene.WASTE_BIN_XY, 0.0),
        "pipette": (*lab_scene.PIPETTE_XY, lab_scene.PIPETTE_YAW_DEG),
        "tray": (*lab_scene.SAMPLE_TRAY_XY, lab_scene.SAMPLE_TRAY_YAW_DEG),
    }


def _default_holder():
    return lab_scene.PLATFORM_CX - 0.05, lab_scene.PLATFORM_CY - 0.15


def _default_centrifuge():
    hx, hy = _default_holder()
    return hx - 0.158 - 0.08 - lab_scene.CENTRIFUGE_EXTENTS[0] / 2, hy


def _rot(yaw_deg):
    a = math.radians(yaw_deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def _box_quad(box, x, y, yaw_deg, extra=0.0):
    """World corners of a local box (x0, x1, y0, y1) at pose (x, y, yaw)."""
    x0, x1, y0, y1 = box
    pts = np.array([[x0 - extra, y0 - extra], [x1 + extra, y0 - extra],
                    [x1 + extra, y1 + extra], [x0 - extra, y1 + extra]])
    return pts @ _rot(yaw_deg).T + [x, y]


def _corners(name, x, y, yaw_deg, extra=0.0, hand=True):
    x0, x1, y0, y1 = _local_box(name)
    rx, ry = HAND_ROOM.get(name, (0.0, 0.0)) if hand else (0.0, 0.0)
    x0, x1, y0, y1 = x0 - rx - extra, x1 + rx + extra, y0 - ry - extra, y1 + ry + extra
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    R = _rot(yaw_deg)
    if name == "centrifuge":          # it turns about its footprint's centre (lab_scene.centrifuge_pos)
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        return (pts - c) @ R.T + c + [x, y]
    return pts @ R.T + [x, y]


def _overlap(p, q):
    """Separating-axis test for two convex quads."""
    for poly in (p, q):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            a, b = p @ n, q @ n
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def _keepouts(name, x, y, yaw):
    """[(quad, names it must stay clear of)] for the moves around an object."""
    if name == "pipette":
        return [(_box_quad(PIPETTE_CAMERA_BOX, x, y, yaw, CLEARANCE / 2), [n for n in NAMES if n != "pipette"])]
    if name == "tip_rack":
        r = READY_BOX_HALF
        dx, dy = lab_scene._turn(yaw, *lab_scene.PIPETTE_READY_FROM_RACK)
        return [(_box_quad((-r, r, -r, r), x + dx, y + dy, 0.0), ["centrifuge"])]
    return []


def _reach_points(name, x, y, yaw):
    """World xy points the arm must reach to work an object at (x, y, yaw)."""
    if name == "holder":
        mm = lab_scene.MM_TO_M
        pts = []
        for r in (0, 3):
            for c in (0, 4):
                dx, dy = lab_scene._turn(yaw, lab_scene.HOLDER_HOLE_X_MM[c] * mm, lab_scene.HOLDER_HOLE_Y_MM[r] * mm)
                pts.append((x + dx, y + dy))
        return pts
    if name == "centrifuge":
        return [tuple(_corners("centrifuge", x, y, yaw, hand=False).mean(axis=0))]   # ~ the rotor
    if name == "pipette":
        a = math.radians(yaw)
        return [(x - 0.11 * math.sin(a), y + 0.11 * math.cos(a))]        # grip point along the body
    return [(x, y)]


def _bearing(x, y, base):
    return math.degrees(math.atan2(y - base[1], x - base[0]))


def _yaw(name, x, y, base, rng):
    """Free for the centrifuge and the bin; else the default orientation
    turned with the bearing from the arm, plus jitter."""
    if name in FREE_YAW:
        return rng.uniform(0.0, 360.0)
    dx, dy, dyaw = defaults()[name]
    return dyaw + _bearing(x, y, base) - _bearing(dx, dy, ARM_DEFAULT) + rng.uniform(-YAW_JITTER[name], YAW_JITTER[name])


def _platform_box():
    return (lab_scene.PLATFORM_CX - lab_scene.PLATFORM_W / 2 + PLATFORM_MARGIN,
            lab_scene.PLATFORM_CX + lab_scene.PLATFORM_W / 2 - PLATFORM_MARGIN,
            lab_scene.PLATFORM_CY - lab_scene.PLATFORM_D / 2 + PLATFORM_MARGIN,
            lab_scene.PLATFORM_CY + lab_scene.PLATFORM_D / 2 - PLATFORM_MARGIN)


def _on_platform(p):
    xlo, xhi, ylo, yhi = _platform_box()
    return p[:, 0].min() >= xlo and p[:, 0].max() <= xhi and p[:, 1].min() >= ylo and p[:, 1].max() <= yhi


def _in_reach(pts, base, name=None):
    hi = REACH_M_OBJ.get(name, REACH_M)
    return all(REACH_MIN <= math.hypot(px - base[0], py - base[1]) <= hi for px, py in pts)


def _fits(name, pose, polys, keep, base):
    """Can `name` stand at pose (x, y, yaw), given what is placed (polys
    {name: quad}, keep [(quad, names)])?"""
    p = _corners(name, *pose, extra=CLEARANCE / 2)
    if not _on_platform(p) or not _in_reach(_reach_points(name, *pose), base, name):
        return None
    if any(_overlap(p, q) for q in polys.values()):
        return None
    if any(name in names and _overlap(p, q) for q, names in keep):
        return None
    mine = _keepouts(name, *pose)
    if any(_overlap(q, polys[n]) for q, names in mine for n in names if n in polys):
        return None
    return p, mine


def sample(rng, tries=TRIES):
    """A random collision-free, reachable layout {name: [x, y, yaw_deg]},
    plus "arm": [x, y] of the arm's base, and the number of attempts it
    took. Falls back to the defaults."""
    xlo, xhi, ylo, yhi = _platform_box()
    for n in range(1, tries + 1):
        base = (ARM_DEFAULT[0] + rng.uniform(*ARM_RANGE["x"]), ARM_DEFAULT[1] + rng.uniform(*ARM_RANGE["y"]))
        order = list(NAMES)
        rng.shuffle(order)
        pl, polys, keep = {}, {}, []
        for name in order:
            for _ in range(OBJ_TRIES):
                x, y = rng.uniform(xlo, xhi), rng.uniform(ylo, yhi)
                pose = (x, y, _yaw(name, x, y, base, rng))
                fit = _fits(name, pose, polys, keep, base)
                if fit:
                    pl[name], polys[name] = pose, fit[0]
                    keep += fit[1]
                    break
            else:
                break
        if len(pl) < len(NAMES):
            continue
        out = {k: [round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]) % 360.0, 2)]
               for k, v in ((k, pl[k]) for k in NAMES)}
        out["arm"] = [round(float(base[0]), 4), round(float(base[1]), 4)]
        if len(tray_spots(rng, n=MIN_TRAY_SPOTS, lay=out)) < MIN_TRAY_SPOTS:
            continue                  # the tray block needs somewhere to move it
        return out, n
    return default_layout(), tries


def default_layout():
    """The fixed arrangement (lab_scene's defaults, arm at its default spot)
    that the pilot data (primitives_v2) was recorded in."""
    out = {k: [float(v[0]), float(v[1]), float(v[2])] for k, v in defaults().items()}
    out["arm"] = list(ARM_DEFAULT[:2])
    return out


def apply(layout):
    """Set lab_scene's placement globals (call before the scene is built)."""
    global _CURRENT
    _CURRENT = layout
    if "arm" in layout:
        lab_scene.ARM_POS = (layout["arm"][0], layout["arm"][1], ARM_DEFAULT[2])
    dh, dc = _default_holder(), _default_centrifuge()
    hx, hy, hyaw = layout["holder"]
    lab_scene.HOLDER_OFFSET = (hx - dh[0], hy - dh[1])
    lab_scene.HOLDER_YAW_DEG = hyaw
    cx, cy, cyaw = layout["centrifuge"]
    lab_scene.CENTRIFUGE_OFFSET = (cx - dc[0], cy - dc[1])
    lab_scene.CENTRIFUGE_YAW_DEG = cyaw
    lab_scene.TIP_RACK_XY = tuple(layout["tip_rack"][:2])
    lab_scene.TIP_RACK_YAW_DEG = layout["tip_rack"][2]
    lab_scene.WASTE_BIN_XY = tuple(layout["waste_bin"][:2])
    lab_scene.WASTE_BIN_YAW_DEG = layout["waste_bin"][2]
    lab_scene.PIPETTE_XY = tuple(layout["pipette"][:2])
    lab_scene.PIPETTE_YAW_DEG = layout["pipette"][2]
    lab_scene.SAMPLE_TRAY_XY = tuple(layout["tray"][:2])
    lab_scene.SAMPLE_TRAY_YAW_DEG = layout["tray"][2]


# landmarks a tray destination is described by (prompts.tray_spot)
TRAY_LANDMARKS = ("tip_rack", "waste_bin", "holder", "centrifuge")
# directions as seen from the arm, which faces +y
TRAY_SIDES = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "front": (0.0, -1.0), "behind": (0.0, 1.0)}


def tray_spots(rng, n=8, tries=400, lay=None):
    """Up to n destinations for the tray in the current (apply()'d) layout,
    each beside another object: [(xy, yaw_rad, key)], key "side:landmark"
    (e.g. "left:tip_rack" -> "to the left of the tip box", prompts.tray_spot).
    The tray lands 0-3 cm clear of the landmark (with the hand's room), on
    the platform, clear of everything else, in reach, and at least 8 cm from
    where it stands now; its yaw follows the bearing rule as in sample().
    lay: a layout other than the current one."""
    lay = lay or _CURRENT
    base = tuple(lay["arm"])
    tray_home = np.array(lay["tray"][:2])
    others = {k: _corners(k, *lay[k], extra=CLEARANCE / 2) for k in NAMES if k != "tray"}
    keep = [q for k in NAMES if k != "tray" for q, names in _keepouts(k, *lay[k]) if "tray" in names]
    out = []
    for _ in range(tries):
        lm, side = rng.choice(TRAY_LANDMARKS), rng.choice(list(TRAY_SIDES))
        d = np.array(TRAY_SIDES[side])
        lq = others[lm]
        across = np.array([-d[1], d[0]])
        c = lq.mean(axis=0)
        half_across = (lq @ across).max() - c @ across
        start = c + across * rng.uniform(-0.6, 0.6) * half_across
        gap = rng.uniform(0.0, 0.03)
        for step in np.arange(0.03, 0.40, 0.005):
            xy = start + d * step
            yaw = _yaw("tray", xy[0], xy[1], base, rng)
            p = _corners("tray", xy[0], xy[1], yaw, extra=CLEARANCE / 2)
            if not _overlap(p, lq):
                xy = xy + d * gap
                p = _corners("tray", xy[0], xy[1], yaw, extra=CLEARANCE / 2)
                break
        else:
            continue
        if (not _on_platform(p) or not _in_reach([tuple(xy)], base) or np.linalg.norm(xy - tray_home) < 0.08
                or any(_overlap(p, q) for q in others.values()) or any(_overlap(p, q) for q in keep)):
            continue
        out.append((xy, math.radians(yaw), f"{side}:{lm}"))
        if len(out) == n:
            break
    return out

"""Split a fused composite object into its parts, guided by a description of
how the parts relate (from LLM scene reasoning, or a hint file).

Depth fusion knows nothing about parts: a rack and the tubes standing in it
come out as one watertight solid. Geometry alone can't say where one object
ends and the next begins; a description alone can't say where the cut is.
Together they can:

  inserted         parts standing in bores of a parent (tubes in a rack,
                   pins in a board, tubes in a rotor). Above the parent's
                   top surface they are separate 3D pieces. Below it they
                   are hidden; each is extended down its bore to the bore
                   floor, MEASURED from the parent's empty bores when any
                   are seen (the cameras see into an empty bore), and the
                   parent gets the bores carved back out.
  resting_inside   parts standing inside an open parent (weights in a tray,
                   a rotor in a centrifuge bowl): in every horizontal slice
                   above the parent's floor they are blobs enclosed by its
                   walls, not touching its outline.
  resting_on_top   a part sitting on the parent (lid, cap): cut where the
                   footprint widens going up (the lid's or cap's overhang).

Works on the object's fine TSDF (reconstruct_depth.py saves it per object):
each part is re-meshed from the field with every other part's voxels
pushed outside, so surfaces facing air keep the fusion's sub-voxel
precision; part-to-part contact surfaces (hidden in any case) are at voxel
resolution.

Limits, by construction: what is hidden stays an estimate -- a tube's shape
inside its bore is its visible cross-section extruded, a lid's skirt over
the box rim is cut by height.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from depth_fusion import FusedGrid, grid_mesh

RELATIONS = ["base", "inserted", "resting_inside", "resting_on_top"]


@dataclass
class Part:
    name: str
    relation: str
    parent: str
    mask: np.ndarray                      # voxels owned (bool grid)
    forced_inside: np.ndarray | None = None   # hidden voxels added by extrusion
    notes: list = field(default_factory=list)
    kinds: list = field(default_factory=list)  # candidate part kinds (several when same relation + parent)


def load_field(path) -> FusedGrid:
    d = np.load(path)
    F = d["field_q"].astype(np.float32) / 127.0
    return FusedGrid(F, np.zeros_like(F), np.zeros(F.shape, np.uint8), d["origin"], float(d["voxel_m"]),
                     float(d["trunc_m"]))


def _components3d(mask, min_voxels):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return []
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    return [lab == i + 1 for i, s in enumerate(sizes) if s >= min_voxels]


def top_of_wide_part(mask: np.ndarray, frac: float = 0.5) -> int:
    """Highest slice whose cross-section is at least frac of the widest one:
    the top of a rack under its tubes, of a rotor under its tubes."""
    A = mask.sum(axis=(0, 1))
    return int(np.where(A >= frac * A.max())[0].max())


def split_inserted(mask: np.ndarray, voxel: float, expected: int | None = None, min_part_mm3: float = 30.0,
                   observed: np.ndarray | None = None, grid: tuple | None = None):
    """(parent mask, [insert masks], [extrusion masks], info). The parent is
    the wide lower body; inserts are the pieces protruding above its top."""
    if observed is None:
        observed = np.ones_like(mask)
    k_top = top_of_wide_part(mask)
    start = k_top + 2                       # two slices clear of the parent's top surface
    above = mask.copy()
    above[:, :, :start] = False
    min_vox = int(min_part_mm3 * 1e-9 / voxel ** 3)
    inserts = [m for m in _components3d(above, min_vox) if m[:, :, start].any()]
    info = {"parent_top_m": None, "k_top": k_top, "found": len(inserts), "expected": expected}
    if grid and expected and len(inserts) < expected:
        # Inserts packed closer than the voxel size fuse into one blob (100 tips
        # hanging in a rack: their collars touch). Split each fused blob on the
        # described grid: voxels go to the nearest cell centre of a grid fitted
        # to the blob's footprint.
        rows, cols, pitch = grid
        if rows > 0 and cols > 0 and pitch > 0:
            split = []
            for m in inserts:
                xy = np.argwhere(m.any(axis=2)).astype(float)
                ext = xy.max(axis=0) - xy.min(axis=0)
                # each insert spans about one pitch of the blob's footprint
                span = ext + 1
                n_a = max(1, int(round(span[0] * voxel / (pitch / 1000))))
                n_b = max(1, int(round(span[1] * voxel / (pitch / 1000))))
                if rows * cols == expected and len(inserts) == 1:
                    # one blob holding the whole described grid: trust the
                    # described counts over a pitch estimated from photos
                    # (10x10 tips came out 9x9 from a 10mm pitch guess)
                    big, small = max(rows, cols), min(rows, cols)
                    n_a, n_b = (big, small) if span[0] >= span[1] else (small, big)
                if n_a * n_b <= 1:
                    split.append(m)
                    continue
                ca = xy[:, 0].min() + (np.arange(n_a) + 0.5) * span[0] / n_a
                cb = xy[:, 1].min() + (np.arange(n_b) + 0.5) * span[1] / n_b
                idx = np.argwhere(m)
                ia = np.abs(idx[:, :1] - ca[None]).argmin(axis=1)
                ib = np.abs(idx[:, 1:2] - cb[None]).argmin(axis=1)
                cell = ia * n_b + ib
                for c in np.unique(cell):
                    sub = np.zeros_like(m)
                    sel = idx[cell == c]
                    sub[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                    if sub.sum() >= min_vox and sub[:, :, start].any():
                        split.append(sub)
            info["grid_split"] = f"{len(inserts)} blob(s) -> {len(split)} on a {rows}x{cols} grid"
            inserts = split
            info["found"] = len(inserts)
    if not inserts:
        return mask, [], [], info

    # Bore floor, measured: empty bores are holes in the parent's top slices
    # that no insert sits in; follow them down until they close.
    top2d = mask[:, :, k_top - 1]
    holes = ndimage.binary_fill_holes(top2d) & ~top2d
    occupied_xy = np.zeros_like(top2d)
    for m in inserts:
        occupied_xy |= m[:, :, start]
    occupied_xy = ndimage.binary_dilation(occupied_xy, iterations=2)
    hl, nh = ndimage.label(holes)
    empty = [hl == i + 1 for i in range(nh) if not (occupied_xy & (hl == i + 1)).any()
             and (hl == i + 1).sum() * voxel ** 2 > 3e-6]       # > 3 mm^2
    floors = []
    for h in empty:
        ring = ndimage.binary_dilation(h, iterations=3) & ~ndimage.binary_dilation(h, iterations=1)
        k = k_top - 1
        # Down while the bore is open (seen free) or unseen, AND still walled
        # by the parent (leaving the parent's material is not a floor: a
        # rotor's bore can exit into air). The floor counts only if a camera
        # SAW it: the bottom of a narrow bore is often never seen, and the
        # start of unseen space is not a floor (pins came out 6.6mm short).
        def open_or_unseen(k):
            return ((~mask[:, :, k] | ~observed[:, :, k]) & h).sum() >= 0.5 * h.sum()
        while k > 0 and open_or_unseen(k) and (mask[:, :, k] & ring).sum() >= 0.6 * ring.sum():
            k -= 1
        if (mask[:, :, k] & observed[:, :, k] & h).sum() >= 0.5 * h.sum():
            floors.append(k + 1)
    k_floor = int(np.median(floors)) if floors else None
    info["empty_bores_seen"] = len(empty)
    info["bore_floor_k"] = k_floor

    parent = mask & ~above
    extrusions = []
    for m in inserts:
        xs = m[:, :, start]
        x2d = ndimage.binary_fill_holes(xs) if not floors else xs      # see note below
        if k_floor is not None:
            lo = k_floor
        else:   # unmeasured: bores in racks/boards run ~60-80% of the holder's height
            k_bot = int(np.nonzero(parent.any(axis=(0, 1)))[0].min())
            lo = int(round(k_top - 0.7 * (k_top - k_bot)))
        ext = np.zeros_like(mask)
        ext[:, :, lo:start] = x2d[:, :, None]
        extrusions.append(ext)
        parent &= ~ext
    # note: with no empty bore to measure, the depth is a 20mm guess and the
    # cross-section is filled (a solid plug) -- flagged in info for the caller.
    info["extrusion"] = "to the measured bore floor" if floors else "70% of holder height (no bore floor seen)"
    return parent, inserts, extrusions, info


def split_resting_inside(mask: np.ndarray, voxel: float, min_part_mm3: float = 200.0,
                         observed: np.ndarray | None = None):
    """(parent mask, [item masks], info). Per slice, 2D components that
    don't touch the object's outer outline are inside it."""
    foot = mask.any(axis=2)
    outline = ndimage.binary_dilation(~ndimage.binary_fill_holes(foot), iterations=2)
    inner = np.zeros_like(mask)
    for k in range(mask.shape[2]):
        sl = mask[:, :, k]
        if not sl.any():
            continue
        lab, n = ndimage.label(sl, structure=np.ones((3, 3)))
        if n < 2:
            continue
        comps = [lab == i for i in range(1, n + 1)]
        filled = [ndimage.binary_fill_holes(c) for c in comps]
        for i, c in enumerate(comps):
            # inside = enclosed by ANOTHER piece of this same slice (a wall
            # ring around it). Per slice, not against the overall footprint:
            # the lab centrifuge's housing tapers, so its upper slices never
            # touch the footprint's outline and were taken for contents.
            if any(j != i and (c & filled[j]).sum() >= 0.9 * c.sum() for j in range(n)):
                inner[:, :, k] |= c
    # Above the parent's rim there is no wall to enclose anything: grow the
    # seeds found below it upward through the voxels connected to them (the
    # tops of weights standing taller than their tray were handed back to
    # the tray, +17mm on its height).
    rim_ks = np.nonzero(inner.any(axis=(0, 1)))[0]
    if len(rim_ks):
        above = mask.copy()
        above[:, :, :rim_ks.max() + 1] = False       # slices above the last enclosed one
        region = inner | above
        lab, n = ndimage.label(region, structure=np.ones((3, 3, 3)))
        seeds = np.unique(lab[inner])
        inner = np.isin(lab, seeds[seeds > 0])
    min_vox = int(min_part_mm3 * 1e-9 / voxel ** 3)
    items = _components3d(inner, min_vox)
    info = {"found": len(items)}
    if items and observed is not None:
        # The cavity floor, where the cameras saw it: in cavity columns not
        # under an item, the highest occupied voxel that was OBSERVED (the
        # fused field is exactly -1 only where no view said anything).
        cavity = ndimage.binary_fill_holes(foot) & ~outline
        under = np.zeros_like(foot)
        for m in items:
            under |= ndimage.binary_dilation(m.any(axis=2), iterations=2)
        cols = cavity & ~under
        tops = []
        for i, j in zip(*np.nonzero(cols)):
            occ_k = np.nonzero(mask[i, j, :] & observed[i, j, :])[0]
            free_above = occ_k.size and not mask[i, j, occ_k.max() + 1:].any()
            if free_above:
                tops.append(occ_k.max())
        if len(tops) >= 10:
            k_floor = int(np.percentile(tops, 25)) + 1      # first slice above the floor
            info["cavity_floor_k"] = k_floor
            for n, m in enumerate(items):
                k_low = int(np.nonzero(m.any(axis=(0, 1)))[0].min())
                if k_low > k_floor:
                    # hidden under the item: its lowest visible cross-section, down to the floor
                    foot_i = m[:, :, k_low]
                    ext = np.zeros_like(m)
                    ext[:, :, k_floor:k_low] = foot_i[:, :, None]
                    items[n] = m | (ext & mask)
            info["extended_to_floor"] = True
    parent = mask.copy()
    for m in items:
        parent &= ~m
    return parent, items, info


def split_resting_on_top(mask: np.ndarray, voxel: float, min_jump: float = 0.05):
    """(parent mask, [top mask], info): cut at the highest slice where the
    footprint grows going up by more than min_jump (a lid's or cap's
    overhang). None if there is no such step."""
    A = mask.sum(axis=(0, 1)).astype(float)
    ks = [k for k in range(2, len(A)) if A[k - 1] > 0 and A[k] > (1 + min_jump) * A[k - 1]
          and A[k:].max() > 0]
    if not ks:
        return mask, [], {"found": 0, "reason": "no footprint step"}
    k = max(ks)
    top = mask.copy()
    top[:, :, :k] = False
    return mask & ~top, [top], {"found": 1, "cut_k": k}


def part_mesh(grid: FusedGrid, own: np.ndarray, all_occ: np.ndarray, forced: np.ndarray | None = None):
    """Mesh one part from the field: other parts' voxels pushed outside,
    forced (hidden, extruded) voxels inside."""
    F = grid.field.copy()
    F[all_occ & ~own] = 1.0
    if forced is not None:
        F[forced] = -1.0
        F[all_occ & ~own & ~forced] = 1.0
    g = FusedGrid(F, grid.weight, grid.free_views, grid.origin, grid.voxel, grid.trunc)
    m = grid_mesh(g, bottom_z=0.0)
    if m is None:
        return None
    parts = m.split(only_watertight=False)
    biggest = max(len(p.faces) for p in parts)
    import trimesh
    return trimesh.util.concatenate([p for p in parts if len(p.faces) >= 0.05 * biggest])


def decompose(grid: FusedGrid, parts_desc: list[dict]) -> tuple[list[Part], list[str]]:
    """parts_desc: [{name, count, relation, parent, ...}] with exactly one
    relation == "base". Handles one level of nesting (tubes inserted in a
    rotor resting inside a housing). Returns parts and log lines."""
    occ = grid.field < 0
    observed = grid.field > -0.995          # exactly -1: no view saw this voxel
    log = []
    base = next(p for p in parts_desc if p["relation"] == "base")
    remaining = occ.copy()
    out: list[Part] = []

    def children(of):
        """Child kinds of `of`, grouped by relation: one geometric split per
        group, its pieces assigned to the group's kinds later (by colour)."""
        groups = {}
        for p in parts_desc:
            if p.get("parent", base["name"]) == of and p["relation"] != "base":
                groups.setdefault(p["relation"], []).append(p)
        return groups

    def label(kinds, i, n):
        stem = kinds[0]["name"] if len(kinds) == 1 else "_or_".join(k["name"] for k in kinds)
        return stem if n == 1 else f"{stem}_{i:02d}"

    def split_level(region_mask, owner_name, owner_rel="base", owner_parent=""):
        """Split owner_name's region into the owner + its children
        (recursively); children go to `out`, the owner is returned."""
        owner_mask = region_mask
        for rel, kinds in children(owner_name).items():
            expected = sum(int(k.get("count", 1)) for k in kinds)
            names = "+".join(k["name"] for k in kinds)
            if rel == "resting_inside":
                owner_mask, items, info = split_resting_inside(owner_mask, grid.voxel, observed=observed)
                log.append(f"{names} resting inside {owner_name}: found {info['found']} (expected {expected})")
                if expected == 1 and len(items) > 1:
                    # described as ONE piece (a rotor = hub + sleeves that fused apart): merge
                    merged = np.zeros_like(items[0])
                    for m in items:
                        merged |= m
                    items = [merged]
                    log.append(f"   merged into 1 (described as one piece)")
                for i, m in enumerate(items):
                    # a nested owner (a rotor with tubes) needs its own name for its children
                    nm = kinds[0]["name"] if len(kinds) == 1 else label(kinds, i, len(items))
                    out.extend(split_level(m, kinds[0]["name"] if len(kinds) == 1 else nm, "resting_inside", owner_name))
                    out[-1].name = label(kinds, i, len(items))
                    out[-1].kinds = [k["name"] for k in kinds]
            elif rel == "inserted":
                g = next(((int(k.get("grid_rows", 0)), int(k.get("grid_cols", 0)), float(k.get("pitch_mm", 0)))
                          for k in kinds if k.get("grid_rows")), None)
                owner_mask, ins, exts, info = split_inserted(owner_mask, grid.voxel, expected, observed=observed, grid=g)
                if info.get("grid_split"):
                    log.append(f"   grid split: {info['grid_split']}")
                log.append(f"{names} inserted in {owner_name}: found {info['found']} (expected "
                           f"{expected}), {info.get('empty_bores_seen', 0)} empty bores seen, "
                           f"extruded {info.get('extrusion')}")
                for i, (m, e) in enumerate(zip(ins, exts)):
                    out.append(Part(label(kinds, i, len(ins)), "inserted", owner_name, m | e,
                                    forced_inside=e & ~m, kinds=[k["name"] for k in kinds]))
            elif rel == "resting_on_top":
                owner_mask, tops, info = split_resting_on_top(owner_mask, grid.voxel)
                log.append(f"{names} on top of {owner_name}: found {info['found']}")
                for i, m in enumerate(tops):
                    out.append(Part(label(kinds, i, len(tops)), "resting_on_top", owner_name, m,
                                    kinds=[k["name"] for k in kinds]))
        return [Part(owner_name, owner_rel, owner_parent, owner_mask, kinds=[owner_name])]

    base_parts = split_level(remaining, base["name"])
    return base_parts + out, log

"""Stage 2a: open-set instance segmentation, per view.

Finds every object-like region in an image WITHOUT assuming what's
there or how many objects exist -- no text prompt, no fixed catalog.
Uses SAM's automatic mask generation (a grid of point prompts over the
whole image, each producing a candidate mask), which is exactly the
"propose everything segmentable" behavior this needs; labeling.py
attaches an open-vocabulary guess to each resulting mask afterward.

Requires the .venv-vision environment (GPU torch + transformers) --
this is intentionally kept separate from the lightweight .venv-cad used
by calibration.py/markers.py/pose_solver.py, which have no business
depending on a multi-GB ML stack.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_PIPELINE = None  # lazy-loaded: model load takes a few seconds, do it once


@dataclass
class InstanceMask:
    mask: np.ndarray       # HxW bool
    score: float           # SAM's predicted IoU/stability score
    bbox_xyxy: tuple[int, int, int, int]
    area_px: int
    centroid_xy: tuple[float, float]


def _get_pipeline(model: str = "facebook/sam-vit-base", device: int = 0):
    global _PIPELINE
    if _PIPELINE is None:
        from transformers import pipeline
        _PIPELINE = pipeline("mask-generation", model=model, device=device)
    return _PIPELINE


def _to_instance(mask: np.ndarray, score: float) -> InstanceMask:
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return InstanceMask(
        mask=mask, score=float(score), bbox_xyxy=(x0, y0, x1, y1),
        area_px=int(mask.sum()), centroid_xy=(float(xs.mean()), float(ys.mean())))


def _contained_fraction(inner: np.ndarray, outer: np.ndarray) -> float:
    inner_area = inner.sum()
    if inner_area == 0:
        return 0.0
    return float(np.logical_and(inner, outer).sum()) / float(inner_area)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    return float(inter) / float(np.logical_or(a, b).sum() or 1)


def _sam_pass(pipe, image_rgb: np.ndarray, pred_iou_thresh: float,
              stability_score_thresh: float) -> list[tuple[np.ndarray, float]]:
    from PIL import Image
    out = pipe(Image.fromarray(image_rgb), pred_iou_thresh=pred_iou_thresh,
               stability_score_thresh=stability_score_thresh)
    return [(np.asarray(m, dtype=bool), float(sc)) for m, sc in zip(out["masks"], out["scores"])]


def generate_masks(
    image_rgb: np.ndarray,
    min_area_px: int = 200,
    max_area_fraction: float = 0.6,
    containment_dedup_thresh: float = 0.9,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.92,
    tiles: int = 1,
    tile_overlap: float = 0.2,
    dedup: bool = True,
) -> list[InstanceMask]:
    """Propose candidate object instance masks for one image.

    tiles > 1 additionally runs SAM on a tiles x tiles grid of overlapping
    crops and maps those masks back to full resolution. SAM resizes every
    input to 1024px, so small low-contrast objects (white labware on a
    white bench, seen from 2m) get too few pixels in a single full-image
    pass; zoomed tiles recover them. (transformers' own crops_n_layers
    option crashes on non-uniform crop sizes, hence doing it here.)

    Filters out:
      - masks too small to be a real object (sensor noise / grid artifacts)
      - masks covering most of the frame (background/wall/bench surface,
        not a discrete object -- `max_area_fraction` of total pixels)
      - near-duplicates: a mask >= `containment_dedup_thresh` contained in
        an already-kept larger mask (SAM re-segmenting a sub-part of the
        same object, e.g. one well of a tray), or with IoU > 0.8 against
        one (the same object found again by an overlapping tile).

    dedup=False returns every proposal; call dedup_masks() yourself AFTER
    removing background masks. Deduplicating first is wrong whenever SAM
    also proposes the surface the objects stand on: that mask contains
    every object on it, so each object's own mask is dropped as "contained",
    and when the surface mask is filtered out later the objects are gone
    (random-object scene: 7 of 24 views lost most objects this way).
    """
    pipe = _get_pipeline()
    h, w = image_rgb.shape[:2]
    total_px = h * w

    raw = _sam_pass(pipe, image_rgb, pred_iou_thresh, stability_score_thresh)
    if tiles > 1:
        for ty, tx, th, tw in tile_rects(h, w, tiles, tile_overlap):
            for m, sc in _sam_pass(pipe, image_rgb[ty:ty + th, tx:tx + tw],
                                   pred_iou_thresh, stability_score_thresh):
                full = np.zeros((h, w), bool)
                full[ty:ty + th, tx:tx + tw] = m
                raw.append((full, sc))

    candidates = []
    for mask, score in raw:
        area = int(mask.sum())
        if area < min_area_px or area > max_area_fraction * total_px:
            continue
        candidates.append(_to_instance(mask, score))
    return dedup_masks(candidates, containment_dedup_thresh) if dedup else candidates


def dedup_masks(candidates: list[InstanceMask], containment_dedup_thresh: float = 0.9) -> list[InstanceMask]:
    # Largest first, so dedup always compares a candidate against masks
    # that are already confirmed to survive.
    candidates = sorted(candidates, key=lambda m: m.area_px, reverse=True)
    kept: list[InstanceMask] = []
    for cand in candidates:
        if any(_contained_fraction(cand.mask, k.mask) >= containment_dedup_thresh
               or _iou(cand.mask, k.mask) > 0.8 for k in kept):
            continue
        kept.append(cand)
    return kept


def tile_rects(h: int, w: int, tiles: int, tile_overlap: float = 0.2) -> list[tuple[int, int, int, int]]:
    """(y, x, height, width) of each crop generate_masks() runs SAM on."""
    th, tw = int(h / tiles * (1 + tile_overlap)), int(w / tiles * (1 + tile_overlap))
    return [(int(ty), int(tx), th, tw) for ty in np.linspace(0, h - th, tiles).astype(int)
            for tx in np.linspace(0, w - tw, tiles).astype(int)]


def drop_tile_clipped(masks: list[InstanceMask], image_shape: tuple[int, int], tiles: int,
                      tile_overlap: float = 0.2, min_edge_px: int = 20) -> list[InstanceMask]:
    """Drop masks that a tile's INTERIOR edge cuts off (as SAM's own crop
    mode does): they lie entirely inside one tile and run along an edge of
    it that isn't the image border. The typical one is the tile's patch of
    the bench surface: each covers under the bench filter's fraction of the
    whole bench, so it survived, and views whose foreground held most of the
    bench carved almost nothing (random-object scene: a 600mm blob merging
    three objects, heights +25 to +100mm). An object crossing a tile edge is
    still found by the full-image pass and by the overlapping neighbour tile.
    Works on cached masks: a full-image-pass mask that lies inside a tile and
    follows its edge line for min_edge_px pixels is not a realistic case."""
    if tiles <= 1:
        return masks
    h, w = image_shape
    rects = tile_rects(h, w, tiles, tile_overlap)
    kept = []
    for m in masks:
        x0, y0, x1, y1 = m.bbox_xyxy
        clipped = False
        for ty, tx, th, tw in rects:
            if not (x0 >= tx and y0 >= ty and x1 <= tx + tw and y1 <= ty + th):
                continue
            edges = []
            if ty > 0:
                edges.append(m.mask[ty, tx:tx + tw])
            if ty + th < h:
                edges.append(m.mask[ty + th - 1, tx:tx + tw])
            if tx > 0:
                edges.append(m.mask[ty:ty + th, tx])
            if tx + tw < w:
                edges.append(m.mask[ty:ty + th, tx + tw - 1])
            if any(e.sum() >= min_edge_px for e in edges):
                clipped = True
                break
        if not clipped:
            kept.append(m)
    return kept

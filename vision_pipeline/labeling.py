"""Stage 2b: attach an open-vocabulary label to each mask from
segmentation.py, as a HINT for shape_priors.py's primitive-family choice
and for naming exported assets -- never a catalog lookup key. Nothing
downstream should branch on an exact string match against this label;
it's evidence, and it can be wrong.

Uses CLIP zero-shot similarity against an editable list of candidate
phrases (DEFAULT_VOCABULARY below) rather than a captioning model: for
this pipeline's purpose (deciding "does this look like a cylinder, a
flat perforated tray, a box, ..." for Stage 4) a scored match against
shape-descriptive phrases is more directly useful than free-form prose,
and it's lighter-weight than running a second captioning model. The
vocabulary is just a list of strings -- extend it freely; nothing else
needs to change when new object types show up in a scene.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_CLIP = None  # (model, processor), lazy-loaded

DEFAULT_VOCABULARY = [
    "a clear cylindrical tube",
    "a conical-bottom test tube",
    "a flat tray with a grid of holes",
    "a flat perforated plate with many small wells",
    "a small rectangular box or bin",
    "a round-topped cylindrical device",
    "a handheld pipette",
    "a thin flat rod or stick",
    "a robotic arm or gripper",
    "a flat empty bench or platform surface",
    "a stack of disposable pipette tips",
    "a small electronic device",
    "a lid or cap",
    # An explicit "none of the above" option. Without this, CLIP's
    # softmax-over-a-closed-list forces a confident-looking score onto
    # whichever phrase is LEAST bad for a mask that doesn't match any of
    # them (e.g. a background blob spanning several objects) -- which
    # reads as a wrong answer instead of the low-confidence non-match it
    # actually is. Masks that score this highest should be treated as
    # unlabeled, not passed to shape_priors.py as if this were a
    # meaningful shape hint.
    "a large region spanning multiple different objects, not one item",
]


@dataclass
class LabelGuess:
    text: str
    score: float


def _get_clip(model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda:0"):
    global _CLIP
    if _CLIP is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        device = device if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_name)
        _CLIP = (model, processor, device)
    return _CLIP


def _crop_masked(image_rgb: np.ndarray, mask: np.ndarray, bbox_xyxy: tuple[int, int, int, int],
                  pad_px: int = 8) -> np.ndarray:
    """Crop to the mask's bbox (with a small margin for context) and
    blank out everything outside the mask so CLIP scores the object, not
    whatever happens to share its bounding box.
    """
    x0, y0, x1, y1 = bbox_xyxy
    h, w = image_rgb.shape[:2]
    x0, y0 = max(0, x0 - pad_px), max(0, y0 - pad_px)
    x1, y1 = min(w, x1 + pad_px), min(h, y1 + pad_px)
    crop = image_rgb[y0:y1, x0:x1].copy()
    mask_crop = mask[y0:y1, x0:x1]
    crop[~mask_crop] = 255  # neutral background, not black (avoids biasing toward "dark" labels)
    return crop


def label_mask(
    image_rgb: np.ndarray, mask: np.ndarray, bbox_xyxy: tuple[int, int, int, int],
    vocabulary: list[str] = DEFAULT_VOCABULARY, top_k: int = 3,
) -> list[LabelGuess]:
    """Zero-shot label a single masked object crop against `vocabulary`.
    Returns the top_k candidates sorted by score (softmax over the full
    vocabulary, so scores are relative confidences, not independent
    probabilities).
    """
    import torch
    from PIL import Image

    model, processor, device = _get_clip()
    crop = _crop_masked(image_rgb, mask, bbox_xyxy)
    if crop.size == 0:
        return []

    inputs = processor(text=vocabulary, images=Image.fromarray(crop),
                        return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()

    order = np.argsort(-probs)[:top_k]
    return [LabelGuess(text=vocabulary[i], score=float(probs[i])) for i in order]

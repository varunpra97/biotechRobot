"""Stage 2 end-to-end on one image: open-set masks (SAM) + open-vocab
labels (CLIP), no platform-pose filtering (use run_segmentation_filtered
for that once a Stage 1 camera pose is available for the image).

Usage:
  .venv-vision/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/run_segmentation.py \
      --image path/to/photo.jpg --out biotech-lab-robotics/vision_pipeline/out/seg_demo
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from labeling import label_mask
from segmentation import generate_masks

RNG = np.random.default_rng(0)


def draw_overlay(image_bgr: np.ndarray, masks_with_labels: list) -> np.ndarray:
    overlay = image_bgr.copy()
    for inst, labels in masks_with_labels:
        color = tuple(int(c) for c in RNG.integers(60, 255, 3))
        colored = np.zeros_like(image_bgr)
        colored[inst.mask] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.45, 0)
        x0, y0, x1, y1 = inst.bbox_xyxy
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        label_text = labels[0].text if labels else "?"
        cv2.putText(overlay, f"{label_text} ({labels[0].score:.2f})" if labels else "?",
                    (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    cv2.LINE_AA)
    return overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise SystemExit(f"could not read {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("running SAM automatic mask generation...")
    masks = generate_masks(image_rgb)
    print(f"{len(masks)} candidate instances after filtering/dedup")

    results = []
    masks_with_labels = []
    print("labeling each instance with CLIP...")
    for inst in masks:
        labels = label_mask(image_rgb, inst.mask, inst.bbox_xyxy)
        masks_with_labels.append((inst, labels))
        results.append({
            "bbox_xyxy": inst.bbox_xyxy,
            "area_px": inst.area_px,
            "sam_score": inst.score,
            "centroid_xy": inst.centroid_xy,
            "labels": [{"text": l.text, "score": l.score} for l in labels],
        })
        top = labels[0] if labels else None
        print(f"  bbox={inst.bbox_xyxy} area={inst.area_px:6d} "
              f"-> {top.text if top else '?'} ({top.score:.2f})" if top else "  no label")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw_overlay(image_bgr, masks_with_labels)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)
    (out_dir / "instances.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir / 'overlay.png'} and {out_dir / 'instances.json'}")


if __name__ == "__main__":
    main()

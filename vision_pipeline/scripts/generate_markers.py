"""Generate printable ArUco marker sheets (one marker per page) from the
same dictionary markers.py detects against (DICT_4X4_50).

Each page also prints a 100mm reference ruler line next to the marker.
Printer "fit to page" scaling is a common source of error -- after
printing, measure the ruler line with an actual ruler. If it doesn't
read 100mm, your printer rescaled the page; measure the marker's own
printed side length directly instead and use THAT measured value (not
the nominal --side-mm) in marker_layout.json. The physical accuracy of
every downstream measurement in this pipeline depends on this number
being the real printed size, not the intended one.

Usage:
  python scripts/generate_markers.py --ids 0 1 2 3 --side-mm 50 --out markers/
Print each PNG at "actual size" / 100% scale (not "fit to page").
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from markers import DEFAULT_DICTIONARY

DPI = 300
QUIET_ZONE_FRACTION = 0.25  # white border around the marker, as a fraction of side length


def mm_to_px(mm: float, dpi: int = DPI) -> int:
    return int(round(mm / 25.4 * dpi))


def make_page(marker_id: int, side_mm: float, dpi: int = DPI) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(DEFAULT_DICTIONARY)
    side_px = mm_to_px(side_mm, dpi)
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)

    quiet_px = int(side_px * QUIET_ZONE_FRACTION)
    ruler_px = mm_to_px(100.0, dpi)
    page_w = side_px + 2 * quiet_px
    page_h = side_px + 2 * quiet_px + mm_to_px(25, dpi)  # extra room for ruler + label

    page = np.full((page_h, page_w), 255, dtype=np.uint8)
    page[quiet_px:quiet_px + side_px, quiet_px:quiet_px + side_px] = marker_img

    ruler_y = quiet_px + side_px + mm_to_px(8, dpi)
    tick_h = mm_to_px(3, dpi)
    cv2.line(page, (quiet_px, ruler_y), (quiet_px + ruler_px, ruler_y), 0, 2)
    cv2.line(page, (quiet_px, ruler_y - tick_h), (quiet_px, ruler_y + tick_h), 0, 2)
    cv2.line(page, (quiet_px + ruler_px, ruler_y - tick_h),
              (quiet_px + ruler_px, ruler_y + tick_h), 0, 2)

    page_color = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    cv2.putText(page_color, f"id={marker_id}  nominal_side={side_mm:.1f}mm",
                (quiet_px, page_h - mm_to_px(4, dpi)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(page_color, "100mm ref -- measure after printing",
                (quiet_px, ruler_y + mm_to_px(6, dpi)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return page_color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--side-mm", type=float, default=50.0,
                     help="nominal printed marker side length; VERIFY with "
                          "a ruler after printing and use the measured "
                          "value in marker_layout.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for marker_id in args.ids:
        page = make_page(marker_id, args.side_mm)
        path = out_dir / f"marker_{marker_id}.png"
        cv2.imwrite(str(path), page, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # Tag the PNG's physical DPI so "print at 100%" tools size it
        # correctly by default; still verify with a ruler regardless.
        try:
            from PIL import Image
            img = Image.open(path)
            img.save(path, dpi=(DPI, DPI))
        except ImportError:
            pass
        print(f"wrote {path} (id={marker_id}, nominal side={args.side_mm}mm "
              f"@ {DPI} DPI -- print at 100% scale, then measure)")


if __name__ == "__main__":
    main()

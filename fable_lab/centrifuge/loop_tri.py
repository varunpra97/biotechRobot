"""Triangulate a point seen in both views:  python loop_tri.py <dir> uA vA uB vB  -> base-frame point (m) and how far
apart the two rays pass (mm).   python loop_tri.py <dir> proj x y z (m)  -> its pixel in each view.
Uses only the camera calibration the server writes (calib.json)."""
import json
import sys
from pathlib import Path

import numpy as np

cal = json.loads((Path(sys.argv[1]) / "calib.json").read_text())


def ray(c, u, v):
    d = ((u - c["W"] / 2) / c["f"] * np.array(c["right"]) - (v - c["H"] / 2) / c["f"] * np.array(c["up"])
         - np.array(c["back"]))
    return np.array(c["pos"]), d / np.linalg.norm(d)


if sys.argv[2] == "proj":
    P = np.array([float(x) for x in sys.argv[3:6]])
    for k, c in cal.items():
        d = P - np.array(c["pos"])
        z = -d @ np.array(c["back"])
        print(k, round(c["W"] / 2 + c["f"] * (d @ np.array(c["right"])) / z, 1),
              round(c["H"] / 2 - c["f"] * (d @ np.array(c["up"])) / z, 1))
else:
    vals = [float(x) for x in sys.argv[2:]]
    for i in range(0, len(vals), 4):
        uA, vA, uB, vB = vals[i:i + 4]
        (p1, d1), (p2, d2) = ray(cal["A"], uA, vA), ray(cal["B"], uB, vB)
        t = np.linalg.lstsq(np.stack([d1, -d2], axis=1), p2 - p1, rcond=None)[0]
        a, b = p1 + t[0] * d1, p2 + t[1] * d2
        print("point_m", [round(float(x), 4) for x in (a + b) / 2], "ray_gap_mm",
              round(float(np.linalg.norm(a - b)) * 1e3, 2))

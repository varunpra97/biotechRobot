"""Compare Stage 1's recovered camera poses against the ground truth
that render_synthetic_capture.py wrote, so "the pipeline worked" is a
number, not an eyeballed overlay.

Usage:
  python scripts/validate_synthetic_capture.py --capture-dir biotech-lab-robotics/vision_pipeline/out/synthetic_capture
(run solve_camera_poses.py into <capture-dir>/poses first)
"""
import argparse
import json
from pathlib import Path

import numpy as np


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = np.asarray(q1) / np.linalg.norm(q1)
    q2 = np.asarray(q2) / np.linalg.norm(q2)
    dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
    return np.degrees(2 * np.arccos(dot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    args = ap.parse_args()

    capture_dir = Path(args.capture_dir)
    ground_truth = {s["name"]: s
                     for s in json.loads((capture_dir / "ground_truth_poses.json").read_text())["shots"]}
    recovered = json.loads((capture_dir / "poses" / "camera_poses.json").read_text())["poses"]
    recovered_by_name = {Path(p["image"]).stem: p for p in recovered}

    print(f"{'shot':20s} {'note':24s} {'status':10s} {'pos_err_mm':11s} {'rot_err_deg':11s}")
    n_pass, n_fail_expected, n_fail_unexpected = 0, 0, 0
    for name, gt in ground_truth.items():
        is_stress = gt["note"].startswith("stress") or gt["note"] == "close detail"
        rec = recovered_by_name.get(name)
        if rec is None:
            status = "no-pose"
            if is_stress:
                n_fail_expected += 1
            else:
                n_fail_unexpected += 1
            print(f"{name:20s} {gt['note']:24s} {status:10s} {'-':11s} {'-':11s}")
            continue

        pos_err_mm = 1000 * np.linalg.norm(
            np.array(rec["position_world"]) - np.array(gt["position_world"]))
        rot_err_deg = quat_angle_deg(rec["quaternion_wxyz"], gt["quaternion_wxyz"])

        # Thresholds are deliberately loose for the stress views (they're
        # *expected* to be degraded, not necessarily garbage) and tight
        # for the ring/top/detail views that represent a real capture.
        threshold_mm, threshold_deg = (50.0, 2.0) if is_stress else (5.0, 0.5)
        ok = pos_err_mm < threshold_mm and rot_err_deg < threshold_deg
        status = "ok" if ok else "DRIFT"
        if ok:
            n_pass += 1
        elif is_stress:
            n_fail_expected += 1
        else:
            n_fail_unexpected += 1

        print(f"{name:20s} {gt['note']:24s} {status:10s} {pos_err_mm:11.2f} {rot_err_deg:11.3f}")

    print(f"\n{n_pass} accurate, {n_fail_expected} degraded-as-expected "
          f"(stress views), {n_fail_unexpected} UNEXPECTED failures "
          f"(non-stress view that should have solved cleanly)")
    if n_fail_unexpected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

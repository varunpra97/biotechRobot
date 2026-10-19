"""Score decomposed parts (decompose_objects.py -> parts.json) against the
composite scene's ground truth (make_composite_scene.py ->
ground_truth_composites.json): part counts, per-part size and position,
material, density and mass, and each composite's total mass.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/score_composites.py \
      --capture-dir CAPTURE [--recon-subdir reconstruction_depth]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

# material families: a predicted name matches a true one if both hit the same family
FAMILIES = {
    "polypropylene": ["polypropylene", r"\bpp\b"],
    "polystyrene": ["polystyrene", r"\bps\b"],
    "abs": [r"\babs\b", "acrylonitrile"],
    "hdpe": ["hdpe", "polyethylene", r"\bpe\b"],
    "aluminium": ["alumin"],
    "stainless_steel": ["steel"],
    "brass": ["brass"],
    "wood_beech": ["wood", "beech", "maple", "birch", "oak", "pine"],
    "rubber": ["rubber", "silicone", "elastomer"],
    "ceramic": ["ceramic", "porcelain", "stoneware"],
    "glass": ["glass", "borosilicate"],
}


def family(name: str) -> str | None:
    """The family of the material named FIRST: "borosilicate glass (possibly
    clear polypropylene)" is glass, "stainless steel (possibly aluminium)"
    is steel -- the hedge doesn't count."""
    n = name.lower().replace("-", " ").replace("_", " ")
    best = None
    for fam, pats in FAMILIES.items():
        for p in pats:
            m = re.search(p, n)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), fam)
    return best[1] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--recon-subdir", default="reconstruction_depth")
    args = ap.parse_args()
    cap = Path(args.capture_dir)
    rec = cap / args.recon_subdir
    gt = json.loads((cap / "ground_truth_composites.json").read_text())
    pred = json.loads((rec / "parts.json").read_text())["parts"]

    by_obj = {}
    for p in pred:
        by_obj.setdefault(p["object"], []).append(p)
    report, rows = {}, []
    for g, comp in gt.items():
        gc = (np.array(comp["bbox_min"]) + np.array(comp["bbox_max"])) / 2
        oid = min(by_obj, key=lambda o: np.linalg.norm(
            np.mean([(np.array(q["bbox_min_m"]) + q["bbox_max_m"])[:2] / 2 for q in by_obj[o]], axis=0) - gc[:2]))
        P, T = by_obj[oid], comp["parts"]
        pc = np.array([(np.array(q["bbox_min_m"]) + q["bbox_max_m"]) / 2 for q in P])
        tc = np.array([t["center_m"] for t in T])
        cost = np.linalg.norm(pc[:, None] - tc[None], axis=2)
        ri, ci = linear_sum_assignment(cost)
        matches = []
        for i, j in zip(ri, ci):
            p, t = P[i], T[j]
            pe = np.array(p["bbox_max_m"]) - p["bbox_min_m"]
            matches.append({
                "pred": p["part"], "truth": t["part"], "role_truth": t["role"],
                "centre_err_mm": round(float(cost[i, j]) * 1000, 1),
                "dims_err_mm": ((pe - np.array(t["extents_m"])) * 1000).round(1).tolist(),
                "material_pred": p["material"]["name"], "material_truth": t["material"],
                "material_ok": family(p["material"]["name"]) == family(t["material"]) and family(t["material"]) is not None,
                "density_err_pct": round((p["material"]["density_kg_m3"] / t["density_kg_m3"] - 1) * 100, 1),
                "mass_pred_g": round(p["mass_kg"] * 1000, 2),
                "mass_truth_g": round(t["mass_kg"] * 1000, 2) if t["mass_kg"] else None,
                "mass_err_pct": round((p["mass_kg"] / t["mass_kg"] - 1) * 100, 1) if t["mass_kg"] else None})
        total_p = sum(q["mass_kg"] for q in P)
        report[g] = {"object": oid, "recipe": comp["recipe"], "parts_found": len(P), "parts_truth": len(T),
                     "total_mass_pred_g": round(total_p * 1000, 1),
                     "total_mass_truth_g": round(comp["total_mass_kg"] * 1000, 1) if comp["total_mass_kg"] else None,
                     "total_mass_err_pct": (round((total_p / comp["total_mass_kg"] - 1) * 100, 1)
                                            if comp["total_mass_kg"] else None),
                     "matches": matches}
        mat_ok = sum(m["material_ok"] for m in matches)
        errs = [abs(m["mass_err_pct"]) for m in matches if m["mass_err_pct"] is not None]
        rows.append((g, len(P), len(T), mat_ok, len(matches),
                     np.median(errs) if errs else float("nan"),
                     report[g]["total_mass_err_pct"] if report[g]["total_mass_err_pct"] is not None else float("nan"),
                     np.median([max(abs(e) for e in m["dims_err_mm"]) for m in matches])))
    (rec / "composite_scores.json").write_text(json.dumps(report, indent=2))
    print(f"{'composite':28s} parts(found/true)  material ok  part mass |err| med  total mass err  part size err med")
    for g, nf, nt, mo, nm, me, te, de in rows:
        print(f"{g:28s} {nf:5d}/{nt:<5d}        {mo:2d}/{nm:<2d}       {me:6.1f}%           {te:+7.1f}%        {de:6.1f} mm")
    print(f"details: {rec / 'composite_scores.json'}")


if __name__ == "__main__":
    main()

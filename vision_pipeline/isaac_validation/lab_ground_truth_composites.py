"""Per-part ground truth for the LAB scene, in score_composites.py's format
(ground_truth_composites.json), from a capture's exported meshes.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/isaac_validation/lab_ground_truth_composites.py --capture-dir CAPTURE

Assemblies, as the simulation defines them (sim_snapshot/lab_scene.py):
  tube_holder   TubeHolder (plastic) + the 20 Tube_r*_c* (glass)
  tip_rack      TipRack (plastic) + Tips (100 tips, one exported mesh: split
                into connected components here)
  centrifuge    CentrifugeBody split into connected pieces: the housing
                (largest) and the rotor (hub + six tilted sleeves). In the
                sim it is ONE static body, lid removed, no tubes loaded.
  others        single parts (tray, waste bin, pipette)

Densities: the sim assigns friction presets, not densities, and gives masses
only to loose parts (TUBE_MASS_KG = 1.5 g, borosilicate ~1.3 g by volume).
Truth here uses handbook densities for the sim's material names -- glass
2230 (borosilicate), plastic 1050, metal 2700 (aluminium-like housing) --
so masses of fixtures are reference values, not the sim's own numbers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

DENSITY = {"glass": 2230.0, "plastic": 1050.0, "metal": 2700.0}
SIM_MATERIAL = {"TubeHolder": "plastic", "Tube": "glass", "TipRack": "plastic", "Tips": "plastic",
                "CentrifugeBody": "metal", "SampleTray": "plastic", "WasteBin": "plastic", "Pipette0": "plastic"}
NAME = {"glass": "borosilicate glass", "plastic": "polypropylene", "metal": "aluminium"}


# lab_scene.py: "580.6 mm^3 of borosilicate glass" per test tube. The exported
# meshes are not watertight, so their own volume can't be trusted.
TUBE_MATERIAL_VOLUME_M3 = 580.6e-9


def part(name, mesh, role, sim_mat):
    if name.startswith("Tube_"):
        vol = TUBE_MATERIAL_VOLUME_M3
    else:
        vol = float(mesh.volume) if mesh.is_volume else None      # unknown: not scored
    lo, hi = mesh.bounds
    return {"name": name, "part": name, "role": role, "material": NAME[sim_mat],
            "density_kg_m3": DENSITY[sim_mat], "volume_m3": vol,
            "mass_kg": vol * DENSITY[sim_mat] if vol is not None else None,
            "extents_m": (hi - lo).tolist(), "center_m": ((lo + hi) / 2).tolist(), "mesh": mesh}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    args = ap.parse_args()
    cap = Path(args.capture_dir)
    gt = json.loads((cap / "ground_truth_meshes.json").read_text())
    load = lambda n: trimesh.load(cap / gt[n]["mesh"], force="mesh")       # noqa: E731

    groups = {}
    groups["tube_holder"] = ("tube_holder", [part("TubeHolder", load("TubeHolder"), "base", "plastic")] +
                             [part(n, load(n), "insert", "glass") for n in sorted(gt) if n.startswith("Tube_")])
    tips = [p for p in load("Tips").split(only_watertight=False) if len(p.faces) > 20]
    groups["tip_rack"] = ("tip_rack", [part("TipRack", load("TipRack"), "base", "plastic")] +
                          [part(f"Tip_{i:03d}", m, "insert", "plastic") for i, m in enumerate(tips)])
    pieces = sorted(load("CentrifugeBody").split(only_watertight=False), key=lambda m: -len(m.faces))
    housing, rotor = pieces[0], trimesh.util.concatenate([p for p in pieces[1:] if len(p.faces) > 4])
    groups["centrifuge"] = ("centrifuge", [part("housing", housing, "base", "metal"),
                                           part("rotor", rotor, "insert", "metal")])
    for n in ("SampleTray", "WasteBin", "Pipette0"):
        if n in gt:
            groups[n] = (n, [part(n, load(n), "base", SIM_MATERIAL[n])])

    out = {}
    for g, (recipe, parts) in groups.items():
        allm = trimesh.util.concatenate([p["mesh"] for p in parts])
        out[g] = {"recipe": recipe, "n_parts": len(parts), "bbox_min": allm.bounds[0].tolist(),
                  "bbox_max": allm.bounds[1].tolist(),
                  "total_mass_kg": (sum(p["mass_kg"] for p in parts) if all(p["mass_kg"] for p in parts) else None),
                  "parts": [{k: v for k, v in p.items() if k != "mesh"} for p in parts],
                  "note": "fixture masses are reference values (handbook density x volume); the sim gives "
                          "masses only to loose parts" if len(parts) > 1 else ""}
        tm = out[g]["total_mass_kg"]
        print(f"{g:14s} {len(parts):3d} parts  " + (f"{tm * 1000:8.1f} g" if tm else "mass not scored (non-watertight truth)"))
    (cap / "ground_truth_composites.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

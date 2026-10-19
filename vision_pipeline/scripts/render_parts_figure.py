"""Figure for split composites: the photo, the reconstructed parts rendered
from the same camera (each part its own colour), and a table per part kind:
count found vs true, predicted vs true material, mean part mass predicted
vs true. Needs decompose_objects.py's parts.json and, for the truth
columns, score_composites.py's composite_scores.json.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/render_parts_figure.py \
      --capture-dir CAPTURE --view el35_az022p5 --out parts.png [--objects obj_00 obj_01]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from geometry import camera_pose_to_rvec_tvec, quaternion_to_rotation_matrix
from render_results_figure import raycast, shade

PALETTE = np.array([[0.89, 0.47, 0.33], [0.35, 0.65, 0.45], [0.36, 0.52, 0.85], [0.85, 0.72, 0.30],
                    [0.66, 0.45, 0.78], [0.30, 0.72, 0.75], [0.90, 0.55, 0.65], [0.55, 0.60, 0.40],
                    [0.95, 0.80, 0.55], [0.50, 0.75, 0.95], [0.75, 0.55, 0.40], [0.60, 0.85, 0.60]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--recon-subdir", default="reconstruction_depth")
    ap.add_argument("--view", required=True)
    ap.add_argument("--objects", nargs="+", help="only these object ids")
    ap.add_argument("--scale", type=float, default=0.35)
    ap.add_argument("--title", default="Composite objects split into parts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    rec = cap / args.recon_subdir
    parts = [p for p in json.loads((rec / "parts.json").read_text())["parts"]
             if not args.objects or p["object"] in args.objects]
    scores = json.loads((rec / "composite_scores.json").read_text()) if (rec / "composite_scores.json").exists() else {}
    truth_of = {}
    for g, s in scores.items():
        for m in s["matches"]:
            truth_of[(s["object"], m["pred"])] = m

    poses_file = rec / "poses" / "camera_poses.json"
    if not poses_file.exists():
        poses_file = cap / "poses" / "camera_poses.json"
    pose = next(p for p in json.loads(poses_file.read_text())["poses"] if p["camera_id"] == args.view)
    rvec, tvec = camera_pose_to_rvec_tvec(quaternion_to_rotation_matrix(np.array(pose["quaternion_wxyz"])),
                                          np.array(pose["position_world"]))
    cam = json.loads((cap / "camera_intrinsics.json").read_text())["cameras"][args.view]
    W, H = cam["image_size"]
    K = np.array(cam["K"], float)
    K[:2] *= args.scale
    wh = (int(W * args.scale), int(H * args.scale))
    photo = cv2.cvtColor(cv2.imread(str(cap / "images" / f"{args.view}.png")), cv2.COLOR_BGR2RGB)
    photo = cv2.resize(photo, wh, interpolation=cv2.INTER_AREA)

    # one colour per part kind within an object; repeated parts get shades of it
    kinds = sorted({(p["object"], p["kind"]) for p in parts})
    kind_col = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(kinds)}
    meshes, cols = [], []
    rng = np.random.default_rng(0)
    for p in parts:
        meshes.append(trimesh.load(rec / p["mesh"], force="mesh"))
        c = kind_col[(p["object"], p["kind"])]
        cols.append(np.clip(c * rng.uniform(0.8, 1.15), 0, 1) if p["relation"] != "base" else c * 0.85)
    hit, gid, nrm, pts, dirs = raycast(meshes, K, rvec, tvec, wh)
    img = (photo.astype(float) / 255) * 0.25 + 0.6
    col = np.array(cols)[np.minimum(gid, len(cols) - 1)] * shade(nrm, dirs)[..., None]
    img = np.where(hit[..., None], col, img)
    ys, xs = np.nonzero(hit)
    pad = int(0.04 * wh[0])
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, wh[1]), max(xs.min() - pad, 0), min(xs.max() + pad, wh[0])

    rows = []
    for (oid, kind) in kinds:
        ps = [p for p in parts if p["object"] == oid and p["kind"] == kind]
        ms = [truth_of.get((oid, p["part"])) for p in ps]
        ms = [m for m in ms if m]
        n_true = None
        for g, s in scores.items():
            if s["object"] == oid:
                tnames = {m["truth"] for m in ms}
                n_true = len(tnames) if ps[0]["relation"] != "base" else 1
        mat_true = sorted({m["material_truth"] for m in ms}) if ms else ["?"]
        mass_p = np.mean([p["mass_kg"] for p in ps]) * 1000
        mt = [m["mass_truth_g"] for m in ms if m.get("mass_truth_g")]
        rows.append([oid, kind[:22], f"{len(ps)}", ps[0]["material"]["name"][:26], "/".join(mat_true)[:22],
                     f"{ps[0]['solid_fraction']:.2f}", f"{mass_p:.1f}", f"{np.mean(mt):.1f}" if mt else "-",
                     f"{(mass_p / np.mean(mt) - 1) * 100:+.0f}%" if mt else "-"])

    fig = plt.figure(figsize=(22, 9 + 0.32 * len(rows)))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.25 + 0.06 * len(rows)])
    a0, a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for a in (a0, a1, a2):
        a.axis("off")
    a0.imshow(photo[y0:y1, x0:x1])
    a0.set_title(f"Input photo ({args.view})", fontsize=13)
    a1.imshow(np.clip(img[y0:y1, x0:x1], 0, 1))
    a1.set_title("Output: each part a separate asset (one colour per kind; repeated parts shaded)", fontsize=13)
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    for (oid, kind) in kinds:
        ps = [p for p in parts if p["object"] == oid and p["kind"] == kind]
        top = np.mean([[(np.array(p["bbox_min_m"]) + p["bbox_max_m"])[0] / 2, (np.array(p["bbox_min_m"]) + p["bbox_max_m"])[1] / 2,
                        p["bbox_max_m"][2]] for p in ps], axis=0)
        pc = R @ top + np.asarray(tvec).reshape(3)
        u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2] - x0, K[1, 1] * pc[1] / pc[2] + K[1, 2] - y0
        if 0 <= u < x1 - x0 and 0 <= v < y1 - y0 and ps[0]["relation"] != "base":
            a1.annotate(f"{len(ps)}x {kind}" if len(ps) > 1 else kind, (u, v), xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                                            ec=kind_col[(oid, kind)], lw=1.5, alpha=0.9))
    t = a2.table(cellText=rows, colLabels=["object", "part kind", "found", "material (predicted)", "material (true)",
                                           "solid frac.", "mass/part pred g", "true g", "err"],
                 loc="upper center", cellLoc="center",
                 colWidths=[0.06, 0.16, 0.05, 0.19, 0.14, 0.07, 0.11, 0.08, 0.06])
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.6)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
        elif c == 1:
            cell.set_facecolor(np.append(kind_col[kinds[r - 1]], 0.45))
    src = sorted({p["description_source"] for p in parts})
    fig.suptitle(f"{args.title}\nparts description: {', '.join(src)}", fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out, dpi=100)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

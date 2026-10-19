"""Results figure for one view of a capture: the input photo and depth, and
the reconstructed 3D models rendered from the same camera with measured
dimensions (and, when ground truth exists, true dimensions and per-pixel
surface error).

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/render_results_figure.py \
      --capture-dir biotech-lab-robotics/vision_pipeline/depth_tests/random_s1 --view el35_az292p5 \
      --recon recon_final_perfect --noisy-recon recon_final_tof --out results.png
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh

from depth_sensor import load_depth
from geometry import camera_pose_to_rvec_tvec, quaternion_to_rotation_matrix

PALETTE = np.array([[0.89, 0.47, 0.33], [0.35, 0.65, 0.45], [0.36, 0.52, 0.85], [0.85, 0.72, 0.30],
                    [0.66, 0.45, 0.78], [0.30, 0.72, 0.75], [0.90, 0.55, 0.65], [0.55, 0.60, 0.40]])


def load_objects(rec: Path):
    out = []
    for o in json.loads((rec / "scene.json").read_text())["objects"]:
        m = trimesh.load(rec / o["mesh"], force="mesh")
        m.apply_translation(o["placement_m"])
        out.append((o, m))
    return out


def raycast(meshes, K, rvec, tvec, wh):
    sc = o3d.t.geometry.RaycastingScene()
    for m in meshes:
        t = o3d.t.geometry.TriangleMesh()
        t.vertex.positions = o3d.core.Tensor(np.asarray(m.vertices, np.float32))
        t.triangle.indices = o3d.core.Tensor(np.asarray(m.faces, np.int32))
        sc.add_triangles(t)
    w, h = wh
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    C = -R.T @ np.asarray(tvec, float).reshape(3)
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3) @ R
    r = sc.cast_rays(o3d.core.Tensor(np.concatenate([np.broadcast_to(C, d.shape), d], 1).astype(np.float32)))
    t = r["t_hit"].numpy()
    hit = np.isfinite(t)
    pts = C + d * np.where(hit, t, 0)[:, None]
    return hit.reshape(h, w), r["geometry_ids"].numpy().reshape(h, w), r["primitive_normals"].numpy().reshape(h, w, 3), \
        pts.reshape(h, w, 3), d.reshape(h, w, 3)


def shade(normals, dirs):
    dn = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    key = np.array([0.3, -0.4, 0.87])
    key /= np.linalg.norm(key)
    return 0.3 + 0.45 * np.abs(np.sum(normals * dn, -1)) + 0.3 * np.clip(normals @ key, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--view", required=True)
    ap.add_argument("--recon", required=True, help="reconstruction subdir to show")
    ap.add_argument("--noisy-recon", help="a second reconstruction (e.g. a simulated noisy sensor) for comparison")
    ap.add_argument("--scale", type=float, default=0.25, help="render at this fraction of the photo resolution")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    rec = cap / args.recon
    poses_file = rec / "poses" / "camera_poses.json"
    if not poses_file.exists():
        poses_file = cap / "poses" / "camera_poses.json"
    pose = next(p for p in json.loads(poses_file.read_text())["poses"] if p["camera_id"] == args.view)
    rvec, tvec = camera_pose_to_rvec_tvec(quaternion_to_rotation_matrix(np.array(pose["quaternion_wxyz"])),
                                          np.array(pose["position_world"]))
    cam = json.loads((cap / "camera_intrinsics.json").read_text())["cameras"][args.view]
    W, H = cam["image_size"]
    s = args.scale
    K = np.array(cam["K"], float)
    Ks = K.copy()
    Ks[:2] *= s
    wh = (int(W * s), int(H * s))

    photo = cv2.cvtColor(cv2.imread(str(cap / "images" / f"{args.view}.png")), cv2.COLOR_BGR2RGB)
    photo = cv2.resize(photo, wh, interpolation=cv2.INTER_AREA)
    manifest = json.loads((cap / "manifest.json").read_text())
    depth = load_depth(cap / f"depth/{args.view}.png", manifest.get("depth", {}).get("scale_m", 1e-4))
    depth = cv2.resize(depth, wh, interpolation=cv2.INTER_NEAREST)

    gt = json.loads((cap / "ground_truth_meshes.json").read_text()) if (cap / "ground_truth_meshes.json").exists() else {}
    gt_meshes = [trimesh.load(cap / g["mesh"], force="mesh") for g in gt.values()]
    gt_scene = None
    if gt_meshes:
        gt_scene = o3d.t.geometry.RaycastingScene()
        for m in gt_meshes:
            t = o3d.t.geometry.TriangleMesh()
            t.vertex.positions = o3d.core.Tensor(np.asarray(m.vertices, np.float32))
            t.triangle.indices = o3d.core.Tensor(np.asarray(m.faces, np.int32))
            gt_scene.add_triangles(t)

    objs = load_objects(rec)
    hit, gid, nrm, pts, dirs = raycast([m for _, m in objs], Ks, rvec, tvec, wh)

    # crop to the objects (with margin)
    ys, xs = np.nonzero(hit)
    pad = int(0.06 * wh[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, wh[0])
    y0, y1 = max(ys.min() - 2 * pad, 0), min(ys.max() + pad, wh[1])
    crop = (slice(y0, y1), slice(x0, x1))

    # reconstructed models, shaded, each its own colour, over a dimmed photo
    sh = shade(nrm, dirs)
    model = (photo.astype(float) / 255) * 0.25 + 0.6
    col = PALETTE[gid % len(PALETTE)] * sh[..., None]
    model = np.where(hit[..., None], col, model)

    def err_map(objs_):
        hit_, gid_, nrm_, pts_, dirs_ = raycast([m for _, m in objs_], Ks, rvec, tvec, wh)
        e = np.full(hit_.shape, np.nan)
        if gt_scene is not None:
            q = pts_[hit_].astype(np.float32)
            e[hit_] = gt_scene.compute_distance(o3d.core.Tensor(q)).numpy() * 1000
        return e, hit_, shade(nrm_, dirs_)

    fig = plt.figure(figsize=(21, 17))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.25, 1])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]),
            fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[2, 2])]
    for a in axes:
        a.axis("off")

    axes[0].imshow(photo[crop])
    axes[0].set_title(f"Input: colour photo ({W}x{H}, view {args.view})", fontsize=12)
    dshow = np.where(depth > 0, depth, np.nan)[crop]
    im = axes[1].imshow(dshow, cmap="turbo")
    axes[1].set_title("Input: depth map (rendered, Z-depth)", fontsize=12)
    cb = fig.colorbar(im, ax=axes[1], fraction=0.035, pad=0.01)
    cb.set_label("m")

    if args.noisy_recon:
        nd = cap / args.noisy_recon / "depth_used" / f"{args.view}.png"
        if nd.exists():
            dn = load_depth(nd)
            sens = json.loads((cap / args.noisy_recon / "scene.json").read_text()).get("sensor_model", {})
            dn = cv2.resize(dn, wh, interpolation=cv2.INTER_NEAREST)
            im2 = axes[2].imshow(np.where(dn > 0, dn, np.nan)[crop], cmap="turbo",
                                 vmin=np.nanmin(dshow), vmax=np.nanmax(dshow))
            name = sens.get("name", "?") if isinstance(sens, dict) else "?"
            res = sens.get("resolution") if isinstance(sens, dict) else None
            axes[2].set_title(f"Input: same depth through a simulated '{name}' sensor\n({res[0]}x{res[1]}, noise, "
                              f"dropouts; white = removed by the edge filter)"
                              if res else f"Input: depth through the '{name}' sensor model", fontsize=12)
            fig.colorbar(im2, ax=axes[2], fraction=0.035, pad=0.01).set_label("m")

    ax = axes[3]
    ax.imshow(np.clip(model[crop], 0, 1))
    ax.set_title(f"Output: reconstructed 3D models ({args.recon}), rendered from the same camera", fontsize=13)
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    rows = []
    for i, (o, m) in enumerate(objs):
        lo, hi = m.bounds
        top = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2]])
        pc = R @ top + np.asarray(tvec).reshape(3)
        u = Ks[0, 0] * pc[0] / pc[2] + Ks[0, 2] - x0
        v = Ks[1, 1] * pc[1] / pc[2] + Ks[1, 2] - y0
        ax.annotate(str(i + 1), (u, v), xytext=(0, 10), textcoords="offset points", ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color="white",
                    bbox=dict(boxstyle="circle,pad=0.25", fc=PALETTE[i % len(PALETTE)] * 0.75, ec="white", lw=1.2))
        d = (hi - lo) * 1000
        sc = o.get("score", {})
        truth = sc.get("ground_truth")
        tname = (truth if isinstance(truth, str) else "+".join(truth)) if truth else "-"
        tname = tname.split("_", 1)[-1] if tname.startswith("Rand") else tname
        t = sc.get("true_dims_mm")
        rows.append([str(i + 1), tname, f"{d[0]:.1f} x {d[1]:.1f} x {d[2]:.1f}",
                     f"{t[0]:.1f} x {t[1]:.1f} x {t[2]:.1f}" if t else "-",
                     f"{max(abs(e) for e in sc['dim_err_mm']):.2f}" if t else "-"])
    tab_ax = axes[6]
    tab_ax.set_title("Measured vs true (bounding box W x D x H, mm)", fontsize=13)
    tbl = tab_ax.table(cellText=rows, colLabels=["#", "object (truth)", "measured", "true", "worst err"],
                       loc="upper center", cellLoc="center", colWidths=[0.06, 0.28, 0.27, 0.27, 0.13])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
        elif c == 0:
            cell.set_facecolor(PALETTE[(r - 1) % len(PALETTE)])
            cell.set_text_props(color="white", fontweight="bold")
    tab_ax.text(0.5, 0.04, "Size error = bounding box of the reconstructed mesh minus the true one.\n"
                "The mushroom's 0.93 mm is the space under its cap, which no camera sees.",
                ha="center", va="bottom", fontsize=10, transform=tab_ax.transAxes)

    vmax = 1.0
    if gt_scene is not None:
        e, h_, sh_ = err_map(objs)
        ax = axes[4]
        base = (photo.astype(float) / 255) * 0.25 + 0.6
        ax.imshow(np.clip(base[crop], 0, 1))
        im3 = ax.imshow(np.ma.masked_invalid(e[crop]), cmap="magma_r", vmin=0, vmax=vmax)
        med = np.nanmedian(e)
        ax.set_title(f"Surface error vs true meshes ({args.recon}): median {med:.3f} mm", fontsize=12)
        fig.colorbar(im3, ax=ax, fraction=0.035, pad=0.01).set_label("distance to true surface, mm")
        if args.noisy_recon:
            e2, _, _ = err_map(load_objects(cap / args.noisy_recon))
            ax = axes[5]
            ax.imshow(np.clip(base[crop], 0, 1))
            im4 = ax.imshow(np.ma.masked_invalid(e2[crop]), cmap="magma_r", vmin=0, vmax=vmax * 3)
            ax.set_title(f"Surface error, {args.noisy_recon}: median {np.nanmedian(e2):.3f} mm (scale 0-3 mm)",
                         fontsize=12)
            fig.colorbar(im4, ax=ax, fraction=0.035, pad=0.01).set_label("distance to true surface, mm")

    fig.suptitle("Camera + depth -> 3D models: Isaac Sim random-object scene, poses from the fiducial markers",
                 fontsize=15)
    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

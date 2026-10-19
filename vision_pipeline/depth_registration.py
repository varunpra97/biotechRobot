"""Camera pose refinement from depth: align every view's depth point cloud to
all the OTHER views' (point-to-plane ICP), a few rounds, keeping the frame
the fiducial markers defined.

Why: marker poses are good to ~1mm median but up to ~5mm on some views
(random-object Isaac capture: median 1.0, max 5.3mm). Fused with those
poses, perfect depth blurred to 0.2-0.3mm median surface error and every
object grew by ~0.4mm, versus 0.03-0.06mm with exact poses. Depth surfaces
from overlapping views constrain relative pose far better than a few
markers do.

Anchoring the frame. Relative alignment alone can't tell where the
platform is, and it can't see a COMMON error: every marker-solved camera
was ~0.6mm too far from the scene (a 0.04% scale bias -- ArUco corners land
slightly inside the true marker edge), which pushed every surface outward
(+0.5mm on every object axis) and is invisible to view-to-view ICP. So each
view is also aligned to geometry known exactly by construction -- the
bench plane (z = 0 defines the frame) and the calibration markers' quads
-- which depth measures in absolute metric units (known_geometry_cloud).
Without that anchor, the mean correction is removed instead (keeps the
markers' frame, keeps their common bias).
"""
from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d


def _pose_matrix(rvec, tvec) -> np.ndarray:
    """4x4 world -> camera."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(tvec, float).reshape(3)
    return T


def _cloud_world(view, w2c: np.ndarray, stride: int, crop_min, crop_max, voxel: float):
    d = view.depth[::stride, ::stride]
    v, u = np.nonzero(d > 0)
    z = d[v, u].astype(np.float64)
    u = u * stride
    v = v * stride
    K = view.K
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
    c2w = np.linalg.inv(w2c)
    pw = pc @ c2w[:3, :3].T + c2w[:3, 3]
    keep = np.all((pw >= crop_min) & (pw <= crop_max), axis=1)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pw[keep]))
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30))
    pcd.orient_normals_towards_camera_location(c2w[:3, 3])
    return pcd


def _rotvec(R):
    return cv2.Rodrigues(R)[0].reshape(3)


def known_geometry_cloud(platform_wd: tuple[float, float], marker_corners: list[np.ndarray],
                         spacing: float = 0.002, bench_margin: float = 0.01):
    """Points + normals on what is known exactly: the bench top (z = 0 over
    the platform) and every marker quad (as laid out in marker_layout.json)."""
    hw, hd = platform_wd[0] / 2 - bench_margin, platform_wd[1] / 2 - bench_margin
    xs, ys = np.arange(-hw, hw, spacing), np.arange(-hd, hd, spacing)
    gx, gy = np.meshgrid(xs, ys)
    pts = [np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)]
    nrm = [np.tile([0.0, 0.0, 1.0], (gx.size, 1))]
    for c in marker_corners:
        c = np.asarray(c, float)                  # TL, TR, BR, BL
        a, b = c[1] - c[0], c[3] - c[0]
        n = np.cross(a, b)
        n /= np.linalg.norm(n)
        if n @ (c.mean(axis=0) - np.array([0, 0, -1.0])) < 0:   # face away from below the bench
            n = -n
        k = max(2, int(np.linalg.norm(a) / spacing))
        s, t = np.meshgrid(np.linspace(0, 1, k), np.linspace(0, 1, k))
        q = c[0] + s.reshape(-1, 1) * a + t.reshape(-1, 1) * b
        pts.append(q)
        nrm.append(np.tile(n, (len(q), 1)))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts)))
    pcd.normals = o3d.utility.Vector3dVector(np.concatenate(nrm))
    return pcd


def refine_poses(views, crop_min, crop_max, rounds: int = 3, voxel: float = 0.002,
                 max_dist: tuple = (0.008, 0.004, 0.002), stride: int | None = None, log=print,
                 anchor=None, anchor_only_rounds: int = 2):
    """Updates views[i].rvec / .tvec in place. crop: the region whose
    geometry is trusted for alignment (the platform and what's on it --
    not distant walls). Returns per-round mean correction sizes."""
    w2c = [_pose_matrix(v.rvec, v.tvec) for v in views]
    if stride is None:
        stride = max(1, int(round(views[0].depth.shape[1] / 1000)))
    history = []
    for r in range(rounds):
        dist = max_dist[min(r, len(max_dist) - 1)]
        clouds = [_cloud_world(v, T, stride, crop_min, crop_max, voxel) for v, T in zip(views, w2c)]
        corr = []
        for i in range(len(views)):
            target = o3d.geometry.PointCloud()
            if not (anchor is not None and r < anchor_only_rounds):
                for j, c in enumerate(clouds):
                    if j != i:
                        target += c
                target = target.voxel_down_sample(voxel)
            if anchor is not None:
                # added after down-sampling: averaged into the other views'
                # (commonly biased) points, the anchor lost its pull
                target += anchor
            res = o3d.pipelines.registration.registration_icp(
                clouds[i], target, dist, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
            corr.append(res.transformation)          # world -> world correction for view i's cloud
        Minv = np.eye(4)
        if anchor is None:     # keep the markers' frame: remove the mean correction
            mean_rv = np.mean([_rotvec(T[:3, :3]) for T in corr], axis=0)
            mean_t = np.mean([T[:3, 3] for T in corr], axis=0)
            M = np.eye(4)
            M[:3, :3] = cv2.Rodrigues(mean_rv)[0]
            M[:3, 3] = mean_t
            Minv = np.linalg.inv(M)
        sizes = []
        for i, T in enumerate(corr):
            Tc = Minv @ T
            # cloud_w' = Tc cloud_w  =>  c2w' = Tc c2w  =>  w2c' = w2c Tc^-1
            w2c[i] = w2c[i] @ np.linalg.inv(Tc)
            sizes.append((np.linalg.norm(Tc[:3, 3]) * 1000, np.degrees(np.linalg.norm(_rotvec(Tc[:3, :3])))))
        sizes = np.array(sizes)
        history.append(sizes)
        log(f"  ICP round {r + 1}: corrections median {np.median(sizes[:, 0]):.2f}mm / {np.median(sizes[:, 1]):.3f}deg, "
            f"max {sizes[:, 0].max():.2f}mm / {sizes[:, 1].max():.3f}deg (max corr dist {dist * 1000:.0f}mm)")
    for v, T in zip(views, w2c):
        v.rvec = cv2.Rodrigues(T[:3, :3])[0]
        v.tvec = T[:3, 3].copy()
    return history

"""CPU ray-cast depth renderer (Open3D) for developing and unit-testing the
depth path without the GPU: exact Z-depth of a set of meshes from an
OpenCV-convention camera. Not a substitute for the Isaac renders (no
colour, so no markers, segmentation or pose solving) -- those stay the
end-to-end test.
"""
from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
import trimesh


class DepthRaycaster:
    def __init__(self, meshes: list[trimesh.Trimesh]):
        self.scene = o3d.t.geometry.RaycastingScene()
        for m in meshes:
            tm = o3d.t.geometry.TriangleMesh()
            tm.vertex.positions = o3d.core.Tensor(np.asarray(m.vertices, np.float32))
            tm.triangle.indices = o3d.core.Tensor(np.asarray(m.faces, np.int32))
            self.scene.add_triangles(tm)

    def render_shaded(self, K: np.ndarray, rvec, tvec, size_wh: tuple[int, int], colors: np.ndarray,
                      supersample: int = 2) -> np.ndarray:
        """Lambert-shaded RGB (uint8), anti-aliased by supersampling: crisp,
        sub-pixel-correct object edges for testing silhouette refinement.
        colors: (n_meshes, 3) in [0, 1], in the order the meshes were added.
        Light: a headlight plus a fixed overhead key light and ambient."""
        w, h = size_wh
        s = supersample
        Ks = K.copy().astype(float)
        Ks[:2] *= s
        Ks[0, 2] = (K[0, 2] + 0.5) * s - 0.5
        Ks[1, 2] = (K[1, 2] + 0.5) * s - 0.5
        R, _ = cv2.Rodrigues(np.asarray(rvec, float))
        C = -R.T @ np.asarray(tvec, float).reshape(3)
        u, v = np.meshgrid(np.arange(w * s, dtype=np.float64), np.arange(h * s, dtype=np.float64))
        dirs = np.stack([(u - Ks[0, 2]) / Ks[0, 0], (v - Ks[1, 2]) / Ks[1, 1], np.ones_like(u)], axis=-1).reshape(-1, 3) @ R
        rays = np.concatenate([np.broadcast_to(C, dirs.shape), dirs], axis=1).astype(np.float32)
        r = self.scene.cast_rays(o3d.core.Tensor(rays))
        hit = np.isfinite(r["t_hit"].numpy())
        n = r["primitive_normals"].numpy()
        gid = np.minimum(r["geometry_ids"].numpy(), len(colors) - 1)
        d = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        key = np.array([0.3, -0.4, 0.87])
        key /= np.linalg.norm(key)
        shade = 0.25 + 0.45 * np.abs(np.sum(n * d, axis=1)) + 0.35 * np.clip(n @ key, 0, 1)
        rgb = np.where(hit[:, None], colors[gid] * shade[:, None], 0.55)
        img = (np.clip(rgb, 0, 1).reshape(h * s, w * s, 3) * 255).astype(np.float32)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).round().astype(np.uint8)

    def render(self, K: np.ndarray, rvec, tvec, size_wh: tuple[int, int]) -> np.ndarray:
        """Z-depth (m), 0 where a ray hits nothing."""
        w, h = size_wh
        R, _ = cv2.Rodrigues(np.asarray(rvec, float))
        t = np.asarray(tvec, float).reshape(3)
        C = -R.T @ t
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        dirs_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
        dirs = dirs_cam.reshape(-1, 3) @ R                       # camera -> world (R^T as row-vector @ R)
        rays = np.concatenate([np.broadcast_to(C, dirs.shape), dirs], axis=1).astype(np.float32)
        t_hit = self.scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(h, w)
        # Rays are unnormalized with unit Z in the camera frame, so t_hit IS z.
        return np.where(np.isfinite(t_hit), t_hit, 0.0).astype(np.float32)


def bench_meshes(platform_w: float, platform_d: float, floor_z: float = -0.9) -> list[trimesh.Trimesh]:
    """The bench top (a slab whose top face is z = 0) and a floor, so every
    ray has a background like in a real room."""
    slab = trimesh.creation.box(extents=(platform_w, platform_d, 0.04))
    slab.apply_translation((0, 0, -0.02))
    floor = trimesh.creation.box(extents=(20, 20, 0.01))
    floor.apply_translation((0, 0, floor_z - 0.005))
    return [slab, floor]

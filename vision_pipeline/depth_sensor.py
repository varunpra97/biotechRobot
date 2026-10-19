"""Depth maps: loading, and a configurable model of a real depth sensor.

Depth throughout the pipeline is Z along the optical axis (OpenCV
convention: distance to the image plane, not along the ray), in meters,
registered to the colour image's camera (same pose; its own K when it is
lower resolution), 0 = no return.

The renderer produces PERFECT depth. A real sensor differs in ways that
matter for reconstruction, and each is modelled explicitly so a run can
say which ones it survived:

  resolution       depth sensors are far below a 12MP colour camera
  random noise     sigma(z) = sigma_at_1m * z^exponent, floored
                   (exponent 2: triangulation -- stereo, structured light;
                    exponent 1: time of flight, roughly)
  mixed pixels     a ToF pixel straddling a depth edge returns a blend of
                   foreground and background -- "flying pixels" floating
                   in free space between the two
  edge fattening   stereo matching bleeds the foreground over the
                   background by about a matching window
  grazing dropout  no return from surfaces seen nearly edge-on
  random dropout   holes (dark, shiny, or absorbent spots)
  scale error      a per-frame multiplicative bias (residual calibration)

PRESETS are representative, not any one product's datasheet. If the real
sensor's spec is known, add a preset with its numbers.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthSensorModel:
    name: str
    resolution: tuple[int, int] | None = None   # (w, h) of the depth map; None = the colour resolution
    sigma_at_1m_m: float = 0.0
    sigma_exponent: float = 2.0
    sigma_floor_m: float = 0.0
    mixed_pixels: bool = False
    edge_fattening_px: float = 0.0               # in depth-map pixels
    grazing_dropout_deg: float = 90.0            # no return beyond this angle between ray and surface normal
    dropout_rate: float = 0.0
    scale_error_sigma: float = 0.0
    min_range_m: float = 0.0
    max_range_m: float = 6.5

    def sigma(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(self.sigma_floor_m, self.sigma_at_1m_m * np.abs(z) ** self.sigma_exponent)


PRESETS = {
    "perfect": DepthSensorModel("perfect"),
    # Metrology-class handheld structured-light / laser-line scanner.
    "structured_light": DepthSensorModel(
        "structured_light", resolution=(2016, 1512), sigma_at_1m_m=0.15e-3, sigma_exponent=2.0,
        sigma_floor_m=0.03e-3, grazing_dropout_deg=78.0, dropout_rate=0.005, scale_error_sigma=2e-4,
        min_range_m=0.3, max_range_m=3.0),
    # Indoor time-of-flight camera / LiDAR depth module ("a laser that
    # measures distance per pixel"), VGA class.
    "tof": DepthSensorModel(
        "tof", resolution=(640, 480), sigma_at_1m_m=1.5e-3, sigma_exponent=1.0, sigma_floor_m=0.5e-3,
        mixed_pixels=True, grazing_dropout_deg=80.0, dropout_rate=0.01, scale_error_sigma=1e-3,
        min_range_m=0.25, max_range_m=5.0),
    # Active stereo (IR projector + two cameras, ~50mm baseline), 1280 wide.
    "active_stereo": DepthSensorModel(
        "active_stereo", resolution=(1280, 960), sigma_at_1m_m=2.4e-3, sigma_exponent=2.0,
        sigma_floor_m=0.2e-3, edge_fattening_px=1.5, grazing_dropout_deg=75.0, dropout_rate=0.02,
        scale_error_sigma=1e-3, min_range_m=0.2, max_range_m=4.0),
    # A phone's rear LiDAR: very low resolution, centimeter-ish noise.
    "phone_lidar": DepthSensorModel(
        "phone_lidar", resolution=(256, 192), sigma_at_1m_m=4e-3, sigma_exponent=1.0, sigma_floor_m=2e-3,
        mixed_pixels=True, grazing_dropout_deg=80.0, dropout_rate=0.01, scale_error_sigma=3e-3,
        min_range_m=0.2, max_range_m=5.0),
}


def load_sensor(name_or_path: str) -> DepthSensorModel:
    """A preset name, or a JSON file with DepthSensorModel's fields."""
    if name_or_path in PRESETS:
        return PRESETS[name_or_path]
    d = json.loads(Path(name_or_path).read_text())
    if d.get("resolution") is not None:
        d["resolution"] = tuple(d["resolution"])
    return DepthSensorModel(**d)


def sensor_to_dict(model: DepthSensorModel) -> dict:
    return asdict(model)


# ---------------------------------------------------------------- I/O

def load_depth(path: str | Path, scale_m: float = 1e-4) -> np.ndarray:
    """Depth in meters (float32, 0 = invalid) from a uint16 PNG (x scale_m),
    a float .npy (meters), or an .npz with a 'depth' array (meters)."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        d = np.load(path).astype(np.float32)
    elif path.suffix.lower() == ".npz":
        d = np.load(path)["depth"].astype(np.float32)
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
        if raw is None:
            raise FileNotFoundError(path)
        d = raw.astype(np.float32) * np.float32(scale_m)
    d[~np.isfinite(d) | (d < 0)] = 0.0
    return d


def save_depth(path: str | Path, depth_m: np.ndarray, scale_m: float = 1e-4) -> None:
    q = np.where(np.isfinite(depth_m) & (depth_m > 0), np.round(depth_m / scale_m), 0)
    q[q > 65535] = 0
    cv2.imwrite(str(path), q.astype(np.uint16))


def scale_intrinsics(K: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    """K for the same camera resampled from src to dst resolution (pixel
    centers at integer coordinates, so cx scales as (cx + 0.5) s - 0.5)."""
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    K2 = np.asarray(K, float).copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] = (K2[0, 2] + 0.5) * sx - 0.5
    K2[1, 2] = (K2[1, 2] + 0.5) * sy - 0.5
    return K2


# ---------------------------------------------------------------- simulation

def _normals_from_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Per-pixel unit normals (camera frame) from central differences of the
    back-projected points; zeros where a neighbor is invalid."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    P = np.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], axis=-1)
    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1] = P[2:] - P[:-2]
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    valid = depth > 0
    ok = valid.copy()
    ok[:, 1:-1] &= valid[:, 2:] & valid[:, :-2]
    ok[1:-1] &= valid[2:] & valid[:-2]
    ok[[0, -1]] = False
    ok[:, [0, -1]] = False
    return np.where(ok[..., None] & (norm > 0), n / np.maximum(norm, 1e-12), 0.0)


def _resample(depth: np.ndarray, dst_wh: tuple[int, int], mixed: bool) -> np.ndarray:
    """Perfect full-resolution depth -> sensor resolution. ToF (mixed=True):
    each sensor pixel integrates its whole footprint, so a footprint
    straddling an edge returns a blend (a flying pixel). Otherwise: the
    depth at the sensor pixel's center."""
    h, w = depth.shape
    if (w, h) == tuple(dst_wh):
        return depth.copy()
    if mixed:
        valid = (depth > 0).astype(np.float32)
        s = cv2.resize(depth * valid, dst_wh, interpolation=cv2.INTER_AREA)
        c = cv2.resize(valid, dst_wh, interpolation=cv2.INTER_AREA)
        return np.where(c > 0.5, s / np.maximum(c, 1e-6), 0.0).astype(np.float32)
    xs = np.clip(np.round((np.arange(dst_wh[0]) + 0.5) * w / dst_wh[0] - 0.5).astype(int), 0, w - 1)
    ys = np.clip(np.round((np.arange(dst_wh[1]) + 0.5) * h / dst_wh[1] - 0.5).astype(int), 0, h - 1)
    return depth[np.ix_(ys, xs)].astype(np.float32)


def simulate(depth_perfect: np.ndarray, K: np.ndarray, model: DepthSensorModel,
             seed: int | str = 0) -> tuple[np.ndarray, np.ndarray]:
    """(sensor depth map, its K) for a perfect depth map rendered with K."""
    if isinstance(seed, str):
        seed = zlib.crc32(seed.encode())
    rng = np.random.default_rng(seed)
    h, w = depth_perfect.shape
    d = np.where(np.isfinite(depth_perfect), depth_perfect, 0.0).astype(np.float32)

    if model.grazing_dropout_deg < 90.0:
        n = _normals_from_depth(d, K)
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1)
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        cos = np.abs(np.sum(n * ray, axis=-1))
        has_n = np.any(n != 0, axis=-1)
        d[has_n & (cos < np.cos(np.radians(model.grazing_dropout_deg)))] = 0.0

    dst = model.resolution or (w, h)
    d = _resample(d, dst, model.mixed_pixels)
    K_out = scale_intrinsics(K, (w, h), dst)

    if model.edge_fattening_px > 0:
        r = int(np.ceil(model.edge_fattening_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        big = np.where(d > 0, d, np.float32(1e6))
        nearest = cv2.erode(big, kernel)              # min filter: the foreground spreads
        jump = (d > 0) & (d - nearest > 4 * model.sigma(d) + 0.005)
        d = np.where(jump, nearest, d)

    valid = (d > 0) & (d >= model.min_range_m) & (d <= model.max_range_m)
    if model.sigma_at_1m_m > 0 or model.sigma_floor_m > 0:
        d = d + rng.standard_normal(d.shape).astype(np.float32) * model.sigma(d).astype(np.float32)
    if model.scale_error_sigma > 0:
        d = d * np.float32(1.0 + rng.normal(0.0, model.scale_error_sigma))
    if model.dropout_rate > 0:
        valid &= rng.random(d.shape) >= model.dropout_rate
    return np.where(valid, d, 0.0).astype(np.float32), K_out


def remove_flying_pixels(depth: np.ndarray, K: np.ndarray, max_slope_deg: float = 80.0,
                         sigma_fn=None) -> np.ndarray:
    """Drop pixels with a depth jump to any 8-neighbor larger than a surface
    tilted at max_slope_deg could produce (plus 4 sigma of noise): mixed /
    flying pixels and both sides of every depth edge. What every depth
    SDK's edge filter does; costs one pixel along each silhouette, which
    the other views and the colour silhouettes cover."""
    d = depth
    valid = d > 0
    footprint = d / K[0, 0]                                   # meters per pixel at that depth
    thr = footprint * np.tan(np.radians(max_slope_deg)) * 1.5
    if sigma_fn is not None:
        thr = thr + 4.0 * sigma_fn(d)
    bad = np.zeros_like(valid)
    p = np.pad(d, 1, mode="edge")
    pv = np.pad(valid, 1, mode="constant", constant_values=False)
    h, w = d.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nb = p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            nbv = pv[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            bad |= nbv & (np.abs(d - nb) > thr)
    return np.where(valid & ~bad, d, 0.0).astype(np.float32)


def with_overrides(model: DepthSensorModel, **kw) -> DepthSensorModel:
    return replace(model, **kw)

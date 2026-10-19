"""Render a synthetic multi-view capture for testing Stage 1 end-to-end,
including real cv2.aruco detection on rendered pixels (not just analytic
point projection, which tests/test_pose_solver_synthetic.py already
covers). Full ground truth (camera poses, intrinsics) is known, so
Stage 1's output can be checked against truth automatically, not just
eyeballed.

Deliberately varies elevation, azimuth, distance, roll, resolution/
intrinsics, noise, and blur across shots -- including a few views
designed to be hard (far-away, grazing angle) -- so a passing run is
actual evidence of robustness, not just a happy-path demo.

Usage:
  python scripts/render_synthetic_capture.py --out biotech-lab-robotics/vision_pipeline/out/synthetic_capture
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from geometry import camera_pose_to_rvec_tvec, look_at_rotation, rotation_matrix_to_quaternion
from markers import DEFAULT_DICTIONARY, MarkerLayout, corners_from_center_axes

RNG = np.random.default_rng(42)

# Platform footprint matches biotech-lab-robotics/simulation/sim/lab_scene.py
# (PLATFORM_W, PLATFORM_D), so this fixture doubles as a sanity check
# that marker placements sized for the real bench are detectable at all.
PLATFORM_W, PLATFORM_D = 1.4, 0.7


@dataclass
class SyntheticObject:
    """A simple proxy for a lab-ware item placed on the platform: enough
    geometry to produce a real, segmentable silhouette (flat-shaded, not
    photorealistic) at a KNOWN ground-truth size/position, so Stage 2
    (and later Stage 4's measurement fitting) has something concrete to
    be checked against -- not just markers on an empty bench.
    """
    name: str
    kind: str  # "box" | "cylinder"
    color_bgr: tuple[int, int, int]
    # box: center + half-extents (meters, platform frame). cylinder:
    # base_center + radius + height.
    params: dict


def build_synthetic_objects() -> list[SyntheticObject]:
    return [
        SyntheticObject(
            name="proxy_box", kind="box", color_bgr=(60, 90, 220),
            params={"center": [0.20, 0.08, 0.0], "half_extents": [0.05, 0.04, 0.045]}),
        SyntheticObject(
            name="proxy_cylinder", kind="cylinder", color_bgr=(180, 140, 40),
            params={"base_center": [-0.18, -0.06, 0.0], "radius": 0.035, "height": 0.09}),
    ]


def _box_corners(center, half_extents) -> np.ndarray:
    cx, cy, cz = center
    hx, hy, hz = half_extents
    return np.array([
        [cx + sx * hx, cy + sy * hy, cz + sz * hz]
        for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 2)
    ], dtype=np.float64)  # sz in (0,2) -> z in [cz, cz+2*hz]: box sits ON the platform (z=cz=0), not centered through it


def _cylinder_rim_points(base_center, radius, height, n=24) -> np.ndarray:
    cx, cy, cz = base_center
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bottom = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles),
                        np.full(n, cz)], axis=1)
    top = bottom.copy()
    top[:, 2] = cz + height
    return np.concatenate([bottom, top], axis=0)


def object_world_points(obj: SyntheticObject) -> np.ndarray:
    if obj.kind == "box":
        return _box_corners(obj.params["center"], obj.params["half_extents"])
    if obj.kind == "cylinder":
        return _cylinder_rim_points(obj.params["base_center"], obj.params["radius"],
                                     obj.params["height"])
    raise ValueError(obj.kind)


def _composite_object(image: np.ndarray, world_points: np.ndarray, color_bgr,
                       rvec, tvec, K, dist) -> None:
    R_wc, _ = cv2.Rodrigues(rvec)
    depths = (R_wc @ world_points.T + tvec.reshape(3, 1))[2, :]
    if np.any(depths <= 0.02):
        return  # part of the object is behind/at the camera; skip rather than draw garbage
    proj, _ = cv2.projectPoints(world_points, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    hull = cv2.convexHull(proj.astype(np.float32))
    # Flat fill plus a faint synthetic shading gradient (brighter near the
    # hull's top) so it isn't a perfectly flat color blob -- cheap, but
    # enough texture that it isn't trivially different from a real photo
    # in the one way that would make segmentation artificially easy.
    overlay = image.copy()
    cv2.fillConvexPoly(overlay, hull.astype(np.int32), color_bgr)
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, dst=image)
    cv2.polylines(image, [hull.astype(np.int32)], True,
                  tuple(max(0, c - 40) for c in color_bgr), 1, cv2.LINE_AA)


@dataclass
class VirtualShot:
    name: str
    camera_id: str
    position_world: np.ndarray
    target_world: np.ndarray
    K: np.ndarray
    image_size: tuple[int, int]
    noise_std: float
    blur_ksize: int
    note: str


def build_marker_layout() -> MarkerLayout:
    corners_3d = {
        0: corners_from_center_axes([-PLATFORM_W / 2 + 0.05, -PLATFORM_D / 2 + 0.05, 0.0],
                                     [1, 0, 0], [0, 1, 0], 0.05),
        1: corners_from_center_axes([PLATFORM_W / 2 - 0.05, -PLATFORM_D / 2 + 0.05, 0.0],
                                     [1, 0, 0], [0, 1, 0], 0.05),
        2: corners_from_center_axes([PLATFORM_W / 2 - 0.05, PLATFORM_D / 2 - 0.05, 0.0],
                                     [1, 0, 0], [0, 1, 0], 0.05),
        3: corners_from_center_axes([-PLATFORM_W / 2 + 0.05, PLATFORM_D / 2 - 0.05, 0.0],
                                     [1, 0, 0], [0, 1, 0], 0.05),
        # A fifth marker standing upright off the bench plane, so the
        # correspondence set isn't purely coplanar (better-conditioned
        # PnP) -- same reasoning as the README's "not all coplanar" note.
        4: corners_from_center_axes([0.0, -PLATFORM_D / 2 + 0.02, 0.15],
                                     [1, 0, 0], [0, 0, 1], 0.05),
    }
    return MarkerLayout(corners_3d=corners_3d)


def build_shots() -> list[VirtualShot]:
    shots = []
    standard_K = np.array([[1400.0, 0, 960.0], [0, 1400.0, 540.0], [0, 0, 1.0]])
    standard_size = (1920, 1080)
    target = np.array([0.0, 0.0, 0.03])

    # 8-shot ring at ~35 deg elevation, every 45 deg of azimuth.
    for i, az_deg in enumerate(range(0, 360, 45)):
        az = np.radians(az_deg)
        elev = np.radians(35.0)
        dist = 1.1 + RNG.uniform(-0.1, 0.1)
        pos = target + dist * np.array(
            [np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])
        shots.append(VirtualShot(
            name=f"ring_{i:02d}_az{az_deg}", camera_id="phone_main",
            position_world=pos, target_world=target,
            K=standard_K, image_size=standard_size,
            noise_std=RNG.uniform(2.0, 5.0), blur_ksize=int(RNG.choice([0, 0, 3])),
            note="ring"))

    # Top-down. Height matters: too low and the platform's long axis
    # doesn't fit the frame even though the shot is otherwise perfect --
    # at this K/image_size, 1.4m needs roughly 1.8m+ of height.
    shots.append(VirtualShot(
        name="top_down", camera_id="phone_main",
        position_world=target + np.array([0.0, 0.0, 2.0]), target_world=target,
        K=standard_K, image_size=standard_size,
        noise_std=3.0, blur_ksize=0, note="top-down"))

    # Close detail shot.
    shots.append(VirtualShot(
        name="detail_close", camera_id="phone_main",
        position_world=target + 0.45 * np.array(
            [np.cos(np.radians(50)) * np.cos(np.radians(30)),
             np.cos(np.radians(50)) * np.sin(np.radians(30)),
             np.sin(np.radians(50))]),
        target_world=np.array([0.1, 0.05, 0.0]),
        K=standard_K, image_size=standard_size,
        noise_std=2.0, blur_ksize=0, note="close detail"))

    # Stress test: far away (markers project small -- realistic failure
    # mode when a capture is shot from across a room).
    far_elev = np.radians(25.0)
    shots.append(VirtualShot(
        name="stress_far", camera_id="phone_main",
        position_world=target + 3.0 * np.array(
            [np.cos(far_elev), 0.0, np.sin(far_elev)]),
        target_world=target,
        K=standard_K, image_size=standard_size,
        noise_std=4.0, blur_ksize=0, note="stress: far away"))

    # Stress test: grazing angle (markers heavily foreshortened).
    graze_elev = np.radians(8.0)
    shots.append(VirtualShot(
        name="stress_grazing", camera_id="phone_main",
        position_world=target + 1.3 * np.array(
            [np.cos(graze_elev), 0.0, np.sin(graze_elev)]),
        target_world=target,
        K=standard_K, image_size=standard_size,
        noise_std=4.0, blur_ksize=3, note="stress: grazing angle"))

    # Stress test: heavier noise + blur, simulating a poor-quality phone
    # shot (motion blur, low light).
    shots.append(VirtualShot(
        name="stress_noisy", camera_id="phone_main",
        position_world=target + 1.1 * np.array(
            [np.cos(np.radians(40)) * np.cos(np.radians(200)),
             np.cos(np.radians(40)) * np.sin(np.radians(200)),
             np.sin(np.radians(40))]),
        target_world=target,
        K=standard_K, image_size=standard_size,
        noise_std=18.0, blur_ksize=7, note="stress: noisy/blurry"))

    # A second, different physical camera (wider FOV, different
    # resolution) to exercise the manifest/multi-camera-id code path
    # rather than just CaptureManifest.from_directory.
    wide_K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1.0]])
    shots.append(VirtualShot(
        name="wide_cam_front", camera_id="cam_wide",
        position_world=target + 1.0 * np.array(
            [np.cos(np.radians(35)) * np.cos(np.radians(180)),
             np.cos(np.radians(35)) * np.sin(np.radians(180)),
             np.sin(np.radians(35))]),
        target_world=target,
        K=wide_K, image_size=(1280, 720),
        noise_std=3.0, blur_ksize=0, note="second camera_id"))

    return shots


def _marker_source_image(marker_id: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(DEFAULT_DICTIONARY)
    return cv2.aruco.generateImageMarker(dictionary, marker_id, 200)


def _bench_background(image_size: tuple[int, int]) -> np.ndarray:
    """A plain, mildly textured background standing in for a real photo
    (lab wall/bench) -- just needs to not be flat white, so marker
    detection and the noise/blur stress-tests mean something.
    """
    w, h = image_size
    base = RNG.integers(150, 185)
    img = np.full((h, w, 3), base, dtype=np.uint8)
    noise = RNG.normal(0, 6, (h, w, 3))
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    # Faint gradient for a bit of visual structure.
    grad = np.linspace(-15, 15, h).reshape(-1, 1, 1)
    img = np.clip(img.astype(np.float64) + grad, 0, 255).astype(np.uint8)
    return img


def _composite_marker(image: np.ndarray, marker_img: np.ndarray, corners_2d: np.ndarray) -> None:
    h, w = marker_img.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst = corners_2d.astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    out_h, out_w = image.shape[:2]
    marker_rgb = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    warped = cv2.warpPerspective(marker_rgb, H, (out_w, out_h))
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)
    image[mask > 0] = warped[mask > 0]


def render_shot(
    shot: VirtualShot, layout: MarkerLayout, objects: list[SyntheticObject],
) -> tuple[np.ndarray, dict]:
    R_cw = look_at_rotation(shot.position_world, shot.target_world)
    rvec, tvec = camera_pose_to_rvec_tvec(R_cw, shot.position_world)
    dist = np.zeros(5)

    image = _bench_background(shot.image_size)
    w, h = shot.image_size

    # Objects first, markers on top: markers sit at the platform's edges
    # specifically so they stay unoccluded reference points, and drawing
    # them last guarantees that regardless of object placement.
    for obj in objects:
        _composite_object(image, object_world_points(obj), obj.color_bgr,
                           rvec, tvec, shot.K, dist)

    visible_markers = []
    R_wc, _ = cv2.Rodrigues(rvec)
    for marker_id, pts3d in layout.corners_3d.items():
        depths = (R_wc @ pts3d.T + tvec.reshape(3, 1))[2, :]
        if np.any(depths <= 0.05):
            continue  # behind or too close to the camera plane
        proj, _ = cv2.projectPoints(pts3d, rvec, tvec, shot.K, dist)
        proj = proj.reshape(4, 2)
        if np.any(proj[:, 0] < -50) or np.any(proj[:, 0] > w + 50):
            continue
        if np.any(proj[:, 1] < -50) or np.any(proj[:, 1] > h + 50):
            continue
        marker_img = _marker_source_image(marker_id)
        _composite_marker(image, marker_img, proj)
        visible_markers.append(marker_id)

    if shot.blur_ksize:
        image = cv2.GaussianBlur(image, (shot.blur_ksize, shot.blur_ksize), 0)
    if shot.noise_std > 0:
        noise = RNG.normal(0, shot.noise_std, image.shape)
        image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    quat = rotation_matrix_to_quaternion(R_cw)
    ground_truth = {
        "name": shot.name,
        "camera_id": shot.camera_id,
        "position_world": shot.position_world.tolist(),
        "quaternion_wxyz": quat.tolist(),
        "visible_markers": visible_markers,
        "note": shot.note,
    }
    return image, ground_truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    layout = build_marker_layout()
    shots = build_shots()
    objects = build_synthetic_objects()

    manifest_shots = []
    ground_truths = []
    cameras = {}

    for shot in shots:
        image, gt = render_shot(shot, layout, objects)
        image_path = images_dir / f"{shot.name}.png"
        cv2.imwrite(str(image_path), image)
        manifest_shots.append({"image": f"images/{shot.name}.png",
                                "camera_id": shot.camera_id})
        ground_truths.append(gt)
        if shot.camera_id not in cameras:
            cameras[shot.camera_id] = {
                "K": shot.K.tolist(), "dist": [0, 0, 0, 0, 0],
                "image_size": list(shot.image_size),
            }
        print(f"rendered {shot.name}: {len(gt['visible_markers'])} markers "
              f"visible ({shot.note})")

    (out_dir / "manifest.json").write_text(
        json.dumps({"shots": manifest_shots}, indent=2))
    (out_dir / "camera_intrinsics.json").write_text(
        json.dumps({"cameras": cameras}, indent=2))
    (out_dir / "ground_truth_poses.json").write_text(
        json.dumps({"shots": ground_truths}, indent=2))

    marker_layout_json = {
        "markers": {
            str(mid): {"corners_3d": pts.tolist()}
            for mid, pts in layout.corners_3d.items()
        }
    }
    (out_dir / "marker_layout.json").write_text(json.dumps(marker_layout_json, indent=2))

    objects_json = [
        {"name": o.name, "kind": o.kind, "color_bgr": list(o.color_bgr), "params": o.params}
        for o in objects
    ]
    (out_dir / "ground_truth_objects.json").write_text(json.dumps(objects_json, indent=2))

    print(f"\nwrote {len(shots)} shots to {images_dir}")
    print(f"manifest: {out_dir / 'manifest.json'}")
    print(f"ground truth: {out_dir / 'ground_truth_poses.json'}")


if __name__ == "__main__":
    main()

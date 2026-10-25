"""Real-time 6-DoF tracking of every object in the scene map from the
robot's cameras (wrist RGB, plus any fixed camera), for the planner.

Per frame, per camera:
  1. raycast the scene at the current estimates -- every object's mesh,
     plus the robot's own hand / fingers / camera bracket (robot_body.py)
     at the measured hand pose -- giving, per pixel, which object is in
     front and how far. Open3D's Embree raycaster, CPU, every 3rd pixel,
     one prebuilt scene per mesh (rays moved into its frame).
  2. occluding contours and creases: pixels of object i whose neighbour is
     farther away (another object, the background, or a step within i,
     such as a bore's rim), refined sub-pixel by casting extra rays along
     the normal until they leave i's near surface. A contour next to the
     robot body, with the robot in front, is dropped -- that is how a tube
     hidden behind the gripper stops being "seen" on the gripper's edges.
     Each sample keeps its 3D point on object i.
  3. search each sample's image normal for the real edge (sub-pixel,
     orientation-checked, as template_fitting._normal_search, vectorized)
     in the full-resolution gradient image, minus the camera's outward
     edge bias (estimated online).
Per object: Gauss-Newton (re-searched 3x with a shrinking window sized
from the current uncertainty) on a 6-DoF twist about the object's centre,
Tukey weights, information capped per view (samples along a contour are
correlated). Every frame is solved against the object's ANCHOR -- its map
pose; or, seated in a holder / rack, its pose relative to that parent
(clamped to the opening's clearance); or, while held, its pose relative to
the hand -- and the frames are fused with a forgetting factor (see
ObjectFilter). Views coarser than max_mm_per_px are not used to refine.

Output (SceneState): per object its world pose, 1-sigma position
uncertainty, how many edge samples supported it this frame, and its
named targets in world coordinates.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
import open3d as o3d

from robot_body import arm_parts, body_parts
from scene_model import SceneObject

ROBOT = -2
NONE = -1


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _rotvec(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + _skew(w)
    k = _skew(w / th)
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * k @ k


@dataclass
class Frame:
    name: str
    gray: np.ndarray
    K: np.ndarray
    T_world_cam: np.ndarray
    T_world_hand: np.ndarray = None       # hand pose at this image's timestamp (robot body rendering)
    q: np.ndarray = None                  # arm joints at this image's timestamp
    hand_mounted: bool = False            # a wrist camera: only the hand, fingers and bracket can be in view


@dataclass
class ObjectState:
    name: str
    cls: str
    T: np.ndarray
    sigma_mm: float                       # 1-sigma position uncertainty (largest axis)
    support: int                          # edge samples used this frame
    updated: bool
    targets: dict = field(default_factory=dict)


# initial 1-sigma (xy m, z m, roll/pitch deg, yaw deg) by class: what the offline map leaves uncertain
PRIOR_SIGMA = {
    "default": (0.003, 0.0007, 0.3, 1.5),
    "tube": (0.003, 0.0007, 1.5, 1.0),     # seated in an 8 mm bore: tilt < ~2 deg, tip on the bore floor
    "tip": (0.003, 0.0007, 1.0, 1.0),
    "bench": (0.002, 0.0005, 0.1, 0.2),
    "seated": (0.0005, 0.0003, 1.0, 5.0),  # relative to its parent: bore / collar clearance
}
# what a seated object physically can do in its opening (m, m, deg tilt): a 7 mm tube in an 8 mm,
# 28 mm deep bore can shift 0.5 mm and lean ~1 deg; edges claiming more (glass, liquid) are wrong
SEATED_LIMIT = (0.0005, 0.0003, 1.0)
# rendered (they occlude) but not refined: the tip rack's 9 mm grid of holes and tips is ambiguous
# at a few px per pitch and pulled it up to 13 mm off; nothing needs the bench's coordinates
NO_REFINE = {"tip_rack", "bench"}
TCP_Z = 0.1034                          # Franka hand frame -> fingertip centre
MAX_GRASP_DIST = 0.03                   # a "held" object must be this close to the fingertip centre


class Renderer:
    """Raycasts objects at given poses, plus robot parts, into a camera.

    Each mesh gets its own raycasting scene, built once in its own frame;
    per frame only the rays inside its projected bounding box are cast,
    after moving them into the mesh's frame -- no per-frame rebuild of a
    quarter-million-triangle scene. A z-buffer combines the results."""

    def __init__(self, objects: list[SceneObject], near=0.012):
        self.near = near                      # the sim cameras' near clip is 0.01 m
        self.obj_scenes = [self._scene(o.mesh) for o in objects]
        self.obj_boxes = [o.mesh.bounds for o in objects]
        self._robot_cache = {}

    @staticmethod
    def _scene(mesh):
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                         o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
        return sc

    def _robot_scene(self, name, mesh):
        key = (name, len(mesh.faces))
        if key not in self._robot_cache:
            self._robot_cache[key] = (self._scene(mesh), mesh.bounds)
        return self._robot_cache[key]

    def cast_object(self, i, K, T_world_cam, T_obj, uv):
        """Depth (camera z; inf on a miss) of object i alone along the rays through pixels uv (N, 2)."""
        dc = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(uv))], 1)
        d = dc @ T_world_cam[:3, :3].T
        Rl = T_obj[:3, :3].T
        o_l = (T_world_cam[:3, 3] + self.near * d - T_obj[:3, 3]) @ Rl.T
        ans = self.obj_scenes[i].cast_rays(o3d.core.Tensor(np.concatenate([o_l, d @ Rl.T], 1).astype(np.float32)))
        return ans["t_hit"].numpy().astype(float) + self.near

    def render(self, K, T_world_cam, size, poses, robot_parts, stride=2, z_max=np.inf):
        """robot_parts: [(name, mesh, T_world_part)]. Objects entirely beyond z_max
        are left out (too far to be refined, and they can't hide anything nearer).
        Returns per grid pixel:
        object index (ROBOT for the robot, NONE for nothing), depth (camera z),
        world point, and the grid's pixel coordinates."""
        w, h = size
        us, vs = np.meshgrid(np.arange(0, w, stride, dtype=np.float32), np.arange(0, h, stride, dtype=np.float32))
        dc = np.stack([(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(us)], -1)
        Rwc, C = T_world_cam[:3, :3], T_world_cam[:3, 3]
        Tcw = np.linalg.inv(T_world_cam)
        dw = dc @ Rwc.T
        ids = np.full(us.shape, NONE, np.int32)
        depth = np.full(us.shape, np.inf, np.float32)
        gh, gw = us.shape
        items = [(i, sc, bb, T) for i, (sc, bb, T) in enumerate(zip(self.obj_scenes, self.obj_boxes, poses))]
        items += [(ROBOT, *self._robot_scene(n, m), T) for n, m, T in robot_parts]
        for i, sc, bb, T in items:
            corners = np.array([[x, y, z] for x in bb[:, 0] for y in bb[:, 1] for z in bb[:, 2]])
            Xc = (corners @ T[:3, :3].T + T[:3, 3]) @ Tcw[:3, :3].T + Tcw[:3, 3]
            if (Xc[:, 2] < self.near).all() or (i != ROBOT and Xc[:, 2].min() > z_max):
                continue
            if (Xc[:, 2] < self.near).any():
                x0, y0, x1, y1 = 0, 0, gw, gh
            else:
                u = K[0, 0] * Xc[:, 0] / Xc[:, 2] + K[0, 2]
                v = K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2]
                x0, x1 = int(max(0, np.floor(u.min() / stride))), int(min(gw, np.ceil(u.max() / stride) + 1))
                y0, y1 = int(max(0, np.floor(v.min() / stride))), int(min(gh, np.ceil(v.max() / stride) + 1))
                if x1 <= x0 or y1 <= y0:
                    continue
            d = dw[y0:y1, x0:x1].reshape(-1, 3)
            Rl = T[:3, :3].T
            o_l = (C + self.near * d - T[:3, 3]) @ Rl.T
            d_l = d @ Rl.T
            ans = sc.cast_rays(o3d.core.Tensor(np.concatenate([o_l, d_l], 1).astype(np.float32)))
            t = ans["t_hit"].numpy().reshape(y1 - y0, x1 - x0) + self.near
            sub = depth[y0:y1, x0:x1]
            nearer = np.isfinite(t) & (t < sub)
            sub[nearer] = t[nearer]
            ids[y0:y1, x0:x1][nearer] = i
        hit = np.isfinite(depth)
        pts = C + dw * np.where(hit, depth, 0)[..., None]
        return ids, depth, pts, us, vs


def occluding_contours(ids, depth, step_m=0.004, slope_k=3.0):
    """Contour pixels of objects (where the neighbour is farther away: another
    object, nothing, or a step within the same object such as a bore rim),
    their outward normals (towards the farther side). A depth step counts
    only if it is much larger than the surface's own slope there (seen on
    the opposite side), so an oblique plane is not an edge. Contours next
    to the robot body are dropped: the robot is in front there."""
    H, W = ids.shape
    contour = np.zeros((H, W), bool)
    nrm = np.zeros((H, W, 2), np.float32)
    pid = np.pad(ids, 1, constant_values=NONE)
    pd = np.pad(depth, 1, constant_values=np.inf)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb_ids = pid[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]       # neighbour at (y + dy, x + dx)
        nb_d = pd[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        op_d = pd[1 - dy:1 - dy + H, 1 - dx:1 - dx + W]         # the opposite neighbour
        slope = np.where(np.isfinite(op_d), np.abs(depth - op_d), 0.0)
        step = nb_d - depth
        edge = (ids >= 0) & (nb_ids != ROBOT) & (step > step_m) & (step > slope_k * slope)
        contour |= edge
        nrm[edge] += (dx, dy)
    near_robot = cv2.dilate((ids == ROBOT).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    contour &= ~near_robot
    ld = np.log(np.minimum(depth, 10.0)).astype(np.float32)
    g = np.stack([cv2.Sobel(ld, cv2.CV_32F, 1, 0, ksize=5), cv2.Sobel(ld, cv2.CV_32F, 0, 1, ksize=5)], -1)
    ok = (np.sum(g * nrm, -1) > 0)                              # agrees with the step's side
    nrm = np.where(ok[..., None], g, nrm)
    n = np.linalg.norm(nrm, axis=-1, keepdims=True)
    return contour, nrm / np.maximum(n, 1e-9)


def _bilinear(img, x, y):
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    xi, yi = x.astype(np.int32), y.astype(np.int32)
    fx, fy = x - xi, y - yi
    return (img[yi, xi] * (1 - fx) * (1 - fy) + img[yi, xi + 1] * fx * (1 - fy)
            + img[yi + 1, xi] * (1 - fx) * fy + img[yi + 1, xi + 1] * fx * fy)


def edge_search(gx, gy, pts, nrm, radius_px, min_grad, step=0.5):
    """Sub-pixel offset along each normal to the nearest local maximum of
    |grad . n| above min_grad (edges crossing the normal don't count),
    refined by a parabola; NaN where none. Vectorized."""
    ts = np.arange(-radius_px, radius_px + step / 2, step)
    xs = pts[:, :1] + ts[None] * nrm[:, :1]
    ys = pts[:, 1:] + ts[None] * nrm[:, 1:]
    sc = np.abs(_bilinear(gx, xs, ys) * nrm[:, :1] + _bilinear(gy, xs, ys) * nrm[:, 1:])
    inside = (xs >= 0) & (xs < gx.shape[1] - 1) & (ys >= 0) & (ys < gx.shape[0] - 1)
    sc = np.where(inside, sc, 0.0)
    a, b, c = sc[:, :-2], sc[:, 1:-1], sc[:, 2:]
    peak = (b >= a) & (b > c) & (b >= min_grad)
    centre = len(ts) // 2
    dist = np.where(peak, np.abs(np.arange(1, len(ts) - 1)[None] - centre), 10 ** 6)
    j = dist.argmin(1)
    has = dist[np.arange(len(pts)), j] < 10 ** 6
    rows = np.arange(len(pts))
    A, B, C = a[rows, j], b[rows, j], c[rows, j]
    den = A - 2 * B + C
    delta = np.where(np.abs(den) > 1e-9, 0.5 * (A - C) / np.where(np.abs(den) > 1e-9, den, 1.0), 0.0)
    off = ts[j + 1] + np.clip(delta, -0.5, 0.5) * step
    return np.where(has, off, np.nan)


def _apply(delta, T, center_local):
    """A world-frame twist (dt, dw) about the object's centre applied to pose T."""
    c = T[:3, :3] @ center_local + T[:3, 3]
    Rd = _rotvec(delta[3:])
    out = np.eye(4)
    out[:3, :3] = Rd @ T[:3, :3]
    out[:3, 3] = Rd @ (T[:3, 3] - c) + c + delta[:3]
    return out


class ObjectFilter:
    """One object's pose: an anchor (the map pose, or -- while held -- its pose
    relative to the hand) with the anchor's uncertainty P0, and a fused
    correction to it. Every frame is solved against the anchor, never
    against the previous estimate, and the per-frame solutions are fused
    with a forgetting factor: a small pose-dependent bias in the edge
    measurements then offsets the estimate by a bounded amount instead of
    walking it away frame after frame."""

    def __init__(self, obj: SceneObject, forget=0.9, parent_T=None):
        self.obj = obj
        seated = parent_T is not None
        sxy, sz, srp, syaw = PRIOR_SIGMA["seated"] if seated else PRIOR_SIGMA.get(obj.cls, PRIOR_SIGMA["default"])
        r, y = math.radians(srp), math.radians(syaw)
        self.forget = forget
        P0 = np.diag([sxy ** 2, sxy ** 2, sz ** 2, r ** 2, r ** 2, y ** 2])
        if seated:
            self.set_anchor(np.linalg.inv(parent_T) @ obj.T, P0, "parent", parent_T)
        else:
            self.set_anchor(obj.T.copy(), P0, "world")
        self.support = 0
        self.updated = False

    def set_anchor(self, A, P0, ref, T_ref=np.eye(4)):
        """ref "world": A is a world pose; "hand": hand-relative; "parent": relative to the parent object."""
        self.A, self.P0, self.ref = A, P0, ref
        self.Lam = np.zeros((6, 6))
        self.eta = np.zeros(6)
        self.delta = np.zeros(6)                      # fused correction, in the reference frame
        self._T_ref = T_ref
        self.T = T_ref @ A
        self.P = P0.copy()

    @staticmethod
    def _B(T_ref):
        B = np.zeros((6, 6))
        B[:3, :3] = B[3:, 3:] = T_ref[:3, :3]
        return B

    def frame(self, T_ref=None):
        """(anchor world pose, prior P0 in world, B: ref-frame twist -> world twist) for this frame."""
        if self.ref != "world" and T_ref is not None:
            self._T_ref = T_ref
        Tr = self._T_ref if self.ref != "world" else np.eye(4)
        B = self._B(Tr)
        Aw = Tr @ self.A
        self.T = _apply(B @ self.delta, Aw, self.obj.center_local)
        return Aw, B @ self.P0 @ B.T, B

    def fuse(self, delta_w, H_w, P0w, B):
        """Add one frame's anchored solution (world twist, its measurement information)."""
        H_r = B.T @ H_w @ B
        eta_r = B.T @ ((np.linalg.inv(P0w) + H_w) @ delta_w)
        self.Lam = self.forget * self.Lam + H_r
        self.eta = self.forget * self.eta + eta_r
        Pinv = np.linalg.inv(self.P0) + self.Lam
        self.P = np.linalg.inv(Pinv)
        self.delta = self.P @ self.eta
        if self.ref == "parent":                     # seated: stay inside the opening's clearance
            lxy, lz, lt = SEATED_LIMIT
            d = self.delta
            d[:2] *= min(1.0, lxy / max(np.linalg.norm(d[:2]), 1e-12))
            d[2] = np.clip(d[2], -lz, lz)
            d[3:5] *= min(1.0, math.radians(lt) / max(np.linalg.norm(d[3:5]), 1e-12))
        Aw = (self._T_ref if self.ref != "world" else np.eye(4)) @ self.A
        self.T = _apply(B @ self.delta, Aw, self.obj.center_local)

    @property
    def center(self):
        return self.T[:3, :3] @ self.obj.center_local + self.T[:3, 3]


class SceneTracker:
    def __init__(self, objects: list[SceneObject], sigma_px=0.5, search_px=6.0, min_grad=4.0,
                 iters=2, max_samples=160, stride=4, n_eff=12, tukey_px=2.0, cam_sigma_px=None, max_mm_per_px=1.5):
        self.objects = objects
        idx = {o.name: i for i, o in enumerate(objects)}
        self.parent = [idx.get(o.parent) for o in objects]
        self.filters = [ObjectFilter(o, parent_T=objects[pi].T if pi is not None else None)
                        for o, pi in zip(objects, self.parent)]
        # parents before children, so a child is placed on its parent's fresh estimate
        self.order = sorted(range(len(objects)), key=lambda i: self.parent[i] is not None)
        self.renderer = Renderer(objects)
        self.sigma_px, self.search_px, self.min_grad = sigma_px, search_px, min_grad
        self.iters, self.max_samples, self.stride = iters, max_samples, stride
        self.n_eff = n_eff          # samples along one contour are correlated: cap each view's information
        self.tukey_px = tukey_px    # residuals beyond this get no weight (a neighbouring edge, not this one)
        self.cam_sigma_px = cam_sigma_px or {}     # per camera; a distant camera's sub-pixel errors are millimetres
        self.max_mm_per_px = max_mm_per_px         # coarser views can't refine a pose; they are not used for it
        self.T_hand_prev = None
        self.q = None                          # arm joint angles (for the links, seen by fixed cameras)
        self.held = None
        self.edge_bias = {}                    # camera -> outward edge offset (px)
        self.rejected_held = None              # last "held" claim refused (object not at the fingertips)
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.timing = {}
        self.rng = np.random.default_rng(0)

    def _robot(self, T_world_hand, finger_pos, q=None):
        if T_world_hand is None:
            return []
        parts = [(n, m, T_world_hand @ Th) for n, m, Th in body_parts(finger_pos)]
        if q is not None:
            parts += arm_parts(q, T_world_hand)
        return parts

    def _update_edge_bias(self, cam, gx, gy, samples, alpha=0.1):
        """A camera's image edges sit a fraction of a pixel outside the
        geometric silhouette (blur, anti-aliasing, the gradient peak of an
        asymmetric step). Uncorrected, that reads as every object being a
        little larger -- i.e. nearer -- and drags poses along the viewing
        direction. Estimated per camera as the median outward offset over
        all objects' contours (pose errors push some contours out and others
        in, so they cancel), smoothed over frames."""
        offs = [edge_search(gx, gy, uv, n, 2.5, self.min_grad) for _, uv, n in samples.values()]
        offs = np.concatenate(offs) if offs else np.zeros(0)
        offs = offs[np.isfinite(offs)]
        if len(offs) < 60:
            return
        b = float(np.median(offs))
        self.edge_bias[cam] = b if cam not in self.edge_bias else (1 - alpha) * self.edge_bias[cam] + alpha * b

    def _ref_pose(self, i, T_world_hand):
        f = self.filters[i]
        if f.ref == "hand":
            return T_world_hand
        if f.ref == "parent":
            return self.filters[self.parent[i]].T
        return None

    def _place_all(self, T_world_hand):
        for i in self.order:
            self.filters[i].frame(self._ref_pose(i, T_world_hand))

    def set_held(self, name, T_world_hand=None):
        """The planner tells which object the gripper holds (None: nothing).
        Grasp: the object's anchor becomes its pose relative to the hand, with
        some slip allowance; release: its world pose where it was let go."""
        if name == self.held:
            return
        if name is not None and T_world_hand is not None:
            # only an object actually between the fingertips can be held
            f = self.filters[[o.name for o in self.objects].index(name)]
            tcp = T_world_hand[:3, :3] @ np.array([0, 0, TCP_Z]) + T_world_hand[:3, 3]
            g = f.obj.world_targets(f.T).get("grasp", f.center)
            if np.linalg.norm(g - tcp) > MAX_GRASP_DIST:
                self.rejected_held = name
                name = None
                if self.held is None:
                    return
        slip = np.diag([0.001 ** 2] * 3 + [math.radians(2.0) ** 2] * 3)
        for f in self.filters:
            if f.obj.name == self.held and T_world_hand is not None:          # released
                f.set_anchor(f.T.copy(), f.P + slip, "world")
            if f.obj.name == name and T_world_hand is not None:                # grasped
                B = f._B(T_world_hand)
                A = np.linalg.inv(T_world_hand) @ f.T
                if f.obj.cls in ("tube", "tip"):
                    # a parallel gripper centres a cylinder between its pads: the axis lands on
                    # the hand's x-z plane (the fingers close along the hand's y)
                    g = A[:3, :3] @ f.obj.targets.get("grasp", f.obj.center_local) + A[:3, 3]
                    A[1, 3] -= g[1]
                f.set_anchor(A, B.T @ f.P @ B + slip, "hand", T_world_hand)
        self.held = name

    def _samples(self, frame: Frame, T_world_hand, finger_pos):
        """Per object: (3D points world, pixel uv, image normals) of its visible contour."""
        poses = [f.T for f in self.filters]
        q = None if frame.hand_mounted else (frame.q if frame.q is not None else self.q)
        robot = self._robot(frame.T_world_hand if frame.T_world_hand is not None else T_world_hand, finger_pos, q)
        rng = np.random.default_rng(len(frame.name))
        h, w = frame.gray.shape
        z_max = self.max_mm_per_px * 1e-3 * frame.K[0, 0]               # beyond this a view can't refine
        ids, depth, pts, us, vs = self.renderer.render(frame.K, frame.T_world_cam, (w, h), poses, robot, self.stride,
                                                       z_max)
        contour, nrm = occluding_contours(ids, depth)
        border = (us > 3) & (us < w - 4) & (vs > 3) & (vs < h - 4)
        contour &= border
        out = {}
        sel = np.nonzero(contour)
        oid = ids[sel]
        for i in np.unique(oid):
            if self.objects[i].cls in NO_REFINE:
                continue
            k = np.nonzero(oid == i)[0]
            if len(k) < 6:
                continue
            if len(k) > self.max_samples:
                k = rng.choice(k, self.max_samples, replace=False)
            yy, xx = sel[0][k], sel[1][k]
            n2 = nrm[yy, xx].astype(float)
            uv0 = np.stack([us[yy, xx], vs[yy, xx]], 1).astype(float)
            # depth on the far side of the step (one grid cell along the normal)
            fy = np.clip(yy + np.round(n2[:, 1]).astype(int), 0, ids.shape[0] - 1)
            fx = np.clip(xx + np.round(n2[:, 0]).astype(int), 0, ids.shape[1] - 1)
            far = np.minimum(depth[fy, fx].astype(float), 1e3)
            by = np.clip(yy - np.round(n2[:, 1]).astype(int), 0, ids.shape[0] - 1)
            bx = np.clip(xx - np.round(n2[:, 0]).astype(int), 0, ids.shape[1] - 1)
            back = depth[by, bx].astype(float)
            d0 = depth[yy, xx].astype(float)
            slope = np.where(np.isfinite(back) & (ids[by, bx] == i), d0 - back, 0.0)   # depth change per grid step
            uv, X = self._refine(i, frame, poses[i], uv0, n2, d0, far, slope)
            out[int(i)] = (X, uv, n2)
        return out, ids

    def _refine(self, i, frame, T_obj, uv0, n2, d0, far, slope, levels=2, m=6):
        """Walk out along each normal to the last ray that still lands on the
        near surface -- object i, at the depth the near surface extrapolates
        to -- so both silhouettes (the far side is something else) and
        creases (a box's top edge with its side face in view: depth keeps
        going but bends) are located to ~stride / m**levels px."""
        lo = np.zeros(len(uv0))
        h = float(self.stride)
        tol = np.maximum(0.0008, 0.3 * np.abs(slope))
        dl = d0.copy()
        for _ in range(levels):
            ss = lo[:, None] + h * np.arange(1, m + 1)[None] / m             # (N, m) beyond lo
            uv = (uv0[:, None, :] + ss[..., None] * n2[:, None, :]).reshape(-1, 2)
            t = self.renderer.cast_object(i, frame.K, frame.T_world_cam, T_obj, uv).reshape(len(uv0), m)
            expect = d0[:, None] + slope[:, None] * ss / self.stride
            on = np.isfinite(t) & (np.abs(t - expect) < tol[:, None] + 0.15 * np.abs(slope)[:, None] * ss / self.stride)
            run = np.cumprod(on, axis=1).astype(bool)                       # leading run of hits
            cnt = run.sum(1)
            has = cnt > 0
            lo = np.where(has, ss[np.arange(len(uv0)), np.maximum(cnt - 1, 0)], lo)
            dl = np.where(has, t[np.arange(len(uv0)), np.maximum(cnt - 1, 0)], dl)
            h = h / m
        uv = uv0 + lo[:, None] * n2
        dc = np.stack([(uv[:, 0] - frame.K[0, 2]) / frame.K[0, 0], (uv[:, 1] - frame.K[1, 2]) / frame.K[1, 1],
                       np.ones(len(uv))], 1)
        X = frame.T_world_cam[:3, 3] + (dc * dl[:, None]) @ frame.T_world_cam[:3, :3].T
        return uv, X

    @staticmethod
    def _project(frame, X):
        Tcw = np.linalg.inv(frame.T_world_cam)
        Xc = X @ Tcw[:3, :3].T + Tcw[:3, 3]
        z = Xc[:, 2]
        uv = np.stack([frame.K[0, 0] * Xc[:, 0] / z + frame.K[0, 2], frame.K[1, 1] * Xc[:, 1] / z + frame.K[1, 2]], 1)
        J = np.zeros((len(X), 2, 3))
        J[:, 0, 0] = frame.K[0, 0] / z
        J[:, 0, 2] = -frame.K[0, 0] * Xc[:, 0] / z ** 2
        J[:, 1, 1] = frame.K[1, 1] / z
        J[:, 1, 2] = -frame.K[1, 1] * Xc[:, 1] / z ** 2
        return uv, J @ Tcw[:3, :3]

    def update(self, frames: list[Frame], T_world_hand=None, finger_pos=(0.04, 0.04), q=None):
        t0 = time.perf_counter()
        self.q = q
        self._place_all(T_world_hand)               # held objects follow the hand, seated ones their parent
        for f in self.filters:
            f.support, f.updated = 0, False
        if T_world_hand is not None:
            self.T_hand_prev = np.asarray(T_world_hand, float).copy()

        # contour samples per camera (object-local points, so they follow pose updates)
        per_obj = {}
        views = {}
        def per_camera(fr):                       # cameras in parallel (Open3D and OpenCV release the GIL)
            samples, _ = self._samples(fr, T_world_hand, finger_pos)
            g = cv2.GaussianBlur(fr.gray.astype(np.float32), (3, 3), 0)
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            return samples, gx, gy

        tr = time.perf_counter()
        results = list(self.pool.map(per_camera, frames))
        t_r = time.perf_counter() - tr
        for fr, (samples, gx, gy) in zip(frames, results):
            views[fr.name] = (gx, gy)
            self._update_edge_bias(fr.name, gx, gy, samples)
            for i, (X, uv, n) in samples.items():
                T = self.filters[i].T
                Xl = (X - T[:3, 3]) @ T[:3, :3]
                per_obj.setdefault(i, []).append((fr, Xl, n))

        t_f = time.perf_counter()
        for i in self.order:
            if i in per_obj and self.objects[i].cls not in NO_REFINE:
                self.filters[i].frame(self._ref_pose(i, T_world_hand))
                self._update_object(self.filters[i], per_obj[i], views)
        self._place_all(T_world_hand)
        self.timing = dict(render_ms=t_r * 1e3, fit_ms=(time.perf_counter() - t_f) * 1e3,
                           total_ms=(time.perf_counter() - t0) * 1e3)
        return self.state()

    def _update_object(self, flt: ObjectFilter, obs, views):
        T0, P0w, B = flt.frame()                      # solve against the anchor ...
        P_inv = np.linalg.inv(P0w)
        delta = B @ flt.delta                         # ... starting from the current estimate
        H = None
        used = 0
        sig_pos = float(np.sqrt(np.linalg.eigvalsh(flt.P[:3, :3]).max()))
        for it in range(self.iters):
            c0 = T0[:3, :3] @ flt.obj.center_local + T0[:3, 3]
            Js, rs = [], []
            for fr, Xl, n2 in obs:
                Xw0 = Xl @ T0[:3, :3].T + T0[:3, 3]
                Xw = (Xw0 - c0) @ _rotvec(delta[3:]).T + c0 + delta[:3]
                uv, Jp = self._project(fr, Xw)
                # search window from the predicted uncertainty in this view, shrinking per iteration
                z = float(np.median((Xw - fr.T_world_cam[:3, 3]) @ fr.T_world_cam[:3, 2]))
                if 1e3 * z / fr.K[0, 0] > self.max_mm_per_px:
                    continue
                radius = float(np.clip(3.0 * sig_pos * fr.K[0, 0] / max(z, 0.05) + 1.0, 2.0, self.search_px))
                radius = max(2.0, radius / (1.0 + it))
                off = edge_search(*views[fr.name], uv, n2, radius, self.min_grad) - self.edge_bias.get(fr.name, 0.0)
                ok = np.isfinite(off) & (np.abs(off) < 0.95 * radius)
                if ok.sum() < 3:
                    continue
                r = -off[ok]                                   # n . (uv - edge)
                u = np.abs(r) / self.tukey_px
                w = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)    # Tukey
                w *= min(1.0, self.n_eff / max(1.0, w.sum()))  # correlated samples: cap the view's information
                wgt = np.sqrt(w) / self.cam_sigma_px.get(fr.name, self.sigma_px)
                Jx = np.einsum("ni,nij->nj", n2[ok], Jp[ok])   # d r / d X
                # d r / d (dt, dw) = [Jx, (X - c) x Jx]   (dX = dt + dw x (X - c))
                Js.append(wgt[:, None] * np.c_[Jx, np.cross(Xw[ok] - (c0 + delta[:3]), Jx)])
                rs.append(wgt * r)
            if not Js:
                break
            J, r = np.concatenate(Js), np.concatenate(rs)
            used = len(r)
            H = J.T @ J + P_inv
            g = J.T @ r + P_inv @ delta
            step = -np.linalg.solve(H, g)
            delta = delta + step
            if np.linalg.norm(step[:3]) < 2e-5:
                break
        if H is None or used < 8:
            return
        # a frame that disagrees wildly with what earlier frames established is an outlier
        d_prev = B @ flt.delta
        P_frame = np.linalg.inv(H)
        diff = delta - d_prev
        if float(diff @ np.linalg.solve(P_frame + B @ flt.P @ B.T, diff)) > 25.0:
            return
        flt.fuse(delta, H - P_inv, P0w, B)
        flt.support, flt.updated = used, True

    def state(self) -> list[ObjectState]:
        out = []
        for f in self.filters:
            s = float(np.sqrt(np.linalg.eigvalsh(f.P[:3, :3]).max()) * 1e3)
            out.append(ObjectState(f.obj.name, f.obj.cls, f.T.copy(), s, f.support, f.updated,
                                   f.obj.world_targets(f.T)))
        return out

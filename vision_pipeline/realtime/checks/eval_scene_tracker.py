"""Replay a recorded episode through the scene tracker and score every
object against the simulator's true poses.

The tracker starts from the scene map with every object's pose disturbed
(xy +-perturb mm, z +-0.5 mm, yaw +-1 deg: what a slightly-off offline map
looks like) and gets only the camera images, the hand pose, finger
openings and joint angles, step by step at 15 Hz, plus which object is in
the gripper (the planner knows what it grasped). Reported, per object
class: the error of its key target point (tube: grasp point; tip: top;
holder: slot openings; others: centre) at the start, after the replay,
and for the episode's target tube just before the grasp; the share of
objects the cameras supported; time per frame.

  .venv-cad/Scripts/python.exe .../eval_scene_tracker.py EPISODE_DIR --cams wrist_top,wrist_side,platform_cam1
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from camera_model import exterior_camera, interpolate_pose7, pose_to_matrix, wrist_cameras
from scene_model import scene_from_sim
from scene_tracker import Frame, SceneTracker

ap = argparse.ArgumentParser()
ap.add_argument("episode")
ap.add_argument("--cams", default="wrist_top,wrist_side")
ap.add_argument("--perturb-mm", type=float, default=3.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", default=None, help="first:last:stride")
ap.add_argument("--image-lag", type=float, default=0.25,
                help="images are this many control steps older than the recorded hand pose (measured: 0.25)")
ap.add_argument("--fixed-cam-sigma", type=float, default=1.5, help="edge sigma (px) for the fixed camera")
ap.add_argument("--stride", type=int, default=4)
ap.add_argument("--iters", type=int, default=2)
ap.add_argument("--out", default=None)
args = ap.parse_args()

ep = Path(args.episode)
meta = json.loads((ep / "meta.json").read_text())
d = np.load(ep / "steps.npz")
n_steps = len(d["hand_pose"])
cams = dict(wrist_cameras())
cams["platform_cam1"] = exterior_camera()
cams = {k: cams[k] for k in args.cams.split(",")}
r, c = meta["layout"]["pick_slot"]
target = f"tube_r{r}_c{c}"

truth0 = scene_from_sim(meta, d["object_poses"][0])
rng = np.random.default_rng(args.seed)
objs = scene_from_sim(meta, d["object_poses"][0])
true_T = {o.name: o.T.copy() for o in objs}
for o in objs:
    if o.cls == "bench" or o.parent:
        continue
    a = math.radians(rng.uniform(-1, 1))
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    c0 = o.T[:3, :3] @ o.center_local + o.T[:3, 3]
    T = o.T.copy()
    T[:3, :3] = R @ T[:3, :3]
    T[:3, 3] = R @ (o.T[:3, 3] - c0) + c0 + np.r_[rng.uniform(-1, 1, 2) * args.perturb_mm, rng.uniform(-0.5, 0.5)] * 1e-3
    o.T = T
by_name = {o.name: o for o in objs}
for o in objs:                       # seated objects: on their (mis-placed) parent, plus their own clearance-size error
    if o.parent:
        P = by_name[o.parent]
        o.T = P.T @ np.linalg.inv(true_T[P.name]) @ true_T[o.name]
        o.T[:3, 3] += np.r_[rng.uniform(-1, 1, 2) * 0.3, 0.0] * 1e-3
trk = SceneTracker(objs, cam_sigma_px={"platform_cam1": args.fixed_cam_sigma}, stride=args.stride, iters=args.iters)


def key_points(obj, T):
    w = obj.world_targets(T)
    if obj.cls == "tube":
        return np.array([w["grasp"]])
    if obj.cls == "tip":
        return np.array([w["top"]])
    if obj.cls == "holder":
        return np.array([v for k, v in w.items() if k.startswith("slot_")])
    if obj.cls == "tip_rack":
        return np.array([v for k, v in w.items() if k.startswith("pos_")])
    return np.array([w.get("center", T[:3, :3] @ obj.center_local + T[:3, 3])])


def errors(states, truth):
    return {s.name: float(np.max(np.linalg.norm(key_points(o, s.T) - key_points(o, t.T), axis=1)) * 1e3)
            for s, o, t in zip(states, objs, truth)}


first, last, stride = (0, n_steps, 1) if not args.steps else [int(v) for v in args.steps.split(":")]
init_err = errors(trk.state(), truth0)
supported = {o.name: 0 for o in objs}
log, times = [], []
pregrasp = None
for k in range(first, min(last, n_steps), stride):
    Th = pose_to_matrix(d["hand_pose"][k])
    fingers = d["finger_position_m"][k]
    gap = float(fingers.sum())
    trk.set_held(target if 0.004 < gap < 0.0085 else None, Th)   # fingers closed on a 7 mm tube
    # the hand (and arm) where it was when the images were taken
    j = max(k - 1, 0)
    Ti = pose_to_matrix(interpolate_pose7(d["hand_pose"][j], d["hand_pose"][k], 1 - args.image_lag)) if k else Th
    qi = (args.image_lag * d["joint_position"][j] + (1 - args.image_lag) * d["joint_position"][k])
    frames = []
    for cn, cam in cams.items():
        g = cv2.imread(str(ep / "images" / f"{cn}_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
        frames.append(Frame(cn, g, cam.K, cam.world_from_cam(Ti), Ti, qi, not cam.fixed))
    t = time.perf_counter()
    states = trk.update(frames, Th, fingers, d["joint_position"][k])
    times.append(dict(trk.timing, wall_ms=(time.perf_counter() - t) * 1e3))
    truth = scene_from_sim(meta, d["object_poses"][k])
    err = errors(states, truth)
    for s in states:
        supported[s.name] += int(s.updated)
    if trk.held is None:
        pregrasp = (k, err[target])
    log.append(dict(k=k, target_err=err[target], held=trk.held,
                    updated=sum(s.updated for s in states), **{f"t_{kk}": v for kk, v in trk.timing.items()}))
    if k % 25 == 0:
        print(f"step {k}: target {target} err {err[target]:.2f} mm, objects updated {log[-1]['updated']}, "
              f"{times[-1]['wall_ms']:.0f} ms", flush=True)

final_err = errors(trk.state(), scene_from_sim(meta, d["object_poses"][min(last, n_steps) - 1]))
by_cls = {}
for o in objs:
    by_cls.setdefault(o.cls, []).append(o.name)
summary = {"episode": ep.name, "cams": list(cams), "target": target, "steps": len(log),
           "target_err_pregrasp_mm": pregrasp, "ms_median": float(np.median([t["wall_ms"] for t in times])),
           "classes": {}}
for cls, names in sorted(by_cls.items()):
    seen = [n for n in names if supported[n] > 0]
    summary["classes"][cls] = dict(
        n=len(names), seen=len(seen),
        init_median_mm=round(float(np.median([init_err[n] for n in names])), 2),
        final_median_mm_seen=round(float(np.median([final_err[n] for n in seen])), 2) if seen else None,
        final_max_mm_seen=round(float(np.max([final_err[n] for n in seen])), 2) if seen else None,
        init_median_mm_seen=round(float(np.median([init_err[n] for n in seen])), 2) if seen else None)
print(json.dumps(summary, indent=1))
if args.out:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{ep.name}_summary.json").write_text(json.dumps(summary, indent=1))
    (out / f"{ep.name}_log.json").write_text(json.dumps(log))
    (out / f"{ep.name}_final_err.json").write_text(json.dumps({"init": init_err, "final": final_err,
                                                               "supported": supported}, indent=1))

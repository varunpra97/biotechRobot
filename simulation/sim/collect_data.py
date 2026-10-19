"""Phase 1 data collector: constraint-based motion planning (Gen2Sim-style)
+ structured multi-modal capture.

Replaces the earlier random-joint-offset collector. Motion is no longer
arbitrary: each episode solves real inverse kinematics (sim/ik_solver.py)
for a sequence of task-space waypoints defined relative to a target
object's own (programmatically read) world pose, with phase-specific
motion constraints:

  Approach:    hover above the target tube's top
  Execution:   straight-line Cartesian descent to just above the tube's
               top surface -- NOT into it. The tubes are solid boxes, not
               hollow containers, so "inserting into" one is not something
               that can be physically correct with this geometry; the
               physically honest version of this task is approaching and
               coming to rest just above the surface, as if aligning a
               pipette tip at the opening without penetrating it.
  Retraction:  straight-line Cartesian vertical lift back up, before any
               lateral motion (here the lift is purely vertical since
               approach/execution share the same XY; a lateral retreat to
               a different object would reuse this same "lift first"
               waypoint before moving sideways)

Targets are defined for the gripper's actual fingertip control point
(lab_scene.get_fingertip_pos()), not the hand frame origin. The hand frame
sits ~8cm above the real fingertip (measured empirically -- see
lab_scene.FINGER_PAD_DEPTH); targeting the hand frame directly drove the
real fingers through whatever the hand was "above", which is what produced
a visually confirmed interpenetration bug in the first version of this
script. Scene objects also now have real collision geometry (previously
none at all -- lab_scene._box()), and the rack's tube spacing was widened
so the gripper can approach one tube without touching its neighbors.

An episode is only saved if every waypoint's IK converged, stayed within
the arm's joint limits, kept the hand above the bench surface, AND the
fingertip (with a gripper-width safety margin) doesn't geometrically
overlap any scene object's bounding box -- a real collision check, not
just a height proxy. Otherwise the episode is discarded and the reason is
reported, never silently written as if it had succeeded.

State-aware metadata (joint names, joint limits, link names, and each task
object's bounding-box size) is extracted programmatically from the live
Articulation and scene (not hand-typed) and saved per episode.

Usage (Isaac venv, repo root on Windows box):
    python sim\\collect_data.py --episodes 1 --out ..\\data\\episodes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ik_solver
import lab_scene

TARGET_OBJECT = "tube_r0_c2"  # nearest row (toward the arm), center column
HOVER_HEIGHT = 0.12         # m above the tube's top -- approach/retraction
DIP_CLEARANCE = 0.015       # m above the tube's top -- execution target (never below it)
BENCH_CLEARANCE = 0.01      # m: minimum allowed hand height above the bench
GRIPPER_RADIUS = 0.045      # m: horizontal safety margin for collision checks,
                            # approximating the open gripper's footprint


def build_waypoints(tube_pos, tube_size):
    """Approach -> straight-line execution descent -> vertical-first
    retraction, as 5 task-space (x, y, z) targets for the fingertip."""
    # tube_pos is the tube's BASE (mesh local origin), not its center --
    # see collect_data_pickplace.py for why this must be a full tube_size[2].
    tube_top = tube_pos[2] + tube_size[2]
    hover = np.array([tube_pos[0], tube_pos[1], tube_top + HOVER_HEIGHT])
    touch = np.array([tube_pos[0], tube_pos[1], tube_top + DIP_CLEARANCE])
    return [
        ("approach", hover),
        ("execution_mid", hover + (touch - hover) * 0.5),
        ("execution_end", touch),
        ("retraction_mid", touch + (hover - touch) * 0.5),
        ("retraction_end", hover),
    ]


def check_collision(fingertip_pos, objects, radius=GRIPPER_RADIUS):
    """Objects whose bounding box (expanded horizontally by `radius`, to
    approximate the gripper's footprint, but NOT vertically -- vertical
    precision is exactly what's being validated) contains fingertip_pos."""
    hits = []
    for name, info in objects.items():
        pos = lab_scene.get_prim_world_pose(info["path"])[:3]
        w, d, h = info["size"]
        lo = pos - np.array([w / 2 + radius, d / 2 + radius, h / 2])
        hi = pos + np.array([w / 2 + radius, d / 2 + radius, h / 2])
        if np.all(fingertip_pos >= lo) and np.all(fingertip_pos <= hi):
            hits.append(name)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--out", default=r"C:\Users\varun\workspace\BiotechRobot\data\episodes")
    ap.add_argument("--start-index", type=int, default=0)
    args = ap.parse_args()

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    from PIL import Image

    arm, cams, objects = lab_scene.build_scene(simulation_app)
    cam_names = sorted(cams.keys())
    obj_names = sorted(objects.keys())

    for name in cam_names:
        _, cam = cams[name]
        cam.add_distance_to_camera_to_frame()
        cam.add_instance_segmentation_to_frame()
    for _ in range(5):
        simulation_app.update()

    hand_idx = arm.get_body_index(lab_scene.HAND_PATH.rsplit("/", 1)[-1])
    joint_limits = np.asarray(arm.get_dof_limits())[0]  # (9, 2)
    bench_top_z = lab_scene.BENCH_Z + 0.04

    def get_fingertip_pos():
        return lab_scene.get_fingertip_pos()

    def step():
        simulation_app.update()

    out_root = Path(args.out)
    n_saved = 0

    for ep in range(args.start_index, args.start_index + args.episodes):
        q = lab_scene.Q_HOME.copy()
        arm.set_joint_positions(q)
        arm.set_joint_position_targets(q)
        for _ in range(10):
            simulation_app.update()

        tube_path = objects[TARGET_OBJECT]["path"]
        tube_size = objects[TARGET_OBJECT]["size"]
        tube_pos = lab_scene.get_prim_world_pose(tube_path)[:3]
        waypoints = build_waypoints(tube_pos, tube_size)

        frames_q, wp_log = [], []
        ok = True
        for phase_name, target in waypoints:
            q, converged, iters, err = ik_solver.solve_ik(
                arm, hand_idx, target, q, joint_limits, get_fingertip_pos, step)
            fingertip_pos = get_fingertip_pos()
            above_bench = bool(fingertip_pos[2] >= (bench_top_z - BENCH_CLEARANCE))
            within_limits = bool(
                np.all(q[:7] >= joint_limits[:7, 0] - 1e-6)
                and np.all(q[:7] <= joint_limits[:7, 1] + 1e-6))
            finite = bool(np.all(np.isfinite(q)))
            collisions = check_collision(fingertip_pos, objects)
            no_collision = len(collisions) == 0
            wp_ok = converged and above_bench and within_limits and finite and no_collision
            ok = ok and wp_ok
            wp_log.append({
                "phase": phase_name, "target": target.tolist(),
                "ik_converged": converged, "ik_iters": iters,
                "ik_final_error_m": err, "above_bench": above_bench,
                "within_joint_limits": within_limits,
                "no_collision": no_collision, "collisions": collisions,
                "ok": wp_ok,
            })
            frames_q.append(q.copy())
            if not wp_ok:
                print(f"  episode {ep}: waypoint '{phase_name}' failed "
                      f"(converged={converged} err={err:.4f} "
                      f"above_bench={above_bench} within_limits={within_limits} "
                      f"collisions={collisions})")
                break

        if not ok:
            print(f"episode {ep}: REJECTED (constraint violation) -- not saved")
            continue

        ep_dir = out_root / f"ep_{ep:04d}"
        (ep_dir / "images").mkdir(parents=True, exist_ok=True)
        (ep_dir / "depth").mkdir(parents=True, exist_ok=True)
        (ep_dir / "segmentation").mkdir(parents=True, exist_ok=True)

        joint_pos, joint_vel, gripper = [], [], []
        ee_pose, fingertip_pose, obj_poses, action = [], [], [], []
        seg_info = None

        for f, q_f in enumerate(frames_q):
            arm.set_joint_positions(q_f)
            arm.set_joint_position_targets(q_f)
            for _ in range(8):
                simulation_app.update()

            for name in cam_names:
                _, cam = cams[name]
                frame = cam.get_current_frame()
                Image.fromarray(frame["rgb"]).save(
                    str(ep_dir / "images" / f"{name}_f{f:04d}.png"))
                depth = frame.get("distance_to_camera")
                if depth is not None:
                    np.save(str(ep_dir / "depth" / f"{name}_f{f:04d}.npy"),
                            np.asarray(depth))
                seg = frame.get("instance_segmentation")
                if seg is not None:
                    np.save(str(ep_dir / "segmentation" / f"{name}_f{f:04d}.npy"),
                            np.asarray(seg["data"]))
                    if seg_info is None:
                        seg_info = seg["info"]

            qp = np.asarray(arm.get_joint_positions())[0]
            qv = np.asarray(arm.get_joint_velocities())[0]
            joint_pos.append(qp[:7])
            joint_vel.append(qv[:7])
            gripper.append(qp[7:9])
            ee_pose.append(lab_scene.get_prim_world_pose(lab_scene.HAND_PATH))
            fingertip_pose.append(lab_scene.get_fingertip_pos())
            obj_poses.append([
                lab_scene.get_prim_world_pose(objects[n]["path"]) for n in obj_names
            ])
            action.append(q_f[:7])

        np.savez(
            str(ep_dir / "trajectory.npz"),
            joint_pos=np.array(joint_pos), joint_vel=np.array(joint_vel),
            gripper=np.array(gripper), ee_pose=np.array(ee_pose),
            fingertip_pose=np.array(fingertip_pose),
            object_poses=np.array(obj_poses), action=np.array(action),
        )

        meta = {
            "task": f"approach_tube_top[{TARGET_OBJECT}]",
            "n_frames": len(frames_q),
            "cameras": cam_names,
            "object_names": obj_names,
            "target_object": TARGET_OBJECT,
            "waypoints": wp_log,
            "sequence_valid": ok,
            "robot": {
                "joint_names": list(arm.dof_names),
                "link_names": list(arm.body_names),
                "joint_limits": joint_limits.tolist(),
            },
            "object_bboxes": {n: objects[n]["size"] for n in obj_names},
        }
        (ep_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        if seg_info is not None:
            (ep_dir / "segmentation_info.json").write_text(
                json.dumps(seg_info, indent=2, default=str))

        n_saved += 1
        print(f"episode {ep}: ACCEPTED, {len(frames_q)} frames -> {ep_dir}")

    print(f"done: {n_saved}/{args.episodes} episodes saved")
    simulation_app.close()


if __name__ == "__main__":
    main()

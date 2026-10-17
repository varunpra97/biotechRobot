"""Phase 1 data collector: constraint-based motion planning (Gen2Sim-style)
+ structured multi-modal capture.

Replaces the earlier random-joint-offset collector. Motion is no longer
arbitrary: each episode solves real inverse kinematics (sim/ik_solver.py)
for a sequence of task-space waypoints defined relative to a target
object's own (programmatically read) world pose, with phase-specific
motion constraints:

  Approach:    hover above the target tube's opening
  Execution:   straight-line Cartesian descent into the opening
  Retraction:  straight-line Cartesian vertical lift back out, before any
               lateral motion (here the lift is purely vertical since
               approach/dip share the same XY; a lateral retreat to a
               different object would reuse this same "lift first"
               waypoint before moving sideways)

An episode is only saved if every waypoint's IK actually converged, stayed
within the arm's joint limits, and kept the hand above the bench surface;
otherwise it is discarded and the reason is reported -- never silently
written as if it had succeeded.

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

TARGET_OBJECT = "tube_1_1"  # a tube roughly in the middle of the 3x4 rack
HOVER_HEIGHT = 0.12         # m above the tube opening -- approach/retraction
DIP_HEIGHT = 0.02           # m above the tube opening -- execution target
BENCH_CLEARANCE = 0.01      # m: minimum allowed hand height above the bench


def build_waypoints(tube_pos, tube_size):
    """Approach -> straight-line execution descent -> vertical-first
    retraction, as 5 task-space (x, y, z) targets."""
    opening_z = tube_pos[2] + tube_size[2] / 2.0
    hover = np.array([tube_pos[0], tube_pos[1], opening_z + HOVER_HEIGHT])
    dip = np.array([tube_pos[0], tube_pos[1], opening_z + DIP_HEIGHT])
    return [
        ("approach", hover),
        ("execution_mid", hover + (dip - hover) * 0.5),
        ("execution_end", dip),
        ("retraction_mid", dip + (hover - dip) * 0.5),
        ("retraction_end", hover),
    ]


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

    def get_hand_pos():
        return lab_scene.get_prim_world_pose(lab_scene.HAND_PATH)[:3]

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
                arm, hand_idx, target, q, joint_limits, get_hand_pos, step)
            hand_pos = get_hand_pos()
            above_bench = bool(hand_pos[2] >= (bench_top_z - BENCH_CLEARANCE))
            within_limits = bool(
                np.all(q[:7] >= joint_limits[:7, 0] - 1e-6)
                and np.all(q[:7] <= joint_limits[:7, 1] + 1e-6))
            finite = bool(np.all(np.isfinite(q)))
            wp_ok = converged and above_bench and within_limits and finite
            ok = ok and wp_ok
            wp_log.append({
                "phase": phase_name, "target": target.tolist(),
                "ik_converged": converged, "ik_iters": iters,
                "ik_final_error_m": err, "above_bench": above_bench,
                "within_joint_limits": within_limits, "ok": wp_ok,
            })
            frames_q.append(q.copy())
            if not wp_ok:
                print(f"  episode {ep}: waypoint '{phase_name}' failed "
                      f"(converged={converged} err={err:.4f} "
                      f"above_bench={above_bench} within_limits={within_limits})")
                break

        if not ok:
            print(f"episode {ep}: REJECTED (constraint violation) -- not saved")
            continue

        ep_dir = out_root / f"ep_{ep:04d}"
        (ep_dir / "images").mkdir(parents=True, exist_ok=True)
        (ep_dir / "depth").mkdir(parents=True, exist_ok=True)
        (ep_dir / "segmentation").mkdir(parents=True, exist_ok=True)

        joint_pos, joint_vel, gripper = [], [], []
        ee_pose, obj_poses, action = [], [], []
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
            obj_poses.append([
                lab_scene.get_prim_world_pose(objects[n]["path"]) for n in obj_names
            ])
            action.append(q_f[:7])

        np.savez(
            str(ep_dir / "trajectory.npz"),
            joint_pos=np.array(joint_pos), joint_vel=np.array(joint_vel),
            gripper=np.array(gripper), ee_pose=np.array(ee_pose),
            object_poses=np.array(obj_poses), action=np.array(action),
        )

        meta = {
            "task": f"dip_into_tube[{TARGET_OBJECT}]",
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

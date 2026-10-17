"""View a collected episode's motion in Isaac Sim.

Builds the same scene as lab_scene.py / collect_data.py, loads a saved
trajectory.npz (the waypoints an episode was accepted with), and smoothly
replays it in a loop so you can watch the motion play out -- including any
object the episode's kinematic attach moved (e.g. a grasped tube), replayed
the same way from the saved object_poses so it doesn't just sit frozen in
its start position while the gripper moves around it. Doesn't solve IK or
write anything -- pure playback of data a collect_data*.py script already
generated and validated.

Usage (Isaac venv, repo root on Windows box):
    python sim\\view_motion.py                        # GUI, loops forever
    python sim\\view_motion.py --camera arm_cam0       # start on this camera
    python sim\\view_motion.py --traj <path\\to\\trajectory.npz>
    python sim\\view_motion.py --headless --once       # sanity check, no window
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import lab_scene

DEFAULT_TRAJ = (
    r"C:\Users\varun\workspace\BiotechRobot\data\episodes\ep_0000\trajectory.npz"
)


def smooth(a, b, t):
    s = 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))
    return a + (b - a) * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=DEFAULT_TRAJ)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--camera", default=None,
                    help="GUI mode: set viewport to this camera on startup")
    ap.add_argument("--steps-per-waypoint", type=int, default=30,
                    help="interpolation steps between each saved waypoint")
    ap.add_argument("--once", action="store_true",
                    help="play the motion once and exit instead of looping")
    args = ap.parse_args()

    traj_path = Path(args.traj)
    if not traj_path.exists():
        raise SystemExit(
            f"trajectory not found: {traj_path}\n"
            f"(generate one first with sim\\collect_data.py)")
    data = np.load(str(traj_path))
    joint_pos = data["joint_pos"]  # (T, 7)
    gripper = data["gripper"]      # (T, 2)
    q_frames = np.concatenate([joint_pos, gripper], axis=1)  # (T, 9)
    print(f"loaded {len(q_frames)}-waypoint trajectory from {traj_path}")

    # Any object the episode's own kinematic attach moved (e.g. a grasped
    # tube) was recorded in object_poses -- replay those too, interpolated
    # the same way as the joints, so it doesn't appear frozen in its start
    # slot while the gripper moves around it.
    obj_names, obj_pos_frames = [], None
    meta_path = traj_path.parent / "meta.json"
    if "object_poses" in data.files and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        obj_names = meta.get("object_names", [])
        obj_pos_frames = data["object_poses"][:, :, :3]  # (T, N, 3), position only
        print(f"replaying {len(obj_names)} object(s) from object_poses")

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless})

    from isaacsim.core.prims import Articulation

    arm, cams, objects = lab_scene.build_scene(simulation_app)
    print(f"scene ready: {len(cams)} cameras")

    if args.camera and not args.headless:
        if args.camera not in cams:
            print(f"Unknown camera '{args.camera}'. Available: "
                  f"{sorted(cams.keys())}")
        else:
            prim_path, _ = cams[args.camera]
            try:
                import omni.kit.viewport.utility as vpu
                vp = vpu.get_active_viewport()
                vp.set_active_camera(prim_path)
                print(f"Viewport set to camera: {args.camera}")
            except Exception as e:
                print(f"Could not set viewport: {e}")

    def set_object_positions(pos_row):
        for name, pos in zip(obj_names, pos_row):
            path = objects[name]["path"]
            lab_scene.set_prim_world_position(path, pos)

    def play_once():
        q_prev = q_frames[0]
        arm.set_joint_positions(q_prev)
        arm.set_joint_position_targets(q_prev)
        if obj_pos_frames is not None:
            set_object_positions(obj_pos_frames[0])
        for _ in range(10):
            simulation_app.update()
        for f, q_target in enumerate(q_frames[1:], start=1):
            obj_prev = obj_pos_frames[f - 1] if obj_pos_frames is not None else None
            obj_target = obj_pos_frames[f] if obj_pos_frames is not None else None
            for s in range(args.steps_per_waypoint):
                t = (s + 1) / args.steps_per_waypoint
                q_cmd = smooth(q_prev, q_target, t)
                arm.set_joint_position_targets(q_cmd)
                if obj_pos_frames is not None:
                    set_object_positions(smooth(obj_prev, obj_target, t))
                simulation_app.update()
                if not simulation_app.is_running():
                    return
            q_prev = q_target

    if args.once or args.headless:
        play_once()
        print("PLAYBACK DONE")
    else:
        while simulation_app.is_running():
            play_once()

    simulation_app.close()


if __name__ == "__main__":
    main()

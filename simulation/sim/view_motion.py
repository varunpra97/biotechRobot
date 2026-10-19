"""View a collected episode's motion in Isaac Sim.

For a pick-and-place episode (collect_data_pickplace.py) this replays the
episode's validated DENSE joint path -- every configuration the collector
solved and collision-checked between keyframes -- not an interpolation
between the 10 keyframes. A held object is re-attached to the LIVE hand
pose with its recorded in-hand transform (position and orientation), and at
the recorded release point it's handed to the physics solver exactly as the
collector did, so the tube slides into its rotor bore under real physics.

(The earlier version interpolated the saved object positions in straight
lines between keyframes while the arm followed a curved joint-space path,
so the tube visibly separated from the gripper and "flew" across the scene,
and it never rotated with the hand.)

Episodes without a dense path (collect_data.py) fall back to smoothly
interpolating the keyframe joint positions.

Usage (Isaac venv, repo root on Windows box):
    python sim\\view_motion.py                        # GUI, loops forever
    python sim\\view_motion.py --camera wrist          # start on this camera
    python sim\\view_motion.py --multi-view            # all cameras tiled at once
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
ATTACHED, DYNAMIC = 1, 2  # collect_data_pickplace.py dense_tube_mode ids


def smooth(a, b, t):
    s = 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))
    return a + (b - a) * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default=DEFAULT_TRAJ)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--camera", default=None,
                    help="GUI mode: set viewport to this camera on startup")
    ap.add_argument("--multi-view", action="store_true",
                    help="GUI mode: show every camera at once, tiled")
    ap.add_argument("--substeps", type=int, default=3,
                    help="app updates per dense-path step (playback speed)")
    ap.add_argument("--steps-per-waypoint", type=int, default=30,
                    help="keyframe-only episodes: interpolation steps per keyframe")
    ap.add_argument("--once", action="store_true",
                    help="play the motion once and exit instead of looping")
    args = ap.parse_args()

    traj_path = Path(args.traj)
    if not traj_path.exists():
        raise SystemExit(f"trajectory not found: {traj_path}\n"
                         f"(generate one first with sim\\collect_data_pickplace.py)")
    data = np.load(str(traj_path))
    meta_path = traj_path.parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    dense = "dense_q" in data.files and "grasp" in meta

    source = meta.get("source_object") if dense else None
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": args.headless})
    update = simulation_app.update

    arm, cams, objects = lab_scene.build_scene(
        simulation_app, rigid_objects=(source,) if source else ())
    print(f"scene ready: {len(cams)} cameras")

    if args.multi_view and not args.headless:
        lab_scene.tile_camera_views(cams)
    elif args.camera and not args.headless:
        if args.camera not in cams:
            print(f"Unknown camera '{args.camera}'. Available: {sorted(cams.keys())}")
        else:
            import omni.kit.viewport.utility as vpu
            vpu.get_active_viewport().set_active_camera(cams[args.camera][0])
            print(f"Viewport set to camera: {args.camera}")

    def set_arm(q):
        arm.set_joint_positions(q)
        arm.set_joint_position_targets(q)

    if dense:
        dq = data["dense_q"]
        mode = data["dense_tube_mode"]
        key = data["dense_keyframe"]
        release_k = meta["grasp"]["release_keyframe"]
        settle = meta["dense_path"]["settle_updates_after_release"]
        T_hand_tube = np.array(meta["grasp"]["tube_in_hand_T"])
        tube_path = objects[source]["path"]
        tube0 = data["object_poses"][0][meta["object_names"].index(source)]
        gripper_bodies = [lab_scene.HAND_PATH, lab_scene.LEFT_FINGER_PATH,
                          lab_scene.RIGHT_FINGER_PATH]
        print(f"loaded {len(dq)}-step validated dense path ({len(data['joint_pos'])} keyframes) "
              f"from {traj_path}")

        def attach_tube():
            h = lab_scene.get_prim_world_pose(lab_scene.HAND_PATH)
            Th = np.eye(4)
            Th[:3, :3] = lab_scene.quat_to_matrix(h[3:7])
            Th[:3, 3] = h[:3]
            T = Th @ T_hand_tube
            lab_scene.set_prim_world_pose(tube_path, T[:3, 3], lab_scene.matrix_to_quat(T[:3, :3]))

        def play_once():
            # reset: tube back in its holder slot, kinematic, pads not filtered
            lab_scene.set_kinematic(tube_path, True)
            lab_scene.set_collision_filter(tube_path, [])
            lab_scene.set_prim_world_pose(tube_path, tube0[:3], tube0[3:7])
            set_arm(dq[0])
            for _ in range(10):
                update()
            for i in range(1, len(dq)):
                if mode[i] == ATTACHED and mode[i - 1] != ATTACHED:
                    lab_scene.set_collision_filter(tube_path, gripper_bodies)
                if mode[i] == DYNAMIC and mode[i - 1] != DYNAMIC:
                    lab_scene.set_collision_filter(tube_path, [])
                    lab_scene.set_kinematic(tube_path, False)
                for s in range(1, args.substeps + 1):
                    set_arm(dq[i - 1] + (dq[i] - dq[i - 1]) * s / args.substeps)
                    if mode[i] == ATTACHED:
                        attach_tube()
                    update()
                    if not simulation_app.is_running():
                        return
                if key[i] == release_k:
                    for _ in range(settle):
                        update()
            p = lab_scene.get_prim_world_pose(tube_path)
            print(f"tube final tip position {np.round(p[:3], 5)} "
                  f"(recorded {np.round(data['object_poses'][-1][meta['object_names'].index(source)][:3], 5)})")
    else:
        q_frames = np.concatenate([data["joint_pos"], data["gripper"]], axis=1)
        print(f"loaded {len(q_frames)}-keyframe trajectory from {traj_path}")

        def play_once():
            q_prev = q_frames[0]
            set_arm(q_prev)
            for _ in range(10):
                update()
            for q_target in q_frames[1:]:
                for s in range(args.steps_per_waypoint):
                    arm.set_joint_position_targets(
                        smooth(q_prev, q_target, (s + 1) / args.steps_per_waypoint))
                    update()
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

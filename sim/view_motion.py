"""View a collected episode's motion in Isaac Sim.

Builds the same scene as lab_scene.py / collect_data.py, loads a saved
trajectory.npz (the 5 approach/execution/retraction waypoints an episode
was accepted with), and smoothly replays it in a loop so you can watch the
dip-into-tube motion play out. Doesn't solve IK or write anything -- pure
playback of data that collect_data.py already generated and validated.

Usage (Isaac venv, repo root on Windows box):
    python sim\\view_motion.py                        # GUI, loops forever
    python sim\\view_motion.py --camera arm_cam0       # start on this camera
    python sim\\view_motion.py --traj <path\\to\\trajectory.npz>
    python sim\\view_motion.py --headless --once       # sanity check, no window
"""
from __future__ import annotations

import argparse
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

    def play_once():
        q_prev = q_frames[0]
        arm.set_joint_positions(q_prev)
        arm.set_joint_position_targets(q_prev)
        for _ in range(10):
            simulation_app.update()
        for q_target in q_frames[1:]:
            for s in range(args.steps_per_waypoint):
                t = (s + 1) / args.steps_per_waypoint
                q_cmd = smooth(q_prev, q_target, t)
                arm.set_joint_position_targets(q_cmd)
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

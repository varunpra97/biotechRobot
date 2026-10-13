"""Isaac Sim pick-and-place: Franka picks up a cube and moves it.

Joint targets are precomputed with validated MuJoCo IK (see scripts), then
played back as smooth joint-space trajectories. Uses the local Menagerie
Franka USD -- no Nucleus server needed.

Usage (Isaac venv):
    .venv-isaac\\Scripts\\Activate.ps1
    cd biotech-lab-robotics
    python sim\\isaac_pick_place.py            # GUI, loops until window closed
    python sim\\isaac_pick_place.py --headless # no window, runs once
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Keyframes: 7 arm joints + 2 gripper fingers. Solved offline with MuJoCo DLS IK
# for hand targets: above [0.45,0,0.25], grasp [0.45,0,0.08], lift [0.45,0,0.3],
# place [0.0,0.45,0.25]. Gripper: open=0.04, closed=0.0.
Q_HOME = np.array([0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04])
Q_ABOVE = np.array([0.008, 0.047, -0.008, -2.627, 0.001, 2.857, 0.741, 0.04, 0.04])
Q_GRASP = np.array([0.010, 0.531, -0.008, -2.421, 0.001, 2.755, 0.741, 0.04, 0.04])
Q_HOLD = np.array([0.010, 0.531, -0.008, -2.421, 0.001, 2.755, 0.741, 0.0, 0.0])
Q_LIFT = np.array([0.007, -0.077, -0.007, -2.656, 0.001, 2.903, 0.741, 0.0, 0.0])
Q_PLACE = np.array([0.331, 0.135, 1.171, -2.587, -0.590, 2.743, 0.741, 0.0, 0.0])
Q_RELEASE = np.array([0.331, 0.135, 1.171, -2.587, -0.590, 2.743, 0.741, 0.04, 0.04])

# (keyframes, steps): move above, descend, close, lift, transit, release, retract
SEQUENCE = [
    (Q_ABOVE, 60),
    (Q_GRASP, 40),
    (Q_HOLD, 20),
    (Q_LIFT, 40),
    (Q_PLACE, 90),
    (Q_RELEASE, 20),
    (Q_HOME, 60),
]
PHASE_NAMES = ["move above cube", "descend", "grasp", "lift", "move to target",
               "release", "retract"]


def smooth(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    s = 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))
    return a + (b - a) * s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    # SimulationApp MUST be instantiated before any other isaacsim import.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    from isaacsim.core.experimental.objects import Cube, DomeLight, GroundPlane
    from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage

    base = Path(r"C:\Users\varun\workspace\BiotechRobot")
    franka_usd = str(base / "biotech-lab-robotics" / "assets" / "panda" / "panda.usda")

    GroundPlane("/World/ground_plane")
    DomeLight("/World/DomeLight").set_intensities(1000)
    add_reference_to_stage(usd_path=franka_usd, prim_path="/World/Franka")

    # Fix: MJCF conversion leaves finger drives with zero stiffness.
    # Set prismatic finger drives so set_dof_position_targets works.
    from pxr import UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if "finger_joint" in str(prim.GetPath()):
            drive = UsdPhysics.DriveAPI.Apply(prim, "linear")
            drive.GetStiffnessAttr().Set(2000.0)
            drive.GetDampingAttr().Set(200.0)

    cube_init = np.array([0.45, 0.0, 0.0258])
    cube_shape = Cube(
        paths="/World/Cube",
        positions=cube_init,
        orientations=np.array([1, 0, 0, 0]),
        sizes=1.0,
        scales=np.array([0.0515, 0.0515, 0.0515]),
        colors="blue",
    )
    GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
    cube = RigidPrim(paths=cube_shape.paths)

    import omni.timeline

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()
    simulation_app.update()

    robot = Articulation("/World/Franka")
    robot.set_dof_positions(Q_HOME)
    robot.set_dof_position_targets(Q_HOME)

    def run_once() -> float:
        q_cur = Q_HOME.copy()
        for (q_goal, n_steps), name in zip(SEQUENCE, PHASE_NAMES):
            print(f"phase: {name}")
            for i in range(n_steps):
                q_cmd = smooth(q_cur, q_goal, (i + 1) / n_steps)
                robot.set_dof_position_targets(q_cmd)
                simulation_app.update()
            q_cur = q_goal.copy()
        # settle, then measure
        for _ in range(30):
            simulation_app.update()
        final = cube.get_world_poses()[0].numpy()[0]
        return float(np.linalg.norm(final[:2] - np.array([0.0, 0.45])))

    if args.headless:
        err = run_once()
        print(f"final cube error to target: {err:.3f} m")
        print("SIM OK" if err < 0.05 else "SIM WEAK")
    else:
        while simulation_app.is_running():
            run_once()
            print("--- restarting ---")
            cube.set_world_poses(positions=cube_init, orientations=np.array([1, 0, 0, 0]))
            robot.set_dof_positions(Q_HOME)
            robot.set_dof_position_targets(Q_HOME)
    simulation_app.close()


if __name__ == "__main__":
    main()

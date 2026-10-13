"""Isaac Sim: 50 parallel Franka pick-and-place instances, randomized starts.

Each of the 50 robots starts from a different random joint configuration
(and random gripper state), then all execute the same pick-and-place
keyframe sequence simultaneously. Joint targets were solved offline with
validated MuJoCo IK.

Usage (Isaac venv):
    .venv-isaac\\Scripts\\Activate.ps1
    cd biotech-lab-robotics
    python sim\\isaac_pick_place_50.py            # GUI
    python sim\\isaac_pick_place_50.py --headless # no window
    python sim\\isaac_pick_place_50.py --n 20     # fewer instances
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Keyframes: 7 arm joints + 2 gripper fingers (MuJoCo-IK solved).
Q_HOME = np.array([0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04])
Q_ABOVE = np.array([0.008, 0.047, -0.008, -2.627, 0.001, 2.857, 0.741, 0.04, 0.04])
Q_GRASP = np.array([0.010, 0.531, -0.008, -2.421, 0.001, 2.755, 0.741, 0.04, 0.04])
Q_HOLD = np.array([0.010, 0.531, -0.008, -2.421, 0.001, 2.755, 0.741, 0.0, 0.0])
Q_LIFT = np.array([0.007, -0.077, -0.007, -2.656, 0.001, 2.903, 0.741, 0.0, 0.0])
Q_PLACE = np.array([0.331, 0.135, 1.171, -2.587, -0.590, 2.743, 0.741, 0.0, 0.0])
Q_RELEASE = np.array([0.331, 0.135, 1.171, -2.587, -0.590, 2.743, 0.741, 0.04, 0.04])

SEQUENCE = [
    (Q_ABOVE, 100, "move above cube"),
    (Q_GRASP, 40, "descend"),
    (Q_HOLD, 20, "grasp"),
    (Q_LIFT, 40, "lift"),
    (Q_PLACE, 90, "move to target"),
    (Q_RELEASE, 20, "release"),
    (Q_HOME, 60, "retract"),
]

# Conservative joint sampling ranges (radians), within Franka limits.
J_RANGES = np.array([
    [-2.5, 2.5],   # j1
    [-1.5, 1.5],   # j2
    [-2.5, 2.5],   # j3
    [-2.9, -0.2],  # j4
    [-2.5, 2.5],   # j5
    [0.2, 3.4],    # j6
    [-2.5, 2.5],   # j7
])


def smooth(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    s = 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))
    return a + (b - a) * s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--same-init", action="store_true",
                    help="all robots start from Q_HOME (debug)")
    args = ap.parse_args()

    # SimulationApp MUST be instantiated before any other isaacsim import.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.objects import Cube, DomeLight, GroundPlane
    from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, UsdGeom

    def set_prim_translate(prim_path: str, pos: np.ndarray):
        prim = stage_utils.get_current_stage().GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    rng = np.random.default_rng(args.seed)
    n = args.n
    cols = int(np.ceil(np.sqrt(n)))
    spacing = 2.5

    base = Path(r"C:\Users\varun\workspace\BiotechRobot")
    franka_usd = str(base / "biotech-lab-robotics" / "assets" / "panda" / "panda.usda")

    GroundPlane("/World/ground_plane")
    DomeLight("/World/DomeLight").set_intensities(1000)

    robots: list[Articulation] = []
    cubes: list[RigidPrim] = []
    q_inits: list[np.ndarray] = []

    for i in range(n):
        row, col = divmod(i, cols)
        bx, by = col * spacing, row * spacing
        rpath = f"/World/Franka_{i}"
        add_reference_to_stage(usd_path=franka_usd, prim_path=rpath)
        set_prim_translate(rpath, np.array([bx, by, 0.0]))

        # Random initial configuration: arm joints uniform in range, gripper random.
        q0 = np.empty(9)
        if args.same_init:
            q0[:] = Q_HOME
        else:
            q0[:7] = rng.uniform(J_RANGES[:, 0], J_RANGES[:, 1])
            q0[7:] = rng.uniform(0.0, 0.04, size=2)
        q_inits.append(q0)

        # Cube at fixed relative offset from this robot's base.
        cpath = f"/World/Cube_{i}"
        cube_shape = Cube(
            paths=cpath,
            positions=np.array([bx + 0.45, by, 0.0258]),
            orientations=np.array([1, 0, 0, 0]),
            sizes=1.0,
            scales=np.array([0.0515, 0.0515, 0.0515]),
            colors="blue",
        )
        GeomPrim(paths=cpath, apply_collision_apis=True)
        cubes.append(RigidPrim(paths=cpath))

    import omni.timeline

    # Initialize all articulations BEFORE playing (avoids interpenetration at origin).
    for i in range(n):
        robot = Articulation(f"/World/Franka_{i}")
        robots.append(robot)

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()
    simulation_app.update()

    for robot, q0 in zip(robots, q_inits):
        robot.set_dof_positions(q0)
        robot.set_dof_position_targets(q0)

    print(f"{n} robots initialized with random configurations")

    def run_once() -> list[float]:
        q_curs = [q.copy() for q in q_inits]
        for q_goal, n_steps, name in SEQUENCE:
            for s in range(n_steps):
                t = (s + 1) / n_steps
                for r, qc in zip(robots, q_curs):
                    r.set_dof_position_targets(smooth(qc, q_goal, t))
                simulation_app.update()
            q_curs = [q_goal.copy() for _ in robots]
        for _ in range(30):
            simulation_app.update()
        errs = []
        for i, cube in enumerate(cubes):
            row, col = divmod(i, cols)
            tx, ty = col * spacing + 0.0, row * spacing + 0.45
            f = cube.get_world_poses()[0].numpy()[0]
            errs.append(float(np.linalg.norm(f[:2] - np.array([tx, ty]))))
        return errs

    if args.headless:
        errs = run_once()
        ok = sum(e < 0.05 for e in errs)
        print(f"done: {ok}/{n} within 5cm, mean err={np.mean(errs):.3f}m")
        print("errs:", [f"{e:.3f}" for e in errs])
    else:
        while simulation_app.is_running():
            errs = run_once()
            ok = sum(e < 0.05 for e in errs)
            print(f"cycle done: {ok}/{n} within 5cm")
            for i, (robot, cube) in enumerate(zip(robots, cubes)):
                row, col = divmod(i, cols)
                bx, by = col * spacing, row * spacing
                cube.set_world_poses(
                    positions=np.array([bx + 0.45, by, 0.0258]),
                    orientations=np.array([1, 0, 0, 0]),
                )
                robot.set_dof_positions(q_inits[i])
                robot.set_dof_position_targets(q_inits[i])
    simulation_app.close()


if __name__ == "__main__":
    main()

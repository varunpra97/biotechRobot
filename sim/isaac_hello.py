"""Isaac Sim hello-world: Franka Panda (from local MuJoCo Menagerie MJCF).

Usage (Isaac venv):
    .venv-isaac\\Scripts\\Activate.ps1
    python sim\\isaac_hello.py            # opens the GUI
    python sim\\isaac_hello.py --headless # no window (for remote/SSH runs)
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--steps", type=int, default=240)
    args = ap.parse_args()

    # SimulationApp MUST be instantiated before any other isaacsim import.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage

    base = Path(r"C:\Users\varun\workspace\BiotechRobot")
    mjcf_path = str(base / "mujoco_menagerie-main" / "franka_emika_panda" / "panda.xml")
    usd_dir = str(base / "biotech-lab-robotics" / "assets")

    importer = MJCFImporter()
    importer.config = MJCFImporterConfig(mjcf_path=mjcf_path, usd_path=usd_dir)
    out = importer.import_mjcf()
    print("converted:", out)

    world = World()
    world.scene.add_default_ground_plane()
    add_reference_to_stage(usd_path=out, prim_path="/World/Franka")

    world.reset()
    if args.headless:
        for _ in range(args.steps):
            world.step(render=True)
        print(f"SIM OK: {args.steps} steps, headless=True")
    else:
        print("GUI open — close the window to stop.")
        while simulation_app.is_running():
            world.step(render=True)
    simulation_app.close()


if __name__ == "__main__":
    main()

# Biotech Lab Robotics — Lab Setup (Isaac Sim)

A static Isaac Sim scene of a biotech lab workstation: one open test-tube
platform and one Franka arm on a pedestal facing it. The arm does not move
yet — this stage is just the scene setup and multi-camera viewing/capture
tooling that later stages (perception, policy) will build on.

## Scene

- **Room**: 10m x 6m x 3m, floor + walls, DomeLight for ambient fill.
- **Platform**: a single open bench (1.4m x 0.7m, no enclosure) holding
  a 3x4 test tube rack, a sample tray with wells, and 2 pipettes, plus an
  overhead work light.
- **Arm**: one Franka Panda on a pedestal facing the platform, base height
  matched to the platform so it can reach everything on it. Held static in
  its home pose — no motion.
- **Cameras** (5 total):
  - `arm_cam0`, `arm_cam1` — behind the robot arm, with the arm and the
    tube platform both in view
  - `platform_cam0`, `platform_cam1` — behind the tube platform, looking
    back at the arm
  - `wrist` — mounted above the arm's wrist, looking down at the platform

## Layout

- `sim/lab_scene.py` — builds the scene; also the GUI viewer and capture tool
- `assets/panda/` — Franka Panda, converted from MuJoCo Menagerie MJCF to
  USD (generated, gitignored)
- `setup_windows.ps1` / `WINDOWS_SETUP.md` — RTX 3080 Ti box setup

## How to run

```powershell
cd C:\Users\varun\workspace\BiotechRobot\biotech-lab-robotics
C:\Users\varun\workspace\BiotechRobot\.venv-isaac\Scripts\Activate.ps1

# View the scene from a specific camera (GUI)
python sim\lab_scene.py --camera arm_cam0
python sim\lab_scene.py --camera arm_cam1
python sim\lab_scene.py --camera platform_cam0
python sim\lab_scene.py --camera platform_cam1
python sim\lab_scene.py --camera wrist

# List available cameras
python sim\lab_scene.py --list-cameras

# Capture a PNG from every camera at once (no window)
python sim\lab_scene.py --capture --out ..\lab_captures

# Headless sanity check (build the scene, no window, no capture)
python sim\lab_scene.py --headless
```

## Status (2026-10-16)

- [x] Open test-tube platform (rack, tray, pipettes) + Franka on a pedestal
- [x] Room, lighting, 5-camera layout
- [x] Static scene (arm held in home pose, no motion)
- [x] Per-camera GUI viewing (`--camera`) and all-camera capture (`--capture`)
- [ ] Arm motion / pick-and-place on this layout
- [ ] Perception, policy work (not started on this scene yet)

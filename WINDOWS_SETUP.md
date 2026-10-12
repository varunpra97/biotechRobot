# Windows setup — RTX 3080 Ti box

Target folder: `C:\Users\varun\workspace\BiotechRobot`

## 1. Software to install (manual downloads)

| # | Software | Where | Notes |
|---|----------|-------|-------|
| 1 | Python 3.12, 3.13, or 3.14, 64-bit | python.org/downloads | ✅ Check **"Add python.exe to PATH"** during install |
| 2 | NVIDIA GPU driver (R545+) | GeForce Experience or nvidia.com/drivers | RTX 3080 Ti; needed for CUDA. No separate CUDA Toolkit required — PyTorch and Warp ship their own CUDA runtime |
| 3 | Git for Windows *(optional)* | git-scm.com | Only if you want version control later |

That's it. Everything else (PyTorch, MuJoCo, Newton, etc.) installs via pip
in step 3. No Visual Studio / C++ build tools needed — all packages are
prebuilt wheels.

## 2. Unzip the project

Unzip `biotech-lab-robotics.zip` so the files land directly in
`C:\Users\varun\workspace\BiotechRobot` (you should see `setup_windows.ps1`
in that folder, not nested one level deeper).

## 3. One-click environment setup

Open PowerShell **in that folder** and run:

```powershell
powershell -ExecutionPolicy Bypass -File setup_windows.ps1
```

This verifies the GPU, creates `.venv`, installs PyTorch (CUDA 12.1),
MuJoCo, Newton/Warp, and runs validation: CUDA visible → MuJoCo arm sim →
camera render test.

## 4. Generate training data + train

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
# keypose-annotated demos (headless OK)
python sim/make_demos.py --episodes 200 --out data/demos
# rendered episodes for the perception net (needs GPU)
python sim/camera_sim.py --episodes 200 --out data/renders
# train perception: images -> (tray, goal) poses
python perception/train_pose_net.py --data data/renders
# train diffusion models
python policies/train_keypose.py --data data/demos
python policies/train_traj.py --data data/demos
```

`train_*.py` are thin training loops to be completed on this box —
`policies/diffusion.py` (models) and `sim/make_demos.py` (data) are done
and validated.

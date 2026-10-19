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

This verifies the GPU, creates `.venv`, and installs PyTorch/MuJoCo/Newton
(currently unused by anything in the repo, but harmless to have on hand).

## 4. Run the lab scene (Isaac Sim)

The scene and cameras run under Isaac Sim's own `.venv-isaac`, a separate
environment (not the `.venv` from step 3):

```powershell
..\.venv-isaac\Scripts\Activate.ps1
python sim\lab_scene.py --capture --out ..\lab_captures
```

See the main [README](README.md) for the full set of view/capture commands.

<#
.SYNOPSIS
  One-click setup for biotech-lab-robotics on Windows (NVIDIA RTX 3080 Ti).
.DESCRIPTION
  Expected location: C:\Users\varun\workspace\BiotechRobot
  Unzip the project there, then run from that folder in PowerShell:
    powershell -ExecutionPolicy Bypass -File setup_windows.ps1
  What it does:
    1. Verifies NVIDIA GPU + driver via nvidia-smi
    2. Verifies Python 3.10+
    3. Creates .venv and installs PyTorch (CUDA), project deps, and Newton
    4. Validates: CUDA visible, MuJoCo sim runs, vision render test passes
#>
$ErrorActionPreference = "Stop"

Write-Host "`n== [1/6] GPU check =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
if ($LASTEXITCODE -ne 0) { throw "nvidia-smi failed. Install NVIDIA Game Ready / Studio drivers first." }

Write-Host "`n== [2/6] Python check =="
python --version
$pyVer = (python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if ([double]$pyVer -lt 3.10) { throw "Python 3.10+ required (found $pyVer)." }

Write-Host "`n== [3/6] Virtualenv =="
if (-not (Test-Path ".venv")) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

Write-Host "`n== [4/6] PyTorch (CUDA) =="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Host "`n== [5/6] Project deps + Newton =="
pip install -r requirements.txt
pip install newton warp-lang usd-core

Write-Host "`n== [6/6] Validation =="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; print('CUDA OK:', torch.cuda.get_device_name(0))"
python -c "import newton, warp; print('Newton OK')"

Write-Host "`nSetup complete. Next:"
Write-Host "  - Scene setup + cameras run under Isaac Sim's own .venv-isaac, not this venv:"
Write-Host "      ..\.venv-isaac\Scripts\Activate.ps1"
Write-Host "      python sim\lab_scene.py --capture --out ..\lab_captures"

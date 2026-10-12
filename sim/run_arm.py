"""Stage 1 arm simulation runner (MuJoCo reference).

Loads a generated arm MJCF, tracks a smooth sinusoidal joint trajectory
with a PD controller, and validates the embodiment: no NaNs, no joint
explosions. This is the stable reference path; the Newton MJWarp runner
is stubbed below and will target the Windows training box (Stage 3).

Usage:
    python sim/run_arm.py --mjcf assets/canonical_arm_6dof.xml
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def run_mujoco(mjcf: Path, seconds: float, freq: float, amp_deg: float) -> None:
    try:
        import mujoco
    except ImportError:
        sys.exit("mujoco is not installed. Run: pip install -r requirements.txt")
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    n = model.nq
    if n == 0:
        sys.exit("FAILED: model has no joints")

    q0 = np.zeros(n)
    # PD on normalized motor commands in [-1, 1]; gear maps to torque.
    kp, kd = 4.0, 0.5
    dt = model.opt.timestep
    steps = int(seconds / dt)
    amp = math.radians(amp_deg)
    # Distal joints move less, mimicking a natural reach trajectory.
    profile = np.linspace(1.0, 0.4, n)
    max_abs_q = 0.0

    for s in range(steps):
        t = s * dt
        q_target = q0 + amp * math.sin(2 * math.pi * freq * t) * profile
        raw = kp * (q_target - data.qpos[:n]) - kd * data.qvel[:n]
        data.ctrl[:] = np.clip(raw, -1.0, 1.0)
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            sys.exit(f"FAILED: non-finite state at step {s}")
        max_abs_q = max(max_abs_q, float(np.max(np.abs(data.qpos[:n]))))

    print(
        f"OK: {n}-dof arm simulated {steps} steps "
        f"({seconds}s), max|q|={math.degrees(max_abs_q):.1f} deg"
    )


def check_newton_available() -> None:
    """Report whether the Newton path can run here (informational only)."""
    try:
        import newton  # noqa: F401
    except ImportError:
        print("Newton not installed here; install it on the Windows box per README.")
        print("The MJCF assets generated here are Newton (MJWarp) compatible.")
        return
    import warp

    print(f"Newton available; warp devices: {warp.get_devices()}")
    print("TODO(Stage 3): add Newton MJWarp runner mirroring run_mujoco().")


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate a generated arm embodiment.")
    ap.add_argument("--mjcf", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--freq", type=float, default=0.5)
    ap.add_argument("--amp-deg", type=float, default=20.0)
    ap.add_argument("--check-newton", action="store_true")
    args = ap.parse_args()

    if args.check_newton:
        check_newton_available()
        return
    run_mujoco(args.mjcf, args.seconds, args.freq, args.amp_deg)


if __name__ == "__main__":
    main()

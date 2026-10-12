"""Procedural serial-arm embodiment generator.

Generates MuJoCo MJCF models for parameterized serial arms (5/6/7 DOF).
MJCF is the shared asset format for this project: it loads in MuJoCo
(Stage 1 stable reference simulator) and in NVIDIA Newton's MuJoCo-Warp
solver path (Stage 3 training target).

Each generated model is accompanied by an embodiment metadata JSON file
used later for morphology-conditioned policies.
"""
from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Joint axis pattern for up to 7 DOF: yaw/pitch/roll arrangement typical of
# small articulated arms.
DEFAULT_AXES_7 = [
    "0 0 1",  # j1 shoulder yaw
    "0 1 0",  # j2 shoulder pitch
    "0 1 0",  # j3 elbow pitch
    "1 0 0",  # j4 wrist roll
    "0 1 0",  # j5 wrist pitch
    "1 0 0",  # j6 wrist roll
    "0 0 1",  # j7 extra yaw (7-dof arms)
]

DEFAULT_RANGES_DEG_7 = [
    (-170, 170),
    (-120, 120),
    (-150, 150),
    (-170, 170),
    (-120, 120),
    (-170, 170),
    (-170, 170),
]

CANONICAL_LENGTHS_6 = [0.28, 0.24, 0.22, 0.12, 0.10, 0.09]


@dataclass
class ArmParams:
    name: str = "proc_arm"
    dof: int = 6
    link_lengths: list = field(default_factory=lambda: list(CANONICAL_LENGTHS_6))
    base_height: float = 0.15
    link_radius: float = 0.035
    tool_length: float = 0.12  # rigid extension beyond the flange (pre-grasp tool)
    tool_radius: float = 0.02
    base_radius: float = 0.09
    pedestal_height: float = 0.35  # rigid mount raising the arm above the bench
    joint_axes: list = field(default_factory=lambda: list(DEFAULT_AXES_7))
    joint_ranges_deg: list = field(
        default_factory=lambda: [list(r) for r in DEFAULT_RANGES_DEG_7]
    )
    gear: float = 160.0
    damping: float = 2.0
    timestep: float = 0.002

    def validate(self) -> None:
        if self.dof not in (5, 6, 7):
            raise ValueError(f"dof must be 5, 6, or 7 (got {self.dof})")
        if len(self.link_lengths) != self.dof:
            raise ValueError("link_lengths must have exactly dof entries")
        if len(self.joint_axes) < self.dof or len(self.joint_ranges_deg) < self.dof:
            raise ValueError("joint_axes / joint_ranges_deg must cover dof")
        if any(L <= 0.02 for L in self.link_lengths):
            raise ValueError("link lengths must be > 0.02 m")
        if self.base_height <= 0.05:
            raise ValueError("base_height must clear the floor")
        for lo, hi in self.joint_ranges_deg[: self.dof]:
            if not lo < hi:
                raise ValueError(f"bad joint range {(lo, hi)}")


def generate_arm_mjcf(p: ArmParams) -> str:
    """Build an MJCF string for the parameterized arm."""
    p.validate()
    mujoco = ET.Element("mujoco", model=p.name)
    ET.SubElement(mujoco, "compiler", angle="degree")
    ET.SubElement(mujoco, "option", timestep=str(p.timestep), gravity="0 0 -9.81")

    default = ET.SubElement(mujoco, "default")
    ET.SubElement(default, "joint", damping=str(p.damping))
    ET.SubElement(default, "geom", density="1000", friction="0.8 0.1 0.1")

    worldbody = ET.SubElement(mujoco, "worldbody")
    ET.SubElement(
        worldbody, "geom", name="floor", type="plane", size="5 5 0.1",
    )

    # Pedestal mount (floor -> arm base), so the shoulder sits above the
    # bench work surface and the elbow clears it during tray handling.
    ET.SubElement(
        worldbody, "geom", name="pedestal", type="cylinder",
        pos=f"0 0 {p.pedestal_height / 2}",
        size=f"{p.base_radius} {p.pedestal_height / 2}",
    )
    base = ET.SubElement(worldbody, "body", name="base",
                         pos=f"0 0 {p.pedestal_height}")
    ET.SubElement(
        base, "geom", name="base_geom", type="cylinder",
        pos=f"0 0 {p.base_height / 2}",
        size=f"{p.base_radius} {p.base_height / 2}",
    )

    parent = base
    parent_offset = p.base_height
    for i in range(p.dof):
        L = p.link_lengths[i]
        body = ET.SubElement(
            parent, "body", name=f"link{i + 1}", pos=f"0 0 {parent_offset}"
        )
        lo, hi = p.joint_ranges_deg[i]
        ET.SubElement(
            body, "joint", name=f"j{i + 1}", type="hinge",
            axis=p.joint_axes[i], range=f"{lo} {hi}",
        )
        ET.SubElement(
            body, "geom", name=f"link{i + 1}_geom", type="capsule",
            fromto=f"0 0 0 0 0 {L}", size=str(p.link_radius),
        )
        if i == p.dof - 1:
            # Rigid tool extending beyond the flange; the control point
            # ("ee") is the tool tip, as on a real gripper.
            tool = ET.SubElement(body, "body", name="tool", pos=f"0 0 {L}")
            ET.SubElement(
                tool, "geom", name="tool_geom", type="capsule",
                fromto=f"0 0 0 0 0 {p.tool_length}",
                size=str(p.tool_radius),
            )
            ET.SubElement(tool, "site", name="ee", pos=f"0 0 {p.tool_length}")
        parent = body
        parent_offset = L

    actuator = ET.SubElement(mujoco, "actuator")
    for i in range(p.dof):
        ET.SubElement(
            actuator, "motor", name=f"m{i + 1}", joint=f"j{i + 1}",
            gear=str(p.gear), ctrlrange="-1 1",
        )

    return '<?xml version="1.0"?>\n' + ET.tostring(mujoco, encoding="unicode")


def embodiment_metadata(p: ArmParams) -> dict:
    """Morphology descriptor consumed by cross-embodiment policies."""
    return {
        "family": "serial_arm",
        "name": p.name,
        "dof": p.dof,
        "link_lengths": list(p.link_lengths[: p.dof]),
        "joint_axes": list(p.joint_axes[: p.dof]),
        "joint_ranges_deg": [list(r) for r in p.joint_ranges_deg[: p.dof]],
        "base_height": p.base_height,
        "link_radius": p.link_radius,
        "gear": p.gear,
        "damping": p.damping,
        "timestep": p.timestep,
    }


def sample_arm_params(rng: random.Random, name: str) -> ArmParams:
    """Sample one arm from the embodiment distribution (wide variety)."""
    dof = rng.choice([5, 6, 7])
    base = CANONICAL_LENGTHS_6[:dof]
    if dof == 7:
        base = base + [0.07]
    if dof == 5:
        base = base[:5]
    lengths = [round(L * rng.uniform(0.8, 1.3), 3) for L in base]
    return ArmParams(
        name=name,
        dof=dof,
        link_lengths=lengths,
        base_height=round(rng.uniform(0.10, 0.22), 3),
        link_radius=round(rng.uniform(0.04, 0.07), 3),
        gear=round(rng.uniform(40.0, 120.0), 1),
        damping=round(rng.uniform(1.0, 4.0), 2),
    )


def params_from_config(path: Path) -> ArmParams:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    return ArmParams(
        name=cfg.get("name", "proc_arm"),
        dof=cfg.get("dof", 6),
        link_lengths=cfg.get("link_lengths", list(CANONICAL_LENGTHS_6)),
        base_height=cfg.get("base_height", 0.15),
        link_radius=cfg.get("link_radius", 0.055),
        base_radius=cfg.get("base_radius", 0.09),
        gear=cfg.get("gear", 80.0),
        damping=cfg.get("damping", 2.0),
        timestep=cfg.get("timestep", 0.002),
    )


def write_embodiment(p: ArmParams, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / f"{p.name}.xml"
    meta_path = out_dir / f"{p.name}.embodiment.json"
    xml_path.write_text(generate_arm_mjcf(p))
    # Validate well-formedness before writing metadata.
    ET.fromstring(xml_path.read_text().split("?>\n", 1)[1])
    meta_path.write_text(json.dumps(embodiment_metadata(p), indent=2))
    return xml_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate procedural arm MJCF embodiments.")
    ap.add_argument("--config", type=Path, help="YAML config for one canonical arm")
    ap.add_argument("--sample", action="store_true", help="sample N random arms")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.config:
        p = params_from_config(args.config)
        path = write_embodiment(p, args.out)
        print(f"wrote {path}")
    if args.sample:
        rng = random.Random(args.seed)
        for k in range(args.count):
            p = sample_arm_params(rng, f"sampled_arm_{args.seed}_{k}")
            path = write_embodiment(p, args.out)
            print(f"wrote {path} (dof={p.dof})")


if __name__ == "__main__":
    main()

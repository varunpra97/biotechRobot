"""A multi-view capture session: which image goes with which camera's
intrinsics. Camera *pose* is not part of the manifest -- it's solved per
image by pose_solver.py from the fiducial markers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Shot:
    image_path: Path
    camera_id: str


@dataclass
class CaptureManifest:
    shots: list[Shot]

    @staticmethod
    def load(path: str | Path) -> "CaptureManifest":
        """Load from JSON: {"shots": [{"image": "...", "camera_id": "..."}]}.
        Relative image paths resolve against the manifest file's directory.
        """
        manifest_path = Path(path)
        data = json.loads(manifest_path.read_text())
        base = manifest_path.parent
        shots = [
            Shot(image_path=(base / s["image"]).resolve(),
                 camera_id=s["camera_id"])
            for s in data["shots"]
        ]
        return CaptureManifest(shots=shots)

    @staticmethod
    def from_directory(
        directory: str | Path, camera_id: str = "default"
    ) -> "CaptureManifest":
        """Convenience for the common case of one camera (e.g. a single
        phone) and a folder of photos: every image in `directory`
        (non-recursive) becomes a shot with the same camera_id.
        """
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = sorted(p for p in Path(directory).iterdir()
                         if p.suffix.lower() in exts)
        return CaptureManifest(
            shots=[Shot(image_path=p, camera_id=camera_id) for p in images])


def resolve_shot_image(capture_dir: str | Path, recorded_path: str) -> Path:
    """The image for a shot of THIS capture: its own images/ folder first.
    Pose files record absolute paths, and after a capture folder is renamed
    a new capture can be written to the old path -- preferring the
    recorded path then silently reads a different capture's images."""
    own = Path(capture_dir) / "images" / Path(recorded_path).name
    return own if own.exists() else Path(recorded_path)

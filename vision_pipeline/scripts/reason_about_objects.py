"""Run LLM scene reasoning (scene_reasoning.py) on every reconstructed object.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/reason_about_objects.py \
      --capture-dir CAPTURE --scene-context biotech-lab-robotics/vision_pipeline/config/scene_context.example.txt [--dry-run]

Needs <capture>/reconstruction/scene.json and camera poses. Writes
<capture>/reasoning/<object id>.json (cached: re-runs skip objects that
already have a result unless --force). --dry-run writes the request text
and image crops instead of calling the API -- review what would be sent
first. Credentials come from the environment (ANTHROPIC_API_KEY, or an
`ant auth login` profile); this script never takes a key as an argument.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from calibration import load_intrinsics
from capture import resolve_shot_image
from geometry import camera_pose_to_rvec_tvec, quaternion_to_rotation_matrix
from scene_reasoning import SYSTEM_PROMPT, analyze_object, build_user_content

MAX_IMAGE_SIDE = 1568  # larger images are downscaled by the API anyway


HULL_NOTE = ("visual hull from silhouettes: overestimates size by roughly 5-25 mm and cannot see "
             "cavities (bores, wells, open interiors); treat dimensions as an envelope")
DEPTH_NOTE = ("multi-view depth fusion: surfaces any camera saw are accurate to about 0.5 mm; space no "
              "camera saw (inside a closed box or bottle, deep narrow bores, under an overhang) is "
              "filled in, so the measured volume overstates hollow objects. Touching objects come out "
              "as one object -- that is why the parts list matters")


def object_summary(obj: dict, method: str = "") -> dict:
    d = obj["dims_m"]
    depth = "depth" in method.lower() or "tsdf" in method.lower()
    out = {
        "id": obj["id"],
        "reconstruction": DEPTH_NOTE if depth else HULL_NOTE,
        "outer_dims_mm": {"x": round(d["x"] * 1000, 1), "y": round(d["y"] * 1000, 1),
                          "z_height": round(d["z"] * 1000, 1)},
        "position_on_bench_m": [round(v, 3) for v in obj["placement_m"][:2]],
        "classifier_label_hints": dict(list(obj.get("label_votes", {}).items())[:3]),
    }
    if obj.get("volume_m3"):
        bbox = d["x"] * d["y"] * d["z"]
        out["measured_volume_cm3"] = round(obj["volume_m3"] * 1e6, 1)
        out["volume_over_bounding_box"] = round(obj["volume_m3"] / bbox, 3) if bbox > 0 else None
    return out


def best_crops(cap: Path, obj: dict, poses: list, intrinsics: dict, n: int, margin: float = 0.15):
    lo, hi = np.array(obj["bbox_min_m"]), np.array(obj["bbox_max_m"])
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    scored = []
    for p in poses:
        cam = intrinsics[p["camera_id"]]
        R_cw = quaternion_to_rotation_matrix(np.array(p["quaternion_wxyz"]))
        rvec, tvec = camera_pose_to_rvec_tvec(R_cw, np.array(p["position_world"]))
        px, _ = cv2.projectPoints(corners, rvec, tvec, cam.K, cam.dist)
        px = px.reshape(-1, 2)
        w, h = cam.image_size
        inside = np.mean((px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h))
        if inside < 0.8:
            continue
        (x0, y0), (x1, y1) = px.min(axis=0), px.max(axis=0)
        scored.append(((x1 - x0) * (y1 - y0), p, (x0, y0, x1, y1), (w, h)))
    scored.sort(key=lambda s: -s[0])
    crops = []
    for _, p, (x0, y0, x1, y1), (w, h) in scored[:n]:
        mx, my = margin * (x1 - x0), margin * (y1 - y0)
        x0, y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
        x1, y1 = int(min(w, x1 + mx)), int(min(h, y1 + my))
        img = cv2.imread(str(resolve_shot_image(cap, p["image"])))[y0:y1, x0:x1]
        s = min(1.0, MAX_IMAGE_SIDE / max(img.shape[:2]))
        if s < 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        crops.append((Path(p["image"]).stem, buf.tobytes()))
    return crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--scene-context", required=True, help="text file describing the scene and expected objects")
    ap.add_argument("--poses-subdir", default=None, help="default: poses_ba if present, else poses")
    ap.add_argument("--views-per-object", type=int, default=3)
    ap.add_argument("--only", nargs="+", help="object ids")
    ap.add_argument("--force", action="store_true", help="re-run objects that already have a result")
    ap.add_argument("--dry-run", action="store_true", help="write requests + crops, don't call the API")
    ap.add_argument("--backend", choices=["auto", "api", "claude-cli"], default="auto",
                    help="api: Claude API (needs API credits). claude-cli: headless Claude Code (`claude -p`) on "
                         "your Claude subscription login. auto: api if API credentials exist, else claude-cli")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--recon-subdir", default="reconstruction",
                    help="reconstruction to use: reconstruction (visual hull) or reconstruction_depth")
    args = ap.parse_args()

    cap = Path(args.capture_dir)
    poses_dir = cap / (args.poses_subdir or ("poses_ba" if (cap / "poses_ba").exists() else "poses"))
    if args.poses_subdir is None and (cap / args.recon_subdir / "poses" / "camera_poses.json").exists():
        poses_dir = cap / args.recon_subdir / "poses"          # depth-ICP refined
    poses = json.loads((poses_dir / "camera_poses.json").read_text())["poses"]
    intrinsics = load_intrinsics(cap / "camera_intrinsics.json")
    scene = json.loads((cap / args.recon_subdir / "scene.json").read_text())
    objects, method = scene["objects"], scene.get("method", "")
    scene_context = Path(args.scene_context).read_text(encoding="utf-8").strip()
    # per reconstruction: object ids are only unique within one
    out = cap / "reasoning" if args.recon_subdir == "reconstruction" else cap / args.recon_subdir / "reasoning"
    out.mkdir(exist_ok=True)

    client, cli = None, None
    backend = args.backend
    if backend == "auto":
        import os
        from scene_reasoning import DEFAULT_KEY_FILE
        api_creds = any(os.environ.get(v) for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                                                     "ANTHROPIC_FEDERATION_RULE_ID")) or DEFAULT_KEY_FILE.exists()
        backend = "api" if api_creds else "claude-cli"
    if not args.dry_run:
        if backend == "api":
            from scene_reasoning import make_client
            client = make_client()
        else:
            from scene_reasoning import find_claude_cli
            cli = find_claude_cli()
            if cli is None:
                raise SystemExit("claude CLI not found: install Claude Code or set CLAUDE_CLI to its path")
            print(f"backend: headless Claude Code ({cli})")

    for obj in objects:
        if args.only and obj["id"] not in args.only:
            continue
        result_path = out / f"{obj['id']}.json"
        if result_path.exists() and not args.force and not args.dry_run:
            print(f"{obj['id']}: cached")
            continue
        summary = object_summary(obj, method)
        crops = best_crops(cap, obj, poses, intrinsics, args.views_per_object)
        if args.dry_run:
            (out / "crops").mkdir(exist_ok=True)
            for view, jpg in crops:
                (out / "crops" / f"{obj['id']}_{view}.jpg").write_bytes(jpg)
            content = build_user_content(scene_context, summary, [c for _, c in crops])
            (out / f"{obj['id']}.request.txt").write_text(
                f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER TEXT:\n{content[0]['text']}\n\n"
                f"+ {len(crops)} images: {[v for v, _ in crops]}\n", encoding="utf-8")
            print(f"{obj['id']}: dry run -> {len(crops)} crops, request written")
            continue

        if cli is not None:
            from scene_reasoning import analyze_object_cli
            (out / "crops").mkdir(exist_ok=True)
            paths = []
            for view, jpg in crops:
                pth = out / "crops" / f"{obj['id']}_{view}.jpg"
                pth.write_bytes(jpg)
                paths.append(pth)
            result = analyze_object_cli(cli, scene_context, summary, paths, model=args.model)
            if result["status"] == "not_logged_in":
                raise SystemExit("Claude Code is not logged in: run `claude` once in a terminal and /login with "
                                 "your Claude account, then re-run. Nothing was sent.")
            result["input"] = {**summary, "views": [v for v, _ in crops]}
            result_path.write_text(json.dumps(result, indent=2))
            a = result.get("analysis")
            print(f"{obj['id']}: {result['status']}" + (
                f" -> {a['identity']['name']}; parts: " + ", ".join(
                    f"{q['count']}x {q['name']} ({q['relation']}, {q['material']['name']}, sf {q['solid_fraction']:.2f})"
                    for q in a["decomposition"]["parts"]) if a else f" ({result.get('reason', '')})"), flush=True)
            continue
        import anthropic
        try:
            result = analyze_object(client, scene_context, summary, [c for _, c in crops], model=args.model)
        except anthropic.AuthenticationError:
            raise SystemExit("No valid Anthropic credentials. Create ~/.anthropic/api_key (see README), set ANTHROPIC_API_KEY, "
                             "or run `ant auth login`, then re-run. Nothing was sent.")
        except anthropic.RateLimitError as e:
            raise SystemExit(f"Rate limited ({e.message}); re-run later -- finished objects are cached.")
        except anthropic.APIStatusError as e:
            print(f"{obj['id']}: API error {e.status_code}: {e.message}")
            continue
        except anthropic.APIConnectionError:
            raise SystemExit("Network error reaching the Anthropic API; re-run later.")
        result["input"] = {**summary, "views": [v for v, _ in crops]}
        result_path.write_text(json.dumps(result, indent=2))
        a = result.get("analysis")
        print(f"{obj['id']}: {result['status']}" + (
            f" -> {a['identity']['name']} [{a['shape_family']}], composite={a['decomposition']['is_composite']}, "
            f"material={a['material']['name']}; parts: " + ", ".join(
                f"{q['count']}x {q['name']} ({q['relation']}, {q['material']['name']}, sf {q['solid_fraction']:.2f})"
                for q in a["decomposition"]["parts"]) if a else ""))


if __name__ == "__main__":
    main()

"""One command: a folder of photos (+ depth maps) of a platform -> simulation
assets (USD with mass, material, collision hulls) for every object on it.

  .venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/run_pipeline.py --capture-dir CAPTURE \
      [--scene-context context.txt] [--simulate-sensor tof] [--ground-truth]

CAPTURE must contain:
  images/                 the photos (any names; .png/.jpg)
  depth/                  optional: one depth map per photo, same stem
                          (uint16 PNG in --depth-scale-m units, or float .npy
                          in meters), registered to the photo
  camera_intrinsics.json  {"cameras": {id: {K, dist, image_size}}}
  marker_layout.json      the calibration markers' 3D corners (platform frame)
  platform_dims.json      {"platform_w": .., "platform_d": ..} (or --platform-w/-d)
A manifest.json pairing photos, depth maps and camera ids is written if
absent (one camera: the single entry in camera_intrinsics.json).

Stages (each a script that can also be run on its own):
  1  camera poses from the markers              solve_camera_poses.py
  2  segmentation + labels (GPU, optional)      segment_capture.py
  3  geometry: depth fusion if depth maps exist  reconstruct_depth.py
     (else the visual hull)                     reconstruct_scene.py
  4  silhouette refinement on the photos        refine_freeform.py
  5  identity, parts, materials (LLM,          reason_about_objects.py
     needs Anthropic credentials; else defaults)
  5b split composites into parts, weigh each    decompose_objects.py
  6  USD + collision + mass export              export_assets.py
GPU stages run only while the GPU is free (isaac_validation/run_when_gpu_free.sh).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VP = HERE.parent
# The workspace holding the virtual environments: the nearest ancestor with
# .venv-cad (the pipeline now lives inside biotech-lab-robotics/, the venvs
# one level above it).
ROOT = next((d for d in VP.parents if (d / ".venv-cad").exists()), VP.parent)
PY_CAD = ROOT / ".venv-cad" / "Scripts" / "python.exe"
PY_VISION = ROOT / ".venv-vision" / "Scripts" / "python.exe"
GPU_WRAP = VP / "isaac_validation" / "run_when_gpu_free.sh"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def run(cmd: list, log: Path, gpu: bool = False) -> int:
    cmd = [str(c) for c in cmd]
    if gpu:
        # Git bash via PATH: a bare "bash" from Windows CreateProcess finds
        # System32\bash.exe first -- WSL, where the guard can't see Windows
        # processes and the venv paths don't run.
        import shutil
        cmd = [shutil.which("bash") or "bash", str(GPU_WRAP)] + cmd
    t0 = time.time()
    print(f"  $ {' '.join(Path(c).name if i == 0 else c for i, c in enumerate(cmd))}", flush=True)
    with open(log, "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    print(f"    -> exit {rc} in {time.time() - t0:.0f}s (log: {log.name})", flush=True)
    return rc


def write_manifest(cap: Path, depth_scale_m: float) -> None:
    cams = json.loads((cap / "camera_intrinsics.json").read_text())["cameras"]
    images = sorted(p for p in (cap / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    shots = []
    for img in images:
        cid = img.stem if img.stem in cams else (next(iter(cams)) if len(cams) == 1 else None)
        if cid is None:
            raise SystemExit(f"{img.name}: no camera id (several cameras in camera_intrinsics.json; "
                             f"name each photo after its camera or write manifest.json)")
        shot = {"image": f"images/{img.name}", "camera_id": cid}
        for ext in (".png", ".npy", ".npz", ".tif", ".tiff"):
            d = cap / "depth" / f"{img.stem}{ext}"
            if d.exists():
                shot["depth"] = f"depth/{d.name}"
                break
        shots.append(shot)
    manifest = {"shots": shots}
    if any("depth" in s for s in shots):
        manifest["depth"] = {"format": "uint16 png or float .npy (m)", "scale_m": depth_scale_m, "invalid": 0,
                             "kind": "z (distance to the image plane), registered to the colour image"}
    (cap / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest.json: {len(shots)} photos, {sum('depth' in s for s in shots)} with depth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-dir", required=True)
    ap.add_argument("--scene-context", help="text describing the scene and expected objects (for the LLM step)")
    ap.add_argument("--simulate-sensor", help="apply a depth_sensor preset to rendered depth (tests only)")
    ap.add_argument("--depth-scale-m", type=float, default=1e-4, help="uint16 depth PNG units")
    ap.add_argument("--platform-w", type=float)
    ap.add_argument("--platform-d", type=float)
    ap.add_argument("--min-markers", type=int, default=3)
    ap.add_argument("--fine-voxel-mm", type=float, default=1.0)
    ap.add_argument("--skip-segmentation", action="store_true", help="no GPU: objects stay unlabeled")
    ap.add_argument("--skip-refine", action="store_true")
    ap.add_argument("--force-refine", action="store_true")
    ap.add_argument("--refine-above-sigma-mm", type=float, default=0.15,
                    help="silhouette-refine depth objects only when the fused depth noise exceeds this")
    ap.add_argument("--ground-truth", action="store_true", help="synthetic captures: score against the truth")
    ap.add_argument("--hints", help="stand-in parts descriptions for decompose_objects.py (testing without the LLM)")
    args = ap.parse_args()

    cap = Path(args.capture_dir).resolve()
    logs = cap / "pipeline_logs"
    logs.mkdir(exist_ok=True)
    if not (cap / "manifest.json").exists():
        write_manifest(cap, args.depth_scale_m)
    manifest = json.loads((cap / "manifest.json").read_text())
    has_depth = any("depth" in s for s in manifest["shots"])
    has_images = (cap / "images").exists() and any((cap / "images").iterdir())
    dims = json.loads((cap / "platform_dims.json").read_text()) if (cap / "platform_dims.json").exists() else {}
    pw, pd = args.platform_w or dims.get("platform_w"), args.platform_d or dims.get("platform_d")
    if pw is None or pd is None:
        raise SystemExit("platform size unknown: platform_dims.json or --platform-w/-d")
    recon = "reconstruction_depth" if has_depth else "reconstruction"
    print(f"capture: {len(manifest['shots'])} shots, depth: {has_depth}, geometry: {recon}")

    print("[1] camera poses")
    if not (cap / "poses" / "camera_poses.json").exists():
        if run([PY_CAD, HERE / "solve_camera_poses.py", "--manifest", cap / "manifest.json",
                "--intrinsics", cap / "camera_intrinsics.json", "--marker-layout", cap / "marker_layout.json",
                "--out-dir", cap / "poses"], logs / "1_poses.log"):
            raise SystemExit("pose solving failed")
    else:
        print("  poses/ exists, reused")

    print("[2] segmentation + labels")
    if args.skip_segmentation or not has_images:
        print("  skipped")
    elif (cap / "segmentation").exists() and len(list((cap / "segmentation").glob("*.npz"))) >= len(
            json.loads((cap / "poses" / "camera_poses.json").read_text())["poses"]):   # posed views get segmented
        print("  segmentation/ complete, reused")
    else:
        run([PY_VISION, "-u", HERE / "segment_capture.py", "--capture-dir", cap, "--platform-w", pw,
             "--platform-d", pd, "--gpu-guard", VP / "isaac_validation" / "gpu_guard.sh"],
            logs / "2_segmentation.log", gpu=True)

    print("[3] geometry")
    gt_flag = ["--ground-truth"] if args.ground_truth else []
    if has_depth:
        cmd = [PY_CAD, "-W", "ignore", HERE / "reconstruct_depth.py", "--capture-dir", cap, "--platform-w", pw,
               "--platform-d", pd, "--min-markers", args.min_markers, "--fine-voxel-mm", args.fine_voxel_mm] + gt_flag
        if args.simulate_sensor:
            cmd += ["--simulate-sensor", args.simulate_sensor]
    else:
        cmd = [PY_CAD, HERE / "reconstruct_scene.py", "--capture-dir", cap, "--platform-w", pw, "--platform-d", pd,
               "--min-markers", args.min_markers]
    if run(cmd, logs / "3_geometry.log"):
        raise SystemExit("reconstruction failed; see the log")
    scene = json.loads((cap / recon / "scene.json").read_text())
    print(f"  {len(scene['objects'])} objects")

    print("[4] silhouette refinement")
    fused_sigma = scene.get("fused_sigma_m")
    if args.skip_refine or not has_images:
        print("  skipped")
    elif has_depth and fused_sigma is not None and fused_sigma < args.refine_above_sigma_mm / 1000 and not args.force_refine:
        # Good depth is already more precise than image edges: refining a
        # perfect-depth start on 2016px images made it worse (0.15 -> 0.4mm).
        print(f"  skipped: fused depth noise {fused_sigma * 1000:.2f}mm < {args.refine_above_sigma_mm}mm "
              f"(depth is already more precise than the image edges; --force-refine to run anyway)")
    else:
        for o in scene["objects"]:
            run([PY_CAD, "-W", "ignore", HERE / "refine_freeform.py", "--capture-dir", cap, "--object", o["id"],
                 "--recon-subdir", recon, "--min-markers", args.min_markers] + gt_flag,
                logs / f"4_refine_{o['id']}.log")

    print("[5] decomposition / material reasoning (LLM)")
    sys.path.insert(0, str(VP))
    from scene_reasoning import DEFAULT_KEY_FILE
    has_creds = any(os.environ.get(v) for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                                                 "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_PROFILE")) \
        or Path(os.environ.get("ANTHROPIC_API_KEY_FILE", DEFAULT_KEY_FILE)).exists() \
        or any((Path(os.environ.get("ANTHROPIC_CONFIG_DIR") or Path(os.environ.get("APPDATA", Path.home()))
                     / "Anthropic") / "credentials").glob("*.json"))      # `ant auth login` profile
    from scene_reasoning import find_claude_cli
    if not args.scene_context:
        print("  skipped: no --scene-context")
    elif not has_creds and find_claude_cli() is None:
        print(f"  skipped: no Anthropic API credentials and no Claude Code CLI; objects stay whole with the "
              f"default material")
    elif not has_creds:
        print("  no API credentials: using headless Claude Code on your Claude subscription")
        run([PY_CAD, HERE / "reason_about_objects.py", "--capture-dir", cap, "--scene-context", args.scene_context,
             "--recon-subdir", recon, "--backend", "claude-cli"], logs / "5_reasoning.log")
    else:
        run([PY_CAD, HERE / "reason_about_objects.py", "--capture-dir", cap, "--scene-context", args.scene_context,
             "--recon-subdir", recon], logs / "5_reasoning.log")
    if has_depth:
        print("[5b] split composites into parts, weigh each")
        cmd = [PY_CAD, "-W", "ignore", HERE / "decompose_objects.py", "--capture-dir", cap, "--recon-subdir", recon]
        if args.hints:
            cmd += ["--hints", args.hints]
        run(cmd, logs / "5b_decompose.log")

    print("[6] assets")
    if run([PY_CAD, "-W", "ignore", HERE / "export_assets.py", "--capture-dir", cap, "--recon-subdir", recon],
           logs / "6_assets.log"):
        raise SystemExit("asset export failed")

    if args.ground_truth:
        summarize_scores(cap, recon, scene)
        if (cap / "ground_truth_composites.json").exists() and (cap / recon / "parts.json").exists():
            run([PY_CAD, HERE / "score_composites.py", "--capture-dir", cap, "--recon-subdir", recon],
                logs / "7_composite_scores.log")
            print((logs / "7_composite_scores.log").read_text())


def summarize_scores(cap: Path, recon: str, scene: dict) -> None:
    rows = []
    for o in scene["objects"]:
        s = o.get("score", {})
        if not s.get("ground_truth"):
            continue
        ref = cap / recon / "refined" / f"{o['id']}.json"
        r = json.loads(ref.read_text()).get("score_refined") if ref.exists() else None
        rows.append((o["id"], s["ground_truth"] if isinstance(s["ground_truth"], str) else "+".join(s["ground_truth"]),
                     s["dim_err_mm"], s["accuracy_mm"]["median"], (r or {}).get("dim_err_mm")))
    print("\nobject  truth                              fused dims err (mm)      acc med   refined dims err")
    for oid, g, d, a, rd in rows:
        print(f"{oid}  {g[:32]:32s}  {str(d):24s} {a:7.3f}   {rd}")
    (cap / "pipeline_logs" / "scores.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

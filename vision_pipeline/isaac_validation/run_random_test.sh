#!/usr/bin/env bash
# Generalization test, end to end: render the random-object scene (waits for
# the shared GPU, re-checking every 15 min), then run the pipeline on the
# images alone -- poses, segmentation, visual hull -- and compare to the
# exact meshes. Run from the BiotechRobot workspace root.
#   bash biotech-lab-robotics/vision_pipeline/isaac_validation/run_random_test.sh [OUT_DIR]
set -euo pipefail
V=${1:-biotech-lab-robotics/vision_pipeline/isaac_validation/out_random}
G=biotech-lab-robotics/vision_pipeline/isaac_validation/gpu_guard.sh

while bash "$G" >/dev/null; do echo "$(date +%T) GPU busy -- waiting 15 min"; sleep 900; done
if [ ! -f "$V/manifest.json" ]; then
  .venv-isaac/Scripts/python.exe biotech-lab-robotics/vision_pipeline/isaac_validation/render_random_scene.py \
      --objects biotech-lab-robotics/vision_pipeline/isaac_validation/random_objects --out "$V"
fi
.venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/solve_camera_poses.py --manifest "$V/manifest.json" \
    --intrinsics "$V/camera_intrinsics.json" --marker-layout "$V/marker_layout.json" --out-dir "$V/poses"
.venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/isaac_validation/check_camera_positions.py --capture-dir "$V"
.venv-vision/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/segment_capture.py --capture-dir "$V" \
    --platform-w 1.4 --platform-d 0.7 --gpu-guard "$G"
.venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/scripts/reconstruct_scene.py --capture-dir "$V" --platform-w 1.4 --platform-d 0.7
.venv-cad/Scripts/python.exe biotech-lab-robotics/vision_pipeline/isaac_validation/compare_to_ground_truth.py --capture-dir "$V"
echo "DONE"

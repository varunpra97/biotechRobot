#!/usr/bin/env bash
# Run a GPU command only while the GPU is free, re-checking every 15 min.
#   bash biotech-lab-robotics/vision_pipeline/isaac_validation/run_when_gpu_free.sh <command> [args...]
# Waits until gpu_guard.sh reports free, runs the command, and if it exits
# 75 (it found the GPU busy part-way and deferred -- segment_capture.py
# --gpu-guard does this between views) waits and runs it again. Never
# sleeps while holding the GPU: the waiting happens out here, in bash.
G="$(dirname "$0")/gpu_guard.sh"
LOG="$(dirname "$0")/gpu_wait.log"
while true; do
  while bash "$G" > /dev/null; do
    echo "$(date '+%F %T') GPU busy -- waiting 15 min before: $*" >> "$LOG"
    sleep 900
  done
  echo "$(date '+%F %T') GPU free -- running: $*" | tee -a "$LOG"
  "$@"
  rc=$?
  [ $rc -ne 75 ] && { echo "$(date '+%F %T') exit $rc: $*" | tee -a "$LOG"; exit $rc; }
done

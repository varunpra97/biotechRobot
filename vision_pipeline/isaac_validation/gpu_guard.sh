#!/usr/bin/env bash
# Is the GPU in use by someone else? Prints the reason and exits 0 if busy,
# exits 1 if free. Used before every GPU job this pipeline starts (Isaac
# renders, SAM segmentation) so it never competes with the other agent.
#
# Busy if either:
#  - a process that isn't ours looks like GPU work: Isaac (.venv-isaac,
#    isaacsim), SAM/CLIP (.venv-vision), the sim repo (biotech-lab-robotics), or any script running
#    from ANOTHER Claude session's scratchpad. The first version of this
#    check only matched "biotech-lab-robotics" and missed the other agent's
#    Isaac jobs launched from its scratchpad -- a segmentation run started
#    on top of them.
#  - or GPU utilization averages > 40% over a few samples (anything else).
# Not counted: the calling job's own process chain (a SAM run checking
# before each view must not see itself) and CPU-only .venv-cad scripts.
# An earlier version skipped every process mentioning vision_pipeline, so two
# sessions of this pipeline never saw each other's SAM runs (~9GB each).
# The process list comes from gpu_procs.ps1.
#
# Fails SAFE: if either check can't run, it reports busy. (Launched from a
# Python job, Git bash gets a minimal PATH without powershell or nvidia-smi;
# the process check then silently found nothing and only the utilization
# check worked.) Hence the explicit PATH.
# /mnt/c too: a caller using plain "bash" from Windows gets WSL's bash.
# Callers pass GPU_GUARD_SELF_PID (their own PID) so their process chain is
# excluded even when it isn't this script's ancestor.
export PATH="/c/Windows/System32/WindowsPowerShell/v1.0:/c/Windows/System32:/mnt/c/Windows/System32/WindowsPowerShell/v1.0:/mnt/c/Windows/System32:/usr/bin:$PATH"

if ! raw=$(powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(dirname "$0")/gpu_procs.ps1" "${GPU_GUARD_SELF_PID:-0}" 2>&1); then
  echo "busy: could not list processes (assuming busy): $raw" | head -3
  exit 0
fi
procs=$(printf '%s\n' "$raw" | tr -d '\r' | grep -v '^$')
if [ -n "$procs" ]; then
  echo "busy: other GPU-type process(es):"
  echo "$procs" | sed 's/^/  /'
  exit 0
fi

total=0
for i in 1 2 3 4 5; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' \r')
  if ! [[ "$u" =~ ^[0-9]+$ ]]; then
    echo "busy: nvidia-smi unavailable (assuming busy)"
    exit 0
  fi
  total=$((total + u))
  sleep 1
done
avg=$((total / 5))
if [ "$avg" -gt 40 ]; then
  echo "busy: GPU utilization averaging ${avg}% with no recognized owner"
  exit 0
fi
echo "free: GPU utilization ${avg}%"
exit 1

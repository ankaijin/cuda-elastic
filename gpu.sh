#!/usr/bin/env bash
# Prints a property of the GPU the workloads run on, for the build and profile
# scripts. Each GPU's binaries and profiles live under <repo>/<name>/<workload>/
# (e.g. rtx3090/, rtx4090/), so runs on different GPUs never overwrite each
# other; the sources stay shared at the top level.
#   gpu.sh name   e.g. rtx4090  (override with GPU=...)
#   gpu.sh arch   e.g. sm_89    (override with ARCH=...)
# The device is the first one in CUDA_VISIBLE_DEVICES (default 0).
set -euo pipefail

device="${CUDA_VISIBLE_DEVICES:-0}"
device="${device%%,*}"
query() { nvidia-smi --query-gpu="$1" --format=csv,noheader -i "$device"; }

case "${1:-}" in
  name)
    if [[ -n "${GPU:-}" ]]; then printf '%s\n' "$GPU"; exit; fi
    # "NVIDIA GeForce RTX 4090" -> rtx4090
    query name | sed -e 's/^NVIDIA //' -e 's/^GeForce //' -e 's/ //g' | tr '[:upper:]' '[:lower:]'
    ;;
  arch)
    if [[ -n "${ARCH:-}" ]]; then printf '%s\n' "$ARCH"; exit; fi
    # compute capability "8.9" -> sm_89
    printf 'sm_%s\n' "$(query compute_cap | tr -d .)"
    ;;
  *)
    echo "Usage: $0 name|arch" >&2
    exit 2
    ;;
esac

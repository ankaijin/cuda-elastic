#!/usr/bin/env bash
# Runs every ncu profile that results/cuda_performance.xlsx uses, on the
# current GPU, writing to <repo>/<gpu>/<workload>/profiles (see gpu.sh). The
# options match the RTX 3090 runs' ncu command lines.
#
# ncu needs GPU performance-counter access; where that is admin-only
# (ERR_NVGPUCTRPERM), run it with sudo and keep your PATH so ncu is found:
#   sudo env PATH="$PATH" CUDA_VISIBLE_DEVICES=0 ./profile_ncu_all.sh
# Files written under sudo are handed back to the invoking user.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GPU_NAME="$("$ROOT/gpu.sh" name)"
OUT="$ROOT/$GPU_NAME"
STAMP="$(date +%Y%m%d-%H%M%S)"
command -v ncu >/dev/null || { echo "error: ncu not in PATH" >&2; exit 127; }
for bin in llama2c/llama2c_cuda_graph loom_msa/build/msa_optimized_large \
           pagerank/gapbs_pagerank_cuda_graph mlperf-anomaly/build/anomaly_mse \
           mlperf-classification/build/kws_postprocess mlperf-classification/build/vww_preprocess; do
  [[ -x "$OUT/$bin" ]] || { echo "error: $OUT/$bin missing; build it first" >&2; exit 1; }
done
MARK="$(mktemp)"
handback() {
  rm -f "$MARK"
  if [[ -n "${SUDO_USER:-}" ]]; then chown -R "$SUDO_USER": "$OUT"; fi
}
trap handback EXIT

echo "### llama2c (graph and direct)"
# Three iterations of rmsnorm + softmax + matmul_coalesced: 9 launches.
mkdir -p "$OUT/llama2c/profiles"
for mode in graph direct; do
  ncu --set basic --launch-skip 0 --launch-count 9 --replay-mode kernel \
    --graph-profiling node --target-processes application-only \
    --export "$OUT/llama2c/profiles/ncu-large-coalesced-b1024-$mode-$STAMP" --force-overwrite \
    "$OUT/llama2c/llama2c_cuda_graph" "$mode" 5 1024 16384 8192 1 1 copy coalesced
done

# The other workloads in both launch modes, with one shared file suffix.
for mode in graph direct; do
  echo "### loom_msa ($mode)"
  (cd "$ROOT/loom_msa" && LAUNCH=$mode TIMESTAMP=$STAMP ./profile_cuda.sh ncu)

  echo "### pagerank ($mode)"
  PAGERANK_LAUNCH=$mode TIMESTAMP=$STAMP "$ROOT/pagerank/profile_cuda.sh" ncu

  echo "### mlperf-tiny anomaly ($mode)"
  (cd "$ROOT/mlperf-anomaly" && LAUNCH=$mode TIMESTAMP=$STAMP ./profile_cuda.sh ncu)

  echo "### mlperf-tiny KWS and VWW ($mode)"
  (cd "$ROOT/mlperf-classification" &&
     APP=kws LAUNCH=$mode TIMESTAMP=$STAMP ./profile_cuda.sh ncu &&
     APP=vww LAUNCH=$mode TIMESTAMP=$STAMP ./profile_cuda.sh ncu)
done

echo "### done; new ncu reports:"
find "$OUT" -name '*.ncu-rep' -newer "$MARK" | sort

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Binaries and profiles go to <repo>/<gpu>/llama2c/ (see gpu.sh).
GPU_DIR="${GPU_DIR:-$SCRIPT_DIR/../$("$SCRIPT_DIR/../gpu.sh" name)/llama2c}"
# VARIANT=baseline (default) profiles llama2c_cuda_graph.cu; VARIANT=sota
# profiles llama2c_sota.cu, built to <repo>/<gpu>/llama2c/build/llama2c_sota
# (the same binary `make -C llama2c sota` builds). The SOTA issues one kernel
# launch per iteration in nocopy mode, so its ncu default launch count is 1.
VARIANT="${VARIANT:-baseline}"
case "$VARIANT" in
  baseline)
    BIN="${BIN:-$GPU_DIR/llama2c_cuda_graph}"
    SRC="$SCRIPT_DIR/llama2c_cuda_graph.cu"
    NVCC_EXTRA=()
    NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-3}"
    NCU_NAME=llama2c
    NSYS_PREFIX= ;;
  sota)
    BIN="${BIN:-$GPU_DIR/build/llama2c_sota}"
    SRC="$SCRIPT_DIR/llama2c_sota.cu"
    NVCC_EXTRA=(-std=c++17 -lineinfo)
    NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"
    NCU_NAME=llama2c-sota
    NSYS_PREFIX=sota- ;;
  *) printf 'error: VARIANT must be baseline or sota\n' >&2; exit 2 ;;
esac
HEADER="$SCRIPT_DIR/llama2c_common.h"
OUT_DIR="${OUT_DIR:-$GPU_DIR/profiles}"
STAMP="$(date +%Y%m%d-%H%M%S)"
NVCC="${NVCC:-nvcc}"
read -r -a NVCC_FLAGS <<< "${NVCC_FLAGS:--arch=$("$SCRIPT_DIR/../gpu.sh" arch)}"

usage() {
  printf 'Usage: %s [build|run|nsys|nsys-node|ncu|sweep|sweep-fill|plot-sweep|plot-nodes|isolate|all] [args...]\n' "$0"
  printf '  env VARIANT=baseline|sota selects the binary (default baseline)\n'
  printf '  run [graph|direct] [launches|auto] [batch] [size] [rows] [repeats] [warmup_ms] [copy|nocopy] [naive|coalesced] [deterministic|random[=SEED]] [--dump=FILE]\n'
  printf '  nsys-node [graph|direct] [launches|auto] [batch] [size] [rows] [repeats] [warmup_ms] [copy|nocopy] [naive|coalesced]\n'
  printf '  sweep / sweep-fill: env SWEEP_CONFIGS, SWEEP_LAUNCHES, SWEEP_MATMUL, SWEEP_TRIALS, SWEEP_WARMUP_MS\n'
  printf '  plot-sweep <sweep.csv> [options]\n  plot-nodes <direct.sqlite> [graph.sqlite] [options]\n'
}
require_tool() { command -v "$1" >/dev/null || { printf 'error: %s is not installed or not in PATH\n' "$1" >&2; exit 127; }; }
build() { require_tool "$NVCC"; mkdir -p "$(dirname "$BIN")"; "$NVCC" "${NVCC_FLAGS[@]}" "${NVCC_EXTRA[@]}" -O3 -o "$BIN" "$SRC"; }
ensure_bin() { [[ -x "$BIN" && "$BIN" -nt "$SRC" && "$BIN" -nt "$HEADER" ]] || build; }
run() { ensure_bin; "$BIN" "$@"; }
profile_nsys() {
  local granularity="$1"; shift
  local mode="${1:-graph}"
  if (($#)); then shift; fi
  require_tool nsys; ensure_bin; mkdir -p "$OUT_DIR"
  nsys profile --trace=cuda,nvtx --cuda-graph-trace="$granularity" --sample=none \
    --cpuctxsw=none --force-overwrite=true --export=sqlite \
    --output="$OUT_DIR/nsys-${NSYS_PREFIX}${granularity}-${mode}-$STAMP" "$BIN" "$mode" "$@"
}
profile_ncu() {
  local mode="${1:-graph}"
  if (($#)); then shift; fi
  require_tool ncu; ensure_bin; mkdir -p "$OUT_DIR"
  ncu --set basic --launch-skip 0 --launch-count "$NCU_LAUNCH_COUNT" --replay-mode kernel \
    --graph-profiling node --target-processes application-only \
    --export "$OUT_DIR/ncu-${NCU_NAME}-${mode}-$STAMP" --force-overwrite "$BIN" "$mode" "$@"
}
# Shared sweep driver: $1 = CSV name prefix, $2 = default configs,
# $3 = default launch count (a number or "auto"), $4 = default matmul variant.
# Each config is batch,size,rows[,repeats[,copy|nocopy]].
run_sweep() {
  ensure_bin; mkdir -p "$OUT_DIR"
  local name="$1"
  local configs="${SWEEP_CONFIGS:-$2}"
  local launches="${SWEEP_LAUNCHES:-$3}"
  local matmul="${SWEEP_MATMUL:-$4}"
  local trials="${SWEEP_TRIALS:-3}"
  local warmup_ms="${SWEEP_WARMUP_MS:-100}"
  local csv="$OUT_DIR/$name-$STAMP.csv"
  local fields=(gpu_ms_per_iter wall_ms_per_iter submit_us_per_iter graph_capture_ms
                graph_instantiate_ms graph_upload_ms rms_waves softmax_waves matmul_waves
                min_waves achieved_gbps valid)
  local header="trial,mode,launches,batch,size,rows,repeats,copies,matmul,warmup_ms"
  header+=",$(IFS=,; printf '%s' "${fields[*]}"),exit_code"
  printf '%s\n' "$header" > "$csv"
  field() { printf '%s\n' "$line" | tr ' ' '\n' | awk -F= -v key="$1" '$1 == key { print $2 }'; }
  for config in $configs; do
    IFS=, read -r batch size rows repeats copies <<< "$config"
    repeats="${repeats:-1}"
    copies="${copies:-copy}"
    for ((trial = 1; trial <= trials; ++trial)); do
      if ((trial % 2 == 1)); then modes=(direct graph); else modes=(graph direct); fi
      for mode in "${modes[@]}"; do
        rc=0
        line="$("$BIN" "$mode" "$launches" "$batch" "$size" "$rows" "$repeats" "$warmup_ms" \
                "$copies" "$matmul" 2>&1)" || rc=$?
        local row="$trial,$mode"
        if ((rc == 0)); then
          # Record the launch count the binary actually used (it differs from
          # the request when launches=auto).
          row+=",$(field launches),$batch,$size,$rows,$repeats,$copies,$matmul,$warmup_ms"
          for key in "${fields[@]}"; do row+=",$(field "$key")"; done
        else
          row+=",$launches,$batch,$size,$rows,$repeats,$copies,$matmul,$warmup_ms"
          row+="$(printf ',%.0s' "${fields[@]:1}"),0"
          printf 'warning: %s failed for %s (%s)\n' "$mode" "$config" "$line" >&2
        fi
        printf '%s,%s\n' "$row" "$rc" >> "$csv"
      done
    done
  done
  printf 'Sweep results: %s\n' "$csv"
  printf 'Plot with: python3 %s/plot_sweep.py %s\n' "$SCRIPT_DIR" "$csv"
}
# Launch-overhead sweep: from tiny single-block kernels to the original size.
sweep() {
  run_sweep sweep "1,16,16,1,nocopy 1,64,64,1,nocopy 1,64,64,16,nocopy 1,256,256,1,copy \
1,256,256,8,nocopy 1,256,256,32,nocopy 1,1024,1024,1,nocopy 1,4096,4096,1,copy \
512,16384,8192,1,copy" 30 naive
}
# GPU-filling sweep: every kernel launches >= 1 full wave (1024 vectors and
# 8192 matmul rows -> ~2 waves on an 82-SM RTX 3090) with the coalesced matmul
# and no host copies, so only launch mode and per-kernel size change.
# Size sets kernel duration (~µs to ~1 ms); repeats set kernels per iteration.
sweep_fill() {
  run_sweep sweep-fill "1024,256,8192,1,nocopy 1024,256,8192,8,nocopy 1024,256,8192,32,nocopy \
1024,1024,8192,1,nocopy 1024,1024,8192,8,nocopy 1024,1024,8192,32,nocopy \
1024,4096,8192,1,nocopy 1024,4096,8192,8,nocopy \
1024,16384,8192,1,nocopy 1024,16384,8192,8,nocopy" auto coalesced
}
isolate() {
  require_tool nsys; require_tool ncu; ensure_bin
  local dir="$OUT_DIR/isolate-$STAMP"
  mkdir -p "$dir"
  python3 - "$BIN" "$dir" <<'PY'
import os, subprocess, sys
bin_path, out = sys.argv[1:]
probes = [
 ('nsys-basic',['nsys','profile','--trace=cuda','--sample=none','--cpuctxsw=none']),
 ('nsys-graph',['nsys','profile','--trace=cuda','--cuda-graph-trace=graph','--sample=none','--cpuctxsw=none']),
 ('ncu-basic',['ncu','--set','basic','--target-processes','application-only']),
 ('ncu-replay',['ncu','--set','basic','--launch-skip','0','--launch-count','3','--replay-mode','kernel','--target-processes','application-only']),
 ('ncu-graph',['ncu','--set','basic','--launch-skip','0','--launch-count','3','--replay-mode','kernel','--graph-profiling','node','--target-processes','application-only']),
]
with open(os.path.join(out,'results.txt'),'w') as summary:
 for name, args in probes:
  output_opt = '--output' if args[0]=='nsys' else '--export'
  overwrite = ['--force-overwrite=true'] if args[0]=='nsys' else ['--force-overwrite']
  cmd=args+overwrite+[output_opt,os.path.join(out,name),bin_path]
  with open(os.path.join(out,name+'.log'),'w') as log:
   p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
  summary.write('%s: exit=%d\n' % (name,p.returncode)); print('%s: exit=%d' % (name,p.returncode))
PY
  printf 'Detailed logs: %s\n' "$dir"
}

case "${1:-all}" in
  build) build ;;
  run) run "${@:2}" ;;
  nsys) profile_nsys graph "${@:2}" ;;
  nsys-node) profile_nsys node "${@:2}" ;;
  ncu) profile_ncu "${@:2}" ;;
  sweep) sweep ;;
  sweep-fill) sweep_fill ;;
  plot-sweep) python3 "$SCRIPT_DIR/plot_sweep.py" "${@:2}" ;;
  plot-nodes) python3 "$SCRIPT_DIR/plot_nodes.py" "${@:2}" ;;
  isolate) isolate ;;
  all) run graph; profile_nsys graph graph; profile_ncu graph ;;
  *) usage >&2; exit 2 ;;
esac

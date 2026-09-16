#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN="${BIN:-$SCRIPT_DIR/llama2c_cuda_graph}"
SRC="$SCRIPT_DIR/llama2c_cuda_graph.cu"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/profiles}"
STAMP="$(date +%Y%m%d-%H%M%S)"
NVCC="${NVCC:-nvcc}"
read -r -a NVCC_FLAGS <<< "${NVCC_FLAGS:-}"

usage() { printf 'Usage: %s [build|run|nsys|ncu|isolate|all]\n' "$0"; }
require_tool() { command -v "$1" >/dev/null || { printf 'error: %s is not installed or not in PATH\n' "$1" >&2; exit 127; }; }
build() { require_tool "$NVCC"; "$NVCC" "${NVCC_FLAGS[@]}" -O3 -o "$BIN" "$SRC"; }
ensure_bin() { [[ -x "$BIN" && "$BIN" -nt "$SRC" ]] || build; }
run() { ensure_bin; "$BIN"; }
profile_nsys() {
  require_tool nsys; ensure_bin; mkdir -p "$OUT_DIR"
  nsys profile --trace=cuda --cuda-graph-trace=graph --sample=none \
    --cpuctxsw=none --force-overwrite=true \
    --output="$OUT_DIR/nsys-$STAMP" "$BIN"
}
profile_ncu() {
  require_tool ncu; ensure_bin; mkdir -p "$OUT_DIR"
  ncu --set basic --launch-skip 0 --launch-count 3 --replay-mode kernel \
    --graph-profiling node --target-processes application-only \
    --export "$OUT_DIR/ncu-llama2c-$STAMP" --force-overwrite "$BIN"
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
  run) run ;;
  nsys) profile_nsys ;;
  ncu) profile_ncu ;;
  isolate) isolate ;;
  all) run; profile_nsys; profile_ncu ;;
  *) usage >&2; exit 2 ;;
esac

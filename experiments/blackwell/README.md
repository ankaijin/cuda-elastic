# RTX 5090 feature experiments

Three experiments evaluate Blackwell features on kilia's RTX 5090 against the
current optimized loom_private implementations:

| Directory | Feature | Workload |
|---|---|---|
| [`packed/`](packed) | Packed byte addition (`add.u8x4`) | `loom-private/vecadd-product` |
| [`tma/`](tma) | TMA bulk copies and a producer warp | `loom-private/multisensor-attention` |
| [`lowprec/`](lowprec) | BF16, MXFP6, and NVFP4 Tensor Cores | `loom-private/llama2c-decoder` classifier |

Each experiment asks one question. It checks correctness, confirms in the
machine code that the feature executes, and measures the benefit against the
production optimized variant. Unchanged performance and regressions are valid
results. Nothing here is promoted into the published suite automatically.

The production sources, contracts, and results are never modified. A control
case compiles the production source in place. A candidate is a copy with a
reviewed change, and a committed `.diff` beside it records the exact
difference (a unit test keeps it current).

## Layout

- `experiments/blackwell/<experiment>/`: sources, `experiment.json` (cases,
  check arguments, SASS features), `Makefile`, and tests.
- `<gpu>/blackwell_experiments/<experiment>/`: raw evidence. `build/` holds
  binaries and is not committed. `results/` holds per-case records and logs,
  `profiles/` holds Nsight reports, and `<stamp>-<action>.json` summarizes one
  run.
- `results/blackwell_experiments/`: the combined report generator
  (`build_report.py`) and each GPU's input manifest (`inputs/<gpu>.json`),
  which selects the run summaries and records each candidate's conclusion.
  Mono generates and commits `README.md` after the raw evidence merges.

## Running an experiment

Use Python 3.10 or newer with `/usr/local/cuda/bin` on `PATH`. Every GPU
process takes `/tmp/cuda-elastic-gpu.lock` itself; do not wrap these commands
in `flock`.

```bash
export PATH=/usr/local/cuda/bin:$PATH
# Fresh build, tests, large checks, SASS inspection; then five paired benches.
python3 results/record_run.py --check "python3 experiments/blackwell/run.py packed" \
  -- python3 experiments/blackwell/run.py packed --action bench --pairs 5
# Confirm an apparent improvement with ten more pairs.
python3 results/record_run.py --check "python3 experiments/blackwell/run.py packed" \
  -- python3 experiments/blackwell/run.py packed --action bench --pairs 10
# One profiled invocation per case (Nsight Systems, then Nsight Compute --set full).
python3 results/record_run.py --check "python3 experiments/blackwell/run.py packed" \
  -- python3 experiments/blackwell/run.py packed --action profile
```

`run.py` actions:

- **check** (the default) forces a fresh build (`make -B`), runs the
  experiment's tests (`make test`), then each case's large check in Direct and
  Graph mode, any extra actions, and a SASS inspection. In `packed` and `tma`,
  every case's output digest must equal the control's.
- **bench** times every case with the harness's protocol (20 iterations, 3
  warmups by default) in Direct and Graph mode. Each pair runs the cases in
  `experiment.json` order on odd pairs and in reverse order on even pairs. It
  never builds. It refuses a binary whose hash differs from its build record.
  A bench whose output digest differs from the same run's check digest fails.
- **profile** captures one end-to-end invocation per case with the harness's
  `--action profile` and exports CSV pages beside each report. It uses Graph
  launches unless `--launch` says otherwise. Where nsys cannot write
  `/tmp/nvidia`, it sets `NSYS_TMPDIR` to `~/.cache/cuda-elastic/nsys-tmp`.

Every record embeds the source revision, the build record, and the GPU's UUID
and driver. The build record lists the exact compiler command, the
architecture target, the specification hash, the SHA-256 of each source and
the binary, and the git revision of any include directory outside the
repository. File names begin with the recorder's run id.

## Interpreting results

Timing records hold the harness's application scope (transfers and kernels)
and compute scope (kernels with resident inputs). The report keeps three kinds
of evidence apart:

- **implementation correctness**: the checks, tests, and digests;
- **numerical quality**: the lower precision accuracy metrics;
- **performance**: paired timings and profiles.

A candidate shows an apparent improvement when its median paired ratio to the
control is below 1 by more than the control's relative half range, and most
pairs improve. Ten more pairs confirm it when at least nine improve and the
median still clears the control's half range. The same rule in the other
direction identifies a regression. Anything else is no measurable benefit.

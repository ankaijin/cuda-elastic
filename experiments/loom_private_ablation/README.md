# loom_private optimization stages

Each loom_private optimized variant combines several optimizations. This
experiment separates them. It starts from the baseline, adds one optimization
at a time, and times and profiles every cumulative combination, so each
optimization's share of the speedup is measured, not estimated.

| Directory | Workload | Stages |
|---|---|---|
| [`pagerank/`](pagerank) | `loom-private/pagerank` | persistent kernel, readiness tags, level-ordered layout, per-GPU tuning |

The results are in `results/loom_private_ablation/opt_results.xlsx`.

## Rules

- Stage 0 is the suite's baseline binary (Direct and Graph). The last stage is
  the suite's optimized binary. Both are built from the unchanged production
  sources by the suite's own Makefile. The check records their hashes, build
  commands, and the nvcc version, and refuses to run if the production sources
  have changes. Bench and profile use exactly the binaries the check built.
- Every stage in between is the previous stage with one more optimization. It
  implements the same contract and passes the same large check and the
  application's full existing test suite. Stage sources link the application's
  `oracle.cpp` and never define their own tolerance or comparison code.
- Output digests: `stages.json` `arithmetic_changes` lists the stages that
  change rounding (`{"stage": N, "why": ..., "gpus": [...]}`, where `gpus` is
  optional). A listed stage may change the digest; every other stage must
  reproduce the previous stage's digest exactly, so a difference starts only
  where it is declared and is then inherited.
- A stage that equals production code apart from one setting is checked
  against the production machine code (`sass_equal`) and against production's
  host-side launch structure (`host_equal`: kernels with grid, block, and
  shared memory; copies with kind and bytes; memsets). The reference is
  production built from its unchanged source with its existing test hooks set
  to the stage's setting: a binary with `"timed": false` and its `"env"`,
  built by the stage Makefile and never timed. Where this GPU's tuning equals
  the setting, a second entry with `"gpus"` compares with the plain production
  binary.
- `scope` (`compute` or `application`) names the timing scope whose savings the
  report's step columns use.
- Timing uses the suite's protocol unchanged: the harness's `--action bench`
  (20 iterations after 3 warmups, `batch_per_iteration_us`), three
  repetitions, median. Each repetition times every case once in a rotated
  order and the Graph baseline once more before and after, which gives the
  drift. Profiling (`tools/profile.py`'s nsys and ncu commands) is a separate
  action. Stages after the baseline run as CUDA Graphs, like the report's
  Optimized column.
- The report flags a step as within noise when its saving is at most the
  larger repetition range (max - min) of its two stages. This is a heuristic,
  not a significance test.
- Production sources, results, and the loom_private workbook are not changed.

## Running

Use Python 3.10 or newer with `/usr/local/cuda/bin` on `PATH`. Each GPU process
takes `/tmp/cuda-elastic-gpu.lock`; do not wrap these commands in `flock`.

```bash
export PATH=/usr/local/cuda/bin:$PATH
python3 results/record_run.py \
  --check "python3 experiments/loom_private_ablation/run.py pagerank" \
  -- bash -c "python3 experiments/loom_private_ablation/run.py pagerank --action bench --repeat 3 && python3 experiments/loom_private_ablation/run.py pagerank --action profile"
```

- **check** (the default) builds the suite's baseline and optimized binaries
  and the stage binaries fresh, runs the stage tests, runs every binary's large
  check in Direct and Graph mode, and compares machine code.
- **bench** never builds. It times every case once per repetition, in stage
  order.
- **profile** captures one end-to-end invocation per case with nsys, then ncu
  `--set basic`, and exports their CSV pages.

Raw evidence goes to `<gpu>/loom_private_ablation/<app>/`: `results/` holds
records and logs, `profiles/` holds Nsight reports, and
`<stamp>-<action>.json` summarizes one run. `build/` is not committed.
`results/loom_private_ablation/inputs/<gpu>.json` selects the runs, and
`python3 results/loom_private_ablation/build_report.py` writes the workbook.

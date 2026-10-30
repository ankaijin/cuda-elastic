# Concurrent development and GPU results

Mono (RTX 3090), octacos (two RTX 4090s), and kilia (RTX 5090) have independent clones. GitHub is their synchronization point. Mono also has `/home/ankai/cuda-integration`, the checkout that combines merged inputs and publishes the generated workbook and tables.

## Branches and worktrees

Use a task branch for each independent change. The initial task worktrees are:

| Machine | Worktree | Branch |
|---|---|---|
| mono | `/home/ankai/cuda-worktrees/mono-small-shapes` | `mono/small-shape-analysis` |
| octacos | `/home/ankai/cuda-worktrees/octacos-baseline` | `octacos/complete-baseline` |
| kilia | `/home/ankai/cuda-worktrees/kilia-baseline` | `kilia/5090-baseline` |

These initially include the infrastructure branch, so the workflow can be tested before its PR merges. Their PRs will include infrastructure changes until that PR reaches main. Existing primary checkouts, older branches, and worktrees stay intact.

After the infrastructure PR merges, start new tasks from current main:

```bash
git fetch origin
git worktree add -b octacos/my-task /home/ankai/cuda-worktrees/my-task origin/main
```

Give each branch one active writer. To test another machine's code, fetch its branch into a separate testing worktree or test a common merged revision. Update a published task branch with `git fetch origin` and `git merge origin/main`; reserve rebase for unpublished commits. Push with `git push -u origin HEAD` and integrate with a PR. Merge finished tasks regularly.

Repository-local defaults on all machines are `pull.ff=only`, `fetch.prune=true`, and `push.default=simple`. Conflict style is `zdiff3` on mono and kilia. Octacos's Git 2.34.1 requires `diff3`. Linked worktrees inherit these settings; fresh clones need them applied again.

## Record an experiment

Start with committed source and a clean worktree. Use Python 3.10 or newer, with `/usr/local/cuda/bin` on PATH. The recorder needs only the standard library. Only the report publisher needs openpyxl and a compatible `ncu`; install its Python dependency with `python3 -m pip install -r results/requirements.txt` if necessary.

Force a fresh build where appropriate. This repository historically tracks some binaries, so `make` alone may reuse an old binary. Validate every implementation you will time using the root README's build/check targets and each workload's `compare_sota.sh`.

After building the required workloads, this illustrates the recording interface:

```bash
export PATH=/usr/local/cuda/bin:$PATH
python3 results/record_run.py --device 0 \
  --check 'flock /tmp/cuda-elastic-gpu.lock bash -c "make -B -C loom_msa check"' \
  -- ./time_all.sh
```

The example check covers loom_msa only. For publication runs, replace it with checks covering all workloads and their direct, graph, and optimized variants. Use matching checks and timing settings across GPUs.

The recorder selects one GPU by UUID, sets its output directory and architecture, and supplies a unique UTC `TIMESTAMP`. Validation must exit successfully before measurement starts. It writes `<gpu>/runs/<run-id>/metadata.json`, `validation.log`, and `measurement.log`. Metadata records source commit, host, GPU UUID, driver, installed tool versions, workload/build environment overrides, commands, return codes, UTC times, and SHA-256 hashes of executable ELF binaries present after validation. Installed tool versions alone do not identify the compiler that produced an existing binary: include fresh builds in the recorded validation command when build provenance matters. Source changes during validation or measurement invalidate the record. Normal profiles remain under `<gpu>/<workload>/profiles/`. Commit raw outputs and metadata before starting another recorded run.

A completed record means the supplied commands succeeded. It neither proves comparability nor selects results for publication. A failed check leaves a failure record and prevents measurement.

Timing scripts already use `/tmp/cuda-elastic-gpu.lock`. Commands without their own locking must use it too; for ncu, use `flock /tmp/cuda-elastic-gpu.lock ./profile_ncu_all.sh` as the measurement command. Never wrap a script that already locks that file: nested `flock` can deadlock. The llama2c, loom_msa, and PageRank comparison scripts already lock internally; the MLPerf classification comparison honors the supplied `GPU_LOCK`. The recorder is a provenance tool, not a GPU scheduler. Keep the default lock path across sessions. This conservatively serializes both 4090s on octacos; independent use requires standardizing device-specific locks across every runner first. Use separate worktrees even when selecting different devices.

If performance counters require administrator access, establish the site's approved profiling access. Keep the recorder under your user account and elevate only the profiling command if needed, preserving PATH, CUDA_VISIBLE_DEVICES, GPU, ARCH, GPU_LOCK, and TIMESTAMP. Do not run the development session as root.

## Select results

Each GPU owns `results/inputs/<gpu>.json`. Its `files` mapping contains exact paths relative to that GPU's output directory. Null means pending. Missing selected files, wildcards, and paths outside that directory fail visibly. Newer logs never replace selections automatically.

The initial 3090/4090 manifests preserve the old report's sources. Legacy profiles keep their existing evidence; migration does not invent historical provenance. After an experiment, select its exact files, then commit raw artifacts, run metadata, and your GPU manifest in one PR. Keep each workload's direct/graph/optimized comparison on matching shapes, inputs, and intended source revisions. Update `toolkit` when replacing results with measurements built by a different toolkit. Document any intentional use of mixed historical sources.

The 5090 has supported summary and ncu detail sheets, with all inputs initially null. Predictions stay disabled because the existing 4090 model does not describe the 5090's memory bus and PCIe assumptions. Kilia currently has CUDA 13.4; it is not labeled as a measured toolchain until a run exists. Octacos currently has CUDA 12.8, while its selected historical results used CUDA 12.5.

## Publish from mono

Merge raw-result and input-manifest PRs first. In `/home/ankai/cuda-integration`, start a fresh report branch:

```bash
git fetch origin
git switch -c reports/update-YYYYMMDD origin/main
export PATH=/usr/local/cuda/bin:$PATH
python3 -m unittest discover -s tests -v
python3 results/build_spreadsheet.py
git diff -- results/README.md
git add results/cuda_performance.xlsx results/README.md
git commit -m "Update combined GPU performance reports"
git push -u origin HEAD
```

Review the workbook and source links, then integrate the report PR before another publisher starts. Keep one publishing branch active at a time. `ncu` must import every selected report version; export portable metrics on the originating machine if necessary. Build after raw results merge so source links can be pinned to commits reachable from `origin/main`.

Keep the workbook tracked. Development/result PRs contribute source inputs; publishing PRs update the generated workbook and tables. Do not use an automatic “ours” merge driver or mark the tracked workbook assume-unchanged. The initial setup PR tests generation but leaves the committed reports unchanged; publish the new 5090 sheet after setup merges.

## Report layout convention

The main results focus on large workloads intended to fully or almost fully use the GPU's relevant compute or memory resources. Support utilization claims with profiling evidence; problem size alone does not establish saturation. Small workloads diagnose launch, dispatch, and latency behavior.

Preserve this convention when adding workloads, GPU results, or report features:

- Put only large application workloads on the primary `RTX 3090`, `RTX 4090`, and `RTX 5090` sheets. Keep these sheets first in the workbook.
- Keep small workload summaries only on `Small RTX 3090`. Do not add small summary sheets for the 4090 or 5090 under the current reporting scope.
- Retain large and small workloads in `Latency RTX 3090` for diagnosis. Include only large workloads in `Latency RTX 4090` and `Latency RTX 5090`.
- Include only large workload comparisons in `GPU analysis`, including kernel comparisons from those large workloads. Match baseline and target GPU rows by workload and stage when filtering; preserve the corresponding prediction and measured speedup.
- In `results/README.md`, present large workload tables for all GPUs first, followed by a separate RTX 3090 small workload diagnostic section.
- Keep `ncu metrics (...)` compact: one row per report, workload, implementation, full kernel name, and launch resource configuration. Show launch count, median/minimum/maximum duration, and median throughput/occupancy metrics; keep missing metrics blank. Link each group to the original report and its individual rows on `NCU launches ...`. These descriptive launch statistics do not represent application latency or aggregate GPU utilization. Never combine different reports or launch configurations.
- Reuse the Direct, Graph, and Agentic column colors for latency rows and NCU rows.
- Preserve raw outputs and input manifests when changing report visibility. Update the generator rather than editing the workbook manually, and keep navigation links valid.

Change this reporting scope only when the user requests it. Run `python3 -m unittest discover -s tests -v` before regenerating the reports. The report layout tests cover sheet ordering, diagnostic scope, latency rows, large workload comparisons, and navigation links. Extend them when adding new report sections or GPU types.

## Private Loom suite

`applications/loom_private/` ports six private Loom applications (commit `4a75f8d`) as new workload identities. It does not change the legacy workloads, their selected inputs, raw results, or generated reports; `python3 applications/loom_private/tools/legacy_freeze.py verify` checks that against `legacy-freeze.json`.

- Fetch assets first with `python3 applications/loom_private/tools/fetch_assets.py`. The cache lives outside the repository (`LOOM_PRIVATE_ASSET_CACHE`); private Loom fixtures are fetched with `gh` at the pinned commit and never committed.
- `python3 applications/loom_private/run_suite.py [APP ...]` builds and runs large correctness checks (Direct and Graph). Benchmarks need `--action bench --launch direct|graph`. Its Makefile targets lock `GPU_LOCK`; do not wrap it in flock.
- Record a run with `python3 results/record_run.py --check "python3 applications/loom_private/run_suite.py" -- python3 applications/loom_private/run_suite.py --no-build --action bench --launch graph`. Its result records carry workload identity, contract hash, dimensions, asset hashes, and timing scope, and share the run id.
- To record the optimized variant, build and check it in `--check` as well (`run_suite.py --variant optimized`), then measure with `--variant optimized --no-build`. Otherwise `make bench` builds inside the measurement and the run's binary hashes miss `<app>_optimized`.
- Each application also has an optimized variant of the same contract (`--variant optimized`, or `VARIANT=optimized` for make targets). It passes the same check; its README section documents the design and the per-GPU parameters. Test-only tuning switches exist only in separate test builds, never in the binaries that are timed.
- Repeat benches with `--repeat N`; the report shows the median. `python3 applications/loom_private/tools/profile.py` records Nsight Systems and Nsight Compute profiles of the baseline (Direct and Graph) and the optimized variant (Graph), with CSV exports beside them. Run it as a recorded measurement; it takes the GPU lock per process. On a host where performance counters are admin-only (octacos), pass `--no-ncu`. Where nsys cannot write `/tmp/nvidia` (kilia), export `NSYS_TMPDIR` to a directory of your own first. Each capture holds exactly one end-to-end invocation: the binaries' `--action profile` runs one warmup invocation, then one between `cudaProfilerStart` and `cudaProfilerStop`.
- Select records and profiles in `results/loom_private/inputs/<gpu>.json` (independent of `results/inputs/`) and regenerate with `python3 results/loom_private/build_report.py`. Timings come only from bench records; correctness checks are listed separately.

## RTX 5090 feature experiments

`experiments/blackwell/` holds independent experiments that compare Blackwell features with the production optimized loom_private variants: packed byte arithmetic, TMA tile loads, and lower precision Tensor Core classifiers. Each one compiles the production source unchanged as its control. `python3 experiments/blackwell/run.py <experiment>` runs fresh builds, tests, checks, and SASS inspection; `--action bench` and `--action profile` measure. Run it under `results/record_run.py`. Raw evidence goes to `<gpu>/blackwell_experiments/`. Each GPU's `results/blackwell_experiments/inputs/<gpu>.json` selects runs and conclusions, and mono publishes `results/blackwell_experiments/README.md` with `build_report.py`. The lower precision experiment needs CUTLASS 4.8.0 at a pinned commit outside the repository (`make -C experiments/blackwell/lowprec cutlass`). See `experiments/blackwell/README.md`.

The loom_private workbook also shows the BF16 classifier on its RTX 5090 summary, Latency, Kernels, and Sources sheets, next to the llama2c decoder. `EXPERIMENT_ROWS` in `results/loom_private/build_report.py` names it, and the rows use the selections in `results/blackwell_experiments/inputs/rtx5090.json`, which must conclude `improved` and include a confirmation bench. The rows show the paired speedup over the FP32 control and the accuracy loss, and are labeled as experiments rather than versions of the contract. The other experiments stay out of the workbook.

## Next experiments from the existing workbook

- **Mono:** investigate optimized small-shape regressions. PageRank's batch per-iteration time is 602.149 us versus Graph's 159.400 us; VWW is 12.097 us versus 8.075 us. Check setup/dispatch costs and consider a size-specific strategy if justified. Repeat anomaly's small-shape single-iteration result: Graph 20.272 us versus Direct 13.616 us. Preserve large-shape baselines, and refresh mono at the same source revision used in new GPU comparisons.
- **Octacos:** complete correctness checks, then collect `time_all.sh` large shapes for direct, graph, and optimized variants, followed by ncu kernel metrics and nsys traces. Small shapes are optional diagnostics and are outside the current 4090 report scope. Its current sheet has older graph-only end-to-end measurements, no ncu metrics, no optimized measurements, and no small-shape timing suite. Select one 4090 and record its UUID.
- **Kilia:** start with fresh sm_120 builds and correctness checks using CUDA 13.4. Then collect the same large workload timing, ncu, and nsys suites at the common revision. Small shapes are outside the current 5090 report scope. No 5090 measurements are published yet. Treat toolchain differences as an experimental variable.

These are next-task recommendations. Setup verification does not run or replace GPU benchmark measurements.

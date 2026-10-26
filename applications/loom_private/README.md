# Private Loom CUDA workloads (suite v1)

Six CUDA workloads ported from private Loom (`PolyArch/loom-private`, commit
`4a75f8d`). Each one checks its GPU output against an independent CPU
reference, and each can be timed with direct kernel launches or with a CUDA
Graph. Results are on the [results page](../../results/loom_private/README.md).

| Directory | Workload at the large size |
|---|---|
| [`pagerank/`](pagerank) | Ordered Gauss–Seidel PageRank: 1,048,576 vertices, 2,097,151 edges, 20 iterations |
| [`attention/`](attention) | Multisensor attention: 25,000 queries × 250,000 context tokens |
| [`decoder/`](decoder) | llama2c decoder: 1,024 sequences × 4 steps, width 16,384, vocabulary 8,192 |
| [`anomaly/`](anomaly) | MLPerf Tiny anomaly model (int8, 10 layers): 104,858 samples × 640 values |
| [`vecadd_memory/`](vecadd_memory) | Integer vector add: 1,048,576 elements |
| [`vecadd_product/`](vecadd_product) | Byte-wise add of 1,048,576 pairs of 64-byte operands |

The older workloads at the repository root (`llama2c/`, `loom_msa/`,
`pagerank/`, `mlperf-*`) are separate and unchanged.
[`legacy-freeze.json`](legacy-freeze.json) holds their hashes, and
`tools/legacy_freeze.py verify` checks them.

## Quick start

Fetch the model, dataset, and fixtures. This needs `gh` access to
`PolyArch/loom-private`.

```bash
python3 applications/loom_private/tools/fetch_assets.py
```

Build and check all six workloads at the large size:

```bash
python3 applications/loom_private/run_suite.py
```

Time one workload. Choose `direct` or `graph`.

```bash
python3 applications/loom_private/run_suite.py --action bench --launch graph pagerank
```

You can also run one application's targets directly:
`make -C applications/loom_private/<app> build`, `check-large`, `test`, or
`bench LAUNCH=graph`.

Every workload also has an optimized version of the same contract. Add
`--variant optimized` to `run_suite.py`, or `VARIANT=optimized` to a `make`
target; the binary is `<app>_optimized`. It passes the same check, and each
application's README explains its design and the parameters chosen for each
GPU. `tools/profile.py` profiles the baseline and the optimized version with
Nsight Systems and Nsight Compute.

Results are written to `<gpu>/loom_private/<app>/results/`. To record a run
with provenance, run it under `results/record_run.py`. See
[docs/concurrent-development.md](../../docs/concurrent-development.md).

## What a check does

1. It runs the CPU reference once.
2. It runs the GPU twice with direct launches and twice with a CUDA Graph.
   Each run starts from poisoned buffers.
3. It passes only if every run matches the reference and all four outputs are
   bit-identical.

## What a benchmark times

- **End to end** (application scope): uploading the inputs, running every
  kernel, and downloading the results.
  - The decoder keeps its weights on the GPU, so it transfers only prompts and
    tokens.
  - The anomaly model is loaded and uploaded once, outside the timed interval.
- **Kernel only** (compute scope): the same kernels without the copies.

Setup, graph capture, warmups, the CPU reference, and output checking are
never timed. Each `contract.json` defines the scopes exactly.

## Command-line options

| Option | Meaning |
|---|---|
| `--action check\|bench\|profile` | Check (the default), time, or run one end-to-end invocation for a profiler (`tools/profile.py`). |
| `--launch direct\|graph\|both` | A check runs both by default. A bench or profile needs `direct` or `graph`. |
| `--preset large` | The only preset. Upstream smoke sizes are documented, never run. |
| `--iterations N`, `--warmup N` | Bench batch length and warmups (defaults 20 and 3). |
| `--set key=value` | Change a dimension. Checks only; tests use it. |
| `--asset name=path` | Use another file for a cached asset. |
| `--json-out PATH` | Also write the result record to a file. |

Exit codes: 0 pass, 1 wrong output, 2 bad arguments or dimensions, 3 missing
or corrupt asset, 70 CUDA error.

Every GPU command takes `/tmp/cuda-elastic-gpu.lock`. Never wrap these
commands in another `flock` on the same file.

## Layout

Each application directory has:

- its CUDA source
- a CPU reference (`oracle.cpp`)
- a `Makefile` that includes [`common.mk`](common.mk)
- tests
- `contract.json`
- a README with design notes

The shared harness is [`common/lp_harness.cuh`](common/lp_harness.cuh).
Binaries go to `<gpu>/loom_private/<app>/build/`. Assets are listed in
[`assets.json`](assets.json). They are cached outside the repository, in
`~/.cache/cuda-elastic/loom-private-assets` or `$LOOM_PRIVATE_ASSET_CACHE`.
No private Loom source is copied here; contracts cite it by commit, path,
and hash.

## Workload contracts

`contract.json` defines each workload. The build embeds the contract's SHA-256
in the binary and in every result record, so each result names the exact
contract it checked.

| Field | Contents |
|---|---|
| `identity` | Workload name, such as `loom-private/pagerank`. |
| `suite_version` | `v1`. |
| `extension_of_upstream` | `false` for a direct port, otherwise a sentence describing the extension. |
| `upstream` | Repository, commit, and each source path with its git blob hash. |
| `large` | The exact large dimensions, including derived counts. |
| `data_generation` | How every input is produced. |
| `outputs` | Every checked output and its byte layout. |
| `tolerances` | How each output is compared, and why. |
| `timing_scope` | What end-to-end and kernel-only timing include. |
| `assets` | Asset names from `assets.json` with their SHA-256 hashes. |
| `reference_only_upstream_configurations` | Upstream smoke and validation settings. Documentation only. |

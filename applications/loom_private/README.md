# Private Loom CUDA workloads (suite v1)

Six CUDA applications ported from private Loom (`PolyArch/loom-private`) at commit
`4a75f8d743ef7fe755822bab0ea0fb78009c145b`. They are new workload identities. The
legacy applications at the repository root (`llama2c/`, `loom_msa/`, `pagerank/`,
`mlperf-anomaly/`, `mlperf-classification/`) and their results stay unchanged
as historical workloads. The suite started from cuda-elastic commit
`4ce57c7c0a333b74e9e6e2bd78142604c4c2cf39`; `legacy-freeze.json` records the hashes
of every legacy application, script, selected input, raw result, and generated
report at that commit, and `tools/legacy_freeze.py verify` checks them.

This delivery covers large workload correctness on the RTX 3090. Timings are
pending, and no optimized variants exist yet.

| Directory | Identity | Large workload |
|---|---|---|
| [`pagerank/`](pagerank) | `loom-private/pagerank` | ordered Gauss–Seidel PageRank, $2^{20}$ vertices, 2,097,151 edges, 20 iterations |
| [`attention/`](attention) | `loom-private/multisensor-attention` | 25,000 query × 250,000 context tokens |
| [`decoder/`](decoder) | `loom-private/llama2c-decoder` | 1,024 sequences, width 16,384, vocabulary 8,192, 4 dependent steps |
| [`anomaly/`](anomaly) | `loom-private/mlperf-tiny-anomaly-model` | 104,858 samples × 640 elements through the 10-layer int8 model |
| [`vecadd_memory/`](vecadd_memory) | `loom-private/vecadd-memory` | 1,048,576 integer elements |
| [`vecadd_product/`](vecadd_product) | `loom-private/vecadd-product` | 1,048,576 records of two 64-byte operands |

## Commands

```bash
python3 applications/loom_private/tools/fetch_assets.py
```

```bash
make -C applications/loom_private/pagerank build check-large
```

```bash
python3 applications/loom_private/run_suite.py
```

`run_suite.py` builds and checks the named applications (all six by default)
at the large preset. Each application's `make test` runs its own small tests.
Upstream smoke and validation configurations are documented in each
`contract.json` as references only; no target runs them.

## Shared conventions

**Layout.** Each application directory holds its CUDA sources, an independent
CPU oracle in separate host files, a `Makefile` that includes
[`common.mk`](common.mk), tests, `contract.json`, and a README. Binaries and
artifacts go under `<gpu>/loom_private/<application>/` (binaries in `build/`).
Asset caches stay outside tracked source (see [`assets.json`](assets.json)).
Private Loom files are referenced by commit, path, and hash; they are not
copied here.

**Command line** (from [`common/lp_harness.cuh`](common/lp_harness.cuh)):
`--preset large`, `--launch direct|graph|both`, `--action check|bench`. The
default is `--preset large --action check --launch both`. Bench needs an explicit
`--launch direct` or `--launch graph`. `--set key=value` overrides a dimension
for tests (check only) and is recorded in the result. `--asset name=path`
replaces a cached asset path. `--json-out PATH` writes the result record.
Exit codes: 0 pass, 1 mismatch, 2 usage or invalid dimensions, 3 missing or
corrupt asset, 70 CUDA error.

**Correctness.** `make check-large` runs the CPU oracle once, then two
invocations in Direct mode and two in Graph mode of the same computation. Each
invocation starts from poisoned outputs and must pass the oracle. All four
output digests must agree bitwise. The result record (`RESULT {...}` on
stdout) carries the workload identity, contract hash, dimensions, asset
hashes, timing scopes, and per-invocation status.

**Timing.** `--action bench` times application scope and compute-only scope.
Application scope includes host↔device transfers for most applications; the
decoder keeps weights and working state resident; the anomaly application
excludes model loading, parsing, and weight upload, and includes each
invocation's sample upload and output download. Allocation, graph construction,
warmups, the CPU oracle, and output checking stay outside every timed interval.

**GPU lock.** Every GPU run takes `/tmp/cuda-elastic-gpu.lock` (`GPU_LOCK`).
The Makefile targets already lock; never wrap them in another `flock` on the
same file.

## Workload contracts

Each `contract.json` has these fields:

- `identity`, `suite_version` (`v1`), `extension_of_upstream` (`false` for a
  direct port; a sentence describing the extension otherwise).
- `upstream`: repository, commit, and each source path with its git blob hash.
- `large`: the exact large dimensions, including derived counts.
- `data_generation`: how every input is produced, precisely enough to
  reimplement.
- `outputs`: every output the check covers and its byte layout.
- `tolerances`: the comparison for each output and where it comes from.
- `timing_scope`: what application and compute scopes include and exclude.
- `assets`: asset names from `assets.json` with their SHA-256, or empty.
- `reference_only_upstream_configurations`: upstream smoke and validation
  settings with their oracle hashes. These are documentation; nothing runs
  them.

The build embeds the contract's SHA-256 in the binary and in every result
record, so a result names the exact contract it validated.

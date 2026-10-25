# MLPerf Tiny anomaly model (`loom-private/mlperf-tiny-anomaly-model`)

Complete int8 inference of the MLPerf Tiny anomaly-detection autoencoder
`ad01_int8.tflite`. The model has ten fully connected layers with widths
640 → 128 → 128 → 128 → 128 → 8 → 128 → 128 → 128 → 128 → 640. Each float32
sample of 640 elements is quantized to int8 and passed through the ten
layers. The output is the 640 int8 codes of the last layer.

This is not the legacy `mlperf-anomaly/` workload, which dequantizes and
computes an MSE. That workload is unrelated and unchanged.

## Large workload

| Quantity | Value |
|---|---|
| Samples | 104,858 (sample $i$ is dataset sample $i \bmod 40$) |
| Elements per sample | 640 |
| Input | 67,109,120 float32 (268,436,480 bytes) |
| Output | 67,109,120 int8 bytes |
| Layers / MACs per sample | 10 / 264,192 |

The large workload is a CUDA extension of the upstream sample contract.
Upstream runs at most the 40 dataset samples. Here, the same per-sample
computation runs on 104,858 samples cycled from the dataset, and the GPU
infers every one of them. Nothing is deduplicated.

Assets (see [`../assets.json`](../assets.json)) are read from the cache:
`mlperf-tiny-ad01-int8-model` (276,976 bytes) and `mlperf-tiny-ad01-dataset`
(102,400 bytes = 40 × 640 little-endian float32). Their sizes and SHA-256
values are checked at setup. The check also reads two reference-only files,
`anomaly-upstream-expected-smoke` and `anomaly-upstream-expected-validation`.

## Per-sample semantics

With `sat(v, z)` defined as upstream does: if $v \le -1024$ the result is
−128, and if $v \ge 1024$ it is 127. Otherwise, add ±0.5 in double, truncate
toward zero, add $z$, and clip to [−128, 127].

- Input: `q = sat(double(x / s0), z0)`, where the division is in float32.
- Each layer: `acc = Σ (a[i] − z_in) · w[o][i]` exactly, then
  `real = (double(acc) · s_in) · s_w + double(bias[o]) · s_bias` in double,
  left to right, then `q = sat(real / s_out, z_out)` in double.
  Layers 1–9 apply ReLU as `max(q, z_out)`.

[`contract.json`](contract.json) states every validation rule, the
accumulation bound, and the rounding argument.

## Mapping to upstream

Upstream: `test/applications/mlperf-tiny-anomaly/runner.cpp` at private Loom
commit `4a75f8d`. It has its own FlatBuffer reader, validates the model, and
runs the layers on the host.

| Upstream | Here |
|---|---|
| FlatBuffer reader and model validation | [`model.cpp`](model.cpp), written from the public TFLite schema; the same checks |
| Per-sample `invoke` | `inference_kernel` in [`anomaly.cu`](anomaly.cu) |
| Size checks on the model and dataset files | [`assets.cpp`](assets.cpp): exact size and SHA-256 |
| `expected-smoke.txt`, `expected-validation.txt` | used only to cross-check the CPU oracle |

The kernel runs one 256-thread block per sample. The activations stay in
shared memory. Warp $w$ computes output rows $w, w + 8, \ldots$, and each lane
reads a strided slice of the weight row. A shuffle tree sums the 32 int32
partials. The int32 sum is exact, because $|acc| \le 255 \cdot 128 \cdot 640
< 2^{31}$. Requantization uses explicit round-to-nearest double intrinsics
(`__dmul_rn`, `__dadd_rn`, `__ddiv_rn`), and the input division uses
`__fdiv_rn`. No step can fuse into an FMA, so the output matches the oracle
bitwise. The weights (264,192 bytes) stay resident in global memory and are
served from cache.

Upstream's smoke (0 warmup, 1 measured) and validation (2 warmup, 2 measured)
configurations are listed in `contract.json` for reference only. No target
runs them.

## Commands

```bash
make -C applications/loom_private/anomaly build
make -C applications/loom_private/anomaly check-large CHECK_JSON=/tmp/lp-anomaly-check.json
make -C applications/loom_private/anomaly test
```

The binary accepts `--set samples=N` with $1 \le N \le 2^{20}$, for tests
only (check action). Values outside that range exit 2.
`--asset name=path` replaces a cached asset.

`make test` first runs the host test [`tests/test_host.cpp`](tests/test_host.cpp),
which needs no GPU. It checks SHA-256 test vectors and that the pinned model
parses into the expected ten layers. It checks that corrupted copies (wrong
identifier, wrong size, bad root offset, schema version 2) are rejected. The
binary never reaches the parser with such copies, because their hash fails
first. A sweep of single-byte corruptions must either parse or be rejected,
never crash. The oracle must reject a non-finite input.

Then [`tests/run_tests.sh`](tests/run_tests.sh) runs these GPU cases:

- Oracle checks for 7, 93, 41 (Direct only), and 1,003 (Graph only) samples.
- Result-record checks: the override, the four asset hashes, and the layer widths.
- A mutant kernel that skips the last sample. It must exit 1 and report the
  mismatch.
- Asset failures that must exit 3: a byte flipped in a copy of the model, a
  missing model, a missing dataset, a truncated dataset, an oversized dataset,
  and a corrupt reference file.
- Invalid dimensions or usage that must exit 2.

## Oracle and tolerance

[`oracle.cpp`](oracle.cpp) is an independent scalar host implementation. It
uses only the parsed model data and the verified dataset. It infers the 40
unique dataset samples once, with int64 accumulation. The expected output for
sample $i$ is oracle sample $i \bmod 40$. Before the GPU runs, the oracle's
bytes for dataset samples 0, 2, and 3 are compared with upstream's reference
bytes (smoke measured sample 0, and validation measured samples 0 and 1 after
two warmups). Only bytes are compared; no upstream configuration is executed.

The comparison is exact. Every one of the 67,109,120 output bytes must equal
the oracle byte. The first five mismatches are printed with their sample,
element, expected value, and actual value.

## Timing scopes

- **Application:** upload all float32 samples from pinned host memory, run
  the inference kernel, and download the int8 outputs.
- **Compute:** the inference kernel with the samples already on the device.

Model loading, hashing, parsing, and validation are outside both scopes, as
are the weight upload, input materialization, allocation, graph capture,
warmups, the oracle, and checking. No timings are claimed here.

## Extension notes

- The kernel is a correctness baseline. A faster variant might stage the
  weights in shared memory, process several samples per block, or use
  `dp4a`. It must keep the exact integer sums and the same sequence of double
  operations to stay bitwise equal to the oracle.
- The dataset has only 40 unique samples. Any variant must still infer every
  sample and must not cache results across samples.
- Non-finite inputs are rejected at setup. A variant that streams
  unvalidated inputs would need a device-side status output.

# loom-private/vecadd-product

A batched CUDA extension of private Loom's `vecadd-product` application
(`test/applications/vecadd-product/vecadd_product.c` at commit
`4a75f8d743ef7fe755822bab0ea0fb78009c145b`). The source paths and their hashes
are listed in [`contract.json`](contract.json); nothing from private Loom is
copied here.

## What upstream does and what this adds

Upstream's product interface caches one 128-byte input (two 64-byte operands).
Each measured sample writes the lane-wise byte sum of the two operands, modulo
256, into its own 64-byte slot of the output. Warm-up samples are not measured;
they rewrite slot 0, which the first measured sample overwrites. Upstream's
host harness accepts at most 64 measured samples, and the same operands are
used for every sample.

This workload is an extension, not a direct port. It runs
$R = 1{,}048{,}576$ independent records, each with its own pair of 64-byte
operands, and one 64-byte output slot per record: a 128 MiB input and a 64 MiB
output.

## What it computes

The fixture is the 128-byte asset `vecadd-product-operands` (the upstream
`input.txt`), $f[0..127]$: the left operand is $f[0..63]$ and the right
operand $f[64..127]$. For record $r$, operand byte $j$ is
$(f[j] + (r \bmod 256)) \bmod 256$, so records repeat with period 256 and
neighbouring records differ in every lane. Output slot $r$, byte $k$, is
$(a_{r,k} + a_{r,64+k}) \bmod 256$, and each record writes only its own slot.
Record 0 equals upstream's single sample.

The records are materialized on the host in setup (untimed), into pinned
memory, and uploaded inside the timed application scope. The kernel uses one
thread per four output bytes and adds four byte lanes at once (the low seven
bits of each lane are added, then each lane's top bit is fixed up), so carries
never cross lanes. Every byte is exact integer arithmetic; no floating point
and no atomics are involved, and Direct and Graph launches agree bitwise.

`--set records=N` accepts $1 \le N \le 2^{24}$ (a 2 GiB input and 1 GiB
output); anything else exits 2. A missing, wrongly sized, or corrupt asset
exits 3; the fixture and the reference file are checked for size and SHA-256 at
setup.

## Commands

```bash
make -C applications/loom_private/vecadd_product build
make -C applications/loom_private/vecadd_product check-large CHECK_JSON=/tmp/lp-vecadd_product-check.json
make -C applications/loom_private/vecadd_product test
```

`check-large` runs the oracle once, then two Direct and two Graph invocations.
Each starts from poisoned outputs. `test` builds the host test, four mutants,
and a slow-warmup binary, and runs them under the GPU lock. No upstream smoke or
validation configuration is run by any target.

## Oracle and checks

[`oracle.cpp`](oracle.cpp) derives every record from the fixture with plain
integer `%` arithmetic and adds the lanes in a byte loop on up to 8
`std::thread` workers. It shares nothing with the kernel: not the packed-word
arithmetic, not the buffer layout. Helper code for SHA-256 and file reading is
in [`fixtures.cpp`](fixtures.cpp), which the oracle does not use.

Every output byte is compared exactly (the comparison is exact, so there is no
tolerance). The first eight mismatches are printed with the record, lane,
expected byte, and actual byte. Beyond the bytes themselves:

- **Upstream cross-check.** Setup parses the reference asset
  `vecadd-product-upstream-expected-smoke` (`measured_sample=0 output=<128
  hex>`). The oracle's record 0 must equal those 64 bytes. This compares
  bytes; it does not run upstream's smoke configuration.
- **Canaries.** The device output buffer has 256 guard bytes on each side of
  the exact $64R$-byte extent. They are filled with a known pattern when the
  outputs are poisoned and must be unchanged after the invocation. Input and
  output sizes are exactly $128R$ and $64R$ bytes.
- **Poison.** Before every invocation the device and host outputs hold a
  position-dependent byte pattern, so a skipped or stale slot fails. The
  device input holds every left operand complemented, so a skipped upload
  changes every output byte.
- **Wraparound and isolation.** Over one 256-record period about half of the
  lanes (8,165 of 16,384) have a true sum above 255. Adjacent slots differ in
  every lane by construction, so a sum landing in a neighbour's slot is caught.

## Tests (`make test`)

Host tests ([`tests/test_host.cpp`](tests/test_host.cpp)):

- SHA-256 known answers (empty, `abc`, the 56-byte two-block message, one
  million `a`) and the padding boundaries.
- The verified reader rejects a wrong size, a wrong hash, and a missing file;
  the reference-line parser rejects malformed lines.
- The oracle's record 0 equals upstream's reference bytes and the plain sum
  of the fixture's two operands.
- For 1,000 records: the closed form
  $(f[k] + f[64+k] + 2(r \bmod 256)) \bmod 256$, period 256, and neighbouring
  slots differing by 2 in every lane. Both wrapping and non-wrapping lanes
  occur.
- The threaded bulk oracle equals the per-record oracle at 1, 2, 7, 255, 257,
  and 10,007 records.

GPU tests ([`tests/run_gpu_tests.sh`](tests/run_gpu_tests.sh), under the GPU
lock):

- The real binary passes (Direct and Graph) at 1, 2, 3, 7, 63, 64, 255, 256,
  257, 1,000, 4,097, 65,537, and 1,000,003 records, plus Direct-only and
  Graph-only runs. The result record names the overrides, dimensions, asset
  hashes, and status.
- **Mutants** of the same source (`-DLP_VECADD_MUTANT=n`) must fail:
  saturating add instead of wraparound; one word written past the output
  extent (caught by the canary); each sum landing in its neighbour's slot;
  records with $r \bmod 256 = 255$ skipped (the poison shows, first at record
  255). The skipping mutant passes at 255 records, before the first skipped
  slot.
- `compute-sanitizer --tool memcheck` finds no out-of-bounds access at 1,000
  and 65,537 records.
- Missing, truncated, extended, and bit-flipped operand files, and a corrupt
  reference file, exit 3. Zero, negative, and above-limit records, a
  non-integer, an unknown key, an unknown preset, and `--set` with bench exit
  2.
- **Warmup exclusion.** A test binary (`-DLP_VECADD_TEST_SLOW_WARMUP`) makes
  the first three application invocations also spin the GPU for at least
  300 ms each. The test asserts that a bench with 3 warmups reports an
  application batch under 200 ms, and that the same bench with 0 warmups, where
  those three invocations land in the timed batch, reports one over 800 ms. So
  bench warmups are outside the timed interval. Every measured invocation also starts from
  poisoned outputs in check mode and must write all slots. This runs the large
  bench with 2 or 3 iterations only for timing records; it does not run the
  oracle and is not a performance claim.

## Timing scopes

- **Application:** upload of the 128 MiB of records from pinned host memory,
  the byte-sum kernel, and download of the 64 MiB of outputs to pinned host
  memory.
- **Compute:** the kernel with the records resident; no transfers.
- **Untimed:** fixture loading and verification, record materialization,
  allocation, poisoning, CUDA Graph construction, warmups, the oracle, and
  output checking.

No timings are reported yet.

## Extension notes

- The record rule (fixture plus record index mod 256) is a deliberately simple
  way to give each record distinct operands; changing it means changing both
  the setup loop in `vecadd_product.cu` and `oracle.cpp`, and the contract.
- The kernel is a straightforward one-pass copy-add and is not tuned. The work
  is bandwidth-bound by the transfers at application scope.

## Optimized variant

[`vecadd_product_optimized.cu`](vecadd_product_optimized.cu) implements the
same contract: the same dimensions, scopes, fixtures, oracle, exact check,
canaries, `--set` handling, asset handling, and poisoning (the workload class
is copied from the baseline). Build and run it with `VARIANT=optimized`:

```bash
make -C applications/loom_private/vecadd_product check-large VARIANT=optimized
make -C applications/loom_private/vecadd_product bench LAUNCH=graph VARIANT=optimized
make -C applications/loom_private/vecadd_product test VARIANT=optimized
```

### Why

The application scope is dominated by PCIe: a 128 MiB upload and a 64 MiB
download, against a kernel of 100 to 240 microseconds. The baseline issues
upload, kernel, and download in sequence on one stream, so the two copy
directions never overlap. The kernel is already close to DRAM bandwidth (ncu
on the 3090: 92 to 93% of peak sustained DRAM throughput; on the 5090: 89%), so
there is nothing to gain there, and the variant keeps the baseline's kernel.

A copy-only microbenchmark (128 MiB up and 64 MiB down, 16 chunks) shows what
overlap can buy. On the 3090 and 4090 the concurrent transfer takes only a few
percent longer than the upload alone, so the two directions are independent.
On the 5090 it takes nearly the sum of the two directions, so uploads and
downloads share a limit and the possible gain is a few percent.

### What changed

- **Chunked pipeline.** The records are split into equal chunks. Every chunk has
  its own stream: it uploads its inputs, runs the kernel over its records, and
  downloads its outputs, so later uploads overlap earlier downloads and
  kernels. The harness stream forks with an event, and joins on one event per
  chunk, so Direct launches and the captured CUDA Graph hold the same pipeline.
  The bytes moved are the baseline's: 128 bytes up and 64 bytes down per
  record.
- **Kernel launches.** Each chunk launches the baseline's kernel over its own
  range of words. The kernel keeps absolute word indices, so the mutants and
  the output-slot isolation (canaries, neighbour-slot sums) see the same
  indices as the baseline. The compute scope is one launch over all records.

### GPU-specific choices

The chunk count is chosen at setup from the compute capability (`chunks_for`).
Measured at 2 to 32 chunks on all three GPUs.

| GPU | Chunks | Reason |
|---|--:|---|
| RTX 3090 (sm_86) | 16 | The time falls steeply up to 8 chunks and is flat from 16 to 32; 16 is the smallest count on the plateau. A smaller last chunk (halving the final chunk) made no measurable difference. |
| RTX 4090 (sm_89) | 16 | Same plateau as the 3090. |
| RTX 5090 (sm_120) | 16 | The uploads and downloads share a limit, so every count from 1 to 32 is within a few percent of the baseline; 16 is at or near the best in Graph and Direct. |

`LP_VECADD_CHUNKS` (1 to 64) overrides the choice. The tests use it to run many
chunk counts; it never changes the result. The count is capped at the number of
records, so every chunk holds at least one record.

### Numerics

Exact integer arithmetic with the baseline's kernel, so every output byte equals
the oracle's, the canaries are untouched, and Direct and Graph outputs agree
bitwise. Nothing is cached between invocations: every invocation uploads all
inputs and downloads all outputs.

### Tests (`make test VARIANT=optimized`)

Everything the baseline's `make test` runs, against the optimized binary
(mutants, memcheck, assets, invalid input, the bench path, and the warmup
exclusion test), plus:

- chunk counts 1, 2, 3, 5, 16, and 64 at record counts 1, 2, 3, 7, 255, 257,
  1,000, 65,537, and 1,000,003 (more chunks than records, uneven chunks, and
  chunk starts off the 256-record period included), in Direct and Graph;
- mutants 5 and 6, which skip the last chunk's upload or download, at 1, 4, and
  64 chunks and at the large size; each must fail the check;
- mutants 1 to 4 again with the records split into chunks, which must still
  fail (the kernel keeps absolute record indices);
- `compute-sanitizer --tool memcheck` with 3 and 64 chunks;
- a bench record that names the variant.

The test binaries carry an `_optimized` suffix next to the baseline's.

# loom-private/vecadd-memory

Element-wise int32 vector addition. This is the CUDA port of private Loom's
`vecadd-memory` application (`test/applications/vecadd-memory/vecadd_memory.c`
at commit `4a75f8d743ef7fe755822bab0ea0fb78009c145b`). The source paths and
their hashes are listed in [`contract.json`](contract.json); nothing from
private Loom is copied here.

It is a new workload identity. It keeps the element count of the legacy
standalone `vec_add.cu` (1,048,576) but adopts private Loom's integer
contract: upstream's inputs, the exact integer check, and the 64-bit checksum.
The upstream default length is 64 and its preset lengths are 4096, 65536, and
131072, so the large size is an extension of upstream, recorded in the
contract's `extension_of_upstream` field.

## What it computes

The large workload has $N = 1{,}048{,}576$ int32 elements, multiplier 2, and
bias 0 (upstream's defaults for the last two).

- **Inputs.** $\text{lhs}[i] = i$ and $\text{rhs}[i] = i \cdot \text{multiplier} + \text{bias}$.
- **Output.** $\text{result}[i] = \text{lhs}[i] + \text{rhs}[i]$, which equals
  $i\,(\text{multiplier} + 1) + \text{bias}$.
- **Checksum.** The sum of all results in a signed 64-bit integer:
  $3 \cdot N(N-1)/2 = 1{,}649{,}265{,}868{,}800$ at the large size. The oracle
  computes it.

`length`, `multiplier`, and `bias` can be overridden with `--set` for tests.
Invalid values exit 2 with a message: a length below 1 or above $2^{27}$, a
multiplier or bias outside int32, or any lhs, rhs, or result value that would
overflow int32. The values are linear in $i$, so the last element holds the
extremes. The guard is exact (128-bit arithmetic) and keeps every add free of
signed overflow. Upstream has no such guard because its sizes are fixed.

## GPU implementation

[`vecadd_memory.cu`](vecadd_memory.cu) launches one kernel with one thread per
element (256-thread blocks) and a bounds guard. That guard is the whole tail
handling; there is no vector-width path, so the only tail is the block size.
The input vectors are generated on the host into pinned memory. There is no
tuning, no atomics, and no floating point, so every invocation is
deterministic, and Direct and Graph launches agree bitwise.

## Commands

```bash
make -C applications/loom_private/vecadd_memory build
make -C applications/loom_private/vecadd_memory check-large CHECK_JSON=/tmp/lp-vecadd_memory-check.json
make -C applications/loom_private/vecadd_memory test
```

`check-large` runs the oracle once, then two Direct and two Graph invocations.
Each invocation starts from poisoned outputs. `test` builds two test-only
binaries and runs them with small `--set length=… --set multiplier=… --set
bias=…` overrides. None of them matches an upstream smoke or validation
configuration.

## Oracle and tolerance

[`oracle.cpp`](oracle.cpp) is an independent host implementation. It builds
both inputs in 64-bit arithmetic, adds them element by element, narrows to
int32, and accumulates the checksum. It shares no code with the kernel or the
workload's input generation. Its `validate` function holds the dimension
rules, which `vecadd_memory.cu` calls so that the host test can check them.

The comparison is exact. Every result element must equal the oracle's, and
the host-side 64-bit sum of the downloaded results must equal the oracle's
checksum. A mismatch exits 1 and prints the first eight indices with
expected and actual values. Poisoning writes the bitwise complement of each
expected value to both the host and the device result buffers, so no element
holds its correct value before the run and a skipped write cannot pass. The
device inputs are set to the bitwise complement of the host inputs, so a
skipped upload of either one changes every sum.

## Tests (`make test`)

Host tests ([`tests/test_host.cpp`](tests/test_host.cpp)):

- The oracle's large checksum is 1,649,265,868,800, and its elements match
  the formula at several indices.
- At eight small dimension sets the oracle matches the closed-form checksum
  and the per-element formula.
- Validation boundaries: lengths 0, −5, 1, $2^{27}$, and $2^{27}+1$; results
  and rhs values exactly at and one past the int32 limits; multiplier and
  bias outside int32; the large length with an overflowing multiplier.

GPU tests ([`tests/run_gpu_tests.sh`](tests/run_gpu_tests.sh), under the GPU
lock):

- **Tail handling.** The binary passes at lengths 1, 2, 255, 256, 257,
  1,000,003, and 5,000, in both launch modes or Direct or Graph alone. The
  result record carries the overrides and dimensions.
- **Other values.** Positive, negative, and zero multipliers and biases, values
  exactly at the int32 limits, and the large length with multiplier 2047
  (checksum far beyond int32).
- **Dropped-element regression.** `vecadd_memory_drop_last_mutant` is the same
  source compiled with `-DLP_VECADD_MUTANT_DROP_LAST`, whose kernel skips the
  last element. It must exit 1 at lengths 1,000,003 and 1 and report the index
  and the checksum mismatch.
- **Invalid input exits 2.** Length 0, −4, and $2^{27}+1$; multiplier or bias
  that overflows the result or the rhs, at small and at large length; negative
  overflow; multiplier or bias outside int32; a non-integer value; an unknown
  key; an unknown preset; and `--set` with bench.

## Timing scopes

- **Application:** upload of lhs and rhs from pinned host memory, the add, and
  download of result to pinned host memory.
- **Compute:** the add with lhs and rhs resident on the device; no transfers.
- **Untimed:** input generation, allocation, CUDA Graph construction,
  warmups, the oracle, and output checks.

No timings are reported yet.

## Extension notes

- The add is memory bound. A faster version could use 128-bit vector loads
  and stores or a grid-stride loop; any vector path needs tail tests at
  lengths that are not a multiple of its width.
- The oracle's `validate` bound of $2^{27}$ elements only limits the pinned and
  device allocations; raising it needs no kernel change up to `INT_MAX`.

## Optimized variant

[`vecadd_memory_optimized.cu`](vecadd_memory_optimized.cu) implements the same
contract: the same dimensions, scopes, oracle, exact check, `--set` handling,
and poisoning (the workload class is copied from the baseline). Build and run
it with `VARIANT=optimized`:

```bash
make -C applications/loom_private/vecadd_memory check-large VARIANT=optimized
make -C applications/loom_private/vecadd_memory bench LAUNCH=graph VARIANT=optimized
make -C applications/loom_private/vecadd_memory test VARIANT=optimized
```

### Why

The application scope is dominated by PCIe: two 4 MiB uploads and a 4 MiB
download, against a kernel of a few microseconds (the 4090 and 5090 serve it
from L2). The baseline issues them in sequence on one stream, so the two
copy directions never overlap. A copy-only microbenchmark (128 MiB up and
64 MiB down, 16 chunks) shows what overlap can buy: on the 3090 and 4090 the
concurrent transfer takes about the same time as the upload alone, so the two
directions are independent. On the 5090 the concurrent transfer takes nearly
the sum of the two directions, so uploads and downloads share a limit, and
only a small gain is available there.

### What changed

- **Chunked pipeline.** Each invocation splits the vectors into chunks whose
  sizes are multiples of four elements. Every chunk has its own stream: it
  uploads, runs the kernel, and downloads, so later uploads overlap earlier
  downloads. The harness stream forks with an event, and joins on one event per
  chunk, so Direct launches and the captured CUDA Graph hold the same pipeline.
  The bytes moved are the baseline's: lhs and rhs up, the result down.
- **One copy for both inputs.** The host and device inputs are one allocation
  each, with rhs one pitch (a multiple of 256 bytes) after lhs, so a single
  `cudaMemcpy2DAsync` of height 2 uploads a chunk of lhs and rhs. That halves
  the upload commands.
- **Kernel.** Each thread loads and stores four elements through 16-byte
  accesses. The last chunk's final (length mod 4) elements take a scalar tail
  in the first thread. The same kernel serves the compute scope, one launch
  over the whole length. The sums are the baseline's integer adds, so the
  output is bit-exact.

### GPU-specific choices

Chosen at setup from the compute capability (`tuning_for`); each was measured
with the chunk counts 1 to 16 and the kernel shapes below on all three GPUs.

| GPU | Chunks | Kernel shape | Reason |
|---|--:|---|---|
| RTX 3090 (sm_86) | 8 | 1 vector per thread, streaming hints | Best chunk count; the kernel is DRAM bound (its inputs do not fit L2), and `ld.global.cs`/`st.global.cs` was the fastest shape. |
| RTX 4090 (sm_89) | 8 | 4 vectors per thread | Best chunk count; the kernel is launch and L2 bound, and 4 vectors per thread was the fastest shape. |
| RTX 5090 (sm_120) | 4 | 2 vectors per thread, streaming hints | Best chunk count (the PCIe directions share a limit, so fewer, larger chunks win); streaming hints with 2 vectors per thread was the fastest shape. |

More chunks than the best add per-copy overhead (about 5 to 10 microseconds
each); fewer leave the first upload and the last download exposed. The kernel
differences are a few percent at most, because the compute scope is a few
microseconds of mostly launch and DRAM latency.

`LP_VECADD_CHUNKS` (1 to 64) and `LP_VECADD_KERNEL` (0, 1, 2 for the rows
above) override the choices. The tests use them to run every path on every GPU;
they never change the result.

### Numerics

Exact integer arithmetic, so the result equals the baseline's and the oracle's
at every element, and the 64-bit checksum equals the oracle's. Direct and Graph
outputs agree bitwise (the harness checks the digests), and nothing is cached
between invocations: every invocation uploads both inputs and downloads the
whole result.

### Tests (`make test VARIANT=optimized`)

Everything the baseline's `make test` runs, against the optimized binary, plus:

- every combination of 4 chunk counts (1, 3, 8, 64; more chunks than vectors
  included), the three kernel shapes, and lengths 1, 5, 257, 5,000, 1,000,003,
  and 1,000,004, in Direct and Graph, and the large size in every kernel shape;
- the drop-last mutant in every kernel shape, at a length whose last element
  is in the 16-byte path and at one whose last element is in the scalar tail;
- two mutants that skip the last chunk's upload or download, at 1, 4, and 64
  chunks and at the large size; each must fail the check;
- `compute-sanitizer --tool memcheck` at several chunk counts and at length 5;
- a short bench in both launch modes, whose record names the variant.

The test binaries carry an `_optimized` suffix next to the baseline's.

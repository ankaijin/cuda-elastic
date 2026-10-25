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

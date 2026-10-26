# llama2c decoder (`loom-private/llama2c-decoder`)

Greedy decoding through llama2.c's last-layer kernels (`rmsnorm`, `matmul`,
`softmax`, `sample_argmax`) on a synthetic model. The workload decodes 1,024
independent sequences, each for 4 dependent steps.

## What it computes

The synthetic model works as follows. The embedding row for token $t$ is 1.0
everywhere except 4.0 at column $t$. The RMSNorm weight is 1.0. Classifier
row $r$ is 0 except for a 1.0 at column $(r + V - 1) \bmod V$. Sequence $b$
starts from token $b \bmod V$. Each step does the following for every
sequence:

1. **rmsnorm** of the current token's embedding row, computed as llama2.c
   does it: $ss = \sum x^2$, then $ss/W + 10^{-5}$, then
   $\text{scale} = 1/\sqrt{ss}$, then $o_j = w_j(\text{scale}\cdot x_j)$.
2. **Dense classifier matmul.** All $V$ rows are multiplied against all $W$
   columns, giving $V$ logits.
3. **softmax**: subtract the maximum, apply `expf`, then divide by the sum.
4. **argmax**: take the first maximum. That index is the next token, and it
   is written back on the GPU for the next step.

For this model the token advances by one each step, modulo $V$. The GPU
nevertheless does the full dense computation at every step and never uses
that shortcut.

| | Large |
|---|---|
| batch $B$ | 1,024 |
| width $W$ | 16,384 |
| vocabulary $V$ (classifier rows) | 8,192 |
| steps | 4 (4,096 generated tokens) |
| model | 1 GiB of float32 (embedding + classifier), resident |
| classifier work | $2^{37}$ multiply-adds per step |

## Mapping to upstream

Upstream is private Loom `4a75f8d`, file `test/applications/llama2c-kernels-smoke.h`.
It wraps llama2.c `run.c` at `350e04fe`. The blob and SHA-256 hashes are in
[`contract.json`](contract.json). Upstream decodes a single sequence. This
port runs a batch of independent sequences, which makes it an extension. The
model, the per-sequence arithmetic, the token check, and the invariant
constants all follow upstream.

Upstream copies the embedding row into an activation buffer before calling
`rmsnorm`. Here the kernel reads the embedding row directly, which gives the
same values. The classifier kernel is a tiled batch matmul. Each output is
accumulated by a single thread, in increasing column order, starting from
`0.0f`, with a separate multiply and add. That is the same order llama2.c's
`matmul` uses.

The rmsnorm square sum and the softmax reductions use fixed tree orders, so
results are deterministic. For this model the square sum is $W + 15$, and the
binary limits $W \le 2^{24} - 16$, so every partial sum is an exact float
integer in any order. The upstream smoke configuration (width 128, 16 rows)
and validation configuration (width 1024, 64 rows) are recorded in the
contract as references only. No target runs them.

## Commands

```bash
make -C applications/loom_private/decoder build
make -C applications/loom_private/decoder check-large   # CHECK_JSON=path for the record
make -C applications/loom_private/decoder test
```

`test` runs [`tests/run_tests.sh`](tests/run_tests.sh) under one GPU lock. It
covers five correct shapes given with `--set batch=… width=… vocab=…
steps=…`. These include prompt and token wraparound, the minimum shape
$V = W = 2$, $V = W$, ragged tiles, and a Graph-only run. It also covers ten
invalid inputs that must exit 2, and three injected faults that must exit 1
and report the offending index with its expected and actual values. The fault
hook is `--set fault=1|2|3`: it corrupts one token, one activation, or one
probability on the host after download, in check mode only.

## Oracle and tolerances

The CPU oracle lives in [`oracle.cpp`](oracle.cpp), is host only, and shares
no code with the kernels. It checks four things:

- **Tokens.** Every generated token must exactly match the analytic
  progression $t \mapsto (t+1) \bmod V$.
- **Normalization invariants**, for every sequence at every step. Upstream
  checks only the final step. The mean of squares of the normalized
  activation must be within $2\times10^{-5} + 1.2\times10^{-7}W$ of 1. The
  probability sum must be within $10^{-6} + 1.2\times10^{-7}V$ of 1. Both sums
  are taken in double. A NaN fails.
- **Dense double evaluation.** For 8 sequences spread across the batch
  (including the first and last), the oracle evaluates every step densely in
  double: every classifier row against every column. GPU normalized values
  and logits must match within $2\times10^{-6}$ relative, and probabilities
  within $2\times10^{-5}$ relative. The contract gives the error budget. At the
  large shape, the worst observed relative errors are about $3\times10^{-8}$
  for normalized values and logits, and $2\times10^{-7}$ for probabilities.
- **Oracle self-check.** The oracle's dense argmax must reproduce the analytic
  progression.

## Timing scopes

- **Application scope:** prompt upload (pinned), device token reset, all
  steps, and the generated-token download (pinned).
- **Compute scope:** the token reset from the resident prompts, plus all
  steps.

Setup generates the weights on the device. Setup is untimed, and the weights
and working buffers stay resident. The per-step normalized, logit, and
probability buffers are check diagnostics. They are copied only in
`check_outputs`, which is untimed, and they are poisoned before every
invocation like every other output. No performance numbers are claimed.

## Extension notes

- The per-step diagnostic buffers take $4 \cdot \text{steps}\cdot B(W+2V)$
  bytes on both the device and the host (512 MiB at the large shape). A
  timing-only build could keep just the last step's buffers.
- The classifier kernel is a plain 64×64 shared-memory tile, with 4×4
  outputs per thread and no FMA. Faster variants must keep the
  per-output accumulation order, or change the contract.
- Real llama2.c weights would make the rmsnorm tree reduction differ from
  the sequential sum by rounding. The bitwise argument above holds only for
  this synthetic model.

## Optimized variant

`make … VARIANT=optimized` builds [`decoder_optimized.cu`](decoder_optimized.cu)
as `decoder_optimized`. It implements the same contract, with the same
dimensions, scopes, inputs, oracle, and check. It reuses the baseline's
rmsnorm and softmax+argmax kernels and replaces only the classifier matmul.

```bash
make -C applications/loom_private/decoder check-large VARIANT=optimized
make -C applications/loom_private/decoder bench VARIANT=optimized LAUNCH=direct
make -C applications/loom_private/decoder test VARIANT=optimized
```

**Why the classifier.** Profiling the baseline (nsys) puts the classifier at
98.7% of GPU time on all three GPUs. Its 64×64 shared-memory tile reaches
about 42% of the FP32 ceiling on the RTX 3090. `--fmad=false` makes every
multiply-add a separate FMUL and FADD, so the ceiling is one multiply-add per
two FP32 lane-cycles.

**Classifier kernel.** It is a register-tiled SGEMM. Both operands are
K-contiguous. 4-byte `cp.async` copies transpose each stage of 8 or 16
columns into k-major shared arrays. Each array is padded by 4 floats, and
each warp copies four 32-byte runs per instruction, so the copies are
coalesced and conflict-free. The copies run 3–4 stages ahead of the
arithmetic. Each thread owns 16 sequences × 8 classifier rows and reads one
column of them with six float4 shared loads. These fragments are
double-buffered in registers, so column k + 1 loads while column k is
multiplied. A block is 8 warps computing a 128 × 256 tile.

Stream-K hybrid schedule:

- The host sizes a persistent grid to the number of resident blocks (the
  occupancy API times the SM count). This keeps waits short but is not needed
  for progress (see below).
- Whole waves of tiles are computed data-parallel.
- The k-stages of the last one to two waves' tiles are split into equal
  contiguous ranges, one per block.
- A tile split between two blocks keeps its exact sequential sum. The block
  whose range ends inside the tile computes that tile's first stages before
  anything else, then publishes the running sums with a flag. The next block
  finishes its own whole tiles, then waits for the flag and continues the
  same sums at the next stage.
- The flag is set with a release store after the block's barrier, and read
  with an acquire load.
- The flags are reset by a memset before every launch. The hand-off buffer
  and the flags are poisoned with the outputs. Poisoned flags read as ready,
  so a launch that skipped the reset would let a consumer that reaches its
  wait early resume from NaN sums. Because heads are published first, that
  rarely happens, so this guards the reset rather than being a reliable test
  of it.
- Progress does not need the whole grid resident. Block b waits only on
  block b − 1, and every block publishes its head before any of its own
  work and before its own wait. So the waits complete whenever blocks are
  dispatched in index order, as the hardware does for a 1-D grid, even with
  one block resident at a time.

**Numerics.** Each logit is still accumulated by one thread, starting at
`0.0f`, in increasing column order, with a separate multiply and add. This is
the baseline's and llama2.c's order, also across a stream-K hand-off. Every
output is therefore bitwise identical to the baseline's. The large digest is
the baseline's `4fcc3e102976547b` on all three GPUs, and the tests compare
digests with the baseline binary.

At the edges:

- Columns past `width` are copied as zeros. Adding `+0` to an accumulator
  that started at `+0.0f` never changes it under IEEE round-to-nearest
  without flush-to-zero, which the build flags do not enable.
- Lines past `batch` or `vocab` are clamped. They only feed outputs that are
  never stored.

The bitwise claim for general weights rests on this sequential-order design,
not on the tests. The synthetic model gives each logit exactly one nonzero
product, so any summation order (even `fmaf`) yields the same bits there.
The digest tests prove that every split and stage boundary covers each
column exactly once, but they cannot detect a reordered sum.

The worst relative errors at the large shape are the baseline's: normalized
values and logits at 3.0e-8 against the 2e-6 tolerance (0.015 of it), and
probabilities at 2.1e-7 against 2e-5 (0.010 of it).

**Per-GPU choices.** The tile is selected at setup from `cudaDeviceProp`.

| GPU | Tile (sequences × rows × columns per stage) | Stages | Schedule |
|---|---|--:|---|
| RTX 3090 (sm_86, 82 SMs) | 128 × 256 × 16 | 3 | stream-K hybrid: 256 tiles on 82 resident blocks would leave a 12% last wave |
| RTX 4090 (sm_89, 128 SMs) | 128 × 256 × 16 | 3 | data parallel: 256 tiles are exactly two waves, so stream-K splits nothing |
| RTX 5090 (sm_120, 170 SMs) | 128 × 256 × 8 | 4 | stream-K hybrid: 256 tiles on 170 blocks would leave a half-empty second wave |

Tile notes:

- 16 × 8 outputs per thread (255 registers, one block per SM) beat
  8 × 8 per thread at two blocks per SM on every GPU. The extra arithmetic
  per shared load and per barrier outweighs the lower warp count.
- On the RTX 5090, 8-column stages measured slightly ahead of 16-column
  stages. On the 3090 and 4090, 16-column stages won because they halve the
  barriers.
- With the chosen tiles, ncu reports the FP32 (FMA) pipe at 81% of peak and
  issue slots at 86% busy on the RTX 3090. On the RTX 5090 the figures are
  80% and 84%. FMUL and FADD counts equal the multiply-adds exactly.
- The packed `f32x2` instructions compile to scalar FMUL/FADD on sm_120 and
  were not used.
- The remaining stalls are spread over register-bank dispatch conflicts,
  instruction-cache misses, and copy-queue throttling.

**Tests** ([`tests/run_optimized_tests.sh`](tests/run_optimized_tests.sh)).
`make test VARIANT=optimized` builds a test-only binary, `decoder_optimized_tuning`,
with `-DLP_DECODER_TEST_TUNING`. That binary honours three environment hooks;
the production `decoder_optimized` never reads the environment.

- `DECODER_OPT_TILE=128x256x16|128x256x8|128x128x8` selects a tile.
- `DECODER_OPT_SCHEDULE=dp|streamk` selects the schedule.
- `DECODER_OPT_CTAS=n` sets the persistent grid, at most the resident blocks.

Bad hook values exit 2. The test binary prints the plan it applied, and the
tests check it. The tests cover:

- All baseline cases, run on the production binary, and a check that the
  production binary ignores the hooks.
- Five shapes with tails in every dimension (odd width, ragged tiles,
  wraparound, and two single-tile shapes). Each runs on every tile
  configuration with both schedules and with forced persistent grids of 1,
  5, 7, and 11 blocks. A grid of 1 and the single-tile shapes run purely
  data-parallel; 5, 7, and 11 blocks split tiles on the multi-tile shapes.
  Each run must apply the forced plan, pass the check, and match the
  baseline binary's digest.
- Checks that grids of 5, 7, and 11 blocks really split tiles.
- Invalid hook values, which must exit 2.
- Two kernel mutants, built from the test binary, which the check must
  reject with exit 1. Mutant 1 drops the stream-K hand-off; mutant 2 drops
  the last stage of every tile.

**Not changed.** rmsnorm and softmax+argmax are the baseline kernels. Together
they are about 2–3% of the optimized time, and changing their reduction trees
would change the probabilities' bits.

# Experiment C: lower precision Tensor Core classifier

**Question.** What accuracy–speed tradeoff results from replacing only the
dense classifier multiplication?

**Workload.** `loom-private/llama2c-decoder`, optimized variant: 1,024
sequences, four dependent steps, width 16,384, vocabulary 8,192. Each step
runs the production rmsnorm, the classifier, and the production
softmax+argmax with on-device token feedback. Only the classifier changes.
[`decoder_lowprec.cu`](decoder_lowprec.cu) compiles the production source
`applications/loom_private/decoder/decoder_optimized.cu` into the same
translation unit, unchanged, with its `main` renamed. So the other kernels,
and the FP32 classifier itself, are the production code.

[`spec.json`](spec.json) is the experimental specification. It defines the
cases, formats, rounding, saturation, zero blocks, scale layouts, fixtures,
checks, and metrics, and every binary embeds its SHA-256. The low precision
cases do not meet the contract's FP32 tolerances by design. Their check is the
specification's implementation check, never a production correctness pass.

| Case | Classifier | Per-step conversion (timed) |
|---|---|---|
| `fp32` (control) | Production register-tiled SGEMM with stream-K | none |
| `bf16` | CUTLASS 2.x `device::Gemm`, Sm80 tensor-op path (`mma.sync`), 128×256×32, 3 stages | FP32 → BF16, round to nearest even |
| `mxfp6` | CUTLASS 4.8.0 SM120 block-scaled GEMM, `mx_float6_t<float_e3m2_t>` × same, 128×128×128 cooperative | E3M2 codes and UE8M0 scales per 32 values |
| `nvfp4` | The same GEMM with `nv_float4_t<float_e2m1_t>`, alpha from device memory | tensor maximum; E2M1 codes and UE4M3 scales per 16 values; alpha |

- All GEMMs accumulate and output in FP32, densely, without split-K.
- Weights are converted once at setup, outside the timed scopes. The
  accuracy mode reports that conversion time separately.
- All cases build for `sm_120f` (`-gencode arch=compute_120f,code=sm_120f`)
  with the production flags, including `--fmad=false`.
- CUTLASS 4.8.0 is pinned at `098de2a652cf8f00fd70b2df54051c7eccbb855a` and
  cached outside the repository. `make cutlass` fetches it, and every build
  checks the commit and a clean checkout.
- CUTLASS 4.8.0 has no SM120 collective for 16-bit inputs, so the BF16 control
  uses CUTLASS's Sm80 `mma.sync` path. GeForce Blackwell runs BF16 Tensor Core
  math through `mma.sync` anyway. Its default 128×256×64 tile needs about
  147 KB of shared memory, over SM120's per-block limit, so the case uses
  CUTLASS's standard 128×256×32 three-stage configuration. No other tile was
  tried.

### Formats in brief

- **MXFP6** follows OCP MX v1.0. Each block of 32 values shares the exponent
  $X = \lfloor \log_2 a \rfloor - 4$, where $a$ is the block's largest
  magnitude and 4 is E3M2's largest exponent. Elements are
  $\mathrm{E3M2}(v \cdot 2^{-X})$, rounded to nearest even and saturated at 28.
  A block whose largest magnitude has a mantissa above 1.75 therefore clips
  its largest element by up to 12.5%. This is a property of the standard
  recipe.
- **NVFP4** uses the two-level recipe. The tensor encode scale is
  $s = 2688 / a_t$, from the tensor's largest magnitude $a_t$. Each block of
  16 values has the UE4M3 scale $f = \mathrm{UE4M3}((a/6) s)$, and elements
  are $\mathrm{E2M1}(v s / f)$. The GEMM multiplies by
  $\alpha = s_A^{-1} s_W^{-1}$, read from device memory each step.
- The device quantizer uses the hardware conversions
  (`cvt.rn.satfinite.{e2m1x2,e3m2x2,e4m3x2}`) and rewrites their −0 results as
  +0. The host reference ([`reference.cpp`](reference.cpp)) finds the same
  codes by searching the value tables. The two must agree bit for bit.
- Scale factors use CUTLASS's `Sm1xxBlkScaledConfig` layout. Both the device
  writer and the host reader index it through CUTLASS's own layout object.

## Checks and tests

The implementation check (`--action check`, spec.json
`implementation_check`) compares each stage with an independent host
computation from the GPU's own inputs to that stage:

- normalized rows against double rmsnorm of the token the GPU fed back;
- converted activations (every step) and weights, bit for bit;
- logits of 8 sequences per step against the double product of the operands
  the GPU multiplied, with a bound that holds for FP32 accumulation in any
  order;
- probabilities against double softmax of the GPU's logits;
- tokens against the first largest probability.

The FP32 control must also pass the production contract check unchanged. In
Direct and Graph mode, invocations of one case must agree bitwise.

`make test` ([`tests/run_tests.sh`](tests/run_tests.sh)) runs:

- [`tests/test_formats.cu`](tests/test_formats.cu): the reference's decoding
  of every code and its sub-byte packing against CUTLASS's own types, plus
  ties, saturation, and zero and tiny blocks;
- `--selftest` per case: the device quantizer against the host reference on
  crafted rows (element and scale ties, saturation, zero, tiny, and
  subnormal-scale blocks, 130 rows so a second scale-factor tile is used);
- the implementation check of every case on the three dense fixtures, a
  synthetic model with ragged tiles (batch 100, vocabulary 200), and one
  sequence;
- six mutants (rounding, scale layout, alpha), each of which must fail both
  the self-test and the check;
- `compute-sanitizer` memcheck of each low precision case.

## Accuracy (`--accuracy`)

Each case reports, for the primary model and for the three dense fixtures
(seeds 1, 2, 3; 64 sequences, width 1,024, vocabulary 128; normal embeddings
and a normal classifier scaled by $1/\sqrt{\text{width}}$):

- **classifier error** on fixed FP32 reference activations (the step-0 rows),
  against double products of the FP32 operands: maximum absolute error,
  normalized RMS error, maximum absolute probability error, mean total
  variation distance, and top-1 agreement;
- **token agreement** over a complete four-step decode, against the
  contract's analytic progression (primary) or a double-precision host decode
  (fixtures);
- the **production FP32 tolerance check** of the primary decode, reported
  separately as pass or fail;
- the weight conversion time (median of five).

The primary classifier has one nonzero weight per row. Its errors reflect only
the activation format, and the dense fixtures are needed to measure general
behavior.

## Results on kilia's RTX 5090

Runs `20261030-190650-kilia-ccdb9517` (check, accuracy, five pairs),
`20261030-191202-kilia-f746f49a` (ten pairs), and
`20261030-191548-kilia-e2d793e7` (profile). They ran at `38dd798` and two
descendants that add only recorded evidence, with CUDA 13.4, driver
615.71.10, and CUTLASS 4.8.0 `098de2a`. The generated report
(`results/blackwell_experiments/`) lists every value. In brief:

**Implementation correctness.** All four cases pass the implementation
check in Direct and Graph mode, with converted weights and every step's
activations bit-identical to the host quantizer. They also pass all 42 tests,
including every mutant and memcheck.

**Speed.** Median Graph application-scope time for the four-step decode, ten
pairs, with every pair faster than the control:

| Case | Time (µs) | Ratio to FP32 | Speedup |
|---|--:|--:|--:|
| `fp32` | 25,285.8 | 1 | 1 |
| `bf16` | 6,802.5 | 0.2691 | 3.7× |
| `mxfp6` | 3,816.6 | 0.1510 | 6.6× |
| `nvfp4` | 1,685.8 | 0.0667 | 15.0× |

Profiled kernel time (one Graph invocation, kernels serialized) splits as
follows:

- `fp32`: the classifier takes 97.8%.
- `bf16`: the GEMM takes 89.8% and the conversion 2.3%.
- `mxfp6`: the GEMM takes 81.5% and the quantization 4.2%.
- `nvfp4`: the GEMM takes 53.0%, the tensor maximum and the quantization
  17.1%, and the unchanged rmsnorm and softmax kernels 29.9%.

Weight conversion at setup takes 0.51 ms (BF16), 0.42 ms (MXFP6), and
0.69 ms (NVFP4).

**Numerical quality.** Classifier error on fixed FP32 reference activations,
and token agreement over the complete decode:

| Case | Primary: normalized RMS error | Primary: tokens | Primary: FP32 tolerance check | Fixtures: normalized RMS error | Fixtures: tokens (of 256) |
|---|--:|--:|---|--:|---|
| `fp32` | 0 | 4,096/4,096 | pass | 5.7e-7 to 5.8e-7 | 256, 256, 256 |
| `bf16` | 4.6e-4 | 4,096/4,096 | fail | 0.0023 | 249, 256, 247 |
| `mxfp6` | 0.124 | 4,096/4,096 | fail | 0.075 to 0.077 | 174, 195, 204 |
| `nvfp4` | 6.0e-8 | 4,096/4,096 | pass | 0.134 to 0.136 | 163, 161, 171 |

- The primary model has one weight of 1.0 per classifier row, and its
  normalized values are $\approx 1$ and $\approx 4$. NVFP4's two-level
  scaling represents them almost exactly. That is why it passes the
  production FP32 tolerance check there. The dense fixtures show its general
  error.
- MXFP6's primary error comes from the OCP shared exponent. A block whose
  maximum is $0.9995 = 1.999 \times 2^{-1}$ gets $X = -5$, so the value
  scales to 31.98 and saturates at 28. The decoded value is 0.875.
- On the fixtures, token agreement falls step by step. A token that differs
  sends that sequence down another path, and each later step compares with
  the reference path.

**Conclusions.** All three candidates are **improved** in speed, beyond
measurement variation, and all are correctly implemented. They trade
accuracy as follows:

- BF16 keeps the fixtures' logits within about 0.2% and changes 0–9 of 256
  tokens, at 3.7× the speed.
- MXFP6 (6.6×) and NVFP4 (15×) change about 20–37% of the fixtures' tokens
  over four steps.

Only the FP32 control meets the contract's tolerances in general. NVFP4 meets
them only on the synthetic primary model. Whether any loss is acceptable is
an application decision. This experiment does not promote a case.

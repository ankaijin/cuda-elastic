// Experiment C (experiments/blackwell/lowprec): the production optimized
// llama2c decoder with only its dense classifier multiplication replaced.
// spec.json defines every case, format, check, and metric.
//
// LP_LOWPREC_CASE selects the classifier at compile time:
//   0 fp32   the production classifier (register-tiled SGEMM, stream-K)
//   1 bf16   activations and weights in BF16, CUTLASS 2.x mma.sync GEMM
//   2 mxfp6  MXFP6 E3M2 with UE8M0 scales per 32, CUTLASS SM120 block-scaled GEMM
//   3 nvfp4  NVFP4 E2M1 with UE4M3 scales per 16 and per-tensor FP32 scales,
//            CUTLASS SM120 block-scaled GEMM with a device alpha
// The production source is compiled into this translation unit unchanged (its
// main is renamed), so rmsnorm, softmax+argmax, the model generator, and the
// FP32 classifier are the production kernels themselves.
//
// Each step: rmsnorm (production) -> activation conversion (low precision
// cases: one kernel, plus a tensor-maximum kernel for NVFP4) -> classifier ->
// softmax+argmax (production) with on-device token feedback. Weights are
// converted once at setup, outside every timed scope.
//
// Modes:
//   (harness)        --action check|bench|profile, as every loom_private app;
//                    the check is spec.json "implementation_check"
//   --accuracy       spec.json "accuracy" on the primary model and the three
//                    fixtures; writes its own JSON record (--json-out PATH)
//   --selftest       the device quantizer against the host reference on
//                    crafted rows (ties, saturation, zero and tiny blocks,
//                    scale-factor tiles past 128 rows); exit 0 or 1
#ifndef LP_LOWPREC_CASE
#error "build with -DLP_LOWPREC_CASE=0 (fp32), 1 (bf16), 2 (mxfp6) or 3 (nvfp4)"
#endif
#define LP_CASE_FP32 0
#define LP_CASE_BF16 1
#define LP_CASE_MXFP6 2
#define LP_CASE_NVFP4 3
#define LP_BLOCK_SCALED (LP_LOWPREC_CASE == LP_CASE_MXFP6 || LP_LOWPREC_CASE == LP_CASE_NVFP4)

// CUTLASS first: the production source defines helpers (cp_async_wait, ...)
// in a global anonymous namespace, which would make CUTLASS's unqualified
// calls ambiguous if they were visible when its templates are defined.
#if LP_LOWPREC_CASE == LP_CASE_BF16
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#elif LP_BLOCK_SCALED
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#endif

#define main lp_production_decoder_main
#include "decoder_optimized.cu"
#undef main

#include "reference.h"

#include <cuda_bf16.h>

#include <functional>
#include <thread>

// Test-only mutants (spec checks must reject them; never in the timed builds):
//   1 element codes rounded toward zero by one step when odd (BF16: truncation)
//   2 each block's scale written at its neighbour's position
//   3 NVFP4 alpha left at 1
#ifndef LP_LOWPREC_MUTANT
#define LP_LOWPREC_MUTANT 0
#endif


namespace {

namespace ref = lowprec_ref;

constexpr const char *kCaseNames[] = {"fp32", "bf16", "mxfp6", "nvfp4"};
constexpr const char *kCaseName = kCaseNames[LP_LOWPREC_CASE];
constexpr long long kFixtureBatch = 64, kFixtureWidth = 1024, kFixtureVocab = 128;
constexpr int kConvertThreads = 256;

// ---------------------------------------------------------------------------
// Classifier GEMMs (spec.json "cases").
#if LP_LOWPREC_CASE == LP_CASE_BF16
using Bf16Gemm = cutlass::gemm::device::Gemm<
    cutlass::bfloat16_t, cutlass::layout::RowMajor, cutlass::bfloat16_t,
    cutlass::layout::ColumnMajor, float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
#elif LP_BLOCK_SCALED
#if LP_LOWPREC_CASE == LP_CASE_MXFP6
using ElementPair = cutlass::mx_float6_t<cutlass::float_e3m2_t>;
constexpr int kAlignment = 128;
constexpr int kSfVec = 32;
constexpr int kBitsPerValue = 6;
#else
using ElementPair = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
constexpr int kAlignment = 32;
constexpr int kSfVec = 16;
constexpr int kBitsPerValue = 4;
#endif
using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float, float,
    cutlass::layout::RowMajor, 4, float, cutlass::layout::RowMajor, 4,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementPair,
    cutlass::layout::RowMajor, kAlignment, ElementPair, cutlass::layout::ColumnMajor, kAlignment,
    float, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,
                                                        CollectiveMainloop, CollectiveEpilogue, void>;
using BlockScaledGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using ScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;
using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename CollectiveMainloop::LayoutSFB;
using DataType = typename ElementPair::DataType;
using ScaleType = typename ElementPair::ScaleFactorType;
#endif

#if LP_LOWPREC_CASE != LP_CASE_FP32
#define LP_CHECK_CUTLASS(call)                                                           \
  do {                                                                                   \
    const cutlass::Status lp_status_ = (call);                                           \
    if (lp_status_ != cutlass::Status::kSuccess)                                         \
      lp::fail(lp::kExitCuda, "%s failed: %s", #call, cutlassGetStatusString(lp_status_)); \
  } while (0)
#endif

// ---------------------------------------------------------------------------
// Device conversion (spec.json "formats", "rounding", "mxfp6", "nvfp4"). The
// hardware conversions round to nearest even and saturate; their -0 results
// for small negative values are rewritten as +0, as the spec requires. The
// host reference (reference.cpp) finds the same codes by table search.

// Eight floats to eight E2M1 codes (value 2i in the low nibble of byte i).
__device__ __forceinline__ uint32_t e2m1x8(const float *v) {
  uint32_t bits;
  asm("{\n .reg .b8 b0, b1, b2, b3;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      " cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      " mov.b32 %0, {b0, b1, b2, b3};\n}"
      : "=r"(bits)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
  const uint32_t nonzero = bits | (bits >> 1) | (bits >> 2); // per nibble, bit 0
  bits &= ~((~nonzero & 0x11111111u) << 3);                  // clear the sign of zeros
#if LP_LOWPREC_MUTANT == 1
  bits &= ~(bits & 0x11111111u & ((bits >> 1) | (bits >> 2)) & 0x11111111u);
#endif
  return bits;
}

// Two floats to two E3M2 codes (bits 0..5 of each returned byte).
__device__ __forceinline__ uint32_t e3m2x2(float lo, float hi) {
  uint16_t bits;
  asm("cvt.rn.satfinite.e3m2x2.f32 %0, %1, %2;" : "=h"(bits) : "f"(hi), "f"(lo));
  uint32_t a = bits & 0x3fu, b = (bits >> 8) & 0x3fu;
  if ((a & 0x1fu) == 0) a = 0;
  if ((b & 0x1fu) == 0) b = 0;
#if LP_LOWPREC_MUTANT == 1
  if ((a & 0x1fu) > 1) a &= ~1u;
  if ((b & 0x1fu) > 1) b &= ~1u;
#endif
  return a | (b << 8);
}

// A non-negative float to UE4M3, and a UE4M3 code to its value.
__device__ __forceinline__ uint32_t ue4m3(float v) {
  uint16_t bits;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(bits) : "f"(0.0f), "f"(v));
  return bits & 0x7fu;
}
__device__ __forceinline__ float ue4m3_value(uint32_t code) {
  const uint32_t e = (code >> 3) & 15u, m = code & 7u;
  return e == 0 ? (float)m * 0x1p-9f : __uint_as_float(((e + 120u) << 23) | (m << 20));
}

#if LP_BLOCK_SCALED
// Scale-factor byte offset of (row, first value of a block), with mutant 2
// moving each block to its neighbour's slot.
template <class Layout>
__device__ __forceinline__ long long scale_offset(const Layout &layout, int row, int block) {
#if LP_LOWPREC_MUTANT == 2
  block ^= 1;
#endif
  return layout(cute::make_coord(row, block * kSfVec, 0));
}
#endif

// The largest |x| over n floats (n a multiple of 4) into *amax_bits (zeroed
// by the caller): non-negative float bits order as unsigned integers.
__global__ void __launch_bounds__(kConvertThreads)
tensor_amax_kernel(const float4 *__restrict__ x, long long n4, unsigned *amax_bits) {
  __shared__ float warp_max[kConvertThreads / 32];
  float m = 0.0f;
  for (long long i = (long long)blockIdx.x * kConvertThreads + threadIdx.x; i < n4;
       i += (long long)gridDim.x * kConvertThreads) {
    const float4 v = x[i];
    m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
  }
  for (int offset = 16; offset > 0; offset /= 2)
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, offset));
  if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x / 32] = m;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < kConvertThreads / 32; ++w) m = fmaxf(m, warp_max[w]);
    atomicMax(amax_bits, __float_as_uint(m));
  }
}

// BF16, round to nearest even (mutant 1: toward zero), four values a thread.
__global__ void __launch_bounds__(kConvertThreads)
to_bf16_kernel(const float4 *__restrict__ x, long long n4, uint2 *__restrict__ out) {
  const long long i = (long long)blockIdx.x * kConvertThreads + threadIdx.x;
  if (i >= n4) return;
  const float4 v = x[i];
#if LP_LOWPREC_MUTANT == 1
  const __nv_bfloat162 lo = __floats2bfloat162_rn(__bfloat162float(__float2bfloat16_rz(v.x)),
                                                  __bfloat162float(__float2bfloat16_rz(v.y)));
  const __nv_bfloat162 hi = __floats2bfloat162_rn(__bfloat162float(__float2bfloat16_rz(v.z)),
                                                  __bfloat162float(__float2bfloat16_rz(v.w)));
#else
  const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y);
  const __nv_bfloat162 hi = __floats2bfloat162_rn(v.z, v.w);
#endif
  out[i] = make_uint2(*reinterpret_cast<const uint32_t *>(&lo),
                      *reinterpret_cast<const uint32_t *>(&hi));
}

#if LP_LOWPREC_CASE == LP_CASE_MXFP6
// One thread per 32-value block of a [rows][width] matrix: the UE8M0 scale
// and 32 E3M2 codes, packed little-endian (value i at bit 6i of the row).
template <class Layout>
__global__ void __launch_bounds__(kConvertThreads)
quantize_mxfp6_kernel(const float *__restrict__ x, int rows, int width,
                      uint8_t *__restrict__ codes, uint8_t *__restrict__ scales, Layout layout) {
  const long long block = (long long)blockIdx.x * kConvertThreads + threadIdx.x;
  const int per_row = width / 32;
  if (block >= (long long)rows * per_row) return;
  const int row = (int)(block / per_row), column_block = (int)(block % per_row);
  const float4 *source = reinterpret_cast<const float4 *>(x + (size_t)row * width) + column_block * 8;
  float v[32];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float4 q = source[i];
    v[4 * i] = q.x, v[4 * i + 1] = q.y, v[4 * i + 2] = q.z, v[4 * i + 3] = q.w;
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
  uint64_t packed[3] = {0, 0, 0};
  uint32_t scale = 0;
  if (amax > 0.0f) {
    // floor(log2(amax)) is the unbiased exponent (any subnormal clamps).
    const int shared = max(-127, (int)((__float_as_uint(amax) >> 23) & 0xffu) - 127 - 4);
    const float multiplier = __uint_as_float((uint32_t)(127 - shared) << 23); // 2^-shared
    scale = (uint32_t)(shared + 127);
#pragma unroll
    for (int i = 0; i < 32; i += 2) {
      const uint32_t two = e3m2x2(v[i] * multiplier, v[i + 1] * multiplier);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const uint64_t code = (two >> (8 * j)) & 0x3fu;
        const int bit = 6 * (i + j), word = bit / 64, offset = bit % 64;
        packed[word] |= code << offset;
        if (offset > 58) packed[word + 1] |= code >> (64 - offset);
      }
    }
  }
  uint64_t *destination = reinterpret_cast<uint64_t *>(codes + (size_t)row * (width / 4 * 3) +
                                                       (size_t)column_block * 24);
  destination[0] = packed[0];
  destination[1] = packed[1];
  destination[2] = packed[2];
  scales[scale_offset(layout, row, column_block)] = (uint8_t)scale;
}
#endif

#if LP_LOWPREC_CASE == LP_CASE_NVFP4
// One thread per 16-value block of a [rows][width] matrix: the UE4M3 scale and
// 16 E2M1 codes (8 bytes). The tensor scales come from *amax_bits; block 0
// thread 0 also writes this tensor's decode scale and, when other_decode is
// given (activations), alpha = decode * other_decode.
template <class Layout>
__global__ void __launch_bounds__(kConvertThreads)
quantize_nvfp4_kernel(const float *__restrict__ x, int rows, int width,
                      const unsigned *__restrict__ amax_bits, const float *other_decode,
                      float *decode_out, float *alpha_out, uint8_t *__restrict__ codes,
                      uint8_t *__restrict__ scales, Layout layout) {
  const float tensor_amax = __uint_as_float(*amax_bits);
  const float encode = tensor_amax > 0.0f ? 2688.0f / tensor_amax : 1.0f;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const float decode = 1.0f / encode;
    *decode_out = decode;
#if LP_LOWPREC_MUTANT == 3
    if (alpha_out) *alpha_out = 1.0f;
#else
    if (alpha_out) *alpha_out = decode * *other_decode;
#endif
  }
  const long long block = (long long)blockIdx.x * kConvertThreads + threadIdx.x;
  const int per_row = width / 16;
  if (block >= (long long)rows * per_row) return;
  const int row = (int)(block / per_row), column_block = (int)(block % per_row);
  const float4 *source = reinterpret_cast<const float4 *>(x + (size_t)row * width) + column_block * 4;
  float v[16];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float4 q = source[i];
    v[4 * i] = q.x, v[4 * i + 1] = q.y, v[4 * i + 2] = q.z, v[4 * i + 3] = q.w;
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
  float target = amax / 6.0f;
  target = target * encode;
  const uint32_t scale = ue4m3(target);
  uint2 packed = make_uint2(0u, 0u);
  if (scale != 0) {
    const float f = ue4m3_value(scale);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float r = v[i] * encode;
      v[i] = r / f;
    }
    packed = make_uint2(e2m1x8(v), e2m1x8(v + 8));
  }
  *reinterpret_cast<uint2 *>(codes + (size_t)row * (width / 2) + (size_t)column_block * 8) = packed;
  scales[scale_offset(layout, row, column_block)] = (uint8_t)scale;
}
#endif

unsigned grid_for(long long items) {
  return (unsigned)((items + kConvertThreads - 1) / kConvertThreads);
}

// Runs fn(i) for i in [0, count) on the host's cores.
void parallel_for(long long count, const std::function<void(long long)> &fn) {
  const unsigned threads = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; ++t)
    pool.emplace_back([&, t] {
      for (long long i = t; i < count; i += threads) fn(i);
    });
  for (std::thread &thread : pool) thread.join();
}

// ---------------------------------------------------------------------------
// The quantized form of one [rows][width] operand on the device, and its host
// copy unpacked to one code per value and one scale code per block.
struct Operand {
  int rows = 0, width = 0;
  uint8_t *codes = nullptr;  // BF16: 2 bytes a value; MXFP6: 3 per 4; NVFP4: 1 per 2
  uint8_t *scales = nullptr; // scale-factor bytes in CUTLASS's layout
  size_t code_bytes = 0, scale_bytes = 0;
  float *decode = nullptr;   // NVFP4 tensor decode scale
  unsigned *amax = nullptr;  // NVFP4 tensor maximum bits
};

struct HostOperand {
  std::vector<uint16_t> codes; // BF16 bits, or E3M2 / E2M1 codes
  std::vector<uint8_t> scales; // [rows][width / block]
  float decode = 1.0f;         // NVFP4
};

class LowPrecisionDecoder final : public lp::Workload {
public:
  LowPrecisionDecoder(long long batch, long long width, long long vocab, long long steps,
                      int fixture)
      : B_(batch), W_(width), V_(vocab), S_(steps), fixture_(fixture) {}

  ~LowPrecisionDecoder() override {
    for (void *p : allocations_) cudaFree(p);
    if (h_prompts_) cudaFreeHost(h_prompts_);
    if (h_generated_) cudaFreeHost(h_generated_);
  }

  std::string identity() const override {
    return std::string("blackwell-experiments/lowprec-decoder-classifier/") + kCaseName;
  }

  lp::JsonObject dimensions() const override {
    using lp::Json;
    return {{"batch", Json::integer(B_)},
            {"width", Json::integer(W_)},
            {"vocab", Json::integer(V_)},
            {"steps", Json::integer(S_)},
            {"fixture", Json::integer(fixture_)},
            {"case", Json::quote(kCaseName)},
            {"generated_tokens", Json::integer(B_ * S_)},
            {"classifier_macs_per_step", Json::integer(B_ * V_ * W_)}};
  }

  std::string application_scope() const override {
    return "prompt upload (pinned H2D), token reset, every step (rmsnorm, activation "
           "conversion with scale generation and layout, classifier GEMM, "
           "softmax+argmax with on-device token feedback), generated-token download; "
           "converted weights resident from setup";
  }
  std::string compute_scope() const override {
    return "token reset and every step (rmsnorm, activation conversion, classifier GEMM, "
           "softmax+argmax)";
  }

  void setup(cudaStream_t stream) override {
    const size_t model = (size_t)(V_ * W_);
    d_embedding_ = alloc<float>(model);
    d_classifier_ = alloc<float>(model);
    d_norm_weight_ = alloc<float>(W_);
    d_prompts_ = alloc<int>(B_);
    d_tokens_ = alloc<int>(B_);
    d_generated_ = alloc<int>(B_ * S_);
    d_normalized_ = alloc<float>(S_ * B_ * W_);
    d_logits_ = alloc<float>(S_ * B_ * V_);
    d_probabilities_ = alloc<float>(S_ * B_ * V_);
    LP_CUDA(cudaMallocHost(&h_prompts_, (size_t)B_ * sizeof(int)));
    LP_CUDA(cudaMallocHost(&h_generated_, (size_t)(B_ * S_) * sizeof(int)));
    if (fixture_ == 0) {
      init_model_kernel<<<1024, 256, 0, stream>>>(d_embedding_, d_classifier_, d_norm_weight_,
                                                  (int)W_, (int)V_);
      LP_CUDA_LAUNCH();
    } else {
      host_fixture_ = ref::make_fixture(fixture_, W_, V_);
      std::vector<float> ones((size_t)W_, 1.0f);
      LP_CUDA(cudaMemcpyAsync(d_embedding_, host_fixture_.embedding.data(), model * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
      LP_CUDA(cudaMemcpyAsync(d_classifier_, host_fixture_.classifier.data(),
                              model * sizeof(float), cudaMemcpyHostToDevice, stream));
      LP_CUDA(cudaMemcpyAsync(d_norm_weight_, ones.data(), (size_t)W_ * sizeof(float),
                              cudaMemcpyHostToDevice, stream));
      LP_CUDA(cudaStreamSynchronize(stream));
    }
    for (long long b = 0; b < B_; ++b) h_prompts_[b] = (int)(b % V_);
    setup_classifier(stream);
    LP_CUDA(cudaStreamSynchronize(stream));
    cudaEvent_t start, stop;
    LP_CUDA(cudaEventCreate(&start));
    LP_CUDA(cudaEventCreate(&stop));
    LP_CUDA(cudaEventRecord(start, stream));
    convert_weights(stream);
    LP_CUDA(cudaEventRecord(stop, stream));
    LP_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    LP_CUDA(cudaEventElapsedTime(&ms, start, stop));
    LP_CUDA(cudaEventDestroy(start));
    LP_CUDA(cudaEventDestroy(stop));
    std::fprintf(stderr, "decoder_lowprec: case %s, fixture %d, weight conversion %.3f ms (setup)\n",
                 kCaseName, fixture_, ms);
  }

  void enqueue_application(cudaStream_t stream) override {
    LP_CUDA(cudaMemcpyAsync(d_prompts_, h_prompts_, (size_t)B_ * sizeof(int),
                            cudaMemcpyHostToDevice, stream));
    enqueue_compute(stream);
    LP_CUDA(cudaMemcpyAsync(h_generated_, d_generated_, (size_t)(B_ * S_) * sizeof(int),
                            cudaMemcpyDeviceToHost, stream));
  }

  void enqueue_compute(cudaStream_t stream) override {
    LP_CUDA(cudaMemcpyAsync(d_tokens_, d_prompts_, (size_t)B_ * sizeof(int),
                            cudaMemcpyDeviceToDevice, stream));
    for (long long s = 0; s < S_; ++s) {
      rmsnorm_kernel<<<(unsigned)B_, kRowThreads, 0, stream>>>(
          normalized(s), d_tokens_, d_embedding_, d_norm_weight_, (int)W_, (int)V_);
      LP_CUDA_LAUNCH();
      enqueue_classifier((int)s, stream);
      softmax_argmax_kernel<<<(unsigned)B_, kRowThreads, 0, stream>>>(
          probabilities(s), logits(s), d_tokens_, d_generated_,
          (int)V_, (int)S_, (int)s);
      LP_CUDA_LAUNCH();
    }
  }

  void poison_outputs(cudaStream_t stream) override {
    // 0xFF: int32 -1 and float NaN, as the production decoder; the converted
    // activations and their scales are intermediates and are poisoned too.
    LP_CUDA(cudaMemsetAsync(d_prompts_, 0xFF, (size_t)B_ * sizeof(int), stream));
    LP_CUDA(cudaMemsetAsync(d_tokens_, 0xFF, (size_t)B_ * sizeof(int), stream));
    LP_CUDA(cudaMemsetAsync(d_generated_, 0xFF, (size_t)(B_ * S_) * sizeof(int), stream));
    LP_CUDA(cudaMemsetAsync(d_normalized_, 0xFF, (size_t)(S_ * B_ * W_) * sizeof(float), stream));
    LP_CUDA(cudaMemsetAsync(d_logits_, 0xFF, (size_t)(S_ * B_ * V_) * sizeof(float), stream));
    LP_CUDA(cudaMemsetAsync(d_probabilities_, 0xFF, (size_t)(S_ * B_ * V_) * sizeof(float), stream));
    for (Operand &a : activations_) {
      LP_CUDA(cudaMemsetAsync(a.codes, 0xFF, a.code_bytes, stream));
      if (a.scales) LP_CUDA(cudaMemsetAsync(a.scales, 0xFF, a.scale_bytes, stream));
      if (a.decode) LP_CUDA(cudaMemsetAsync(a.decode, 0xFF, sizeof(float), stream));
    }
    if (d_alpha_) LP_CUDA(cudaMemsetAsync(d_alpha_, 0xFF, (size_t)S_ * sizeof(float), stream));
    if (d_partial_) {
      LP_CUDA(cudaMemsetAsync(d_partial_, 0xFF, partial_count_ * sizeof(float), stream));
      LP_CUDA(cudaMemsetAsync(d_ready_, 0xFF, (size_t)grid_ * sizeof(int), stream));
    }
    std::memset(h_generated_, 0xFF, (size_t)(B_ * S_) * sizeof(int));
  }

  // Host state for the check: the converted weights from the host quantizer
  // (compared once with the device's) and the dequantized weights.
  void compute_reference() override {
    host_weights_ = host_weight_values();
#if LP_LOWPREC_CASE != LP_CASE_FP32
    LP_CUDA(cudaDeviceSynchronize());
    const HostOperand expected = quantize_on_host(host_weights_.data(), (int)V_);
    const HostOperand actual = download(weights_, layout_rows_weights());
    weights_match_ = compare_operands("weights", expected, actual, (int)V_);
    dequantized_weights_ = dequantize(expected, (int)V_);
    weight_decode_ = expected.decode;
#endif
    if (fixture_ == 0 && LP_LOWPREC_CASE == LP_CASE_FP32)
      production_reference_ = decoder_oracle::compute(shape());
  }

  bool check_outputs() override {
    download_diagnostics();
    return implementation_check();
  }

  uint64_t output_digest() const override {
    // Generated tokens, then every diagnostic output (normalized, logits,
    // probabilities) and the converted activations, all as downloaded.
    auto self = const_cast<LowPrecisionDecoder *>(this);
    self->download_diagnostics();
    uint64_t digest = lp::fnv1a64(h_generated_, (size_t)(B_ * S_) * sizeof(int));
    digest = lp::fnv1a64(h_normalized_.data(), h_normalized_.size() * sizeof(float), digest);
    digest = lp::fnv1a64(h_logits_.data(), h_logits_.size() * sizeof(float), digest);
    digest = lp::fnv1a64(h_probabilities_.data(), h_probabilities_.size() * sizeof(float), digest);
    for (const std::vector<uint8_t> &bytes : h_activation_bytes_)
      digest = lp::fnv1a64(bytes.data(), bytes.size(), digest);
    return digest;
  }

  // --- used by the accuracy and self-test modes -------------------------
  long long batch() const { return B_; }
  long long width() const { return W_; }
  long long vocab() const { return V_; }
  long long steps() const { return S_; }
  int fixture() const { return fixture_; }
  const ref::Fixture &host_fixture() const { return host_fixture_; }
  const int32_t *generated() const { return h_generated_; }
  float *normalized(long long s) const { return d_normalized_ + (size_t)(s * B_ * W_); }
  float *logits(long long s) const { return d_logits_ + (size_t)(s * B_ * V_); }
  float *probabilities(long long s) const { return d_probabilities_ + (size_t)(s * B_ * V_); }

  // The classifier of step s alone on whatever normalized(s) holds.
  void enqueue_classifier(int s, cudaStream_t stream) {
#if LP_LOWPREC_CASE == LP_CASE_FP32
    if (plan_.sk_iterations > 0)
      LP_CUDA(cudaMemsetAsync(d_ready_, 0, (size_t)grid_ * sizeof(int), stream));
    classifier_<<<(unsigned)grid_, (unsigned)tile_->threads, tile_->smem_bytes, stream>>>(
        logits(s), normalized(s), d_classifier_, d_partial_, d_ready_, (int)B_, (int)W_,
        (int)V_, plan_);
    LP_CUDA_LAUNCH();
#else
    convert(normalized(s), (int)B_, activations_[s], d_alpha_ + s, stream);
#if LP_LOWPREC_CASE == LP_CASE_BF16
    LP_CHECK_CUTLASS(bf16_gemm_[s](stream));
#else
    LP_CHECK_CUTLASS(gemm_[s].run(stream));
#endif
#endif
  }

  // Converts the weights again (the accuracy mode times this).
  void convert_weights(cudaStream_t stream) {
#if LP_LOWPREC_CASE != LP_CASE_FP32
    convert(d_classifier_, (int)V_, weights_, nullptr, stream);
#else
    (void)stream;
#endif
  }

  // The device conversion of a [rows][width] FP32 matrix, then downloaded
  // and unpacked, with the host quantizer's result (self test).
  bool selftest_convert(const std::vector<float> &matrix, int rows) {
#if LP_LOWPREC_CASE == LP_CASE_FP32
    (void)matrix, (void)rows;
    return true;
#else
    float *d_matrix = alloc<float>((long long)rows * W_);
    LP_CUDA(cudaMemcpy(d_matrix, matrix.data(), matrix.size() * sizeof(float),
                       cudaMemcpyHostToDevice));
    Operand operand = make_operand(rows, false);
    float *d_other = alloc<float>(1), *d_alpha = alloc<float>(1);
    const float other = 0.75f;
    LP_CUDA(cudaMemcpy(d_other, &other, sizeof other, cudaMemcpyHostToDevice));
#if LP_LOWPREC_CASE == LP_CASE_NVFP4
    weights_decode_for_selftest_ = d_other;
#endif
    convert(d_matrix, rows, operand, d_alpha, nullptr);
    LP_CUDA(cudaDeviceSynchronize());
    const HostOperand expected = quantize_on_host(matrix.data(), rows);
    const HostOperand actual = download(operand, rows);
    bool pass = compare_operands("self-test", expected, actual, rows);
#if LP_LOWPREC_CASE == LP_CASE_NVFP4
    float alpha = 0.0f;
    LP_CUDA(cudaMemcpy(&alpha, d_alpha, sizeof alpha, cudaMemcpyDeviceToHost));
    const float want = expected.decode * other;
    if (alpha != want) {
      std::fprintf(stderr, "decoder_lowprec: self-test alpha %.9g, expected %.9g\n", alpha, want);
      pass = false;
    }
#endif
    return pass;
#endif
  }

private:
  template <class T> T *alloc(long long count) {
    void *p = nullptr;
    LP_CUDA(cudaMalloc(&p, (size_t)count * sizeof(T)));
    allocations_.push_back(p);
    return static_cast<T *>(p);
  }

  decoder_oracle::Shape shape() const { return {B_, W_, V_, S_}; }


#if LP_BLOCK_SCALED
  LayoutSFA layout_sfa(int rows) const {
    return ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(rows, (int)V_, (int)W_, 1));
  }
  LayoutSFB layout_sfb() const {
    return ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape((int)B_, (int)V_, (int)W_, 1));
  }
#endif
  int layout_rows_weights() const { return (int)V_; }

  // A device operand for `rows` rows (weights: the SFB layout).
  Operand make_operand(int rows, bool weights) {
    Operand o;
    o.rows = rows;
    o.width = (int)W_;
    const long long values = (long long)rows * W_;
#if LP_LOWPREC_CASE == LP_CASE_BF16
    (void)weights;
    o.code_bytes = (size_t)values * 2;
#elif LP_BLOCK_SCALED
    o.code_bytes = (size_t)values * kBitsPerValue / 8;
    o.scale_bytes = weights ? (size_t)cute::size(cute::filter_zeros(layout_sfb()))
                            : (size_t)cute::size(cute::filter_zeros(layout_sfa(rows)));
    o.scales = alloc<uint8_t>((long long)o.scale_bytes);
    // Padding scale bytes (rows past the last 128-row tile) stay zero.
    LP_CUDA(cudaMemset(o.scales, 0, o.scale_bytes));
#if LP_LOWPREC_CASE == LP_CASE_NVFP4
    o.decode = alloc<float>(1);
    o.amax = alloc<unsigned>(1);
#endif
#endif
    o.codes = alloc<uint8_t>((long long)o.code_bytes);
    return o;
  }

  // FP32 [rows][W] at x -> operand o; alpha (NVFP4 activations) gets the
  // product of the two tensors' decode scales.
  void convert(const float *x, int rows, Operand &o, float *alpha, cudaStream_t stream) {
    const long long values = (long long)rows * W_;
#if LP_LOWPREC_CASE == LP_CASE_BF16
    (void)alpha;
    to_bf16_kernel<<<grid_for(values / 4), kConvertThreads, 0, stream>>>(
        reinterpret_cast<const float4 *>(x), values / 4, reinterpret_cast<uint2 *>(o.codes));
    LP_CUDA_LAUNCH();
#elif LP_LOWPREC_CASE == LP_CASE_MXFP6
    (void)alpha;
    if (rows == (int)V_ && o.codes == weights_.codes)
      quantize_mxfp6_kernel<<<grid_for(values / 32), kConvertThreads, 0, stream>>>(
          x, rows, (int)W_, o.codes, o.scales, layout_sfb());
    else
      quantize_mxfp6_kernel<<<grid_for(values / 32), kConvertThreads, 0, stream>>>(
          x, rows, (int)W_, o.codes, o.scales, layout_sfa(rows));
    LP_CUDA_LAUNCH();
#elif LP_LOWPREC_CASE == LP_CASE_NVFP4
    LP_CUDA(cudaMemsetAsync(o.amax, 0, sizeof(unsigned), stream));
    const unsigned amax_grid = (unsigned)std::min<long long>(grid_for(values / 4), 2048);
    tensor_amax_kernel<<<amax_grid, kConvertThreads, 0, stream>>>(
        reinterpret_cast<const float4 *>(x), values / 4, o.amax);
    LP_CUDA_LAUNCH();
    const bool weights = rows == (int)V_ && o.codes == weights_.codes;
    const float *other = weights ? nullptr
                         : weights_decode_for_selftest_ ? weights_decode_for_selftest_
                                                        : weights_.decode;
    float *alpha_out = weights ? nullptr : alpha;
    if (weights)
      quantize_nvfp4_kernel<<<grid_for(values / 16), kConvertThreads, 0, stream>>>(
          x, rows, (int)W_, o.amax, other, o.decode, alpha_out, o.codes, o.scales, layout_sfb());
    else
      quantize_nvfp4_kernel<<<grid_for(values / 16), kConvertThreads, 0, stream>>>(
          x, rows, (int)W_, o.amax, other, o.decode, alpha_out, o.codes, o.scales,
          layout_sfa(rows));
    LP_CUDA_LAUNCH();
#else
    (void)x, (void)rows, (void)o, (void)alpha, (void)stream, (void)values;
#endif
  }

  void setup_classifier(cudaStream_t stream) {
    (void)stream;
#if LP_LOWPREC_CASE == LP_CASE_FP32
    // The production plan_classifier() without its test hooks.
    int device = 0;
    LP_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    LP_CUDA(cudaGetDeviceProperties(&prop, device));
    const int arch = prop.major * 10 + prop.minor;
    const std::string name = arch == 120 ? "128x256x8" : "128x256x16";
    for (const TileVariant &v : tile_variants())
      if (name == v.name) tile_ = &v;
    classifier_ = tile_->kernel;
    LP_CUDA(cudaFuncSetAttribute(classifier_, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)tile_->smem_bytes));
    int per_sm = 0;
    LP_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, classifier_, tile_->threads,
                                                          tile_->smem_bytes));
    const long long ctas = (long long)per_sm * prop.multiProcessorCount;
    const long long tiles_m = (B_ + tile_->bm - 1) / tile_->bm;
    const long long tiles = tiles_m * ((V_ + tile_->bn - 1) / tile_->bn);
    const long long k_tiles = (W_ + tile_->bk - 1) / tile_->bk;
    plan_ = {(int)tiles_m, (int)k_tiles, 1, 0, 0};
    grid_ = tiles;
    if (tiles > ctas) {
      const long long waves = tiles % ctas == 0 ? tiles / ctas : tiles / ctas - 1;
      grid_ = ctas;
      plan_.dp_waves = (int)waves;
      plan_.sk_first_tile = waves * ctas;
      plan_.sk_iterations = (tiles - waves * ctas) * k_tiles;
    }
    if (plan_.sk_iterations > 0) {
      partial_count_ = (size_t)grid_ * tile_->bm * tile_->bn;
      d_partial_ = alloc<float>((long long)partial_count_);
      d_ready_ = alloc<int>(grid_);
    }
#else
    weights_ = make_operand((int)V_, true);
    for (long long s = 0; s < S_; ++s) activations_.push_back(make_operand((int)B_, false));
#if LP_LOWPREC_CASE == LP_CASE_NVFP4
    d_alpha_ = alloc<float>(S_);
#endif
    const int M = (int)B_, N = (int)V_, K = (int)W_;
#if LP_LOWPREC_CASE == LP_CASE_BF16
    bf16_gemm_.resize(S_);
    for (long long s = 0; s < S_; ++s) {
      typename Bf16Gemm::Arguments arguments(
          {M, N, K}, {reinterpret_cast<cutlass::bfloat16_t *>(activations_[s].codes), K},
          {reinterpret_cast<cutlass::bfloat16_t *>(weights_.codes), K}, {logits(s), N},
          {logits(s), N}, {1.0f, 0.0f});
      LP_CHECK_CUTLASS(bf16_gemm_[s].can_implement(arguments));
      LP_CHECK_CUTLASS(bf16_gemm_[s].initialize(arguments, nullptr, stream));
    }
#else
    gemm_.resize(S_);
    const auto stride_a = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, 1});
    const auto stride_b = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, 1});
    const auto stride_c = cutlass::make_cute_packed_stride(typename GemmKernel::StrideC{}, {M, N, 1});
    for (long long s = 0; s < S_; ++s) {
      typename BlockScaledGemm::Arguments arguments{
          cutlass::gemm::GemmUniversalMode::kGemm,
          {M, N, K, 1},
          {reinterpret_cast<DataType *>(activations_[s].codes), stride_a,
           reinterpret_cast<DataType *>(weights_.codes), stride_b,
           reinterpret_cast<ScaleType *>(activations_[s].scales), layout_sfa(M),
           reinterpret_cast<ScaleType *>(weights_.scales), layout_sfb()},
          {{1.0f, 0.0f}, logits(s), stride_c, logits(s), stride_c}};
#if LP_LOWPREC_CASE == LP_CASE_NVFP4
      arguments.epilogue.thread.alpha_ptr = d_alpha_ + s;
#endif
      if (BlockScaledGemm::get_workspace_size(arguments) != 0)
        lp::fail(lp::kExitCuda, "the block-scaled GEMM unexpectedly needs a workspace");
      LP_CHECK_CUTLASS(gemm_[s].can_implement(arguments));
      LP_CHECK_CUTLASS(gemm_[s].initialize(arguments, nullptr, stream));
    }
#endif
#endif
  }

  // Host FP32 weights, [vocab][width].
  std::vector<float> host_weight_values() const {
    if (fixture_ != 0) return host_fixture_.classifier;
    std::vector<float> w((size_t)(V_ * W_), 0.0f);
    for (long long r = 0; r < V_; ++r) w[(size_t)(r * W_ + (r + V_ - 1) % V_)] = 1.0f;
    return w;
  }

  // Host quantizer over [rows][W] (spec.json), in parallel over rows.
  HostOperand quantize_on_host(const float *x, int rows) const {
    HostOperand h;
    const long long values = (long long)rows * W_;
    h.codes.resize((size_t)values);
#if LP_LOWPREC_CASE == LP_CASE_BF16
    parallel_for(rows, [&](long long r) {
      for (long long k = 0; k < W_; ++k)
        h.codes[(size_t)(r * W_ + k)] = ref::to_bf16(x[r * W_ + k]);
    });
#elif LP_BLOCK_SCALED
    const long long per_row = W_ / kSfVec;
    h.scales.resize((size_t)(rows * per_row));
#if LP_LOWPREC_CASE == LP_CASE_NVFP4
    float amax = 0.0f;
    for (long long i = 0; i < values; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float encode = ref::nvfp4_encode_scale(amax);
    h.decode = ref::nvfp4_decode_scale(encode);
#endif
    parallel_for(rows, [&](long long r) {
      std::vector<uint8_t> codes((size_t)W_);
#if LP_LOWPREC_CASE == LP_CASE_MXFP6
      ref::quantize_mxfp6(x + r * W_, W_, codes.data(), &h.scales[(size_t)(r * per_row)]);
#else
      ref::quantize_nvfp4(x + r * W_, W_, encode, codes.data(), &h.scales[(size_t)(r * per_row)]);
#endif
      for (long long k = 0; k < W_; ++k) h.codes[(size_t)(r * W_ + k)] = codes[(size_t)k];
    });
#else
    (void)x;
#endif
    return h;
  }

  // Downloads and unpacks a device operand (scales through CUTLASS's layout).
  HostOperand download(const Operand &o, int layout_rows) const {
    HostOperand h;
    std::vector<uint8_t> bytes(o.code_bytes);
    LP_CUDA(cudaMemcpy(bytes.data(), o.codes, o.code_bytes, cudaMemcpyDeviceToHost));
    const long long values = (long long)o.rows * W_;
    h.codes.resize((size_t)values);
#if LP_LOWPREC_CASE == LP_CASE_BF16
    (void)layout_rows;
    std::memcpy(h.codes.data(), bytes.data(), o.code_bytes);
#elif LP_BLOCK_SCALED
    std::vector<uint8_t> scales(o.scale_bytes);
    LP_CUDA(cudaMemcpy(scales.data(), o.scales, o.scale_bytes, cudaMemcpyDeviceToHost));
    const long long row_bytes = W_ * kBitsPerValue / 8, per_row = W_ / kSfVec;
    h.scales.resize((size_t)(o.rows * per_row));
    const bool weights = o.codes == weights_.codes;
    const auto sfa = layout_sfa(layout_rows);
    const auto sfb = layout_sfb();
    parallel_for(o.rows, [&](long long r) {
      const uint8_t *row = bytes.data() + r * row_bytes;
      for (long long k = 0; k < W_; ++k)
        h.codes[(size_t)(r * W_ + k)] =
            (uint16_t)(kBitsPerValue == 6 ? ref::unpack_e3m2(row, k) : ref::unpack_e2m1(row, k));
      for (long long b = 0; b < per_row; ++b)
        h.scales[(size_t)(r * per_row + b)] =
            scales[(size_t)(weights ? sfb(cute::make_coord((int)r, (int)(b * kSfVec), 0))
                                    : sfa(cute::make_coord((int)r, (int)(b * kSfVec), 0)))];
    });
#if LP_LOWPREC_CASE == LP_CASE_NVFP4
    LP_CUDA(cudaMemcpy(&h.decode, o.decode, sizeof(float), cudaMemcpyDeviceToHost));
#endif
#else
    (void)layout_rows, (void)values;
#endif
    return h;
  }

  // Exact comparison; prints the first differences.
  bool compare_operands(const char *what, const HostOperand &expected, const HostOperand &actual,
                        int rows) const {
    long long code_errors = 0, scale_errors = 0;
    for (size_t i = 0; i < expected.codes.size(); ++i)
      if (expected.codes[i] != actual.codes[i]) {
        if (code_errors < 5)
          std::fprintf(stderr, "decoder_lowprec: %s code mismatch row %lld column %lld: "
                               "expected 0x%x, actual 0x%x\n",
                       what, (long long)(i / W_), (long long)(i % W_), expected.codes[i],
                       actual.codes[i]);
        ++code_errors;
      }
    for (size_t i = 0; i < expected.scales.size(); ++i)
      if (expected.scales[i] != actual.scales[i]) {
        if (scale_errors < 5)
          std::fprintf(stderr, "decoder_lowprec: %s scale mismatch block %zu: expected 0x%x, "
                               "actual 0x%x\n",
                       what, i, expected.scales[i], actual.scales[i]);
        ++scale_errors;
      }
    const bool decode_ok = expected.decode == actual.decode;
    if (!decode_ok)
      std::fprintf(stderr, "decoder_lowprec: %s tensor decode scale %.9g, expected %.9g\n", what,
                   actual.decode, expected.decode);
    std::fprintf(stderr, "decoder_lowprec: %s (%d rows): %lld code and %lld scale mismatches\n",
                 what, rows, code_errors, scale_errors);
    return code_errors == 0 && scale_errors == 0 && decode_ok;
  }

  // Dequantized values (float; every product is exact) without the NVFP4
  // tensor decode scale, which joins alpha.
  std::vector<float> dequantize(const HostOperand &h, int rows) const {
    std::vector<float> out((size_t)((long long)rows * W_));
    parallel_for(rows, [&](long long r) {
      for (long long k = 0; k < W_; ++k) {
        const size_t i = (size_t)(r * W_ + k);
#if LP_LOWPREC_CASE == LP_CASE_BF16
        out[i] = (float)ref::bf16_value(h.codes[i]);
#elif LP_LOWPREC_CASE == LP_CASE_MXFP6
        out[i] = (float)ref::dequantize_mxfp6((uint8_t)h.codes[i],
                                              h.scales[(size_t)(r * (W_ / 32) + k / 32)]);
#elif LP_LOWPREC_CASE == LP_CASE_NVFP4
        out[i] = (float)ref::dequantize_nvfp4_unscaled((uint8_t)h.codes[i],
                                                       h.scales[(size_t)(r * (W_ / 16) + k / 16)]);
#else
        (void)h, (void)i;
#endif
      }
    });
    return out;
  }

  void download_diagnostics() {
    h_normalized_.resize((size_t)(S_ * B_ * W_));
    h_logits_.resize((size_t)(S_ * B_ * V_));
    h_probabilities_.resize((size_t)(S_ * B_ * V_));
    LP_CUDA(cudaMemcpy(h_normalized_.data(), d_normalized_, h_normalized_.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));
    LP_CUDA(cudaMemcpy(h_logits_.data(), d_logits_, h_logits_.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));
    LP_CUDA(cudaMemcpy(h_probabilities_.data(), d_probabilities_,
                       h_probabilities_.size() * sizeof(float), cudaMemcpyDeviceToHost));
    h_activation_bytes_.clear();
    for (const Operand &a : activations_) {
      std::vector<uint8_t> bytes(a.code_bytes + a.scale_bytes + (a.decode ? sizeof(float) : 0));
      LP_CUDA(cudaMemcpy(bytes.data(), a.codes, a.code_bytes, cudaMemcpyDeviceToHost));
      if (a.scales)
        LP_CUDA(cudaMemcpy(bytes.data() + a.code_bytes, a.scales, a.scale_bytes,
                           cudaMemcpyDeviceToHost));
      if (a.decode)
        LP_CUDA(cudaMemcpy(bytes.data() + a.code_bytes + a.scale_bytes, a.decode, sizeof(float),
                           cudaMemcpyDeviceToHost));
      h_activation_bytes_.push_back(std::move(bytes));
    }
  }

  const float *embedding_row(long long token) const {
    if (fixture_ != 0) return &host_fixture_.embedding[(size_t)(token * W_)];
    synthetic_row_.assign((size_t)W_, 1.0f);
    synthetic_row_[(size_t)token] = 4.0f;
    return synthetic_row_.data();
  }

  // spec.json "implementation_check".
  bool implementation_check() {
    bool pass = true;
    // Normalized rows against rmsnorm in double of the token the GPU fed back.
    long long normalized_errors = 0, token_range_errors = 0;
    double worst_normalized = 0.0;
    std::vector<double> row((size_t)W_);
    for (long long s = 0; s < S_; ++s)
      for (long long b = 0; b < B_; ++b) {
        const long long token = s == 0 ? h_prompts_[b] : h_generated_[b * S_ + s - 1];
        if (token < 0 || token >= V_) {
          ++token_range_errors;
          continue;
        }
        if (fixture_ == 0) {
          const float *x = embedding_row(token);
          ref::rmsnorm_double(x, W_, row.data());
        } else {
          ref::rmsnorm_double(&host_fixture_.embedding[(size_t)(token * W_)], W_, row.data());
        }
        const float *o = &h_normalized_[(size_t)((s * B_ + b) * W_)];
        for (long long j = 0; j < W_; ++j) {
          const double error = std::fabs((double)o[j] - row[(size_t)j]);
          const double bound = 2e-6 * std::fabs(row[(size_t)j]) + 1e-12;
          if (!(error <= bound)) {
            if (normalized_errors < 5)
              std::fprintf(stderr, "decoder_lowprec: normalized mismatch step %lld sequence %lld "
                                   "column %lld: expected %.9g, actual %.9g\n",
                           s, b, j, row[(size_t)j], (double)o[j]);
            ++normalized_errors;
          } else {
            worst_normalized = std::max(worst_normalized, error / bound);
          }
        }
      }
    // Converted activations, bit for bit, every step.
    bool activations_ok = true;
    std::vector<std::vector<float>> dequantized_activations(S_);
    std::vector<float> alpha((size_t)S_, 1.0f);
#if LP_LOWPREC_CASE != LP_CASE_FP32
    for (long long s = 0; s < S_; ++s) {
      const HostOperand expected =
          quantize_on_host(&h_normalized_[(size_t)(s * B_ * W_)], (int)B_);
      const HostOperand actual = download(activations_[s], (int)B_);
      char what[32];
      std::snprintf(what, sizeof what, "activations step %lld", s);
      activations_ok = compare_operands(what, expected, actual, (int)B_) && activations_ok;
      dequantized_activations[s] = dequantize(expected, (int)B_);
#if LP_LOWPREC_CASE == LP_CASE_NVFP4
      alpha[(size_t)s] = expected.decode * weight_decode_;
      float device_alpha = 0.0f;
      LP_CUDA(cudaMemcpy(&device_alpha, d_alpha_ + s, sizeof(float), cudaMemcpyDeviceToHost));
      if (device_alpha != alpha[(size_t)s]) {
        std::fprintf(stderr, "decoder_lowprec: alpha step %lld %.9g, expected %.9g\n", s,
                     device_alpha, alpha[(size_t)s]);
        activations_ok = false;
      }
#endif
    }
#endif
    // Logits of sampled sequences against the double product of the operands
    // the GPU multiplied (rigorous bound for FP32 accumulation in any order).
    std::vector<long long> sampled;
    const long long count = std::min<long long>(8, B_);
    for (long long i = 0; i < count; ++i)
      sampled.push_back(count == 1 ? 0 : i * (B_ - 1) / (count - 1));
    const double u = 0x1p-23, gamma = (double)W_ * u / (1.0 - (double)W_ * u);
    std::vector<long long> logit_errors((size_t)(count * S_), 0);
    std::vector<double> worst_logit((size_t)(count * S_), 0.0);
    const std::vector<float> &weights =
        LP_LOWPREC_CASE == LP_CASE_FP32 ? host_weights_ : dequantized_weights_;
    parallel_for(count * S_, [&](long long job) {
      const long long i = job / S_, s = job % S_, b = sampled[(size_t)i];
      const float *a = LP_LOWPREC_CASE == LP_CASE_FP32
                           ? &h_normalized_[(size_t)((s * B_ + b) * W_)]
                           : &dequantized_activations[(size_t)s][(size_t)(b * W_)];
      const double scale = alpha[(size_t)s];
      for (long long r = 0; r < V_; ++r) {
        const float *w = &weights[(size_t)(r * W_)];
        double sum = 0.0, magnitude = 0.0;
        for (long long k = 0; k < W_; ++k) {
          const double product = (double)a[k] * (double)w[k];
          sum += product;
          magnitude += std::fabs(product);
        }
        const double expected = sum * scale;
        const double actual = h_logits_[(size_t)((s * B_ + b) * V_ + r)];
        const double bound = gamma * magnitude * std::fabs(scale) + u * std::fabs(expected) + 1e-30;
        const double error = std::fabs(actual - expected);
        if (!(error <= bound)) {
          if (logit_errors[(size_t)job] == 0)
            std::fprintf(stderr, "decoder_lowprec: logit mismatch step %lld sequence %lld row "
                                 "%lld: expected %.9g, actual %.9g, bound %.3g\n",
                         s, b, r, expected, actual, bound);
          ++logit_errors[(size_t)job];
        } else if (magnitude > 0) {
          worst_logit[(size_t)job] =
              std::max(worst_logit[(size_t)job], error / (magnitude * std::fabs(scale)));
        }
      }
    });
    long long total_logit_errors = 0;
    double worst_relative_logit = 0.0;
    for (size_t j = 0; j < logit_errors.size(); ++j) {
      total_logit_errors += logit_errors[j];
      worst_relative_logit = std::max(worst_relative_logit, worst_logit[j]);
    }
    // Probabilities and tokens from the GPU's own logits.
    long long probability_errors = 0, token_errors = 0;
    std::vector<double> logits_row((size_t)V_), probabilities((size_t)V_);
    for (long long s = 0; s < S_; ++s)
      for (long long b = 0; b < B_; ++b) {
        const float *l = &h_logits_[(size_t)((s * B_ + b) * V_)];
        const float *p = &h_probabilities_[(size_t)((s * B_ + b) * V_)];
        for (long long r = 0; r < V_; ++r) logits_row[(size_t)r] = l[r];
        ref::softmax_double(logits_row.data(), V_, probabilities.data());
        long long best = 0;
        for (long long r = 0; r < V_; ++r) {
          const double bound = 2e-5 * std::fabs(probabilities[(size_t)r]) + 1e-12;
          if (!(std::fabs((double)p[r] - probabilities[(size_t)r]) <= bound)) {
            if (probability_errors < 5)
              std::fprintf(stderr, "decoder_lowprec: probability mismatch step %lld sequence "
                                   "%lld row %lld: expected %.9g, actual %.9g\n",
                           s, b, r, probabilities[(size_t)r], (double)p[r]);
            ++probability_errors;
          }
          if (p[r] > p[best]) best = r;
        }
        if (h_generated_[b * S_ + s] != best) {
          if (token_errors < 5)
            std::fprintf(stderr, "decoder_lowprec: token step %lld sequence %lld is %d, the "
                                 "first largest probability is at %lld\n",
                         s, b, h_generated_[b * S_ + s], best);
          ++token_errors;
        }
      }
    pass = normalized_errors == 0 && token_range_errors == 0 && activations_ok && weights_match_ &&
           total_logit_errors == 0 && probability_errors == 0 && token_errors == 0;
    std::fprintf(stderr,
                 "decoder_lowprec: implementation check %s: normalized %lld mismatches (worst "
                 "%.3g of the bound), %lld tokens out of range; converted weights %s, "
                 "activations %s; logits %lld mismatches over %lld sequences x %lld steps "
                 "(worst |error| / sum|terms| %.3g, bound %.3g); probabilities %lld mismatches; "
                 "tokens %lld not the first largest probability\n",
                 pass ? "PASS" : "FAIL", normalized_errors, worst_normalized, token_range_errors,
                 weights_match_ ? "exact" : "MISMATCH", activations_ok ? "exact" : "MISMATCH",
                 total_logit_errors, count, S_, worst_relative_logit, gamma, probability_errors,
                 token_errors);
    if (LP_LOWPREC_CASE == LP_CASE_FP32 && fixture_ == 0) {
      // The control must also satisfy the production contract unchanged.
      const decoder_oracle::Observed observed{h_generated_, h_normalized_.data(), h_logits_.data(),
                                              h_probabilities_.data()};
      const bool production = decoder_oracle::check(shape(), production_reference_, observed);
      pass = pass && production;
    }
    return pass;
  }

  long long B_, W_, V_, S_;
  int fixture_;
  std::vector<void *> allocations_;
  float *d_embedding_ = nullptr, *d_classifier_ = nullptr, *d_norm_weight_ = nullptr;
  int *d_prompts_ = nullptr, *d_tokens_ = nullptr, *d_generated_ = nullptr;
  float *d_normalized_ = nullptr, *d_logits_ = nullptr, *d_probabilities_ = nullptr;
  int *h_prompts_ = nullptr, *h_generated_ = nullptr;
  ref::Fixture host_fixture_;
  // FP32 control: the production classifier and its plan.
  const TileVariant *tile_ = nullptr;
  ClassifierFn classifier_ = nullptr;
  GemmPlan plan_{};
  long long grid_ = 0;
  float *d_partial_ = nullptr;
  int *d_ready_ = nullptr;
  size_t partial_count_ = 0;
  // Low precision: converted operands and GEMMs, one per step.
  Operand weights_;
  std::vector<Operand> activations_;
  float *d_alpha_ = nullptr;
  const float *weights_decode_for_selftest_ = nullptr;
#if LP_LOWPREC_CASE == LP_CASE_BF16
  std::vector<Bf16Gemm> bf16_gemm_;
#elif LP_BLOCK_SCALED
  std::vector<BlockScaledGemm> gemm_;
#endif
  // Host-side check state.
  std::vector<float> host_weights_, dequantized_weights_;
  float weight_decode_ = 1.0f;
  bool weights_match_ = true;
  decoder_oracle::Reference production_reference_;
  std::vector<float> h_normalized_, h_logits_, h_probabilities_;
  std::vector<std::vector<uint8_t>> h_activation_bytes_;
  mutable std::vector<float> synthetic_row_;
};

// Dimensions with the contract's limits and this experiment's alignment.
std::unique_ptr<LowPrecisionDecoder> make_lowprec(long long batch, long long width,
                                                  long long vocab, long long steps,
                                                  long long fixture) {
  if (fixture < 0 || fixture > 3)
    lp::fail(lp::kExitUsage, "fixture must be 0 (primary model) or 1, 2, 3 (got %lld)", fixture);
  if (batch < 1 || batch > (1LL << 20))
    lp::fail(lp::kExitUsage, "batch must be in [1, 1048576] (got %lld)", batch);
  if (width < 2 || width > (1LL << 24) - 16)
    lp::fail(lp::kExitUsage, "width must be in [2, 16777200] (got %lld)", width);
  if (vocab < 2 || vocab > width)
    lp::fail(lp::kExitUsage, "vocab must satisfy 2 <= vocab <= width (got %lld, %lld)", vocab,
             width);
  if (steps < 2 || steps > 1024)
    lp::fail(lp::kExitUsage, "steps must be in [2, 1024] (got %lld)", steps);
  if (LP_LOWPREC_CASE != LP_CASE_FP32 && (width % 128 != 0 || vocab % 4 != 0))
    lp::fail(lp::kExitUsage, "low precision cases need width %% 128 == 0 and vocab %% 4 == 0 "
                             "(got width %lld, vocab %lld)", width, vocab);
  const double device = 4.0 * (2.0 * (double)vocab * (double)width) +
                        4.0 * (double)steps * (double)batch * ((double)width + 2.0 * (double)vocab);
  if (device > 16.0 * (1LL << 30))
    lp::fail(lp::kExitUsage, "dimensions need %.1f GiB of device memory (limit 16)",
             device / (1LL << 30));
  return std::make_unique<LowPrecisionDecoder>(batch, width, vocab, steps, (int)fixture);
}

std::unique_ptr<lp::Workload> make_workload(lp::Options &options) {
  const long long fixture = options.take_int("fixture", 0);
  const bool dense = fixture != 0;
  const long long batch = options.take_int("batch", dense ? kFixtureBatch : kLargeBatch);
  const long long width = options.take_int("width", dense ? kFixtureWidth : kLargeWidth);
  const long long vocab = options.take_int("vocab", dense ? kFixtureVocab : kLargeVocab);
  const long long steps = options.take_int("steps", kLargeSteps);
  if (options.preset != "large") return nullptr;
  return make_lowprec(batch, width, vocab, steps, fixture);
}

// ---------------------------------------------------------------------------
// --accuracy (spec.json "accuracy").

std::string metrics_json(const ref::ErrorMetrics &m) {
  using lp::Json;
  return lp::render({{"rows", Json::integer(m.rows)},
                     {"max_abs_error", Json::number(m.max_abs_error)},
                     {"normalized_rms_error", Json::number(m.normalized_rms_error)},
                     {"max_abs_probability_error", Json::number(m.max_abs_probability_error)},
                     {"mean_total_variation", Json::number(m.mean_total_variation)},
                     {"top1_agree", Json::integer(m.top1_agree)}});
}

std::string accuracy_dataset(int fixture, cudaStream_t stream, double *weight_ms) {
  using lp::Json;
  auto workload = fixture == 0
                      ? make_lowprec(kLargeBatch, kLargeWidth, kLargeVocab, kLargeSteps, 0)
                      : make_lowprec(kFixtureBatch, kFixtureWidth, kFixtureVocab, kLargeSteps,
                                     fixture);
  LowPrecisionDecoder &d = *workload;
  d.setup(stream);
  LP_CUDA(cudaStreamSynchronize(stream));
  const long long B = d.batch(), W = d.width(), V = d.vocab(), S = d.steps();

  // Weight conversion time: median of five (setup cost, untimed elsewhere).
  if (weight_ms) {
    std::vector<double> times;
    cudaEvent_t start, stop;
    LP_CUDA(cudaEventCreate(&start));
    LP_CUDA(cudaEventCreate(&stop));
    for (int i = 0; i < 5; ++i) {
      LP_CUDA(cudaEventRecord(start, stream));
      d.convert_weights(stream);
      LP_CUDA(cudaEventRecord(stop, stream));
      LP_CUDA(cudaEventSynchronize(stop));
      float ms = 0.0f;
      LP_CUDA(cudaEventElapsedTime(&ms, start, stop));
      times.push_back(ms);
    }
    LP_CUDA(cudaEventDestroy(start));
    LP_CUDA(cudaEventDestroy(stop));
    *weight_ms = lp::median(times);
  }

  // Fixed FP32 reference activations: step-0 rows, rmsnorm in float on the host.
  std::vector<float> embedding_row((size_t)W), activations((size_t)(B * W));
  auto embedding = [&](long long token) -> const float * {
    if (fixture != 0) return &d.host_fixture().embedding[(size_t)(token * W)];
    std::fill(embedding_row.begin(), embedding_row.end(), 1.0f);
    embedding_row[(size_t)token] = 4.0f;
    return embedding_row.data();
  };
  for (long long b = 0; b < B; ++b)
    ref::rmsnorm_float(embedding(b % V), W, &activations[(size_t)(b * W)]);
  // Ordered on the stream: a pageable cudaMemcpy may return before its DMA
  // lands, and the legacy stream does not order the non-blocking one.
  LP_CUDA(cudaMemcpyAsync(d.normalized(0), activations.data(),
                          activations.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
  d.enqueue_classifier(0, stream);
  LP_CUDA(cudaStreamSynchronize(stream));
  LP_CUDA(cudaGetLastError());
  std::vector<float> logits((size_t)(B * V));
  LP_CUDA(cudaMemcpy(logits.data(), d.logits(0), logits.size() * sizeof(float),
                     cudaMemcpyDeviceToHost));
  std::vector<double> reference((size_t)(B * V)), actual((size_t)(B * V));
  for (size_t i = 0; i < logits.size(); ++i) actual[i] = logits[i];
  parallel_for(B, [&](long long b) {
    const float *x = &activations[(size_t)(b * W)];
    for (long long r = 0; r < V; ++r) {
      double sum = 0.0;
      if (fixture == 0) {
        sum = (double)x[(r + V - 1) % V]; // the one nonzero weight, 1.0
      } else {
        const float *w = &d.host_fixture().classifier[(size_t)(r * W)];
        for (long long k = 0; k < W; ++k) sum += (double)x[k] * (double)w[k];
      }
      reference[(size_t)(b * V + r)] = sum;
    }
  });
  const ref::ErrorMetrics classifier = ref::compare_logits(reference, actual, B, V);

  // Complete four-step decode, one application-scope invocation.
  d.poison_outputs(stream);
  d.enqueue_application(stream);
  LP_CUDA(cudaStreamSynchronize(stream));
  LP_CUDA(cudaGetLastError());
  std::vector<int32_t> expected_tokens((size_t)(B * S));
  if (fixture == 0) {
    for (long long b = 0; b < B; ++b)
      for (long long s = 0; s < S; ++s)
        expected_tokens[(size_t)(b * S + s)] = (int32_t)((b % V + s + 1) % V);
  } else {
    parallel_for(B, [&](long long b) {
      std::vector<double> o((size_t)W), l((size_t)V);
      long long token = b % V;
      for (long long s = 0; s < S; ++s) {
        ref::rmsnorm_double(&d.host_fixture().embedding[(size_t)(token * W)], W, o.data());
        for (long long r = 0; r < V; ++r) {
          const float *w = &d.host_fixture().classifier[(size_t)(r * W)];
          double sum = 0.0;
          for (long long k = 0; k < W; ++k) sum += o[(size_t)k] * (double)w[k];
          l[(size_t)r] = sum;
        }
        token = ref::argmax_first(l.data(), V);
        expected_tokens[(size_t)(b * S + s)] = (int32_t)token;
      }
    });
  }
  long long agree = 0;
  std::string per_step = "[";
  for (long long s = 0; s < S; ++s) {
    long long step_agree = 0;
    for (long long b = 0; b < B; ++b)
      step_agree += d.generated()[b * S + s] == expected_tokens[(size_t)(b * S + s)];
    agree += step_agree;
    per_step += (s ? ", " : "") + Json::integer(step_agree);
  }
  per_step += "]";

  lp::JsonObject out = {
      {"dataset", Json::quote(fixture == 0 ? "primary" : "fixture" + std::to_string(fixture))},
      {"batch", Json::integer(B)}, {"width", Json::integer(W)}, {"vocab", Json::integer(V)},
      {"steps", Json::integer(S)},
      {"classifier_error", metrics_json(classifier)},
      {"tokens", lp::render({{"agree", Json::integer(agree)}, {"total", Json::integer(B * S)},
                             {"per_step_agree", per_step},
                             {"reference", Json::quote(fixture == 0 ? "analytic progression"
                                                                    : "double host decode")}})}};
  if (fixture == 0) {
    // The production FP32 tolerance check of this decode.
    const decoder_oracle::Shape shape{B, W, V, S};
    const decoder_oracle::Reference production = decoder_oracle::compute(shape);
    std::vector<float> normalized((size_t)(S * B * W)), all_logits((size_t)(S * B * V)),
        probabilities((size_t)(S * B * V));
    LP_CUDA(cudaMemcpy(normalized.data(), d.normalized(0), normalized.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));
    LP_CUDA(cudaMemcpy(all_logits.data(), d.logits(0), all_logits.size() * sizeof(float),
                       cudaMemcpyDeviceToHost));
    LP_CUDA(cudaMemcpy(probabilities.data(), d.probabilities(0),
                       probabilities.size() * sizeof(float), cudaMemcpyDeviceToHost));
    const decoder_oracle::Observed observed{d.generated(), normalized.data(), all_logits.data(),
                                            probabilities.data()};
    std::fprintf(stderr, "decoder_lowprec: production FP32 tolerance check of the %s decode "
                         "(reported, not required):\n", kCaseName);
    const bool pass = decoder_oracle::check(shape, production, observed);
    out.push_back({"production_fp32_tolerance_pass", Json::boolean(pass)});
  }
  std::fprintf(stderr, "decoder_lowprec: accuracy %s fixture %d: max |error| %.3g, normalized "
                       "RMS %.3g, max |probability error| %.3g, tokens %lld/%lld\n",
               kCaseName, fixture, classifier.max_abs_error, classifier.normalized_rms_error,
               classifier.max_abs_probability_error, agree, B * S);
  return lp::render(out);
}

int run_accuracy(int argc, char **argv) {
  std::string json_out;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--accuracy") continue;
    if (flag == "--json-out" && i + 1 < argc) json_out = argv[++i];
    else lp::fail(lp::kExitUsage, "--accuracy takes only --json-out PATH (got %s)", flag.c_str());
  }
  cudaStream_t stream;
  LP_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  double weight_ms = 0.0;
  std::string datasets = "[";
  for (int fixture = 0; fixture <= 3; ++fixture)
    datasets += (fixture ? ", " : "") + accuracy_dataset(fixture, stream, fixture == 0 ? &weight_ms : nullptr);
  datasets += "]";
  cudaDeviceProp properties{};
  int device = 0;
  LP_CUDA(cudaGetDevice(&device));
  LP_CUDA(cudaGetDeviceProperties(&properties, device));
  using lp::Json;
  const std::string line = lp::render(
      {{"schema", Json::quote("blackwell_experiments.lowprec_accuracy/1")},
       {"identity", Json::quote(std::string("blackwell-experiments/lowprec-decoder-classifier/") +
                                kCaseName)},
       {"case", Json::quote(kCaseName)},
       {"spec_sha256", Json::quote(LP_CONTRACT_SHA256)},
       {"device", Json::quote(properties.name)},
       {"weight_conversion_ms_median_of_5", Json::number(weight_ms)},
       {"datasets", datasets},
       {"status", Json::quote("accuracy")}});
  std::printf("RESULT %s\n", line.c_str());
  if (!json_out.empty()) {
    FILE *file = std::fopen(json_out.c_str(), "w");
    if (!file || std::fprintf(file, "%s\n", line.c_str()) < 0 || std::fclose(file))
      lp::fail(lp::kExitUsage, "cannot write %s", json_out.c_str());
  }
  LP_CUDA(cudaStreamDestroy(stream));
  return 0;
}

// ---------------------------------------------------------------------------
// --selftest: the device conversion of crafted rows against the host quantizer.
int run_selftest() {
  const long long width = 256;
  const int rows = 130; // a second 128-row scale-factor tile, partly filled
  auto workload = make_lowprec(rows, width, 128, 2, 0);
  std::vector<float> m((size_t)(rows * width), 0.0f);
  // Element ties and saturation at unit block scale. NVFP4: the tensor
  // maximum 2688 (row 0) makes the encode scale 1, and a block maximum of 6
  // makes the block scale 1. MXFP6: a block maximum of 16 makes 2^X = 1.
  const float e2m1_cases[] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f, 5.5f,
                              -0.25f, -0.75f, -5.0f, 0.24f, -0.1f, 0.0f, 6.0f, 7.0f};
  const float e3m2_cases[] = {0.03125f, 0.09375f, 0.15625f, 0.21875f, 0.28125f, 26.0f, 30.0f,
                              31.9f, -0.03125f, -0.01f, 0.0f, 1.125f, 1.375f, 2.75f, 16.0f, -27.9f};
  m[0] = 2688.0f;
  for (int i = 0; i < 16; ++i) m[(size_t)(1 * width + i)] = e2m1_cases[i];      // block max 7
  for (int i = 0; i < 16; ++i) m[(size_t)(2 * width + i)] = e2m1_cases[i] * 6.0f / 7.0f;
  for (int i = 0; i < 16; ++i) m[(size_t)(3 * width + i)] = e3m2_cases[i];      // block max 31.9
  for (int i = 0; i < 16; ++i) m[(size_t)(3 * width + 16 + i)] = e3m2_cases[i] / 2.0f;
  m[(size_t)(4 * width)] = 16.0f;
  for (int i = 1; i < 32; ++i) m[(size_t)(4 * width + i)] = e3m2_cases[i % 16];
  // Tiny and zero blocks: a block scale that rounds to zero, subnormal scales.
  for (int i = 0; i < 16; ++i) m[(size_t)(5 * width + i)] = 1e-6f * (float)(i - 8);
  for (int i = 0; i < 16; ++i) m[(size_t)(5 * width + 16 + i)] = 0.004f * (float)(i - 8);
  for (int i = 0; i < 32; ++i) m[(size_t)(5 * width + 64 + i)] = 1e-30f * (float)(i + 1);
  // Scale ties: block maxima whose UE4M3 target falls between two codes.
  for (int i = 0; i < 16; ++i) m[(size_t)(6 * width + 16 * i)] = 6.0f * (1.0f + 0.0625f * (float)i);
  // Random rows everywhere else, including rows 128 and 129, with magnitudes
  // from 2^-9 to 4 (below 2688, so the NVFP4 encode scale stays 1).
  uint64_t state = 7;
  for (int r = 7; r < rows; ++r)
    for (long long k = 0; k < width; ++k) {
      state = state * 6364136223846793005ull + 1442695040888963407ull;
      const double u = (double)(state >> 11) * 0x1p-53;
      m[(size_t)(r * width + k)] = (float)((u - 0.5) * std::ldexp(1.0, (int)(r % 12) - 8));
    }
  const bool pass = workload->selftest_convert(m, rows);
  std::printf("%s: %s quantizer self-test\n", pass ? "PASS" : "FAIL", kCaseName);
  return pass ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--accuracy") return run_accuracy(argc, argv);
    if (std::string(argv[i]) == "--selftest") return run_selftest();
  }
  return lp::run_main(argc, argv, make_workload);
}

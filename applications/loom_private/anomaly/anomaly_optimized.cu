// loom-private/mlperf-tiny-anomaly-model, optimized variant: the same contract
// as anomaly.cu (same inputs, outputs, oracle, check, and scopes), computed
// with int8 tensor cores over tiles of samples and with the transfers
// pipelined against the inference.
//
// Setup (not timed): as in the baseline, plus the packed weights are padded
// for the tensor-core layout, the requantization constants (zero-point
// corrections from the weight row sums, binary32 scales) are packed, and the
// per-GPU pipeline chunk count is chosen from cudaDeviceProp.
//
// Per invocation (application scope): the samples are split into chunks. For
// each chunk, a side stream uploads its float32 inputs, the harness stream
// runs tiled_inference_kernel on it, and a second side stream downloads its
// int8 outputs, so uploads, inference, and downloads of different chunks
// overlap. Every input byte is uploaded and every output byte downloaded
// inside the call, as in the baseline.
//
// tiled_inference_kernel: one block per tile of 16 samples. It runs the ten
// layers as int8 GEMMs with mma.sync m16n8k32 (exact int32 accumulation),
// quantizing the inputs chunk by chunk into shared memory for layer 1,
// keeping the activations in shared memory, and writing layer 10's codes to
// the output.
//
// Exactness. The tensor cores multiply raw int8 activations a by int8 weights
// w; the zero point is applied afterwards: sum (a - z) w = sum a w - z sum w,
// with sum w the row's weight sum. Every term is an integer and
// |sum a w|, |z sum w| <= 128 * 128 * 640 = 10,485,760, so the int32 result
// equals the baseline's accumulator exactly (integer addition is exact in any
// order). Each code then comes from a binary32 fast path that proves it equal
// to the baseline's arithmetic, or else from the baseline's requantize() and
// input quantization, unchanged (requant.cuh; tests/test_requant.cu checks
// the fast paths exhaustively). So the output is bitwise equal to the
// baseline and to the CPU oracle.
#include "assets.h"
#include "lp_harness.cuh"
#include "model.h"
#include "oracle.h"
#include "requant.cuh"

#include <cmath>

namespace {

using anomaly_model::kDatasetBytes;
using anomaly_model::kDatasetSamples;
using anomaly_model::kLayerCount;
using anomaly_model::kModelBytes;
using anomaly_model::kSampleElements;

constexpr long long kLargeSamples = 104858;
constexpr long long kMaxSamples = 1ll << 20;

struct AssetSpec {
  const char *name;
  size_t bytes;
  const char *sha256;
};
constexpr AssetSpec kModelAsset = {
    "mlperf-tiny-ad01-int8-model", kModelBytes,
    "87cf24194ef93d1d9b11a591d805526b98008e351655d29883c825c9c106ba24"};
constexpr AssetSpec kDatasetAsset = {
    "mlperf-tiny-ad01-dataset", kDatasetBytes,
    "31bc130d27e3732e1c09db946ccc7bfa130f98739bcd90cfa39d590f61f4d6fa"};
// Reference only: upstream oracle bytes, compared with the CPU oracle.
constexpr AssetSpec kSmokeAsset = {
    "anomaly-upstream-expected-smoke", 1306,
    "212195d59044c9d88ebd46de3eb445ed10e3be98311b5449ea240837d8e10897"};
constexpr AssetSpec kValidationAsset = {
    "anomaly-upstream-expected-validation", 2612,
    "f1bc2c6d5c498d352b62457ab4220db463aa573afb2d7c459a3299e072331d80"};

using anomaly_requant::LayerParams;

struct ModelParams {
  float input_scale;
  int input_zero_point;
  LayerParams layers[kLayerCount];
  float input_reciprocal; // binary32 1 / input_scale (requant.cuh)
};

// The padded tensor-core layout of one layer: weights [n_tiles * 8][k_padded]
// row-major, zero beyond the real extents, at padded_offset in the padded
// weight array. fast holds the layer's binary32 requantization constants
// (requant.cuh make_fast_layer).
struct GemmLayer {
  int k_padded, n_tiles, padded_offset;
  anomaly_requant::FastLayer fast;
};
// The requantization constants of output columns 2p and 2p + 1 of a layer, at
// pair (bias_offset / 2 + p): the zero-point corrections z_in * sum_i w[o][i]
// and the binary32 bias terms beta (requant.cuh).
struct __align__(16) Epilogue {
  int correction[2];
  float beta[2];
};
struct GemmParams {
  GemmLayer layers[kLayerCount];
};

// Tile geometry. A block of kThreads threads infers a tile of samples. Layer 1
// streams its quantized inputs through two staging buffers of kKStep codes
// per sample; layers 1-9 write their outputs (at most kNarrowWidth codes per
// sample) to two alternating activation buffers; layer 10 writes its codes
// straight to the global output. Rows of the activation buffers are
// kNarrowStride bytes and rows of the staging buffers kKStep bytes, so the
// 16-byte fragment loads of a quarter warp (2 rows x 4 lanes) hit distinct
// banks. K is padded to multiples of kKStep.
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kKStep = 64;
constexpr int kNarrowWidth = 128;
constexpr int kNarrowStride = kNarrowWidth + 64;
static_assert(kNarrowStride % 128 == 64 && kKStep % 128 == 64, "bank layout");
static_assert(kSampleElements % kKStep == 0, "whole staging chunks");

// The input quantization and the requantization of one code (requant.cuh).
// The test build LP_ANOMALY_BASELINE_ARITHMETIC uses the baseline's
// functions for every code, which are the fallback paths of the fast forms.
__device__ __forceinline__ int quantize_code(float x, const ModelParams &params) {
#ifdef LP_ANOMALY_BASELINE_ARITHMETIC
  return anomaly_requant::quantize_input(x, params.input_scale, params.input_zero_point);
#else
  return anomaly_requant::quantize_input_fast(x, params.input_reciprocal, params.input_scale,
                                              params.input_zero_point);
#endif
}

__device__ __forceinline__ int requantize_code(int acc, const int32_t *bias, float beta,
                                               float margin_base, const GemmLayer &gemm,
                                               const LayerParams &layer) {
#ifdef LP_ANOMALY_BASELINE_ARITHMETIC
  (void)beta;
  (void)margin_base;
  (void)gemm;
  return anomaly_requant::requantize(acc, *bias, layer);
#else
  return anomaly_requant::requantize_fast(acc, bias, beta, margin_base, gemm.fast.kappa,
                                          gemm.fast.lambda, gemm.fast.code_min, layer);
#endif
}

// c += A * B for one m16n8k32 tile: A row-major 16x32 int8, B column-major
// 32x8 int8, C 16x8 int32.
__device__ __forceinline__ void mma_s8(int (&c)[4], unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0,
                                       unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// A warp's unit is one 16-sample m-tile by up to kNT 8-wide n-tiles of a
// layer. Lane (group, quad) holds rows `group` and `group` + 8 of the m-tile
// and columns 2 quad, 2 quad + 1 of each n-tile (the mma C layout).
//
// One 64-wide K step of a unit: each lane loads 16 bytes of each of its two A
// rows and of each B row (weight row), and the two mma steps take words
// (x, y) and (z, w). A and B use the same lane-to-k assignment, so each
// product a[k] w[k] is formed exactly once (the sum runs over a permutation
// of k, exact for integers).
template <int kNT>
__device__ __forceinline__ void load_b(uint4 (&b)[kNT], const int8_t *w, int w_tile_stride,
                                       int n_valid) {
#pragma unroll
  for (int t = 0; t < kNT; ++t)
    if (t < n_valid) b[t] = __ldg(reinterpret_cast<const uint4 *>(w + t * w_tile_stride));
}

template <int kNT>
__device__ __forceinline__ void unit_k_step(int (&acc)[kNT][4], const int8_t *a_lo,
                                            const int8_t *a_hi, const uint4 (&b)[kNT],
                                            int n_valid) {
  const uint4 lo = *reinterpret_cast<const uint4 *>(a_lo);
  const uint4 hi = *reinterpret_cast<const uint4 *>(a_hi);
#pragma unroll
  for (int t = 0; t < kNT; ++t) {
    if (t >= n_valid) break;
    mma_s8(acc[t], lo.x, hi.x, lo.y, hi.y, b[t].x, b[t].y);
    mma_s8(acc[t], lo.z, hi.z, lo.w, hi.w, b[t].z, b[t].w);
  }
}

// Requantizes a unit's accumulators and stores the int8 codes, two adjacent
// columns per 16-bit store, to rows of `out` (shared activations, or the
// global output with only rows < valid_rows stored).
template <int kNT, bool kGlobal>
__device__ __forceinline__ void store_unit(const int (&acc)[kNT][4], int first_tile, int n_valid,
                                           int row, int8_t *out, int out_stride,
                                           int valid_rows,
                                           const Epilogue *__restrict__ epilogue,
                                           const int32_t *__restrict__ biases,
                                           const LayerParams &layer, const GemmLayer &gemm) {
  const int quad = threadIdx.x % 4;
#pragma unroll
  for (int t = 0; t < kNT; ++t) {
    if (t >= n_valid) break;
    const int o = (first_tile + t) * 8 + quad * 2;
    const Epilogue e = epilogue[layer.bias_offset / 2 + o / 2];
    const int32_t *bias = biases + layer.bias_offset + o;
    int codes[4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float margin_base = anomaly_requant::fast_margin_base(e.beta[j]);
      codes[j] = requantize_code(acc[t][j] - e.correction[j], bias + j, e.beta[j],
                                 margin_base, gemm, layer);
      codes[2 + j] = requantize_code(acc[t][2 + j] - e.correction[j], bias + j,
                                     e.beta[j], margin_base, gemm, layer);
    }
    const unsigned short lo_pair = (unsigned short)((codes[0] & 0xff) | (codes[1] & 0xff) << 8);
    const unsigned short hi_pair = (unsigned short)((codes[2] & 0xff) | (codes[3] & 0xff) << 8);
    if (!kGlobal || row < valid_rows)
      *reinterpret_cast<unsigned short *>(out + (size_t)row * out_stride + o) = lo_pair;
    if (!kGlobal || row + 8 < valid_rows)
      *reinterpret_cast<unsigned short *>(out + (size_t)(row + 8) * out_stride + o) = hi_pair;
  }
}

// One fully connected layer of the tile whose inputs are on chip:
// out = requant(in * W^T).
template <int kTile, int kNT, bool kGlobal>
__device__ __forceinline__ void tile_layer(const int8_t *in, int8_t *out, int out_stride,
                                           int valid_rows,
                                           const int8_t *__restrict__ weights,
                                           const Epilogue *__restrict__ epilogue,
                                           const int32_t *__restrict__ biases,
                                           const LayerParams &layer,
                                           const GemmLayer &gemm) {
  constexpr int kMTiles = kTile / 16;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int group = lane / 4, quad = lane % 4;
  const int chunks = (gemm.n_tiles + kNT - 1) / kNT;
  for (int unit = warp; unit < kMTiles * chunks; unit += kWarps) {
    const int m_tile = unit % kMTiles, chunk = unit / kMTiles;
    const int row = m_tile * 16 + group;
    const int n_valid = min(kNT, gemm.n_tiles - chunk * kNT);
    const int8_t *a_lo = in + row * kNarrowStride + quad * 16;
    const int8_t *w = weights + gemm.padded_offset +
                      (size_t)(chunk * kNT * 8 + group) * gemm.k_padded + quad * 16;
    int acc[kNT][4] = {};
    for (int k = 0; k < gemm.k_padded; k += kKStep) {
      uint4 b[kNT];
      load_b<kNT>(b, w + k, 8 * gemm.k_padded, n_valid);
      unit_k_step<kNT>(acc, a_lo + k, a_lo + 8 * kNarrowStride + k, b, n_valid);
    }
    store_unit<kNT, kGlobal>(acc, chunk * kNT, n_valid, row, out, out_stride, valid_rows,
                             epilogue, biases, layer, gemm);
  }
}

// Loads (kRegs float4 per thread) and quantizes one kKStep-wide column chunk
// of the tile's inputs; rows past the end of the samples are zero.
template <int kTile>
struct InputChunk {
  static constexpr int kRegs = kTile * (kKStep / 4) / kThreads;
  static_assert(kRegs >= 1 && kTile * (kKStep / 4) % kThreads == 0, "whole float4 loads");
  float4 values[kRegs];

  __device__ __forceinline__ void load(const float *__restrict__ x, int rows, int chunk) {
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      const int v = threadIdx.x + i * kThreads;
      const int r = v / (kKStep / 4), c = v % (kKStep / 4);
      values[i] = r < rows ? __ldg(reinterpret_cast<const float4 *>(
                                 x + (size_t)r * kSampleElements + chunk * kKStep) + c)
                           : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  }
  __device__ __forceinline__ void store(int8_t *stage, const ModelParams &params) const {
#pragma unroll
    for (int i = 0; i < kRegs; ++i) {
      const int v = threadIdx.x + i * kThreads;
      const int r = v / (kKStep / 4), c = v % (kKStep / 4);
      *reinterpret_cast<char4 *>(stage + r * kKStep + 4 * c) =
          make_char4((signed char)quantize_code(values[i].x, params),
                     (signed char)quantize_code(values[i].y, params),
                     (signed char)quantize_code(values[i].z, params),
                     (signed char)quantize_code(values[i].w, params));
    }
  }
};

// Infers samples [begin, end) of the `samples` in input and output, one block
// per tile of kTile samples (the last tile may be partial; its missing rows
// are computed on zeros and never stored). kNT = kTile / 8, so
// layer 1 (at most 128 outputs) has at most one unit per warp, and each
// warp keeps its layer-1 accumulators while the input chunks stream through.
template <int kTile>
__global__ void __launch_bounds__(kThreads)
tiled_inference_kernel(const float *__restrict__ input, int begin, int end, int samples,
                       const int8_t *__restrict__ weights,
                       const Epilogue *__restrict__ epilogue,
                       const int32_t *__restrict__ biases, const ModelParams params,
                       const GemmParams gemm, int8_t *__restrict__ output) {
  constexpr int kNT = kTile / 8;
  constexpr int kMTiles = kTile / 16;
  constexpr int kChunks = (int)kSampleElements / kKStep;
  // Layer 1 gives each warp one unit: kWarps / kMTiles chunks of kNT n-tiles
  // per m-tile must cover its (at most kNarrowWidth) outputs.
  static_assert(kWarps % kMTiles == 0 && kWarps / kMTiles * kNT * 8 >= kNarrowWidth,
                "layer 1 schedule covers kNarrowWidth outputs");
  __shared__ __align__(16) int8_t smem[kTile * (2 * kNarrowStride + 2 * kKStep)];
  int8_t *act[2] = {smem, smem + kTile * kNarrowStride}; // [kTile][kNarrowStride]
  int8_t *stage[2] = {smem + 2 * kTile * kNarrowStride,  // [kTile][kKStep]
                      smem + 2 * kTile * kNarrowStride + kTile * kKStep};
  const int first = begin + blockIdx.x * kTile;
  const int rows = min(kTile, end - first);
#ifdef LP_ANOMALY_MUTANT_SKIP_LAST_SAMPLE
  // Test-only mutant: the last sample is never written, so its output stays
  // poisoned and the check must fail.
  const int stored_rows = first + rows == samples ? rows - 1 : rows;
#else
  (void)samples;
  const int stored_rows = rows;
#endif
  const float *x = input + (size_t)first * kSampleElements;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int group = lane / 4, quad = lane % 4;

  // Layer 1: the input chunk c + 1 is loaded into registers while the mma
  // steps consume chunk c from shared memory.
  {
    const GemmLayer &g = gemm.layers[0];
    const int m_tile = warp % kMTiles, chunk = warp / kMTiles;
    const int row = m_tile * 16 + group;
    const int n_valid = min(kNT, g.n_tiles - chunk * kNT);
    const int8_t *w = weights + g.padded_offset +
                      (size_t)(chunk * kNT * 8 + group) * g.k_padded + quad * 16;
    InputChunk<kTile> pending;
    pending.load(x, rows, 0);
    pending.store(stage[0], params);
    __syncthreads();
    int acc[kNT][4] = {};
#pragma unroll 1
    for (int c = 0; c < kChunks; ++c) {
      if (c + 1 < kChunks) pending.load(x, rows, c + 1);
      uint4 b[kNT];
      load_b<kNT>(b, w + c * kKStep, 8 * g.k_padded, n_valid);
      const int8_t *a_lo = stage[c % 2] + row * kKStep + quad * 16;
      unit_k_step<kNT>(acc, a_lo, a_lo + 8 * kKStep, b, n_valid);
      if (c + 1 < kChunks) pending.store(stage[(c + 1) % 2], params);
      __syncthreads();
    }
    if (n_valid > 0)
      store_unit<kNT, false>(acc, chunk * kNT, n_valid, row, act[0], kNarrowStride, kTile,
                             epilogue, biases, params.layers[0], g);
    __syncthreads();
  }
  // Layers 2-9 alternate between the activation buffers; layer 10 writes the
  // global output.
#pragma unroll 1
  for (int l = 1; l + 1 < (int)kLayerCount; ++l) {
    tile_layer<kTile, kNT, false>(act[(l + 1) % 2], act[l % 2], kNarrowStride, kTile,
                                  weights, epilogue, biases, params.layers[l],
                                  gemm.layers[l]);
    __syncthreads();
  }
  tile_layer<kTile, kNT, true>(act[(kLayerCount) % 2], output + (size_t)first * kSampleElements,
                               (int)kSampleElements, stored_rows, weights, epilogue, biases,
                               params.layers[kLayerCount - 1], gemm.layers[kLayerCount - 1]);
}

// Samples per block. 16 was fastest on all three GPUs: larger tiles reuse
// each weight fragment across more samples but need more shared memory and
// registers per block, and the lower occupancy cost more than the reuse saved.
constexpr int kBlockSamples = 16;

// Application-scope pipeline chunks for this GPU (README "Optimized
// variant"). The RTX 5090 (cc 12.x, PCIe 5) uploads more slowly while a
// download runs, so fewer, larger chunks overlap fewer downloads with the
// uploads; on the PCIe 4 RTX 3090 and 4090 the upload rate does not drop, and
// more chunks shorten the pipeline's fill and drain. Tests force other counts
// with -DLP_ANOMALY_FORCE_CHUNKS=N.
int select_chunks(const cudaDeviceProp &properties) {
#ifdef LP_ANOMALY_FORCE_CHUNKS
  (void)properties;
  return LP_ANOMALY_FORCE_CHUNKS;
#else
  return properties.major >= 12 ? 16 : 32;
#endif
}

// Asset paths, resolved from --asset (or the cache) by the factory so that the
// harness can reject unknown --asset names before setup.
struct AssetPaths {
  std::string model, dataset, smoke, validation;
};

std::string asset_path(lp::Options &options, const AssetSpec &spec) {
  return options.asset(spec.name, lp::asset_cache() + "/" + spec.name);
}

std::vector<uint8_t> load_asset(const std::string &path, const AssetSpec &spec) {
  std::vector<uint8_t> bytes;
  std::string error;
  if (!anomaly_assets::load_verified(spec.name, path, spec.bytes, spec.sha256,
                                     bytes, error))
    lp::fail(lp::kExitAsset, "%s", error.c_str());
  return bytes;
}

class Anomaly final : public lp::Workload {
public:
  Anomaly(lp::Action action, AssetPaths paths, long long samples)
      : action_(action), paths_(std::move(paths)), samples_(samples) {}

  ~Anomaly() override {
    for (cudaEvent_t event : events_) cudaEventDestroy(event);
    if (upload_stream_) cudaStreamDestroy(upload_stream_);
    if (download_stream_) cudaStreamDestroy(download_stream_);
    cudaFreeHost(h_input_);
    cudaFreeHost(h_output_);
    cudaFree(d_input_);
    cudaFree(d_output_);
    cudaFree(d_padded_weights_);
    cudaFree(d_epilogue_);
    cudaFree(d_biases_);
  }

  std::string identity() const override {
    return "loom-private/mlperf-tiny-anomaly-model";
  }

  lp::JsonObject dimensions() const override {
    using lp::Json;
    std::string widths = "[" + Json::integer(kSampleElements);
    long long macs = 0;
    for (const auto &layer : model_.layers) {
      widths += ", " + Json::integer(layer.output_count);
      macs += (long long)layer.input_count * layer.output_count;
    }
    widths += "]";
    return {{"samples", Json::integer(samples_)},
            {"sample_elements", Json::integer(kSampleElements)},
            {"dataset_samples", Json::integer(kDatasetSamples)},
            {"layers", Json::integer(kLayerCount)},
            {"layer_widths", widths},
            {"macs_per_sample", Json::integer(macs)},
            {"input_float32_elements", Json::integer(elements())},
            {"output_int8_bytes", Json::integer(elements())}};
  }

  lp::JsonObject asset_hashes() const override { return hashes_; }

  std::string application_scope() const override {
    return "upload the float32 samples [samples][640] from pinned host memory; "
           "int8 inference of every sample through the 10 FC layers; download "
           "the int8 outputs [samples][640]";
  }
  std::string compute_scope() const override {
    return "int8 inference of every sample through the 10 FC layers with the "
           "samples resident on the device; no host-device transfers";
  }

  void setup(cudaStream_t stream) override {
    model_bytes_ = load_asset(paths_.model, kModelAsset);
    dataset_ = load_asset(paths_.dataset, kDatasetAsset);
    hashes_ = {{kModelAsset.name, lp::Json::quote(kModelAsset.sha256)},
               {kDatasetAsset.name, lp::Json::quote(kDatasetAsset.sha256)}};
    if (action_ == lp::Action::Check) {
      // Needed only by the oracle cross-check; bench does not read them.
      smoke_ = load_asset(paths_.smoke, kSmokeAsset);
      validation_ = load_asset(paths_.validation, kValidationAsset);
      hashes_.push_back({kSmokeAsset.name, lp::Json::quote(kSmokeAsset.sha256)});
      hashes_.push_back(
          {kValidationAsset.name, lp::Json::quote(kValidationAsset.sha256)});
    }
    try {
      model_ = anomaly_model::parse(model_bytes_);
    } catch (const anomaly_model::ModelError &error) {
      lp::fail(lp::kExitAsset, "model asset rejected: %s", error.what());
    }
    for (size_t i = 0; i < kDatasetSamples * kSampleElements; ++i)
      if (!std::isfinite(anomaly_model::load_le_float(dataset_.data() + 4 * i)))
        lp::fail(lp::kExitAsset, "dataset sample %zu element %zu is not finite",
                 i / kSampleElements, i % kSampleElements);

    // Pack the layers for the kernel, as the baseline does, plus the padded
    // tensor-core layout and the requantization constants. The tile keeps
    // intermediate activations of at most kNarrowWidth codes per sample on
    // chip.
    std::vector<int8_t> padded;
    std::vector<int32_t> biases;
    std::vector<Epilogue> epilogue;
    int weight_count = 0;
    params_.input_scale = model_.input.scale;
    params_.input_zero_point = (int)model_.input.zero_point;
    params_.input_reciprocal = anomaly_requant::fast_input_reciprocal(model_.input.scale);
    for (size_t l = 0; l < kLayerCount; ++l) {
      const anomaly_model::Layer &layer = model_.layers[l];
      const bool last = l + 1 == kLayerCount;
      if (layer.output_count % 8 != 0 ||
          (!last && layer.output_count > (uint32_t)kNarrowWidth))
        lp::fail(lp::kExitAsset,
                 "model asset rejected: layer %zu has %u outputs; the optimized "
                 "variant needs a multiple of 8, at most %d before the last layer",
                 l + 1, layer.output_count, kNarrowWidth);
      LayerParams &p = params_.layers[l];
      p = anomaly_requant::make_layer_params(layer, weight_count, (int)biases.size());
      weight_count += (int)layer.weights.size();
      biases.insert(biases.end(), layer.biases.begin(), layer.biases.end());

      GemmLayer &g = gemm_.layers[l];
      std::vector<float> betas(layer.output_count);
      g.fast = anomaly_requant::make_fast_layer(p, layer.biases.data(), betas.data());
      g.k_padded = (int)((layer.input_count + kKStep - 1) / kKStep * kKStep);
      g.n_tiles = (int)(layer.output_count / 8);
      g.padded_offset = (int)padded.size();
      padded.resize(padded.size() + (size_t)layer.output_count * g.k_padded, 0);
      for (uint32_t o = 0; o < layer.output_count; ++o) {
        int32_t sum = 0; // |sum| <= 128 * 640
        for (uint32_t i = 0; i < layer.input_count; ++i) {
          const int8_t w = layer.weights[(size_t)o * layer.input_count + i];
          padded[g.padded_offset + (size_t)o * g.k_padded + i] = w;
          sum += w;
        }
        if (o % 2 == 0) epilogue.push_back({});
        epilogue.back().correction[o % 2] = p.input_zero_point * sum;
        epilogue.back().beta[o % 2] = betas[o];
      }
    }
    LP_CUDA(cudaMalloc(&d_padded_weights_, padded.size()));
    LP_CUDA(cudaMalloc(&d_biases_, biases.size() * sizeof(int32_t)));
    LP_CUDA(cudaMalloc(&d_epilogue_, epilogue.size() * sizeof(Epilogue)));
    LP_CUDA(cudaMemcpyAsync(d_padded_weights_, padded.data(), padded.size(),
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_biases_, biases.data(), biases.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_epilogue_, epilogue.data(), epilogue.size() * sizeof(Epilogue),
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaStreamSynchronize(stream)); // the host vectors go out of scope

    // Materialize the cycled inputs: sample i is dataset sample i mod 40.
    LP_CUDA(cudaMallocHost(&h_input_, input_bytes()));
    LP_CUDA(cudaMallocHost(&h_output_, output_bytes()));
    for (long long s = 0; s < samples_; ++s) {
      const uint8_t *source = dataset_.data() + (s % kDatasetSamples) * kSampleElements * 4;
      for (size_t i = 0; i < kSampleElements; ++i)
        h_input_[s * kSampleElements + i] = anomaly_model::load_le_float(source + 4 * i);
    }
    LP_CUDA(cudaMalloc(&d_input_, input_bytes()));
    LP_CUDA(cudaMalloc(&d_output_, output_bytes()));

    // The per-GPU chunk count, and the pipeline streams and events.
    int device = 0;
    cudaDeviceProp properties{};
    LP_CUDA(cudaGetDevice(&device));
    LP_CUDA(cudaGetDeviceProperties(&properties, device));
    const int chunks = select_chunks(properties);
    // Chunks are whole tiles; the last chunk takes the remainder.
    const long long tiles = (samples_ + kBlockSamples - 1) / kBlockSamples;
    const long long chunk_tiles = (tiles + chunks - 1) / chunks;
    for (long long first = 0; first < samples_; first += chunk_tiles * kBlockSamples)
      chunk_starts_.push_back(first);
    chunk_starts_.push_back(samples_);
    std::fprintf(stderr,
                 "anomaly: optimized configuration for %s (cc %d.%d): %d samples "
                 "per block, %zu pipeline chunks\n",
                 properties.name, properties.major, properties.minor, kBlockSamples,
                 chunk_starts_.size() - 1);
    LP_CUDA(cudaStreamCreateWithFlags(&upload_stream_, cudaStreamNonBlocking));
    LP_CUDA(cudaStreamCreateWithFlags(&download_stream_, cudaStreamNonBlocking));
    events_.resize(3 + 2 * (chunk_starts_.size() - 1));
    for (cudaEvent_t &event : events_)
      LP_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }

  // Fork two side streams from the harness stream; chunk c is uploaded on
  // upload_stream_, inferred on the harness stream after its upload, and
  // downloaded on download_stream_ after its inference; then both side
  // streams join the harness stream. The same dependencies hold in Direct
  // mode and in the captured graph.
  void enqueue_application(cudaStream_t stream) override {
    cudaEvent_t fork = events_[0], upload_join = events_[1], download_join = events_[2];
    LP_CUDA(cudaEventRecord(fork, stream));
    LP_CUDA(cudaStreamWaitEvent(upload_stream_, fork, 0));
    LP_CUDA(cudaStreamWaitEvent(download_stream_, fork, 0));
    const size_t chunks = chunk_starts_.size() - 1;
    for (size_t c = 0; c < chunks; ++c) {
      const long long first = chunk_starts_[c], count = chunk_starts_[c + 1] - first;
      const size_t offset = (size_t)first * kSampleElements;
      const size_t elements = (size_t)count * kSampleElements;
      cudaEvent_t uploaded = events_[3 + 2 * c], inferred = events_[4 + 2 * c];
      LP_CUDA(cudaMemcpyAsync(d_input_ + offset, h_input_ + offset,
                              elements * sizeof(float), cudaMemcpyHostToDevice,
                              upload_stream_));
      LP_CUDA(cudaEventRecord(uploaded, upload_stream_));
      LP_CUDA(cudaStreamWaitEvent(stream, uploaded, 0));
      launch(stream, first, count);
      LP_CUDA(cudaEventRecord(inferred, stream));
      LP_CUDA(cudaStreamWaitEvent(download_stream_, inferred, 0));
      LP_CUDA(cudaMemcpyAsync(h_output_ + offset, d_output_ + offset, elements,
                              cudaMemcpyDeviceToHost, download_stream_));
    }
    LP_CUDA(cudaEventRecord(upload_join, upload_stream_));
    LP_CUDA(cudaEventRecord(download_join, download_stream_));
    LP_CUDA(cudaStreamWaitEvent(stream, upload_join, 0));
    LP_CUDA(cudaStreamWaitEvent(stream, download_join, 0));
  }

  void enqueue_compute(cudaStream_t stream) override { launch(stream, 0, samples_); }

  // The device input copy and the device and host outputs are overwritten.
  // The output poison byte alternates between 0xa5 and 0x5a, so an output
  // byte that is never written fails the check in at least one of each mode's
  // two invocations, whatever its expected value.
  void poison_outputs(cudaStream_t stream) override {
    const int poison = (poison_count_++ % 2) ? 0x5a : 0xa5;
    LP_CUDA(cudaMemsetAsync(d_input_, 0xff, input_bytes(), stream));
    LP_CUDA(cudaMemsetAsync(d_output_, poison, output_bytes(), stream));
    std::memset(h_output_, poison ^ 0xff, output_bytes());
  }

  void compute_reference() override {
    try {
      reference_ = anomaly_oracle::infer_dataset(model_, dataset_);
    } catch (const std::exception &error) {
      lp::fail(lp::kExitAsset, "CPU oracle rejected the inputs: %s", error.what());
    }
    // Cross-check against upstream's oracle bytes: smoke measured sample 0 is
    // dataset sample 0; validation (2 warmups) measured samples 0 and 1 are
    // dataset samples 2 and 3.
    struct Expected {
      const AssetSpec *asset;
      const std::vector<uint8_t> *text;
      size_t rows, first_dataset_sample;
    };
    const Expected files[] = {{&kSmokeAsset, &smoke_, 1, 0},
                              {&kValidationAsset, &validation_, 2, 2}};
    for (const Expected &file : files) {
      std::vector<std::vector<uint8_t>> rows;
      std::string error;
      if (!anomaly_oracle::parse_expected(*file.text, file.rows, rows, error))
        lp::fail(lp::kExitAsset, "%s: %s", file.asset->name, error.c_str());
      for (size_t r = 0; r < rows.size(); ++r) {
        const size_t sample = file.first_dataset_sample + r;
        const int8_t *mine = reference_.data() + sample * kSampleElements;
        for (size_t i = 0; i < kSampleElements; ++i)
          if ((uint8_t)mine[i] != rows[r][i])
            lp::fail(lp::kExitMismatch,
                     "CPU oracle disagrees with %s for dataset sample %zu at "
                     "element %zu: upstream 0x%02x, oracle 0x%02x",
                     file.asset->name, sample, i, rows[r][i], (uint8_t)mine[i]);
      }
    }
    std::fprintf(stderr,
                 "anomaly: CPU oracle matches upstream reference bytes for "
                 "dataset samples 0, 2, 3\n");
  }

  bool check_outputs() override {
    long long mismatches = 0;
    for (long long s = 0; s < samples_; ++s) {
      const int8_t *actual = h_output_ + s * kSampleElements;
      const int8_t *expected = reference_.data() + (s % kDatasetSamples) * kSampleElements;
      if (std::memcmp(actual, expected, kSampleElements) == 0) continue;
      for (size_t i = 0; i < kSampleElements; ++i) {
        if (actual[i] == expected[i]) continue;
        if (mismatches < 5)
          std::fprintf(stderr,
                       "anomaly: mismatch at sample %lld (dataset sample %lld) "
                       "element %zu (byte %lld): expected %d (0x%02x), actual "
                       "%d (0x%02x)\n",
                       s, s % (long long)kDatasetSamples, i,
                       s * (long long)kSampleElements + (long long)i,
                       expected[i], (uint8_t)expected[i], actual[i],
                       (uint8_t)actual[i]);
        ++mismatches;
      }
    }
    if (mismatches)
      std::fprintf(stderr, "anomaly: %lld of %lld output bytes mismatch\n",
                   mismatches, elements());
    else
      std::fprintf(stderr, "anomaly: all %lld output bytes match the oracle\n",
                   elements());
    return mismatches == 0;
  }

  uint64_t output_digest() const override {
    return lp::fnv1a64(h_output_, output_bytes());
  }

private:
  long long elements() const { return samples_ * (long long)kSampleElements; }
  size_t input_bytes() const { return (size_t)elements() * sizeof(float); }
  size_t output_bytes() const { return (size_t)elements(); }

  // Infers samples [first, first + count) on `stream`.
  void launch(cudaStream_t stream, long long first, long long count) {
    const unsigned blocks = (unsigned)((count + kBlockSamples - 1) / kBlockSamples);
    tiled_inference_kernel<kBlockSamples><<<blocks, kThreads, 0, stream>>>(
        d_input_, (int)first, (int)(first + count), (int)samples_, d_padded_weights_,
        d_epilogue_, d_biases_, params_, gemm_, d_output_);
    LP_CUDA_LAUNCH();
  }

  const lp::Action action_;
  const AssetPaths paths_;
  long long samples_;
  std::vector<uint8_t> model_bytes_, dataset_, smoke_, validation_;
  lp::JsonObject hashes_;
  anomaly_model::Model model_;
  ModelParams params_{};
  GemmParams gemm_{};
  std::vector<long long> chunk_starts_; // chunk c is [starts[c], starts[c + 1])
  cudaStream_t upload_stream_ = nullptr, download_stream_ = nullptr;
  std::vector<cudaEvent_t> events_; // fork, joins, then per chunk 2
  int poison_count_ = 0;
  float *h_input_ = nullptr, *d_input_ = nullptr;
  int8_t *h_output_ = nullptr, *d_output_ = nullptr;
  int8_t *d_padded_weights_ = nullptr;
  int32_t *d_biases_ = nullptr;
  Epilogue *d_epilogue_ = nullptr;
  std::vector<int8_t> reference_; // [40][640]
};

std::unique_ptr<lp::Workload> make_workload(lp::Options &options) {
  if (options.preset != "large") return nullptr;
  const long long samples = options.take_int("samples", kLargeSamples);
  if (samples < 1 || samples > kMaxSamples)
    lp::fail(lp::kExitUsage, "samples must be in [1, %lld], got %lld", kMaxSamples,
             samples);
  AssetPaths paths{asset_path(options, kModelAsset), asset_path(options, kDatasetAsset),
                   asset_path(options, kSmokeAsset), asset_path(options, kValidationAsset)};
  return std::make_unique<Anomaly>(options.action, std::move(paths), samples);
}

} // namespace

int main(int argc, char **argv) { return lp::run_main(argc, argv, make_workload); }

// loom-private/multisensor-attention: audio queries attend over IMU context.
//
// Pipeline per invocation (application scope):
//   upload audio [Q][3] and IMU [C][3] from pinned host memory
//   project_audio  -> query     [Q][4] = (x, y, z, x + y)
//   project_imu    -> key/value [C][4] = (x, y, z, x - y)
//   attention      -> attention [Q][4], one block per query row, two streaming
//                     passes over the context (row maximum, then exponentials
//                     and weighted sums); no score matrix is stored
//   residual_statistics -> output [Q][4] = attention + query * 0.125 and the
//                     four statistics, in one block with a fixed reduction order
//   download output and statistics
//
// The kernels are adapted from the legacy loom_msa/msa_baseline.cu streaming
// baseline (one block per query row, fixed-shape shared-memory trees), with
// the context length a runtime value and the value pass fused into a single
// sweep. Every reduction has a fixed order, so results are bitwise
// reproducible across invocations and between Direct and Graph launches.
#include "lp_harness.cuh"
#include "oracle.h"

#include <cmath>

namespace {

constexpr int kSensorLanes = 3;
constexpr int kFeatureLanes = 4;
constexpr int kStatistics = 4;

constexpr long long kLargeQueryTokens = 25000;
constexpr long long kLargeContextTokens = 250000;
// Query weights 4t + lane + 1 stay exact in float up to 2^24.
constexpr long long kMaxQueryTokens = 1ll << 22;
// Sensor ordinals 3t + lane stay far from 32-bit wraparound of ordinal * 37.
constexpr long long kMaxContextTokens = 1ll << 24;

constexpr int kProjectionThreads = 256;
constexpr int kAttentionThreads = 256;
constexpr int kStatisticsThreads = 1024;

float variant_scale(int variant) {
  return variant == 1 ? 0.5f : variant == 2 ? 1.5f : 1.0f;
}

// Host input generator (upstream's sensor_sample generalized to separate
// query and context extents).
float sensor_sample(unsigned ordinal, unsigned phase, float scale) {
  const unsigned mixed = (ordinal * 37u + phase * 101u) % 251u;
  return ((float)((int)mixed - 125) * (1.0f / 128.0f)) * scale;
}

__global__ void project_audio_kernel(const float *__restrict__ audio, int tokens,
                                     float4 *__restrict__ query) {
  const int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  const float x = audio[token * kSensorLanes + 0];
  const float y = audio[token * kSensorLanes + 1];
  const float z = audio[token * kSensorLanes + 2];
  query[token] = make_float4(x, y, z, x + y);
}

__global__ void project_imu_kernel(const float *__restrict__ imu, int tokens,
                                   float4 *__restrict__ key_value) {
  const int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  const float x = imu[token * kSensorLanes + 0];
  const float y = imu[token * kSensorLanes + 1];
  const float z = imu[token * kSensorLanes + 2];
  key_value[token] = make_float4(x, y, z, x - y);
}

// Upstream's score: the lane products summed in lane order, then halved.
__device__ __forceinline__ float half_score(const float4 q, const float4 k) {
  float dot = 0.0f;
  dot += q.x * k.x;
  dot += q.y * k.y;
  dot += q.z * k.z;
  dot += q.w * k.w;
  return dot * 0.5f;
}

// One block per query row. Thread i visits context columns i, i + T, ... in
// increasing order; partials combine through a fixed binary tree.
__global__ void __launch_bounds__(kAttentionThreads)
attention_kernel(const float4 *__restrict__ query,
                 const float4 *__restrict__ key_value, int context_tokens,
                 float4 *__restrict__ attention) {
  __shared__ float maximum_partials[kAttentionThreads];
  __shared__ float sum_partials[5][kAttentionThreads];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const float4 q = query[row];

  float maximum = -INFINITY;
  for (int column = tid; column < context_tokens; column += kAttentionThreads)
    maximum = fmaxf(maximum, half_score(q, key_value[column]));
  maximum_partials[tid] = maximum;
  __syncthreads();
  for (int stride = kAttentionThreads / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      maximum_partials[tid] =
          fmaxf(maximum_partials[tid], maximum_partials[tid + stride]);
    __syncthreads();
  }
  const float row_maximum = maximum_partials[0];

  float denominator = 0.0f, v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
  for (int column = tid; column < context_tokens; column += kAttentionThreads) {
    const float4 k = key_value[column];
    const float p = expf(half_score(q, k) - row_maximum);
    denominator += p;
    v0 += p * k.x;
    v1 += p * k.y;
    v2 += p * k.z;
    v3 += p * k.w;
  }
  sum_partials[0][tid] = denominator;
  sum_partials[1][tid] = v0;
  sum_partials[2][tid] = v1;
  sum_partials[3][tid] = v2;
  sum_partials[4][tid] = v3;
  __syncthreads();
  for (int stride = kAttentionThreads / 2; stride > 0; stride /= 2) {
    if (tid < stride)
      for (int i = 0; i < 5; ++i)
        sum_partials[i][tid] += sum_partials[i][tid + stride];
    __syncthreads();
  }
  if (tid == 0) {
    const float d = sum_partials[0][0];
    attention[row] = make_float4(sum_partials[1][0] / d, sum_partials[2][0] / d,
                                 sum_partials[3][0] / d, sum_partials[4][0] / d);
  }
}

// One block. Thread i visits flat elements i, i + T, ... (element = 4t + lane)
// in increasing order, writes the residual, and accumulates the statistics in
// float as upstream does; partials combine through a fixed binary tree. At the
// large preset the float sums stay within about 0.5% of the tolerance against
// the double oracle (contract.json "tolerances").
__global__ void __launch_bounds__(kStatisticsThreads)
residual_statistics_kernel(const float *__restrict__ query,
                           const float *__restrict__ attention, int elements,
                           float *__restrict__ output,
                           float *__restrict__ statistics) {
  __shared__ float energy_partials[kStatisticsThreads];
  __shared__ float sum_partials[kStatisticsThreads];
  __shared__ float weighted_partials[kStatisticsThreads];
  __shared__ float maximum_partials[kStatisticsThreads];
  const int tid = threadIdx.x;
  float energy = 0.0f, sum = 0.0f, weighted = 0.0f;
  float maximum = 0.0f;
  for (int element = tid; element < elements; element += kStatisticsThreads) {
    const float projection = query[element];
    const float value = attention[element];
    const float magnitude = value < 0.0f ? -value : value;
    energy += projection * projection;
    sum += value;
    weighted += (float)(element + 1) * value;
    if (magnitude > maximum) maximum = magnitude;
    output[element] = value + projection * 0.125f;
  }
  energy_partials[tid] = energy;
  sum_partials[tid] = sum;
  weighted_partials[tid] = weighted;
  maximum_partials[tid] = maximum;
  __syncthreads();
  for (int stride = kStatisticsThreads / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      energy_partials[tid] += energy_partials[tid + stride];
      sum_partials[tid] += sum_partials[tid + stride];
      weighted_partials[tid] += weighted_partials[tid + stride];
      if (maximum_partials[tid + stride] > maximum_partials[tid])
        maximum_partials[tid] = maximum_partials[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    statistics[0] = energy_partials[0];
    statistics[1] = sum_partials[0];
    statistics[2] = weighted_partials[0];
    statistics[3] = maximum_partials[0];
  }
}

// Upstream's tolerance, evaluated in double against the double oracle; NaN
// fails.
bool within_tolerance(double actual, double expected) {
  return std::fabs(actual - expected) <= 1.0e-4 * (1.0 + std::fabs(expected));
}

class Attention final : public lp::Workload {
public:
  Attention(long long query_tokens, long long context_tokens, int variant)
      : Q_(query_tokens), C_(context_tokens), variant_(variant),
        scale_(variant_scale(variant)) {}

  ~Attention() override {
    cudaFreeHost(h_audio_);
    cudaFreeHost(h_imu_);
    cudaFreeHost(h_output_);
    cudaFreeHost(h_statistics_);
    cudaFree(d_audio_);
    cudaFree(d_imu_);
    cudaFree(d_query_);
    cudaFree(d_key_value_);
    cudaFree(d_attention_);
    cudaFree(d_output_);
    cudaFree(d_statistics_);
  }

  std::string identity() const override {
    return "loom-private/multisensor-attention";
  }

  lp::JsonObject dimensions() const override {
    using lp::Json;
    return {{"query_tokens", Json::integer(Q_)},
            {"context_tokens", Json::integer(C_)},
            {"sensor_lanes", Json::integer(kSensorLanes)},
            {"feature_lanes", Json::integer(kFeatureLanes)},
            {"statistics", Json::integer(kStatistics)},
            {"input_variant", Json::integer(variant_)},
            {"input_scale", Json::number(scale_)},
            {"score_pairs", Json::integer(Q_ * C_)},
            {"output_elements", Json::integer(Q_ * kFeatureLanes)}};
  }

  std::string application_scope() const override {
    return "upload audio [Q][3] and IMU [C][3] float32 from pinned host memory; "
           "audio and IMU projections; streaming attention; residual and "
           "statistics; download output [Q][4] and the 4 statistics";
  }
  std::string compute_scope() const override {
    return "audio and IMU projections, streaming attention, residual and "
           "statistics with the sensor inputs resident on the device; no "
           "host-device transfers";
  }

  void setup(cudaStream_t) override {
    LP_CUDA(cudaMallocHost(&h_audio_, audio_bytes()));
    LP_CUDA(cudaMallocHost(&h_imu_, imu_bytes()));
    LP_CUDA(cudaMallocHost(&h_output_, output_bytes()));
    LP_CUDA(cudaMallocHost(&h_statistics_, statistics_bytes()));
    for (long long t = 0; t < Q_; ++t)
      for (int lane = 0; lane < kSensorLanes; ++lane)
        h_audio_[t * kSensorLanes + lane] =
            sensor_sample((unsigned)(t * kSensorLanes + lane), 0u, scale_);
    for (long long t = 0; t < C_; ++t)
      for (int lane = 0; lane < kSensorLanes; ++lane)
        h_imu_[t * kSensorLanes + lane] =
            sensor_sample((unsigned)(t * kSensorLanes + lane), 1u, scale_);
    LP_CUDA(cudaMalloc(&d_audio_, audio_bytes()));
    LP_CUDA(cudaMalloc(&d_imu_, imu_bytes()));
    LP_CUDA(cudaMalloc(&d_query_, output_bytes()));
    LP_CUDA(cudaMalloc(&d_key_value_, (size_t)C_ * kFeatureLanes * sizeof(float)));
    LP_CUDA(cudaMalloc(&d_attention_, output_bytes()));
    LP_CUDA(cudaMalloc(&d_output_, output_bytes()));
    LP_CUDA(cudaMalloc(&d_statistics_, statistics_bytes()));
  }

  void enqueue_application(cudaStream_t stream) override {
    LP_CUDA(cudaMemcpyAsync(d_audio_, h_audio_, audio_bytes(),
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_imu_, h_imu_, imu_bytes(), cudaMemcpyHostToDevice,
                            stream));
    enqueue_compute(stream);
    LP_CUDA(cudaMemcpyAsync(h_output_, d_output_, output_bytes(),
                            cudaMemcpyDeviceToHost, stream));
    LP_CUDA(cudaMemcpyAsync(h_statistics_, d_statistics_, statistics_bytes(),
                            cudaMemcpyDeviceToHost, stream));
  }

  void enqueue_compute(cudaStream_t stream) override {
    const int q = (int)Q_, c = (int)C_;
    project_audio_kernel<<<(q + kProjectionThreads - 1) / kProjectionThreads,
                           kProjectionThreads, 0, stream>>>(d_audio_, q,
                                                            (float4 *)d_query_);
    LP_CUDA_LAUNCH();
    project_imu_kernel<<<(c + kProjectionThreads - 1) / kProjectionThreads,
                         kProjectionThreads, 0, stream>>>(
        d_imu_, c, (float4 *)d_key_value_);
    LP_CUDA_LAUNCH();
    attention_kernel<<<q, kAttentionThreads, 0, stream>>>(
        (const float4 *)d_query_, (const float4 *)d_key_value_, c,
        (float4 *)d_attention_);
    LP_CUDA_LAUNCH();
    residual_statistics_kernel<<<1, kStatisticsThreads, 0, stream>>>(
        d_query_, d_attention_, q * kFeatureLanes, d_output_, d_statistics_);
    LP_CUDA_LAUNCH();
  }

  // Every buffer the invocation writes, including the uploaded inputs and the
  // intermediates, is set to all-ones bytes (a NaN pattern); host outputs too.
  void poison_outputs(cudaStream_t stream) override {
    LP_CUDA(cudaMemsetAsync(d_audio_, 0xff, audio_bytes(), stream));
    LP_CUDA(cudaMemsetAsync(d_imu_, 0xff, imu_bytes(), stream));
    LP_CUDA(cudaMemsetAsync(d_query_, 0xff, output_bytes(), stream));
    LP_CUDA(cudaMemsetAsync(d_key_value_, 0xff,
                            (size_t)C_ * kFeatureLanes * sizeof(float), stream));
    LP_CUDA(cudaMemsetAsync(d_attention_, 0xff, output_bytes(), stream));
    LP_CUDA(cudaMemsetAsync(d_output_, 0xff, output_bytes(), stream));
    LP_CUDA(cudaMemsetAsync(d_statistics_, 0xff, statistics_bytes(), stream));
    std::memset(h_output_, 0xff, output_bytes());
    std::memset(h_statistics_, 0xff, statistics_bytes());
  }

  void compute_reference() override {
    attention_oracle::compute(Q_, C_, (double)scale_, reference_);
  }

  bool check_outputs() override {
    static const char *const kNames[kStatistics] = {
        "projection_energy", "attention_sum", "weighted_sum", "max_abs_attention"};
    bool pass = true;
    // Residual output, every element.
    long long failures = 0;
    double worst_ratio = 0.0;
    for (long long i = 0; i < Q_ * kFeatureLanes; ++i) {
      const double actual = h_output_[i], expected = reference_.output[i];
      const double ratio =
          std::fabs(actual - expected) / (1.0e-4 * (1.0 + std::fabs(expected)));
      if (!(ratio <= worst_ratio)) worst_ratio = ratio;
      if (!within_tolerance(actual, expected)) {
        if (failures < 5)
          std::fprintf(stderr,
                       "attention: mismatch output[%lld][%lld] (flat %lld): "
                       "expected %.9g, actual %.9g\n",
                       i / kFeatureLanes, i % kFeatureLanes, i, expected, actual);
        ++failures;
      }
    }
    if (failures) {
      std::fprintf(stderr, "attention: %lld of %lld output elements mismatch\n",
                   failures, Q_ * kFeatureLanes);
      pass = false;
    }
    std::fprintf(stderr, "attention: output worst |error|/tolerance %.3g\n",
                 worst_ratio);
    // Statistics and upstream's combined checksum (float, upstream's order).
    const float *s = h_statistics_;
    const float combined = s[2] + 3.0f * s[0] + 7.0f * s[1] + 11.0f * s[3];
    for (int i = 0; i <= kStatistics; ++i) {
      const double actual = i < kStatistics ? s[i] : combined;
      const double expected =
          i < kStatistics ? reference_.statistics[i] : reference_.combined;
      const char *label = i < kStatistics ? kNames[i] : "combined_checksum";
      const double ratio =
          std::fabs(actual - expected) / (1.0e-4 * (1.0 + std::fabs(expected)));
      const bool ok = within_tolerance(actual, expected);
      std::fprintf(stderr,
                   "attention: %s%s: expected %.9g, actual %.9g, "
                   "|error|/tolerance %.3g\n",
                   ok ? "" : "mismatch ", label, expected, actual, ratio);
      pass = pass && ok;
    }
    return pass;
  }

  uint64_t output_digest() const override {
    return lp::fnv1a64(h_statistics_, statistics_bytes(),
                       lp::fnv1a64(h_output_, output_bytes()));
  }

private:
  size_t audio_bytes() const { return (size_t)Q_ * kSensorLanes * sizeof(float); }
  size_t imu_bytes() const { return (size_t)C_ * kSensorLanes * sizeof(float); }
  size_t output_bytes() const { return (size_t)Q_ * kFeatureLanes * sizeof(float); }
  size_t statistics_bytes() const { return kStatistics * sizeof(float); }

  long long Q_, C_;
  int variant_;
  float scale_;
  float *h_audio_ = nullptr, *h_imu_ = nullptr, *h_output_ = nullptr,
        *h_statistics_ = nullptr;
  float *d_audio_ = nullptr, *d_imu_ = nullptr, *d_query_ = nullptr,
        *d_key_value_ = nullptr, *d_attention_ = nullptr, *d_output_ = nullptr,
        *d_statistics_ = nullptr;
  attention_oracle::Reference reference_;
};

std::unique_ptr<lp::Workload> make_workload(lp::Options &options) {
  if (options.preset != "large") return nullptr;
  const long long q = options.take_int("query_tokens", kLargeQueryTokens);
  const long long c = options.take_int("context_tokens", kLargeContextTokens);
  const long long variant = options.take_int("input_variant", 0);
  if (q < 1 || q > kMaxQueryTokens)
    lp::fail(lp::kExitUsage, "query_tokens must be in [1, %lld], got %lld",
             kMaxQueryTokens, q);
  if (c < 1 || c > kMaxContextTokens)
    lp::fail(lp::kExitUsage, "context_tokens must be in [1, %lld], got %lld",
             kMaxContextTokens, c);
  if (variant < 0 || variant > 2)
    lp::fail(lp::kExitUsage,
             "input_variant must be 0 (scale 1.0), 1 (0.5) or 2 (1.5), got %lld",
             variant);
  return std::make_unique<Attention>(q, c, (int)variant);
}

} // namespace

int main(int argc, char **argv) { return lp::run_main(argc, argv, make_workload); }

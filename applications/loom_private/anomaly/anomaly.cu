// loom-private/mlperf-tiny-anomaly-model: int8 inference of the MLPerf Tiny
// ad01 autoencoder (ten fully connected layers, 640 -> ... -> 640) over a
// large batch of samples cycled from the pinned 40-sample dataset.
//
// Setup (not timed): read and SHA-256-verify the model and dataset, parse and
// validate the model, upload weights and biases, and materialize the
// [samples][640] float32 input in pinned host memory (sample i is dataset
// sample i mod 40).
//
// Per invocation (application scope):
//   upload the float32 inputs [samples][640] from pinned host memory
//   inference_kernel: one block per sample; quantize the 640 inputs, then the
//     ten layers with int8 activations in shared memory (one warp per output
//     row, exact int32 dot products, double requantization)
//   download the int8 outputs [samples][640]
//
// Every sample is inferred on the GPU; nothing is deduplicated across the
// cycled samples. Integer dot products are exact, and the requantization uses
// explicitly rounded IEEE double operations, so the result is bitwise
// deterministic and bitwise equal to the CPU oracle.
#include "assets.h"
#include "lp_harness.cuh"
#include "model.h"
#include "oracle.h"

#include <cmath>

namespace {

using anomaly_model::kDatasetBytes;
using anomaly_model::kDatasetSamples;
using anomaly_model::kLayerCount;
using anomaly_model::kModelBytes;
using anomaly_model::kSampleElements;

constexpr long long kLargeSamples = 104858;
constexpr long long kMaxSamples = 1ll << 20;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

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

// Kernel-side layer description (passed by value as a kernel parameter).
struct LayerParams {
  int input_count, output_count;
  int weight_offset, bias_offset; // into the packed weight / bias arrays
  float input_scale, weight_scale, bias_scale, output_scale;
  int input_zero_point, output_zero_point;
  int relu;
};
struct ModelParams {
  float input_scale;
  int input_zero_point;
  LayerParams layers[kLayerCount];
};

// Upstream's saturating int8 conversion of a scaled value: |v| >= 1024
// saturates; otherwise add +-0.5 in double, truncate toward zero, add the
// zero point, and clip to [-128, 127].
__device__ __forceinline__ int saturate_int8(double value, int zero_point) {
  if (value <= -1024.0) return -128;
  if (value >= 1024.0) return 127;
  const double biased = value < 0.0 ? __dsub_rn(value, 0.5) : __dadd_rn(value, 0.5);
  const long long shifted = (long long)biased + zero_point; // cvt.rzi
  return shifted <= -128 ? -128 : shifted >= 127 ? 127 : (int)shifted;
}

// real = (double(acc) * in_scale) * w_scale + double(bias) * bias_scale,
// then real / out_scale; every step one IEEE-754 binary64 round-to-nearest.
__device__ __forceinline__ int requantize(int acc, int bias, const LayerParams &layer) {
  const double scaled_acc =
      __dmul_rn(__dmul_rn((double)acc, (double)layer.input_scale),
                (double)layer.weight_scale);
  const double scaled_bias = __dmul_rn((double)bias, (double)layer.bias_scale);
  const double real = __dadd_rn(scaled_acc, scaled_bias);
  int code = saturate_int8(__ddiv_rn(real, (double)layer.output_scale),
                           layer.output_zero_point);
  if (layer.relu && code < layer.output_zero_point) code = layer.output_zero_point;
  return code;
}

// One block per sample. Activations live in shared memory; warp w computes
// output rows w, w + 8, ...; lane l reads inputs l, l + 32, ... (coalesced
// weight row reads), and a shuffle tree adds the 32 partials. All of it is
// int32 integer arithmetic, exact for these extents: |in - zp| <= 255,
// |w| <= 128, at most 640 terms, so |acc| <= 20,889,600 < 2^31.
__global__ void __launch_bounds__(kThreads)
inference_kernel(const float *__restrict__ input, const int8_t *__restrict__ weights,
                 const int32_t *__restrict__ biases, const ModelParams params,
                 int8_t *__restrict__ output) {
  __shared__ int8_t activations[2][kSampleElements];
  const size_t sample = blockIdx.x;
#ifdef LP_ANOMALY_MUTANT_SKIP_LAST_SAMPLE
  // Test-only mutant: the last sample is never written, so its output stays
  // poisoned and the check must fail.
  if (blockIdx.x == gridDim.x - 1) return;
#endif
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const float *x = input + sample * kSampleElements;
  for (int i = threadIdx.x; i < (int)kSampleElements; i += kThreads) {
    const float ratio = __fdiv_rn(x[i], params.input_scale); // float32 division
    activations[0][i] = (int8_t)saturate_int8((double)ratio, params.input_zero_point);
  }
  __syncthreads();
  int current = 0;
#pragma unroll 1
  for (int l = 0; l < (int)kLayerCount; ++l) {
    const LayerParams &layer = params.layers[l];
    const int8_t *in = activations[current];
    int8_t *out = activations[current ^ 1];
    for (int o = warp; o < layer.output_count; o += kWarps) {
      const int8_t *row = weights + layer.weight_offset + (size_t)o * layer.input_count;
      int acc = 0;
      for (int i = lane; i < layer.input_count; i += 32)
        acc += ((int)in[i] - layer.input_zero_point) * (int)row[i];
      for (int offset = 16; offset > 0; offset /= 2)
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
      if (lane == 0)
        out[o] = (int8_t)requantize(acc, biases[layer.bias_offset + o], layer);
    }
    __syncthreads();
    current ^= 1;
  }
  int8_t *y = output + sample * kSampleElements;
  for (int i = threadIdx.x; i < (int)kSampleElements; i += kThreads)
    y[i] = activations[current][i];
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
    cudaFreeHost(h_input_);
    cudaFreeHost(h_output_);
    cudaFree(d_input_);
    cudaFree(d_output_);
    cudaFree(d_weights_);
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

    // Pack the layers for the kernel.
    std::vector<int8_t> weights;
    std::vector<int32_t> biases;
    params_.input_scale = model_.input.scale;
    params_.input_zero_point = (int)model_.input.zero_point;
    for (size_t l = 0; l < kLayerCount; ++l) {
      const anomaly_model::Layer &layer = model_.layers[l];
      LayerParams &p = params_.layers[l];
      p.input_count = (int)layer.input_count;
      p.output_count = (int)layer.output_count;
      p.weight_offset = (int)weights.size();
      p.bias_offset = (int)biases.size();
      p.input_scale = layer.input.scale;
      p.weight_scale = layer.weight.scale;
      p.bias_scale = layer.bias.scale;
      p.output_scale = layer.output.scale;
      p.input_zero_point = (int)layer.input.zero_point;
      p.output_zero_point = (int)layer.output.zero_point;
      p.relu = layer.relu ? 1 : 0;
      weights.insert(weights.end(), layer.weights.begin(), layer.weights.end());
      biases.insert(biases.end(), layer.biases.begin(), layer.biases.end());
    }
    LP_CUDA(cudaMalloc(&d_weights_, weights.size()));
    LP_CUDA(cudaMalloc(&d_biases_, biases.size() * sizeof(int32_t)));
    LP_CUDA(cudaMemcpyAsync(d_weights_, weights.data(), weights.size(),
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_biases_, biases.data(), biases.size() * sizeof(int32_t),
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
  }

  void enqueue_application(cudaStream_t stream) override {
    LP_CUDA(cudaMemcpyAsync(d_input_, h_input_, input_bytes(),
                            cudaMemcpyHostToDevice, stream));
    enqueue_compute(stream);
    LP_CUDA(cudaMemcpyAsync(h_output_, d_output_, output_bytes(),
                            cudaMemcpyDeviceToHost, stream));
  }

  void enqueue_compute(cudaStream_t stream) override {
    inference_kernel<<<(unsigned)samples_, kThreads, 0, stream>>>(
        d_input_, d_weights_, d_biases_, params_, d_output_);
    LP_CUDA_LAUNCH();
  }

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

  const lp::Action action_;
  const AssetPaths paths_;
  long long samples_;
  std::vector<uint8_t> model_bytes_, dataset_, smoke_, validation_;
  lp::JsonObject hashes_;
  anomaly_model::Model model_;
  ModelParams params_{};
  int poison_count_ = 0;
  float *h_input_ = nullptr, *d_input_ = nullptr;
  int8_t *h_output_ = nullptr, *d_output_ = nullptr;
  int8_t *d_weights_ = nullptr;
  int32_t *d_biases_ = nullptr;
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

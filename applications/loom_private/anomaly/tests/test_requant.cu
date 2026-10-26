// Exactness test of the optimized variant's binary32 fast paths (requant.cuh),
// run by `make test VARIANT=optimized`. Each GPU run takes the GPU lock in
// run_optimized_tests.sh.
//   test_requant <asset cache directory>
//
// It loads and verifies the pinned model, derives the layer parameters and
// fast-path constants with the requant.cuh helpers that anomaly_optimized.cu
// uses, and checks on the GPU:
//   1. input quantization: for every finite binary32 x for which
//      quantize_input_try_fast() accepts, its code equals quantize_input(x);
//   2. requantization: for every output of every layer and every accumulator
//      in [-20,889,600, 20,889,600] (the contract's accumulation bound, so
//      every value the kernel can produce) for which requantize_try_fast()
//      accepts, its code equals requantize(acc).
// Every accepted code is compared with a direct call of the baseline
// function. The test also requires that it visited exactly the expected
// number of arguments.
// Exit 0 when everything matches, 1 on any mismatch or coverage error, 3 for
// a missing or corrupt model.
#include "../assets.h"
#include "../model.h"
#include "../requant.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using anomaly_model::kLayerCount;
using anomaly_requant::FastLayer;
using anomaly_requant::LayerParams;

constexpr long long kAccBound = 20889600; // 255 * 128 * 640

#define CHECK_CUDA(call)                                                         \
  do {                                                                           \
    cudaError_t error_ = (call);                                                 \
    if (error_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "test_requant: CUDA error %s at %s:%d\n",            \
                   cudaGetErrorString(error_), __FILE__, __LINE__);              \
      std::exit(70);                                                             \
    }                                                                            \
  } while (0)

// Order-preserving map from 32-bit keys to binary32 values: key(x) < key(y)
// iff x < y (with -0 < +0).
__host__ __device__ unsigned key_of(float x) {
  unsigned bits;
  std::memcpy(&bits, &x, 4);
  return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}
__host__ __device__ float float_of(unsigned key) {
  const unsigned bits = key & 0x80000000u ? key & 0x7fffffffu : ~key;
  float x;
  std::memcpy(&x, &bits, 4);
  return x;
}

struct Counters {
  unsigned long long checked, accepted, mismatches;
  long long first_argument;
  int first_output, first_expected, first_actual;
};

__device__ void record_mismatch(Counters *counters, int output, long long argument,
                                int expected, int actual) {
  if (atomicAdd(&counters->mismatches, 1ull) == 0) {
    counters->first_output = output;
    counters->first_argument = argument;
    counters->first_expected = expected;
    counters->first_actual = actual;
  }
}

// --- Input quantization: every finite binary32 value -----------------------

struct InputParams {
  float scale, reciprocal;
  int zero_point;
};

__global__ void input_exhaustive(InputParams p, unsigned lo_key, unsigned long long count,
                                 Counters *counters) {
  unsigned long long checked = 0, accepted = 0;
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
       i < count; i += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned key = lo_key + (unsigned)i;
    const float x = float_of(key);
    int code;
    ++checked;
    if (!anomaly_requant::quantize_input_try_fast(x, p.reciprocal, p.zero_point, code))
      continue;
    ++accepted;
    const int expected = anomaly_requant::quantize_input(x, p.scale, p.zero_point);
    if (code != expected) record_mismatch(counters, -1, key, expected, code);
  }
  atomicAdd(&counters->checked, checked);
  atomicAdd(&counters->accepted, accepted);
}

// --- Requantization: every output and accumulator --------------------------

struct OutputParams {
  int layer, bias;
  float beta;
};
struct LayerConstants {
  LayerParams params;
  FastLayer fast;
};

// Block (o, part): output o, accumulators [-bound + part * span, ...).
__global__ void acc_exhaustive(const OutputParams *outputs, const LayerConstants *layers,
                               int parts, Counters *counters) {
  const int o = blockIdx.x / parts, part = blockIdx.x % parts;
  const OutputParams out = outputs[o];
  const LayerConstants layer = layers[out.layer];
  const float margin_base = anomaly_requant::fast_margin_base(out.beta);
  const long long total = 2 * kAccBound + 1, span = (total + parts - 1) / parts;
  const long long begin = -kAccBound + part * span;
  const long long end = min(begin + span, kAccBound + 1);
  unsigned long long checked = 0, accepted = 0;
  for (long long acc = begin + threadIdx.x; acc < end; acc += blockDim.x) {
    int code;
    ++checked;
    if (!anomaly_requant::requantize_try_fast((int)acc, out.beta, margin_base,
                                              layer.fast.kappa, layer.fast.lambda,
                                              layer.params.output_zero_point,
                                              layer.fast.code_min, code))
      continue;
    ++accepted;
    const int expected = anomaly_requant::requantize((int)acc, out.bias, layer.params);
    if (code != expected) record_mismatch(counters, o, acc, expected, code);
  }
  atomicAdd(&counters->checked, checked);
  atomicAdd(&counters->accepted, accepted);
}

Counters *new_counters() {
  Counters *counters;
  CHECK_CUDA(cudaMallocManaged(&counters, sizeof(Counters)));
  std::memset(counters, 0, sizeof(Counters));
  return counters;
}

bool report(const std::string &name, const Counters &c, unsigned long long expected,
            float ms) {
  const bool ok = c.mismatches == 0 && c.checked == expected && c.accepted > 0;
  std::printf("%s %s: %llu arguments (expected %llu), fast path accepted %llu (%.6f%%), "
              "%llu mismatches (%.0f ms)\n",
              ok ? "PASS" : "FAIL", name.c_str(), c.checked, expected, c.accepted,
              100.0 * (double)c.accepted / (double)c.checked, c.mismatches, ms);
  if (c.mismatches)
    std::printf("    first mismatch: output %d, argument %lld: fast %d, baseline %d\n",
                c.first_output, c.first_argument, c.first_actual, c.first_expected);
  return ok;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: test_requant <asset cache directory>\n");
    return 2;
  }
  const std::string name = "mlperf-tiny-ad01-int8-model";
  std::vector<uint8_t> bytes;
  std::string error;
  if (!anomaly_assets::load_verified(
          name, std::string(argv[1]) + "/" + name, anomaly_model::kModelBytes,
          "87cf24194ef93d1d9b11a591d805526b98008e351655d29883c825c9c106ba24", bytes, error)) {
    std::fprintf(stderr, "test_requant: %s\n", error.c_str());
    return 3;
  }
  const anomaly_model::Model model = anomaly_model::parse(bytes);

  const InputParams input{model.input.scale,
                          anomaly_requant::fast_input_reciprocal(model.input.scale),
                          (int)model.input.zero_point};
  std::vector<LayerConstants> layers(kLayerCount);
  std::vector<OutputParams> outputs;
  for (size_t l = 0; l < kLayerCount; ++l) {
    const anomaly_model::Layer &layer = model.layers[l];
    std::vector<float> betas(layer.output_count);
    layers[l].params = anomaly_requant::make_layer_params(layer, 0, 0);
    layers[l].fast =
        anomaly_requant::make_fast_layer(layers[l].params, layer.biases.data(), betas.data());
    if (layers[l].fast.kappa == 0.0f) {
      std::fprintf(stderr, "test_requant: layer %zu has no fast path\n", l + 1);
      return 1;
    }
    for (uint32_t o = 0; o < layer.output_count; ++o)
      outputs.push_back({(int)l, layer.biases[o], betas[o]});
  }
  if (input.reciprocal == 0.0f) {
    std::fprintf(stderr, "test_requant: the input quantization has no fast path\n");
    return 1;
  }

  bool ok = true;
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  float ms = 0;

  // 1. Input quantization over every finite binary32 value: keys from
  // -FLT_MAX to FLT_MAX (2^32 - 2^24 values; the NaN and infinity patterns
  // are outside).
  {
    const unsigned lo_key = key_of(-3.40282347e38f), hi_key = key_of(3.40282347e38f);
    const unsigned long long count = (unsigned long long)hi_key - lo_key + 1;
    Counters *counters = new_counters();
    CHECK_CUDA(cudaEventRecord(start));
    input_exhaustive<<<4096, 256>>>(input, lo_key, count, counters);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ok = report("input quantization, every finite binary32 value", *counters,
                (1ull << 32) - (1ull << 24), ms) &&
         ok;
    CHECK_CUDA(cudaFree(counters));
  }

  // 2. Requantization of every output over the whole accumulator bound.
  {
    const int count = (int)outputs.size();
    OutputParams *d_outputs;
    LayerConstants *d_layers;
    Counters *counters = new_counters();
    CHECK_CUDA(cudaMalloc(&d_outputs, outputs.size() * sizeof(OutputParams)));
    CHECK_CUDA(cudaMalloc(&d_layers, layers.size() * sizeof(LayerConstants)));
    CHECK_CUDA(cudaMemcpy(d_outputs, outputs.data(), outputs.size() * sizeof(OutputParams),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_layers, layers.data(), layers.size() * sizeof(LayerConstants),
                          cudaMemcpyHostToDevice));
    constexpr int kParts = 64;
    CHECK_CUDA(cudaEventRecord(start));
    acc_exhaustive<<<count * kParts, 256>>>(d_outputs, d_layers, kParts, counters);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    // 1,672 outputs for the pinned model, each over 2 * 20,889,600 + 1 values.
    if (count != 1672) {
      std::fprintf(stderr, "test_requant: expected 1672 layer outputs, found %d\n", count);
      ok = false;
    }
    ok = report("requantization, " + std::to_string(count) +
                    " outputs x every accumulator in [-20889600, 20889600]",
                *counters, 1672ull * (2 * kAccBound + 1), ms) &&
         ok;
    CHECK_CUDA(cudaFree(d_outputs));
    CHECK_CUDA(cudaFree(d_layers));
    CHECK_CUDA(cudaFree(counters));
  }
  return ok ? 0 : 1;
}

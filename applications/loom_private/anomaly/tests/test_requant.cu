// Exactness test of the optimized variant's binary32 fast paths (requant.cuh),
// run by `make test VARIANT=optimized`. Each GPU run takes the GPU lock in
// run_optimized_tests.sh.
//   test_requant <asset cache directory>
//
// It loads and verifies the pinned model, derives the layer parameters and
// fast-path constants as anomaly_optimized.cu does, and checks on the GPU:
//   1. input quantization: for every finite binary32 x for which
//      quantize_input_try_fast() accepts, its code equals quantize_input(x);
//   2. requantization: for every output of every layer and every accumulator
//      in [-20,889,600, 20,889,600] (the contract's accumulation bound, so
//      every value the kernel can produce) for which requantize_try_fast()
//      accepts, its code equals requantize(acc).
// The expected codes come from the baseline functions through their
// thresholds. Both are monotone non-decreasing in their argument (every step
// is a correctly rounded operation with a positive scale, a truncation, or a
// clip), so the code at an argument is -128 plus the number of thresholds at
// or below it. The thresholds (the least argument with code >= c, for
// c = -127..127) are found by bisection with the baseline function itself,
// and the tables are cross-checked against direct baseline evaluations at
// strided arguments and around every threshold.
// Exit 0 when everything matches, 1 on any mismatch, 3 for a missing or
// corrupt model.
#include "../assets.h"
#include "../model.h"
#include "../requant.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using anomaly_requant::LayerParams;
using anomaly_model::kLayerCount;

constexpr int kCodes = 255;              // thresholds for codes -127..127
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

// Order-preserving map between binary32 values and 32-bit keys: key(x) <
// key(y) iff x < y (with -0 < +0, which quantize identically).
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

// The code -128 + #{thresholds <= value}: an upper-bound search in a sorted
// table of kCodes entries.
template <typename T>
__device__ int code_from_table(const T *table, T value) {
  int lo = 0, hi = kCodes;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (table[mid] <= value) lo = mid + 1;
    else hi = mid;
  }
  return lo - 128;
}

struct Counters {
  unsigned long long checked, accepted, mismatches, table_checked, table_mismatches;
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

// --- Input quantization ----------------------------------------------------

struct InputParams {
  float scale, reciprocal;
  int zero_point;
};

// Thresholds: the least finite key with quantize_input >= c.
__global__ void input_thresholds(InputParams p, unsigned lo_key, unsigned hi_key,
                                 unsigned *table) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x - 127;
  if (c > 127) return;
  unsigned long long lo = lo_key, hi = (unsigned long long)hi_key + 1;
  while (lo < hi) {
    const unsigned long long mid = (lo + hi) / 2;
    if (anomaly_requant::quantize_input(float_of((unsigned)mid), p.scale, p.zero_point) >= c)
      hi = mid;
    else
      lo = mid + 1;
  }
  table[c + 127] = (unsigned)lo; // hi_key + 1 when no finite x reaches c
}

__global__ void input_exhaustive(InputParams p, unsigned lo_key, unsigned long long count,
                                 const unsigned *table_in, Counters *counters) {
  __shared__ unsigned table[kCodes];
  for (int i = threadIdx.x; i < kCodes; i += blockDim.x) table[i] = table_in[i];
  __syncthreads();
  unsigned long long checked = 0, accepted = 0;
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
       i < count; i += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned key = lo_key + (unsigned)i;
    int code;
    ++checked;
    if (!anomaly_requant::quantize_input_try_fast(float_of(key), p.reciprocal, p.zero_point,
                                                  code))
      continue;
    ++accepted;
    const int expected = code_from_table(table, key);
    if (code != expected) record_mismatch(counters, -1, key, expected, code);
  }
  atomicAdd(&counters->checked, checked);
  atomicAdd(&counters->accepted, accepted);
}

// Direct baseline evaluations against the table: strided keys, and +-256
// keys around every threshold.
__global__ void input_table_check(InputParams p, unsigned lo_key, unsigned hi_key,
                                  const unsigned *table, Counters *counters) {
  const unsigned long long span = (unsigned long long)hi_key - lo_key + 1;
  const unsigned long long strided = span / 65521 + 1, windows = kCodes * 513ull;
  for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
       i < strided + windows; i += (unsigned long long)gridDim.x * blockDim.x) {
    long long key;
    if (i < strided) {
      key = (long long)lo_key + (long long)(i * 65521);
    } else {
      const unsigned long long w = i - strided;
      key = (long long)table[w / 513] + (long long)(w % 513) - 256;
    }
    if (key < lo_key || key > hi_key) continue;
    const int direct =
        anomaly_requant::quantize_input(float_of((unsigned)key), p.scale, p.zero_point);
    atomicAdd(&counters->table_checked, 1ull);
    if (direct != code_from_table(table, (unsigned)key))
      atomicAdd(&counters->table_mismatches, 1ull);
  }
}

// --- Requantization --------------------------------------------------------

struct OutputParams {
  int layer, bias;
  float beta;
};
struct LayerConstants {
  LayerParams params;
  float kappa, lambda;
  int code_min;
};

// Thresholds: per output, the least accumulator in [-bound, bound + 1] with
// requantize >= c.
__global__ void acc_thresholds(const OutputParams *outputs, const LayerConstants *layers,
                               int output_count, int *tables) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= (long long)output_count * kCodes) return;
  const int o = (int)(index / kCodes), c = (int)(index % kCodes) - 127;
  const OutputParams out = outputs[o];
  const LayerParams &layer = layers[out.layer].params;
  long long lo = -kAccBound, hi = kAccBound + 1;
  while (lo < hi) {
    const long long mid = lo + (hi - lo) / 2;
    if (anomaly_requant::requantize((int)mid, out.bias, layer) >= c) hi = mid;
    else lo = mid + 1;
  }
  tables[index] = (int)lo;
}

// Block (o, part): output o, accumulators [-bound + part * span, ...).
__global__ void acc_exhaustive(const OutputParams *outputs, const LayerConstants *layers,
                               const int *tables, int parts, Counters *counters) {
  __shared__ int table[kCodes];
  const int o = blockIdx.x / parts, part = blockIdx.x % parts;
  for (int i = threadIdx.x; i < kCodes; i += blockDim.x) table[i] = tables[o * kCodes + i];
  __syncthreads();
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
    if (!anomaly_requant::requantize_try_fast((int)acc, out.beta, margin_base, layer.kappa,
                                              layer.lambda, layer.params.output_zero_point,
                                              layer.code_min, code))
      continue;
    ++accepted;
    const int expected = code_from_table(table, (int)acc);
    if (code != expected) record_mismatch(counters, o, acc, expected, code);
  }
  atomicAdd(&counters->checked, checked);
  atomicAdd(&counters->accepted, accepted);
}

// Direct baseline evaluations against the tables: every 9973rd accumulator,
// and +-64 around every threshold, of every output.
__global__ void acc_table_check(const OutputParams *outputs, const LayerConstants *layers,
                                const int *tables, int output_count, Counters *counters) {
  constexpr long long kStride = 9973, kWindow = 129;
  const long long strided = (2 * kAccBound) / kStride + 1;
  const long long per_output = strided + kCodes * kWindow;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
       i < output_count * per_output; i += (long long)gridDim.x * blockDim.x) {
    const int o = (int)(i / per_output);
    const long long j = i % per_output;
    const long long acc = j < strided ? -kAccBound + j * kStride
                                      : (long long)tables[o * kCodes + (j - strided) / kWindow] +
                                            (j - strided) % kWindow - kWindow / 2;
    if (acc < -kAccBound || acc > kAccBound) continue;
    const OutputParams out = outputs[o];
    const int direct =
        anomaly_requant::requantize((int)acc, out.bias, layers[out.layer].params);
    atomicAdd(&counters->table_checked, 1ull);
    if (direct != code_from_table(tables + o * kCodes, (int)acc))
      atomicAdd(&counters->table_mismatches, 1ull);
  }
}

Counters *new_counters() {
  Counters *counters;
  CHECK_CUDA(cudaMallocManaged(&counters, sizeof(Counters)));
  std::memset(counters, 0, sizeof(Counters));
  return counters;
}

bool report(const char *name, const Counters &c) {
  const bool ok = c.mismatches == 0 && c.table_mismatches == 0 && c.table_checked > 0 &&
                  c.accepted > 0;
  std::printf("%s %s: %llu arguments, fast path accepted %llu (%.6f%%), %llu mismatches; "
              "threshold table cross-checked at %llu arguments, %llu mismatches\n",
              ok ? "PASS" : "FAIL", name, c.checked, c.accepted,
              100.0 * (double)c.accepted / (double)c.checked, c.mismatches,
              c.table_checked, c.table_mismatches);
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

  // The constants as anomaly_optimized.cu derives them.
  const InputParams input{model.input.scale,
                          anomaly_requant::fast_input_reciprocal(model.input.scale),
                          (int)model.input.zero_point};
  std::vector<LayerConstants> layers(kLayerCount);
  std::vector<OutputParams> outputs;
  for (size_t l = 0; l < kLayerCount; ++l) {
    const anomaly_model::Layer &layer = model.layers[l];
    LayerParams &p = layers[l].params;
    p = {};
    p.input_count = (int)layer.input_count;
    p.output_count = (int)layer.output_count;
    p.input_scale = layer.input.scale;
    p.weight_scale = layer.weight.scale;
    p.bias_scale = layer.bias.scale;
    p.output_scale = layer.output.scale;
    p.input_zero_point = (int)layer.input.zero_point;
    p.output_zero_point = (int)layer.output.zero_point;
    p.relu = layer.relu ? 1 : 0;
    std::vector<float> betas(layer.output_count);
    layers[l].kappa = anomaly_requant::fast_constants(p, layer.biases.data(), betas.data());
    layers[l].lambda = layers[l].kappa * 0x1p-20f;
    layers[l].code_min = p.relu ? p.output_zero_point : -128;
    if (layers[l].kappa == 0.0f) {
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

  // 1. Input quantization over every finite binary32 value.
  {
    const unsigned lo_key = key_of(-3.40282347e38f), hi_key = key_of(3.40282347e38f);
    unsigned *table;
    Counters *counters = new_counters();
    CHECK_CUDA(cudaMalloc(&table, kCodes * sizeof(unsigned)));
    CHECK_CUDA(cudaEventRecord(start));
    input_thresholds<<<1, 256>>>(input, lo_key, hi_key, table);
    input_exhaustive<<<4096, 256>>>(input, lo_key, (unsigned long long)hi_key - lo_key + 1,
                                    table, counters);
    input_table_check<<<1024, 256>>>(input, lo_key, hi_key, table, counters);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ok = report("input quantization, every finite binary32 value", *counters) && ok;
    std::printf("    (%.0f ms)\n", ms);
    CHECK_CUDA(cudaFree(table));
    CHECK_CUDA(cudaFree(counters));
  }

  // 2. Requantization of every output over the whole accumulator bound.
  {
    const int count = (int)outputs.size();
    OutputParams *d_outputs;
    LayerConstants *d_layers;
    int *tables;
    Counters *counters = new_counters();
    CHECK_CUDA(cudaMalloc(&d_outputs, outputs.size() * sizeof(OutputParams)));
    CHECK_CUDA(cudaMalloc(&d_layers, layers.size() * sizeof(LayerConstants)));
    CHECK_CUDA(cudaMalloc(&tables, (size_t)count * kCodes * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_outputs, outputs.data(), outputs.size() * sizeof(OutputParams),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_layers, layers.data(), layers.size() * sizeof(LayerConstants),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(start));
    acc_thresholds<<<(count * kCodes + 255) / 256, 256>>>(d_outputs, d_layers, count, tables);
    constexpr int kParts = 64;
    acc_exhaustive<<<count * kParts, 256>>>(d_outputs, d_layers, tables, kParts, counters);
    acc_table_check<<<4096, 256>>>(d_outputs, d_layers, tables, count, counters);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    const std::string what = "requantization, " + std::to_string(count) +
                             " outputs x every accumulator in [-20889600, 20889600]";
    ok = report(what.c_str(), *counters) && ok;
    std::printf("    (%.0f ms)\n", ms);
    CHECK_CUDA(cudaFree(d_outputs));
    CHECK_CUDA(cudaFree(d_layers));
    CHECK_CUDA(cudaFree(tables));
    CHECK_CUDA(cudaFree(counters));
  }
  return ok ? 0 : 1;
}

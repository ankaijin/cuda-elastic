// CUDA port of mlperf_tiny_anomaly_mse_kernel, the kernel Loom compiles for
// mlperf-tiny-anomaly-detection (loom/mlperf-tiny-anomaly-smoke.h):
//
//   for i in [0, count):
//     difference = DequantizeInt8ToFloat(output[i], scale, zeroPoint) - input[i]
//     differenceSum += difference * difference
//   return differenceSum / count
//
// DequantizeInt8ToFloat is upstream mlperf-tiny's
// static_cast<float>(value - zero_point) * scale (util/quantization_helpers.h).
//
// The default run (--count=16 --inputs=smoke) is Loom's smoke workload and
// oracle: |mse - 0.015625| <= 1e-6. Larger --count values extend the smoke
// input pattern; 640 is upstream's real kInputSize (5 slices x 128 features).
//
// The GPU sums in a fixed tree order instead of sequentially, so the result
// is checked against a double-precision reference with a relative tolerance.
// The sequential float result of the Loom loop is reported next to it.
// Built with --fmad=false to match Loom's -ffp-contract=off.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t error__ = (call);                                      \
    if (error__ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error: %s at %s:%d\n",                      \
                   cudaGetErrorString(error__), __FILE__, __LINE__);         \
      return EXIT_FAILURE;                                                   \
    }                                                                        \
  } while (0)

namespace {

// Smoke workload constants from loom/mlperf-tiny-anomaly-smoke.h.
constexpr size_t kSmokeCount = 16;  // kInputSize
constexpr float kSmokeScale = 0.25f;
constexpr int kSmokeZeroPoint = 0;
constexpr float kSmokeExpected = 0.015625f;
constexpr float kSmokeTolerance = 1.0e-6f;

// Random inputs use a nonzero zero point so it is exercised.
constexpr float kRandomScale = 0.0625f;
constexpr int kRandomZeroPoint = -3;

// Tree-order float sum against the double reference.
constexpr double kRelativeTolerance = 1.0e-5;
constexpr double kAbsoluteTolerance = 1.0e-7;

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kBlocksPerSm = 8;

// ---------------------------------------------------------------------------
// Host ports of the Loom code, used as oracles.

float DequantizeInt8ToFloat(int8_t value, float scale, int zeroPoint) {
  return static_cast<float>(value - zeroPoint) * scale;
}

// Verbatim loop of mlperf_tiny_anomaly_mse_kernel.
float loom_mse(const int8_t* output, const float* input, size_t count,
               float scale, int zeroPoint) {
  float differenceSum = 0.0f;
  for (size_t index = 0; index < count; ++index) {
    const float converted =
        DequantizeInt8ToFloat(output[index], scale, zeroPoint);
    const float difference = converted - input[index];
    differenceSum += difference * difference;
  }
  return differenceSum / static_cast<float>(count);
}

double reference_mse(const int8_t* output, const float* input, size_t count,
                     float scale, int zeroPoint) {
  double differenceSum = 0.0;
  for (size_t index = 0; index < count; ++index) {
    const double difference =
        static_cast<double>(DequantizeInt8ToFloat(output[index], scale,
                                                  zeroPoint)) -
        static_cast<double>(input[index]);
    differenceSum += difference * difference;
  }
  return differenceSum / static_cast<double>(count);
}

// Generalizes the smoke main()'s input loop to any count.
void fill_smoke(int8_t* output, float* input, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    const int value = static_cast<int>(index % 17) - 8;
    input[index] = static_cast<float>(value) * 0.25f;
    output[index] = static_cast<int8_t>(value + ((index & 3U) == 0));
  }
}

void fill_random(int8_t* output, float* input, size_t count,
                 unsigned seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> quantized(-128, 127);
  std::uniform_real_distribution<float> noise(-4.0f, 4.0f);
  for (size_t index = 0; index < count; ++index) {
    output[index] = static_cast<int8_t>(quantized(generator));
    input[index] =
        DequantizeInt8ToFloat(output[index], kRandomScale, kRandomZeroPoint) +
        noise(generator) * kRandomScale;
  }
}

// ---------------------------------------------------------------------------
// Device kernels.

__device__ __forceinline__ void accumulate(float& sum, int8_t output,
                                           float input, float scale,
                                           int zeroPoint) {
  const float difference =
      static_cast<float>(output - zeroPoint) * scale - input;
  sum += difference * difference;
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

// Block-wide sum; the result is valid in thread 0. Requires kThreads threads.
__device__ __forceinline__ float block_sum(float value) {
  __shared__ float warp_partials[kWarps];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0)
    warp_partials[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < kWarps ? warp_partials[lane] : 0.0f;
    value = warp_sum(value);
  }
  return value;
}

// Each block reduces a grid-stride slice of the squared differences to one
// partial sum. The body loads 4 elements per thread (char4 + float4); the
// remaining count % 4 elements are picked up by the first threads of the grid.
__global__ void __launch_bounds__(kThreads)
    mse_partial_kernel(const int8_t* __restrict__ output,
                       const float* __restrict__ input, size_t count,
                       float scale, int zeroPoint,
                       float* __restrict__ partials) {
  const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t vectorCount = count / 4;
  const char4* output4 = reinterpret_cast<const char4*>(output);
  const float4* input4 = reinterpret_cast<const float4*>(input);

  float sum = 0.0f;
  for (size_t index = thread; index < vectorCount; index += stride) {
    const char4 o = output4[index];
    const float4 x = input4[index];
    accumulate(sum, o.x, x.x, scale, zeroPoint);
    accumulate(sum, o.y, x.y, scale, zeroPoint);
    accumulate(sum, o.z, x.z, scale, zeroPoint);
    accumulate(sum, o.w, x.w, scale, zeroPoint);
  }
  for (size_t index = vectorCount * 4 + thread; index < count;
       index += stride)
    accumulate(sum, output[index], input[index], scale, zeroPoint);

  sum = block_sum(sum);
  if (threadIdx.x == 0)
    partials[blockIdx.x] = sum;
}

__global__ void __launch_bounds__(kThreads)
    mse_finalize_kernel(const float* __restrict__ partials, int partialCount,
                        size_t count, float* __restrict__ result) {
  float sum = 0.0f;
  for (int index = threadIdx.x; index < partialCount; index += blockDim.x)
    sum += partials[index];
  sum = block_sum(sum);
  if (threadIdx.x == 0)
    *result = sum / static_cast<float>(count);
}

// ---------------------------------------------------------------------------
// Host harness.

struct Options {
  size_t count = kSmokeCount;
  std::string inputs = "smoke";
  int replays = 10000;
  unsigned seed = 1;
};

bool parse_options(int argc, char** argv, Options& options) {
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto value = [&](const char* prefix) -> const char* {
      const size_t length = std::strlen(prefix);
      return argument.compare(0, length, prefix) == 0
                 ? argument.c_str() + length
                 : nullptr;
    };
    if (const char* v = value("--count="))
      options.count = std::strtoull(v, nullptr, 10);
    else if (const char* v = value("--inputs="))
      options.inputs = v;
    else if (const char* v = value("--replays="))
      options.replays = std::atoi(v);
    else if (const char* v = value("--seed="))
      options.seed = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
    else {
      std::fprintf(stderr,
                   "usage: %s [--count=N] [--inputs=smoke|random] "
                   "[--replays=R] [--seed=S]\n",
                   argv[0]);
      return false;
    }
  }
  if (options.count == 0 || options.replays <= 0 ||
      (options.inputs != "smoke" && options.inputs != "random")) {
    std::fprintf(stderr, "invalid options\n");
    return false;
  }
  return true;
}

// Average microseconds per replay of a captured graph, after warm-up.
int time_graph(cudaGraphExec_t graph, cudaStream_t stream, int replays,
               double& microseconds) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int replay = 0; replay < std::min(replays, 100); ++replay)
    CUDA_CHECK(cudaGraphLaunch(graph, stream));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int replay = 0; replay < replays; ++replay)
    CUDA_CHECK(cudaGraphLaunch(graph, stream));
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  microseconds = 1000.0 * milliseconds / replays;
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, options))
    return EXIT_FAILURE;
  const size_t count = options.count;
  const bool smoke = options.inputs == "smoke";
  const float scale = smoke ? kSmokeScale : kRandomScale;
  const int zeroPoint = smoke ? kSmokeZeroPoint : kRandomZeroPoint;

  int device = 0, smCount = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(
      cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
  const size_t work = std::max<size_t>(count / 4, 1);
  const int blocks = static_cast<int>(std::min<size_t>(
      (work + kThreads - 1) / kThreads,
      static_cast<size_t>(smCount) * kBlocksPerSm));

  int8_t* hostOutput = nullptr;
  float* hostInput = nullptr;
  float* hostResult = nullptr;
  CUDA_CHECK(cudaMallocHost(&hostOutput, count * sizeof(int8_t)));
  CUDA_CHECK(cudaMallocHost(&hostInput, count * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&hostResult, sizeof(float)));
  if (smoke)
    fill_smoke(hostOutput, hostInput, count);
  else
    fill_random(hostOutput, hostInput, count, options.seed);

  int8_t* deviceOutput = nullptr;
  float* deviceInput = nullptr;
  float* devicePartials = nullptr;
  float* deviceResult = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceOutput, count * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&deviceInput, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&devicePartials, blocks * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceResult, sizeof(float)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Graph 1: kernels only, inputs already resident.
  // Graph 2: end to end, H2D of both inputs + kernels + D2H of the result.
  cudaGraph_t graphs[2];
  cudaGraphExec_t execs[2];
  for (int endToEnd = 0; endToEnd < 2; ++endToEnd) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    if (endToEnd) {
      CUDA_CHECK(cudaMemcpyAsync(deviceOutput, hostOutput,
                                 count * sizeof(int8_t),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(deviceInput, hostInput, count * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
    }
    mse_partial_kernel<<<blocks, kThreads, 0, stream>>>(
        deviceOutput, deviceInput, count, scale, zeroPoint, devicePartials);
    mse_finalize_kernel<<<1, kThreads, 0, stream>>>(devicePartials, blocks,
                                                    count, deviceResult);
    if (endToEnd)
      CUDA_CHECK(cudaMemcpyAsync(hostResult, deviceResult, sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamEndCapture(stream, &graphs[endToEnd]));
    CUDA_CHECK(cudaGraphInstantiate(&execs[endToEnd], graphs[endToEnd], 0));
  }

  // Correctness: one end-to-end replay from a poisoned result.
  *hostResult = NAN;
  CUDA_CHECK(cudaGraphLaunch(execs[1], stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());
  const float gpu = *hostResult;
  const float loom = loom_mse(hostOutput, hostInput, count, scale, zeroPoint);
  const double reference =
      reference_mse(hostOutput, hostInput, count, scale, zeroPoint);
  const double error = std::fabs(static_cast<double>(gpu) - reference);
  const bool withinTolerance =
      error <= kRelativeTolerance * std::fabs(reference) + kAbsoluteTolerance;
  const bool smokeShape = smoke && count == kSmokeCount;
  const bool smokeOracle =
      !smokeShape || std::fabs(gpu - kSmokeExpected) <= kSmokeTolerance;

  double kernelMicroseconds = 0.0, endToEndMicroseconds = 0.0;
  if (time_graph(execs[0], stream, options.replays, kernelMicroseconds) ||
      time_graph(execs[1], stream, options.replays, endToEndMicroseconds))
    return EXIT_FAILURE;
  const double bytes = static_cast<double>(count) * (sizeof(int8_t) +
                                                     sizeof(float));

  std::printf("mlperf-anomaly mse: count=%zu inputs=%s scale=%g zero_point=%d "
              "blocks=%d replays=%d\n",
              count, options.inputs.c_str(), scale, zeroPoint, blocks,
              options.replays);
  std::printf("  gpu          %.9g\n", gpu);
  std::printf("  loom loop    %.9g (host, sequential float)%s\n", loom,
              gpu == loom ? ", bitwise equal to gpu" : "");
  std::printf("  reference    %.9g (host, double)\n", reference);
  if (smokeShape)
    std::printf("  smoke oracle %s (|mse - %g| <= %g)\n",
                smokeOracle ? "pass" : "FAIL", kSmokeExpected,
                kSmokeTolerance);
  std::printf("  check        %s (abs err %.3e vs double)\n",
              withinTolerance ? "pass" : "FAIL", error);
  std::printf("  kernels      %.3f us/iter (graph replay)\n",
              kernelMicroseconds);
  std::printf("  end-to-end   %.3f us/iter (H2D + kernels + D2H, graph "
              "replay)\n",
              endToEndMicroseconds);
  std::printf("  bandwidth    %.1f GB/s (kernels, %zu B/element)\n",
              bytes / (kernelMicroseconds * 1.0e3),
              sizeof(int8_t) + sizeof(float));
  std::printf("  graph total  %.3f s (replays x (kernels + end-to-end), "
              "excluding warm-up)\n",
              options.replays * (kernelMicroseconds + endToEndMicroseconds) /
                  1.0e6);

  for (int index = 0; index < 2; ++index) {
    CUDA_CHECK(cudaGraphExecDestroy(execs[index]));
    CUDA_CHECK(cudaGraphDestroy(graphs[index]));
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(deviceOutput));
  CUDA_CHECK(cudaFree(deviceInput));
  CUDA_CHECK(cudaFree(devicePartials));
  CUDA_CHECK(cudaFree(deviceResult));
  CUDA_CHECK(cudaFreeHost(hostOutput));
  CUDA_CHECK(cudaFreeHost(hostInput));
  CUDA_CHECK(cudaFreeHost(hostResult));
  return withinTolerance && smokeOracle ? EXIT_SUCCESS : EXIT_FAILURE;
}

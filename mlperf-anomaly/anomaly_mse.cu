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
//
// --launch=graph (the default) captures the kernels-only and end-to-end work
// into CUDA graphs and replays them; --launch=direct enqueues the same stream
// operations directly on every iteration. Each timed iteration is an NVTX
// range (anomaly.kernels, anomaly.end_to_end). --dump=<file> writes the count
// and the float32 result of the correctness pass (the format anomaly_mse_sota
// --dump uses for its set 0), for compare_sota.sh.
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

#if defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define ANOMALY_HAS_NVTX 1
#  endif
#endif
#ifndef ANOMALY_HAS_NVTX
#  define ANOMALY_HAS_NVTX 0
#endif

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

// Isolated end-to-end iterations whose median is the single-iteration time.
constexpr int kSingleIterations = 20;

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
  std::string launch = "graph";  // graph | direct
  int replays = 10000;
  unsigned seed = 1;
  std::string dump;
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
    else if (const char* v = value("--launch="))
      options.launch = v;
    else if (const char* v = value("--replays="))
      options.replays = std::atoi(v);
    else if (const char* v = value("--seed="))
      options.seed = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
    else if (const char* v = value("--dump="))
      options.dump = v;
    else {
      std::fprintf(stderr,
                   "usage: %s [--count=N] [--inputs=smoke|random] "
                   "[--launch=graph|direct] [--replays=R] [--seed=S] "
                   "[--dump=FILE]\n",
                   argv[0]);
      return false;
    }
  }
  if (options.count == 0 || options.replays <= 0 ||
      (options.inputs != "smoke" && options.inputs != "random") ||
      (options.launch != "graph" && options.launch != "direct")) {
    std::fprintf(stderr, "invalid options\n");
    return false;
  }
  return true;
}

void nvtx_push(const char* label) {
#if ANOMALY_HAS_NVTX
  nvtxRangePushA(label);
#else
  (void)label;
#endif
}

void nvtx_pop() {
#if ANOMALY_HAS_NVTX
  nvtxRangePop();
#endif
}

// Captures the stream work of enqueue() (an int() callable returning
// EXIT_SUCCESS) into an executable graph.
template <typename Enqueue>
int capture_graph(Enqueue enqueue, cudaStream_t stream,
                  cudaGraphExec_t* exec) {
  cudaGraph_t graph;
  CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  if (enqueue() != EXIT_SUCCESS)
    return EXIT_FAILURE;
  CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(exec, graph, 0));
  CUDA_CHECK(cudaGraphDestroy(graph));
  return EXIT_SUCCESS;
}

// Average microseconds per iteration, after warm-up. launch() enqueues one
// iteration on stream (a graph replay or the direct stream operations) and
// returns EXIT_SUCCESS; each timed iteration is an NVTX range named range.
template <typename Launch>
int time_launches(Launch launch, cudaStream_t stream, int replays,
                  const char* range, double& microseconds) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int replay = 0; replay < std::min(replays, 100); ++replay)
    if (launch() != EXIT_SUCCESS)
      return EXIT_FAILURE;
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int replay = 0; replay < replays; ++replay) {
    nvtx_push(range);
    const int status = launch();
    nvtx_pop();
    if (status != EXIT_SUCCESS)
      return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  microseconds = 1000.0 * milliseconds / replays;
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return EXIT_SUCCESS;
}

// Median microseconds of one iteration launched by itself: the stream is
// drained before each of kSingleIterations iterations, so the iteration's
// launch cost is exposed instead of overlapping the previous iteration's work
// as in time_launches' back-to-back loop. Each is an NVTX range
// named range.
template <typename Launch>
int time_single(Launch launch, cudaStream_t stream, const char* range,
                double& microseconds) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<double> samples;
  for (int iteration = 0; iteration < kSingleIterations; ++iteration) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtx_push(range);
    CUDA_CHECK(cudaEventRecord(start, stream));
    const int status = launch();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    nvtx_pop();
    if (status != EXIT_SUCCESS)
      return EXIT_FAILURE;
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    samples.push_back(1000.0 * milliseconds);
  }
  std::sort(samples.begin(), samples.end());
  microseconds = (samples[(kSingleIterations - 1) / 2] +
                  samples[kSingleIterations / 2]) / 2.0;
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

  // endToEnd 0: kernels only, inputs already resident.
  // endToEnd 1: H2D of both inputs + kernels + D2H of the result.
  const auto enqueue = [&](int endToEnd) {
    if (endToEnd) {
      CUDA_CHECK(cudaMemcpyAsync(deviceOutput, hostOutput,
                                 count * sizeof(int8_t),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(deviceInput, hostInput, count * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
    }
    mse_partial_kernel<<<blocks, kThreads, 0, stream>>>(
        deviceOutput, deviceInput, count, scale, zeroPoint, devicePartials);
    CUDA_CHECK(cudaGetLastError());
    mse_finalize_kernel<<<1, kThreads, 0, stream>>>(devicePartials, blocks,
                                                    count, deviceResult);
    CUDA_CHECK(cudaGetLastError());
    if (endToEnd)
      CUDA_CHECK(cudaMemcpyAsync(hostResult, deviceResult, sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    return EXIT_SUCCESS;
  };
  const bool graph = options.launch == "graph";
  cudaGraphExec_t execs[2] = {};
  for (int endToEnd = 0; graph && endToEnd < 2; ++endToEnd)
    if (capture_graph([&] { return enqueue(endToEnd); }, stream,
                      &execs[endToEnd]))
      return EXIT_FAILURE;
  // One iteration: a graph replay, or the same work enqueued directly.
  const auto launch = [&](int endToEnd) {
    if (!graph)
      return enqueue(endToEnd);
    CUDA_CHECK(cudaGraphLaunch(execs[endToEnd], stream));
    return EXIT_SUCCESS;
  };

  // Correctness: one end-to-end iteration from a poisoned result.
  *hostResult = NAN;
  if (launch(1))
    return EXIT_FAILURE;
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
  if (!options.dump.empty()) {
    FILE* file = std::fopen(options.dump.c_str(), "w");
    if (!file) {
      std::fprintf(stderr, "cannot write %s\n", options.dump.c_str());
      return EXIT_FAILURE;
    }
    uint32_t bits;
    std::memcpy(&bits, &gpu, sizeof(bits));
    std::fprintf(file, "count=%zu set=0 result=%.9g bits=0x%08x\n", count, gpu,
                 bits);
    std::fclose(file);
  }

  double kernelMicroseconds = 0.0, endToEndMicroseconds = 0.0;
  double singleMicroseconds = 0.0;
  if (time_launches([&] { return launch(0); }, stream, options.replays,
                    "anomaly.kernels", kernelMicroseconds) ||
      time_launches([&] { return launch(1); }, stream, options.replays,
                    "anomaly.end_to_end", endToEndMicroseconds) ||
      time_single([&] { return launch(1); }, stream,
                  "anomaly.end_to_end.single", singleMicroseconds))
    return EXIT_FAILURE;
  const double bytes = static_cast<double>(count) * (sizeof(int8_t) +
                                                     sizeof(float));

  std::printf("mlperf-anomaly mse: count=%zu inputs=%s scale=%g zero_point=%d "
              "blocks=%d launch=%s replays=%d\n",
              count, options.inputs.c_str(), scale, zeroPoint, blocks,
              options.launch.c_str(), options.replays);
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
  std::printf("  kernels      %.3f us/iter (%s)\n", kernelMicroseconds,
              graph ? "graph replay" : "direct launch");
  std::printf("  end-to-end   %.3f us/iter (H2D + kernels + D2H, %s)\n",
              endToEndMicroseconds, graph ? "graph replay" : "direct launch");
  std::printf("  bandwidth    %.1f GB/s (kernels, %zu B/element)\n",
              bytes / (kernelMicroseconds * 1.0e3),
              sizeof(int8_t) + sizeof(float));
  std::printf("  batch        %.3f ms (%d back-to-back end-to-end "
              "iterations)\n",
              options.replays * endToEndMicroseconds / 1.0e3, options.replays);
  std::printf("  single       %.3f us (median of %d end-to-end iterations, "
              "each launched alone)\n",
              singleMicroseconds, kSingleIterations);
  std::printf("  timed total  %.3f s (replays x (kernels + end-to-end), "
              "excluding warm-up)\n",
              options.replays * (kernelMicroseconds + endToEndMicroseconds) /
                  1.0e6);

  for (cudaGraphExec_t exec : execs)
    if (exec)
      CUDA_CHECK(cudaGraphExecDestroy(exec));
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

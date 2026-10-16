// CUDA port of mlperf_tiny_kws_postprocess_kernel, the kernel Loom compiles
// for mlperf-tiny-keyword-spotting (loom/mlperf-tiny-kws-smoke.h):
//
//   for i in [0, count):
//     probabilities[i] = DequantizeInt8ToFloat(quantized[i], scale, zeroPoint)
//     if probabilities[i] > probabilities[best]: best = i
//   return best
//
// i.e. dequantize, then the first index of the maximum. DequantizeInt8ToFloat
// is upstream mlperf-tiny's static_cast<float>(value - zero_point) * scale.
//
// The port runs --batch independent rows of --count categories, one warp per
// row. The default (--count=12 --batch=1 --inputs=smoke) is Loom's smoke
// workload; 12 is also upstream's real kCategoryCount. Every row is checked
// bitwise (probabilities and index) against a host copy of the Loom loop, and
// row 0 of the smoke inputs against the smoke oracle.
#include <cfloat>
#include <climits>
#include <cstdint>
#include <random>
#include <vector>

#include "classification_harness.cuh"

namespace {

// Smoke workload constants from loom/mlperf-tiny-kws-smoke.h and the
// classification stub's output_scale() / output_zero_point().
constexpr size_t kSmokeCount = 12;  // kCategoryCount
constexpr int8_t kSmokeQuantized[kSmokeCount] = {-8, -4, 1, 3, 7, 12,
                                                 5,  2,  0, 9, 4, 6};
constexpr float kSmokeScale = 0.125f;
constexpr int kSmokeZeroPoint = 0;

// Random inputs use a nonzero zero point so it is exercised. "ties" draws
// from a narrow range so most rows have a repeated maximum.
constexpr float kRandomScale = 0.0390625f;
constexpr int kRandomZeroPoint = -7;
constexpr int kTiesRange = 3;

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;

float DequantizeInt8ToFloat(int8_t value, float scale, int zeroPoint) {
  return static_cast<float>(value - zeroPoint) * scale;
}

// Verbatim loop of mlperf_tiny_kws_postprocess_kernel.
int loom_postprocess(const int8_t* quantized, float* probabilities, int count,
                     float scale, int zeroPoint) {
  int best = 0;
  for (int index = 0; index < count; ++index) {
    probabilities[index] =
        DequantizeInt8ToFloat(quantized[index], scale, zeroPoint);
    if (probabilities[index] > probabilities[best])
      best = index;
  }
  return best;
}

// Row r is the smoke vector rotated left by r, repeated to fill --count, so
// the winning index varies by row (and repeats when count > 12).
void fill_smoke(int8_t* quantized, size_t count, size_t batch) {
  for (size_t row = 0; row < batch; ++row)
    for (size_t index = 0; index < count; ++index)
      quantized[row * count + index] =
          kSmokeQuantized[(index + row) % kSmokeCount];
}

void fill_random(int8_t* quantized, size_t elements, int low, int high,
                 unsigned seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> distribution(low, high);
  for (size_t index = 0; index < elements; ++index)
    quantized[index] = static_cast<int8_t>(distribution(generator));
}

// One warp per row. Each lane scans its strided elements in increasing order
// keeping the first maximum, then the warp merges (value, index) pairs,
// preferring the larger value and, on equal values, the smaller index. That
// reproduces the sequential "first index of the maximum".
__global__ void __launch_bounds__(kThreads)
    kws_postprocess_kernel(const int8_t* __restrict__ quantized,
                           float* __restrict__ probabilities,
                           int* __restrict__ best, int count, size_t batch,
                           float scale, int zeroPoint) {
  const int lane = threadIdx.x & 31;
  const size_t row =
      static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
  if (row >= batch)
    return;  // uniform per warp
  const size_t base = row * count;

  float bestValue = -INFINITY;
  int bestIndex = INT_MAX;
  for (int index = lane; index < count; index += 32) {
    const float probability =
        static_cast<float>(quantized[base + index] - zeroPoint) * scale;
    probabilities[base + index] = probability;
    if (probability > bestValue) {
      bestValue = probability;
      bestIndex = index;
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float otherValue =
        __shfl_down_sync(0xffffffffu, bestValue, offset);
    const int otherIndex = __shfl_down_sync(0xffffffffu, bestIndex, offset);
    if (otherValue > bestValue ||
        (otherValue == bestValue && otherIndex < bestIndex)) {
      bestValue = otherValue;
      bestIndex = otherIndex;
    }
  }
  if (lane == 0)
    best[row] = bestIndex;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  options.count = kSmokeCount;
  if (!parse_options(argc, argv, options, /*allowBatch=*/true,
                     "smoke random ties"))
    return EXIT_FAILURE;
  if (options.count > INT_MAX) {
    std::fprintf(stderr, "--count must fit in int\n");
    return EXIT_FAILURE;
  }
  const int count = static_cast<int>(options.count);
  const size_t batch = options.batch;
  const size_t elements = options.count * batch;
  const bool smoke = options.inputs == "smoke";
  const float scale = smoke ? kSmokeScale : kRandomScale;
  const int zeroPoint = smoke ? kSmokeZeroPoint : kRandomZeroPoint;
  const unsigned blocks =
      static_cast<unsigned>((batch + kWarpsPerBlock - 1) / kWarpsPerBlock);

  int8_t* hostQuantized = nullptr;
  float* hostProbabilities = nullptr;
  int* hostBest = nullptr;
  CUDA_CHECK(cudaMallocHost(&hostQuantized, elements * sizeof(int8_t)));
  CUDA_CHECK(cudaMallocHost(&hostProbabilities, elements * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&hostBest, batch * sizeof(int)));
  if (smoke)
    fill_smoke(hostQuantized, options.count, batch);
  else if (options.inputs == "random")
    fill_random(hostQuantized, elements, -128, 127, options.seed);
  else
    fill_random(hostQuantized, elements, -kTiesRange, kTiesRange,
                options.seed);

  int8_t* deviceQuantized = nullptr;
  float* deviceProbabilities = nullptr;
  int* deviceBest = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceQuantized, elements * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&deviceProbabilities, elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&deviceBest, batch * sizeof(int)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  // endToEnd 0: kernel only. 1: H2D + kernel + D2H.
  const auto enqueue = [&](int endToEnd) {
    if (endToEnd)
      CUDA_CHECK(cudaMemcpyAsync(deviceQuantized, hostQuantized,
                                 elements * sizeof(int8_t),
                                 cudaMemcpyHostToDevice, stream));
    kws_postprocess_kernel<<<blocks, kThreads, 0, stream>>>(
        deviceQuantized, deviceProbabilities, deviceBest, count, batch, scale,
        zeroPoint);
    CUDA_CHECK(cudaGetLastError());
    if (endToEnd) {
      CUDA_CHECK(cudaMemcpyAsync(hostProbabilities, deviceProbabilities,
                                 elements * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(hostBest, deviceBest, batch * sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
    }
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

  // Correctness: one end-to-end iteration into poisoned host outputs.
  std::memset(hostProbabilities, 0xff, elements * sizeof(float));
  std::memset(hostBest, 0xff, batch * sizeof(int));
  if (launch(1))
    return EXIT_FAILURE;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());

  std::vector<float> expected(options.count);
  size_t mismatchedRows = 0, tiedRows = 0;
  for (size_t row = 0; row < batch; ++row) {
    const size_t base = row * options.count;
    const int expectedBest = loom_postprocess(hostQuantized + base,
                                              expected.data(), count, scale,
                                              zeroPoint);
    bool match = hostBest[row] == expectedBest &&
                 std::memcmp(hostProbabilities + base, expected.data(),
                             options.count * sizeof(float)) == 0;
    int maximumCount = 0;
    for (int index = 0; index < count; ++index)
      maximumCount += expected[index] == expected[expectedBest];
    tiedRows += maximumCount > 1;
    if (!match && mismatchedRows++ < 5)
      std::fprintf(stderr, "row %zu: gpu best %d, loom best %d\n", row,
                   hostBest[row], expectedBest);
  }
  // The smoke main()'s check, on row 0 (the unrotated smoke vector).
  const bool smokeShape = smoke && options.count == kSmokeCount;
  const bool smokeOracle = !smokeShape || (hostBest[0] == 5 &&
                                           hostProbabilities[5] == 1.5f &&
                                           hostProbabilities[0] == -1.0f);

  double kernelMicroseconds = 0.0, endToEndMicroseconds = 0.0;
  if (time_launches([&] { return launch(0); }, stream, options.replays,
                    "kws.kernels", kernelMicroseconds) ||
      time_launches([&] { return launch(1); }, stream, options.replays,
                    "kws.end_to_end", endToEndMicroseconds))
    return EXIT_FAILURE;

  std::printf("mlperf-classification kws postprocess: count=%d batch=%zu "
              "inputs=%s scale=%g zero_point=%d launch=%s replays=%d\n",
              count, batch, options.inputs.c_str(), scale, zeroPoint,
              options.launch.c_str(), options.replays);
  std::printf("  row 0        best=%d p[best]=%g\n", hostBest[0],
              hostProbabilities[hostBest[0]]);
  if (smokeShape)
    std::printf("  smoke oracle %s (best == 5, p[5] == 1.5, p[0] == -1)\n",
                smokeOracle ? "pass" : "FAIL");
  std::printf("  check        %s (%zu/%zu rows bitwise equal to loom loop; "
              "%zu rows with a tied maximum)\n",
              mismatchedRows == 0 ? "pass" : "FAIL", batch - mismatchedRows,
              batch, tiedRows);
  std::printf("  kernel       %.3f us/iter (%s)\n", kernelMicroseconds,
              graph ? "graph replay" : "direct launch");
  std::printf("  end-to-end   %.3f us/iter (H2D + kernel + D2H, %s)\n",
              endToEndMicroseconds, graph ? "graph replay" : "direct launch");
  std::printf("  throughput   %.1f M rows/s (kernel)\n",
              static_cast<double>(batch) / kernelMicroseconds);
  print_timed_total(options.replays, kernelMicroseconds,
                    endToEndMicroseconds);

  for (cudaGraphExec_t exec : execs)
    if (exec)
      CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(deviceQuantized));
  CUDA_CHECK(cudaFree(deviceProbabilities));
  CUDA_CHECK(cudaFree(deviceBest));
  CUDA_CHECK(cudaFreeHost(hostQuantized));
  CUDA_CHECK(cudaFreeHost(hostProbabilities));
  CUDA_CHECK(cudaFreeHost(hostBest));
  return mismatchedRows == 0 && smokeOracle ? EXIT_SUCCESS : EXIT_FAILURE;
}

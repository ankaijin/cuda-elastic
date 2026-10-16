// CUDA port of mlperf_tiny_vww_preprocess_kernel, the kernel Loom compiles
// for mlperf-tiny-visual-wake-words (loom/mlperf-tiny-vww-smoke.h):
//
//   for i in [0, count):
//     centered[i] = static_cast<int8_t>(static_cast<int>(input[i]) - 128)
//
// For every byte x, (x - 128) mod 256 == x ^ 0x80, so the kernel flips the top
// bit of each byte, 16 bytes per thread (uint4), with a scalar tail using the
// original expression.
//
// The default (--count=16 --inputs=smoke) is Loom's smoke workload; 27648 is
// upstream's real kVwwInputSize (96 x 96 x 3). Output is checked bytewise
// against a host copy of the Loom loop, which is also the smoke check.
#include <cstdint>
#include <random>
#include <vector>

#include "classification_harness.cuh"

namespace {

// Smoke input from loom/mlperf-tiny-vww-smoke.h.
constexpr size_t kSmokeCount = 16;  // kVwwInputSize
constexpr uint8_t kSmokeInput[kSmokeCount] = {0,   1,   7,   31,  63,  95,
                                              126, 127, 128, 129, 160, 192,
                                              224, 248, 254, 255};

constexpr int kThreads = 256;
constexpr int kBlocksPerSm = 8;

// Verbatim loop of mlperf_tiny_vww_preprocess_kernel.
void loom_preprocess(const uint8_t* input, int8_t* centered, size_t count) {
  for (size_t index = 0; index < count; ++index)
    centered[index] = static_cast<int8_t>(static_cast<int>(input[index]) - 128);
}

// The smoke vector repeated to fill --count.
void fill_smoke(uint8_t* input, size_t count) {
  for (size_t index = 0; index < count; ++index)
    input[index] = kSmokeInput[index % kSmokeCount];
}

void fill_random(uint8_t* input, size_t count, unsigned seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> distribution(0, 255);
  for (size_t index = 0; index < count; ++index)
    input[index] = static_cast<uint8_t>(distribution(generator));
}

__global__ void __launch_bounds__(kThreads)
    vww_preprocess_kernel(const uint8_t* __restrict__ input,
                          int8_t* __restrict__ centered, size_t count) {
  const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t vectorCount = count / 16;
  const uint4* input16 = reinterpret_cast<const uint4*>(input);
  uint4* centered16 = reinterpret_cast<uint4*>(centered);
  for (size_t index = thread; index < vectorCount; index += stride) {
    uint4 bytes = input16[index];
    bytes.x ^= 0x80808080u;
    bytes.y ^= 0x80808080u;
    bytes.z ^= 0x80808080u;
    bytes.w ^= 0x80808080u;
    centered16[index] = bytes;
  }
  for (size_t index = vectorCount * 16 + thread; index < count;
       index += stride)
    centered[index] =
        static_cast<int8_t>(static_cast<int>(input[index]) - 128);
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  options.count = kSmokeCount;
  if (!parse_options(argc, argv, options, /*allowBatch=*/false,
                     "smoke random"))
    return EXIT_FAILURE;
  const size_t count = options.count;
  const bool smoke = options.inputs == "smoke";
  const size_t work = std::max<size_t>(count / 16, 1);
  const int blocks = static_cast<int>(std::min<size_t>(
      (work + kThreads - 1) / kThreads,
      static_cast<size_t>(sm_count()) * kBlocksPerSm));

  uint8_t* hostInput = nullptr;
  int8_t* hostCentered = nullptr;
  CUDA_CHECK(cudaMallocHost(&hostInput, count));
  CUDA_CHECK(cudaMallocHost(&hostCentered, count));
  if (smoke)
    fill_smoke(hostInput, count);
  else
    fill_random(hostInput, count, options.seed);

  uint8_t* deviceInput = nullptr;
  int8_t* deviceCentered = nullptr;
  CUDA_CHECK(cudaMalloc(&deviceInput, count));
  CUDA_CHECK(cudaMalloc(&deviceCentered, count));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  // endToEnd 0: kernel only. 1: H2D + kernel + D2H.
  const auto enqueue = [&](int endToEnd) {
    if (endToEnd)
      CUDA_CHECK(cudaMemcpyAsync(deviceInput, hostInput, count,
                                 cudaMemcpyHostToDevice, stream));
    vww_preprocess_kernel<<<blocks, kThreads, 0, stream>>>(
        deviceInput, deviceCentered, count);
    CUDA_CHECK(cudaGetLastError());
    if (endToEnd)
      CUDA_CHECK(cudaMemcpyAsync(hostCentered, deviceCentered, count,
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

  // Correctness: one end-to-end iteration into a poisoned host output.
  std::memset(hostCentered, 0x55, count);
  if (launch(1))
    return EXIT_FAILURE;
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());
  std::vector<int8_t> expected(count);
  loom_preprocess(hostInput, expected.data(), count);
  size_t mismatches = 0;
  for (size_t index = 0; index < count; ++index)
    if (hostCentered[index] != expected[index] && mismatches++ < 5)
      std::fprintf(stderr, "element %zu: input %d, gpu %d, loom %d\n", index,
                   hostInput[index], hostCentered[index], expected[index]);

  double kernelMicroseconds = 0.0, endToEndMicroseconds = 0.0;
  if (time_launches([&] { return launch(0); }, stream, options.replays,
                    "vww.kernels", kernelMicroseconds) ||
      time_launches([&] { return launch(1); }, stream, options.replays,
                    "vww.end_to_end", endToEndMicroseconds))
    return EXIT_FAILURE;

  std::printf("mlperf-classification vww preprocess: count=%zu inputs=%s "
              "blocks=%d launch=%s replays=%d\n",
              count, options.inputs.c_str(), blocks, options.launch.c_str(),
              options.replays);
  std::printf("  check        %s (%zu/%zu bytes equal to loom loop)\n",
              mismatches == 0 ? "pass" : "FAIL", count - mismatches, count);
  std::printf("  kernel       %.3f us/iter (%s)\n", kernelMicroseconds,
              graph ? "graph replay" : "direct launch");
  std::printf("  end-to-end   %.3f us/iter (H2D + kernel + D2H, %s)\n",
              endToEndMicroseconds, graph ? "graph replay" : "direct launch");
  std::printf("  bandwidth    %.1f GB/s (kernel, 2 B/element)\n",
              2.0 * static_cast<double>(count) / (kernelMicroseconds * 1.0e3));
  print_timed_total(options.replays, kernelMicroseconds,
                    endToEndMicroseconds);

  for (cudaGraphExec_t exec : execs)
    if (exec)
      CUDA_CHECK(cudaGraphExecDestroy(exec));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(deviceInput));
  CUDA_CHECK(cudaFree(deviceCentered));
  CUDA_CHECK(cudaFreeHost(hostInput));
  CUDA_CHECK(cudaFreeHost(hostCentered));
  return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

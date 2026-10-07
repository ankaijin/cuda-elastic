// Shared host harness for the mlperf-tiny classification kernel ports
// (kws_postprocess.cu, vww_preprocess.cu): error checking, option parsing,
// and graph-replay timing, so both kernels are measured the same way.
//
// Both binaries capture two CUDA graphs: kernels only (inputs resident) and
// end to end (H2D + kernels + D2H). Each is warmed up, then replayed
// --replays times between CUDA events.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t error__ = (call);                                      \
    if (error__ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error: %s at %s:%d\n",                      \
                   cudaGetErrorString(error__), __FILE__, __LINE__);         \
      return EXIT_FAILURE;                                                   \
    }                                                                        \
  } while (0)

constexpr int kWarmupReplays = 100;

struct Options {
  size_t count = 0;  // set by each binary to its smoke size
  size_t batch = 1;
  std::string inputs = "smoke";
  int replays = 10000;
  unsigned seed = 1;
};

// Parses --count, --batch (if allowBatch), --inputs (one of validInputs,
// space separated), --replays and --seed.
inline bool parse_options(int argc, char** argv, Options& options,
                          bool allowBatch, const char* validInputs) {
  const std::string valid = std::string(" ") + validInputs + " ";
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
    else if (const char* v = allowBatch ? value("--batch=") : nullptr)
      options.batch = std::strtoull(v, nullptr, 10);
    else if (const char* v = value("--inputs="))
      options.inputs = v;
    else if (const char* v = value("--replays="))
      options.replays = std::atoi(v);
    else if (const char* v = value("--seed="))
      options.seed = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
    else {
      std::fprintf(stderr,
                   "usage: %s [--count=N]%s [--inputs={%s}] [--replays=R] "
                   "[--seed=S]\n",
                   argv[0], allowBatch ? " [--batch=B]" : "", validInputs);
      return false;
    }
  }
  if (options.count == 0 || options.batch == 0 || options.replays <= 0 ||
      valid.find(" " + options.inputs + " ") == std::string::npos) {
    std::fprintf(stderr, "invalid options\n");
    return false;
  }
  return true;
}

// Average microseconds per replay of a captured graph, after warm-up.
inline int time_graph(cudaGraphExec_t graph, cudaStream_t stream, int replays,
                      double& microseconds) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int replay = 0; replay < std::min(replays, kWarmupReplays); ++replay)
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

// Measured time of both graphs' timed replays, excluding warm-up.
inline void print_graph_total(int replays, double kernelMicroseconds,
                              double endToEndMicroseconds) {
  std::printf("  graph total  %.3f s (replays x (kernel + end-to-end), "
              "excluding warm-up)\n",
              replays * (kernelMicroseconds + endToEndMicroseconds) / 1.0e6);
}

inline int sm_count() {
  int device = 0, count = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount,
                             device) != cudaSuccess)
    return 1;
  return count;
}

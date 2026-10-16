// Shared host harness for the mlperf-tiny classification kernel ports
// (kws_postprocess.cu, vww_preprocess.cu): error checking, option parsing,
// and launch timing, so both kernels are measured the same way.
//
// Both binaries run two pieces of work: kernels only (inputs resident) and
// end to end (H2D + kernels + D2H). With --launch=graph (the default) each is
// captured once into a CUDA graph and replayed; with --launch=direct the same
// stream operations are enqueued directly on every iteration. Either way each
// is warmed up, then run --replays times between CUDA events, each iteration
// inside an NVTX range.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__has_include)
#  if __has_include(<nvtx3/nvToolsExt.h>)
#    include <nvtx3/nvToolsExt.h>
#    define CLASSIFICATION_HAS_NVTX 1
#  endif
#endif
#ifndef CLASSIFICATION_HAS_NVTX
#  define CLASSIFICATION_HAS_NVTX 0
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

constexpr int kWarmupReplays = 100;

struct Options {
  size_t count = 0;  // set by each binary to its smoke size
  size_t batch = 1;
  std::string inputs = "smoke";
  std::string launch = "graph";  // graph | direct
  int replays = 10000;
  unsigned seed = 1;
};

// Parses --count, --batch (if allowBatch), --inputs (one of validInputs,
// space separated), --launch, --replays and --seed.
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
    else if (const char* v = value("--launch="))
      options.launch = v;
    else if (const char* v = value("--replays="))
      options.replays = std::atoi(v);
    else if (const char* v = value("--seed="))
      options.seed = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
    else {
      std::fprintf(stderr,
                   "usage: %s [--count=N]%s [--inputs={%s}] "
                   "[--launch={graph,direct}] [--replays=R] [--seed=S]\n",
                   argv[0], allowBatch ? " [--batch=B]" : "", validInputs);
      return false;
    }
  }
  if (options.count == 0 || options.batch == 0 || options.replays <= 0 ||
      valid.find(" " + options.inputs + " ") == std::string::npos ||
      (options.launch != "graph" && options.launch != "direct")) {
    std::fprintf(stderr, "invalid options\n");
    return false;
  }
  return true;
}

inline void nvtx_push(const char* label) {
#if CLASSIFICATION_HAS_NVTX
  nvtxRangePushA(label);
#else
  (void)label;
#endif
}

inline void nvtx_pop() {
#if CLASSIFICATION_HAS_NVTX
  nvtxRangePop();
#endif
}

// Captures the stream work of enqueue() (an int() callable returning
// EXIT_SUCCESS) into an executable graph.
template <typename Enqueue>
inline int capture_graph(Enqueue enqueue, cudaStream_t stream,
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
inline int time_launches(Launch launch, cudaStream_t stream, int replays,
                         const char* range, double& microseconds) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int replay = 0; replay < std::min(replays, kWarmupReplays); ++replay)
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

// Measured time of both timed loops, excluding warm-up.
inline void print_timed_total(int replays, double kernelMicroseconds,
                              double endToEndMicroseconds) {
  std::printf("  timed total  %.3f s (replays x (kernel + end-to-end), "
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

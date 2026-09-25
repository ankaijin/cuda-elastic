// Shared benchmark harness for the multi-sensor attention CUDA variants.
//
// Every variant (msa_baseline.cu, msa_optimized.cu) provides only its kernels
// and an enqueue function. Everything that must be held constant for a fair
// comparison lives here: problem shape, input data, H2D/D2H inclusion, graph
// capture, warm-up policy, replay count, timing, and the correctness oracle.
//
// The oracle is a direct port of main.c's loops, generalized to
// TOKEN_COUNT x CONTEXT_COUNT and accumulated in double precision.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#ifndef BATCH_COUNT
#define BATCH_COUNT 1
#endif
#ifndef TOKEN_COUNT
#define TOKEN_COUNT 1000
#endif
#ifndef CONTEXT_COUNT
#define CONTEXT_COUNT 50000
#endif
#ifndef GRAPH_REPLAYS
#define GRAPH_REPLAYS 5000
#endif

enum {
  SENSOR_LANE_COUNT = 3,
  FEATURE_LANE_COUNT = 4,
  STATISTIC_COUNT = 4,
};

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t error__ = (call);                                      \
    if (error__ != cudaSuccess) {                                            \
      std::cerr << "CUDA error: " << cudaGetErrorString(error__)             \
                << " at " << __FILE__ << ":" << __LINE__ << '\n';           \
      return EXIT_FAILURE;                                                    \
    }                                                                         \
  } while (0)

// Tolerances for changed summation order against the double reference.
constexpr double kOutputAbsoluteTolerance = 1.0e-4;
constexpr double kOutputRelativeTolerance = 1.0e-4;
constexpr double kStatisticAbsoluteTolerance = 1.0e-4;
constexpr double kStatisticRelativeTolerance = 1.0e-4;

constexpr size_t kQuerySensorElements =
    static_cast<size_t>(BATCH_COUNT) * TOKEN_COUNT * SENSOR_LANE_COUNT;
constexpr size_t kContextSensorElements =
    static_cast<size_t>(BATCH_COUNT) * CONTEXT_COUNT * SENSOR_LANE_COUNT;
constexpr size_t kFeatureElements =
    static_cast<size_t>(BATCH_COUNT) * TOKEN_COUNT * FEATURE_LANE_COUNT;
constexpr size_t kContextFeatureElements =
    static_cast<size_t>(BATCH_COUNT) * CONTEXT_COUNT * FEATURE_LANE_COUNT;
constexpr size_t kStatisticsElements =
    static_cast<size_t>(BATCH_COUNT) * STATISTIC_COUNT;

struct PipelineBuffers {
  float* h_audio;
  float* h_imu;
  float* h_output;
  float* h_statistics;
  float* d_audio;
  float* d_imu;
  float* d_projected_audio;
  float* d_projected_imu;
  float* d_attention;
  float* d_output;
  float* d_statistics;
};

// Enqueues one full pipeline iteration (H2D, kernels, D2H) during capture.
// audio_stream is the capture origin; imu_stream has already been forked from
// it and must be joined back before returning. imu_ready is a spare event for
// that join.
using EnqueuePipeline = int (*)(const PipelineBuffers& buffers,
                                cudaStream_t audio_stream,
                                cudaStream_t imu_stream,
                                cudaEvent_t imu_ready);

enum class InputMode { kRepeat, kRandom };

inline void fill_inputs(InputMode mode, float* audio, float* imu) {
  const float audio_seed[4][SENSOR_LANE_COUNT] = {
      {1.0f, 0.0f, 0.5f},
      {0.5f, 1.0f, -0.5f},
      {-1.0f, 0.5f, 1.0f},
      {0.25f, -0.75f, 0.5f},
  };
  const float imu_seed[4][SENSOR_LANE_COUNT] = {
      {0.5f, 1.0f, 0.0f},
      {-0.5f, 0.25f, 1.0f},
      {1.0f, -0.5f, 0.5f},
      {0.0f, 0.75f, -1.0f},
  };
  if (mode == InputMode::kRepeat) {
    for (int batch = 0; batch < BATCH_COUNT; ++batch) {
      for (int token = 0; token < TOKEN_COUNT; ++token)
        std::memcpy(audio + (batch * TOKEN_COUNT + token) * SENSOR_LANE_COUNT,
                    audio_seed[token % 4], sizeof(audio_seed[0]));
      for (int token = 0; token < CONTEXT_COUNT; ++token)
        std::memcpy(imu + (batch * CONTEXT_COUNT + token) * SENSOR_LANE_COUNT,
                    imu_seed[token % 4], sizeof(imu_seed[0]));
    }
    return;
  }

  // Deterministic, nonrepeating values in [-1, 1).
  uint32_t state = 0x9e3779b9u;
  auto next = [&state]() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<float>(state >> 8) * (2.0f / 16777216.0f) - 1.0f;
  };
  for (size_t index = 0; index < kQuerySensorElements; ++index)
    audio[index] = next();
  for (size_t index = 0; index < kContextSensorElements; ++index)
    imu[index] = next();
}

// Direct port of main.c semantics: projections in float exactly as written,
// attention and statistics accumulated in double.
inline void compute_reference(const float* audio, const float* imu,
                              std::vector<double>& output,
                              std::vector<double>& statistics) {
  output.assign(kFeatureElements, 0.0);
  statistics.assign(kStatisticsElements, 0.0);
  std::vector<float> projected_audio(TOKEN_COUNT * FEATURE_LANE_COUNT);
  std::vector<float> projected_imu(CONTEXT_COUNT * FEATURE_LANE_COUNT);
  std::vector<double> attention(TOKEN_COUNT * FEATURE_LANE_COUNT);
  std::vector<double> scores(CONTEXT_COUNT);

  for (int batch = 0; batch < BATCH_COUNT; ++batch) {
    for (int token = 0; token < TOKEN_COUNT; ++token) {
      const float* in =
          audio + (batch * TOKEN_COUNT + token) * SENSOR_LANE_COUNT;
      float* out = &projected_audio[token * FEATURE_LANE_COUNT];
      out[0] = in[0];
      out[1] = in[1];
      out[2] = in[2];
      out[3] = in[0] + in[1];
    }
    for (int token = 0; token < CONTEXT_COUNT; ++token) {
      const float* in =
          imu + (batch * CONTEXT_COUNT + token) * SENSOR_LANE_COUNT;
      float* out = &projected_imu[token * FEATURE_LANE_COUNT];
      out[0] = in[0];
      out[1] = in[1];
      out[2] = in[2];
      out[3] = in[0] - in[1];
    }

    for (int row = 0; row < TOKEN_COUNT; ++row) {
      const float* query = &projected_audio[row * FEATURE_LANE_COUNT];
      double maximum = 0.0;
      for (int column = 0; column < CONTEXT_COUNT; ++column) {
        const float* key = &projected_imu[column * FEATURE_LANE_COUNT];
        double dot = 0.0;
        for (int lane = 0; lane < FEATURE_LANE_COUNT; ++lane)
          dot += static_cast<double>(query[lane]) * key[lane];
        scores[column] = dot * 0.5;
        if (column == 0 || scores[column] > maximum)
          maximum = scores[column];
      }
      double denominator = 0.0;
      for (int column = 0; column < CONTEXT_COUNT; ++column) {
        scores[column] = std::exp(scores[column] - maximum);
        denominator += scores[column];
      }
      for (int lane = 0; lane < FEATURE_LANE_COUNT; ++lane) {
        double value = 0.0;
        for (int column = 0; column < CONTEXT_COUNT; ++column)
          value += scores[column] * projected_imu[column * FEATURE_LANE_COUNT +
                                                  lane];
        attention[row * FEATURE_LANE_COUNT + lane] = value / denominator;
      }
    }

    double projection_energy = 0.0;
    double attention_sum = 0.0;
    double weighted_sum = 0.0;
    double maximum_magnitude = 0.0;
    for (int token = 0; token < TOKEN_COUNT; ++token) {
      for (int lane = 0; lane < FEATURE_LANE_COUNT; ++lane) {
        const int index = token * FEATURE_LANE_COUNT + lane;
        const double projection = projected_audio[index];
        const double value = attention[index];
        const double magnitude = value < 0.0 ? -value : value;
        projection_energy += projection * projection;
        attention_sum += value;
        weighted_sum += static_cast<double>(index + 1) * value;
        if (magnitude > maximum_magnitude)
          maximum_magnitude = magnitude;
        output[batch * TOKEN_COUNT * FEATURE_LANE_COUNT + index] =
            value + projection * 0.125;
      }
    }
    statistics[batch * STATISTIC_COUNT + 0] = projection_energy;
    statistics[batch * STATISTIC_COUNT + 1] = attention_sum;
    statistics[batch * STATISTIC_COUNT + 2] = weighted_sum;
    statistics[batch * STATISTIC_COUNT + 3] = maximum_magnitude;
  }
}

inline bool within_tolerance(double actual, double expected,
                             double absolute_tolerance,
                             double relative_tolerance) {
  // Written so that NaN fails.
  return std::fabs(actual - expected) <=
         absolute_tolerance + relative_tolerance * std::fabs(expected);
}

inline bool verify(InputMode mode, const float* audio, const float* imu,
                   const float* output, const float* statistics) {
  std::vector<double> expected_output;
  std::vector<double> expected_statistics;
  compute_reference(audio, imu, expected_output, expected_statistics);

  bool passed = true;
  double worst_output_error = 0.0;
  size_t failures = 0;
  for (size_t index = 0; index < kFeatureElements; ++index) {
    const double error = std::fabs(output[index] - expected_output[index]);
    if (!(error <= worst_output_error))
      worst_output_error = error;
    if (!within_tolerance(output[index], expected_output[index],
                          kOutputAbsoluteTolerance,
                          kOutputRelativeTolerance)) {
      if (failures++ < 5)
        std::cerr << "output[" << index << "] = " << output[index]
                  << ", expected " << expected_output[index] << '\n';
      passed = false;
    }
  }
  double worst_statistic_error = 0.0;
  for (size_t index = 0; index < kStatisticsElements; ++index) {
    const double expected = expected_statistics[index];
    const double relative =
        std::fabs(statistics[index] - expected) /
        (std::fabs(expected) > 1.0 ? std::fabs(expected) : 1.0);
    if (!(relative <= worst_statistic_error))
      worst_statistic_error = relative;
    if (!within_tolerance(statistics[index], expected,
                          kStatisticAbsoluteTolerance,
                          kStatisticRelativeTolerance)) {
      std::cerr << "statistics[" << index << "] = " << statistics[index]
                << ", expected " << expected << '\n';
      passed = false;
    }
  }
  std::cout << "reference max |output error|: " << worst_output_error << '\n'
            << "reference max statistic error (rel): "
            << worst_statistic_error << '\n';

#if BATCH_COUNT == 1 && TOKEN_COUNT == 4 && CONTEXT_COUNT == 4
  // Regression against the constants hard-coded in main.c, with its
  // tolerance. Only meaningful for main.c's own inputs.
  if (mode == InputMode::kRepeat) {
    const float combined = statistics[2] + 3.0f * statistics[0] +
                           7.0f * statistics[1] + 11.0f * statistics[3];
    const float actual[5] = {statistics[0], statistics[1], statistics[2],
                             statistics[3], combined};
    const float golden[5] = {9.62500f, 3.93196f, 24.58165f, 0.73098f,
                             89.02110f};
    for (int index = 0; index < 5; ++index) {
      if (!(std::fabs(actual[index] - golden[index]) <= 1.0e-4f)) {
        std::cerr << "main.c constant " << index << ": " << actual[index]
                  << ", expected " << golden[index] << '\n';
        passed = false;
      }
    }
    std::cout << "main.c 4x4 constants: " << (passed ? "match" : "MISMATCH")
              << '\n';
  }
#else
  (void)mode;
#endif
  return passed;
}

inline int run_benchmark(int argc, char** argv, const char* variant_name,
                         EnqueuePipeline enqueue_pipeline) {
  InputMode mode = InputMode::kRepeat;
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--inputs=random") == 0) {
      mode = InputMode::kRandom;
    } else if (std::strcmp(argv[index], "--inputs=repeat") == 0) {
      mode = InputMode::kRepeat;
    } else {
      std::cerr << "Usage: " << argv[0] << " [--inputs=repeat|random]\n";
      return EXIT_FAILURE;
    }
  }

  PipelineBuffers buffers{};
  CUDA_CHECK(cudaMallocHost(&buffers.h_audio,
                            kQuerySensorElements * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&buffers.h_imu,
                            kContextSensorElements * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&buffers.h_output,
                            kFeatureElements * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&buffers.h_statistics,
                            kStatisticsElements * sizeof(float)));
  fill_inputs(mode, buffers.h_audio, buffers.h_imu);

  CUDA_CHECK(cudaMalloc(&buffers.d_audio,
                        kQuerySensorElements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.d_imu,
                        kContextSensorElements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.d_projected_audio,
                        kFeatureElements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.d_projected_imu,
                        kContextFeatureElements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.d_attention,
                        kFeatureElements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.d_output, kFeatureElements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buffers.d_statistics,
                        kStatisticsElements * sizeof(float)));

  cudaStream_t audio_stream = nullptr;
  cudaStream_t imu_stream = nullptr;
  cudaEvent_t capture_start = nullptr;
  cudaEvent_t imu_ready = nullptr;
  cudaEvent_t timing_start = nullptr;
  cudaEvent_t timing_stop = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&audio_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&imu_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&capture_start, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&imu_ready, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreate(&timing_start));
  CUDA_CHECK(cudaEventCreate(&timing_stop));

  CUDA_CHECK(cudaStreamBeginCapture(audio_stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(cudaEventRecord(capture_start, audio_stream));
  CUDA_CHECK(cudaStreamWaitEvent(imu_stream, capture_start, 0));
  if (enqueue_pipeline(buffers, audio_stream, imu_stream, imu_ready) !=
      EXIT_SUCCESS)
    return EXIT_FAILURE;
  CUDA_CHECK(cudaStreamEndCapture(audio_stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  // Warm-up policy: one launch, then GRAPH_REPLAYS timed launches.
  CUDA_CHECK(cudaGraphLaunch(graph_exec, audio_stream));
  CUDA_CHECK(cudaStreamSynchronize(audio_stream));
  CUDA_CHECK(cudaEventRecord(timing_start, audio_stream));
  for (int replay = 0; replay < GRAPH_REPLAYS; ++replay)
    CUDA_CHECK(cudaGraphLaunch(graph_exec, audio_stream));
  CUDA_CHECK(cudaEventRecord(timing_stop, audio_stream));
  CUDA_CHECK(cudaStreamSynchronize(audio_stream));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, timing_start, timing_stop));

  const float* statistics = buffers.h_statistics;
  const float combined = statistics[2] + 3.0f * statistics[0] +
                         7.0f * statistics[1] + 11.0f * statistics[3];
  std::cout << "variant: " << variant_name << " (" << BATCH_COUNT << "x"
            << TOKEN_COUNT << "x" << CONTEXT_COUNT << ", inputs="
            << (mode == InputMode::kRepeat ? "repeat" : "random") << ")\n"
            << "attention checksum: " << statistics[2] << '\n'
            << "projection energy: " << statistics[0] << '\n'
            << "attention sum: " << statistics[1] << '\n'
            << "attention max: " << statistics[3] << '\n'
            << "combined checksum: " << combined << '\n'
            << "graph replay average (us): "
            << 1000.0f * elapsed_ms / GRAPH_REPLAYS << '\n';

  const bool passed = verify(mode, buffers.h_audio, buffers.h_imu,
                             buffers.h_output, buffers.h_statistics);

  CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaEventDestroy(timing_stop));
  CUDA_CHECK(cudaEventDestroy(timing_start));
  CUDA_CHECK(cudaEventDestroy(imu_ready));
  CUDA_CHECK(cudaEventDestroy(capture_start));
  CUDA_CHECK(cudaStreamDestroy(imu_stream));
  CUDA_CHECK(cudaStreamDestroy(audio_stream));
  CUDA_CHECK(cudaFree(buffers.d_statistics));
  CUDA_CHECK(cudaFree(buffers.d_output));
  CUDA_CHECK(cudaFree(buffers.d_attention));
  CUDA_CHECK(cudaFree(buffers.d_projected_imu));
  CUDA_CHECK(cudaFree(buffers.d_projected_audio));
  CUDA_CHECK(cudaFree(buffers.d_imu));
  CUDA_CHECK(cudaFree(buffers.d_audio));
  CUDA_CHECK(cudaFreeHost(buffers.h_statistics));
  CUDA_CHECK(cudaFreeHost(buffers.h_output));
  CUDA_CHECK(cudaFreeHost(buffers.h_imu));
  CUDA_CHECK(cudaFreeHost(buffers.h_audio));

  std::cout << (passed ? "PASSED" : "FAILED") << '\n';
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

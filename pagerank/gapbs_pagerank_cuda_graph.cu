// CUDA Graph representation of the small GAPBS PageRank smoke workload.
//
// This is an executable example of the dataflow represented by
// externals/gapbs/src/pr.cc and gapbs_pagerank_kernel(...).  It is not a
// replacement for Loom's source-level application or its generated kernels.
// The graph is specialized to the smoke fixture: four vertices, eight fixed
// PageRank iterations, and a small CSR graph.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(err__)); \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

constexpr int kVertices = 1 << 20;
constexpr int kEdgesPerVertex = 16;
constexpr int kEdges = kVertices * kEdgesPerVertex;
constexpr int kIterations = 20;
constexpr float kDamping = 0.85f;

// One thread owns one destination vertex.  Each update reads the previous
// score vector and writes a distinct element of the next vector, so no atomic
// operation (and no artificial cross-vertex dependency) is required.
__global__ void pagerank_update(const int* row_offsets, const int* columns,
                                const int* out_degrees,
                                const float* previous, float* next,
                                int vertices, float damping) {
  int dst = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst >= vertices) return;

  float incoming = 0.0f;
  for (int edge = row_offsets[dst]; edge < row_offsets[dst + 1]; ++edge) {
    const int source = columns[edge];
    incoming += previous[source] / out_degrees[source];
  }
  next[dst] = (1.0f - damping) / vertices + damping * incoming;
}

__global__ void initialize_scores(float* scores, int vertices) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < vertices) scores[i] = 1.0f / vertices;
}

int main() {
  // Generated regular CSR fixture: every destination has sixteen incoming
  // edges.  The graph is large enough to exercise the RTX 3090, while the
  // deterministic pattern keeps runs reproducible.
  std::vector<int> h_row_offsets(kVertices + 1);
  std::vector<int> h_columns(kEdges);
  std::vector<int> h_out_degrees(kVertices, kEdgesPerVertex);
  for (int vertex = 0; vertex <= kVertices; ++vertex)
    h_row_offsets[vertex] = vertex * kEdgesPerVertex;
  for (int dst = 0; dst < kVertices; ++dst)
    for (int edge = 0; edge < kEdgesPerVertex; ++edge)
      h_columns[dst * kEdgesPerVertex + edge] =
          (dst * 17 + edge * 104729) & (kVertices - 1);
  std::vector<float> h_result(kVertices);

  int *d_row_offsets = nullptr, *d_columns = nullptr, *d_out_degrees = nullptr;
  float *d_scores_a = nullptr, *d_scores_b = nullptr;
  CUDA_CHECK(cudaMalloc(&d_row_offsets, h_row_offsets.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_columns, h_columns.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out_degrees, h_out_degrees.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_scores_a, sizeof(float) * kVertices));
  CUDA_CHECK(cudaMalloc(&d_scores_b, sizeof(float) * kVertices));

  cudaStream_t compute_stream = nullptr;
  cudaStream_t offsets_stream = nullptr;
  cudaStream_t columns_stream = nullptr;
  cudaStream_t degrees_stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&offsets_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&columns_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&degrees_stream, cudaStreamNonBlocking));
  cudaEvent_t capture_start = nullptr;
  cudaEvent_t offsets_ready = nullptr, columns_ready = nullptr, degrees_ready = nullptr;
  CUDA_CHECK(cudaEventCreateWithFlags(&capture_start, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&offsets_ready, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&columns_ready, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&degrees_ready, cudaEventDisableTiming));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;

  // Begin capture on the compute stream.  Each independent H2D copy runs on
  // its own stream; the three event/wait pairs join those branches into this
  // capture and allow the copies to overlap on capable hardware.  The waits
  // are the only edges needed before initialization.  PageRank iterations
  // remain serial because iteration i+1 depends on iteration i's score vector.
  CUDA_CHECK(cudaStreamBeginCapture(compute_stream, cudaStreamCaptureModeGlobal));
  CUDA_CHECK(cudaEventRecord(capture_start, compute_stream));
  // Waiting on an event recorded by the capturing stream joins each transfer
  // stream to the same graph capture before its independent copy is issued.
  CUDA_CHECK(cudaStreamWaitEvent(offsets_stream, capture_start, 0));
  CUDA_CHECK(cudaStreamWaitEvent(columns_stream, capture_start, 0));
  CUDA_CHECK(cudaStreamWaitEvent(degrees_stream, capture_start, 0));
  CUDA_CHECK(cudaMemcpyAsync(d_row_offsets, h_row_offsets.data(),
                             h_row_offsets.size() * sizeof(int),
                             cudaMemcpyHostToDevice, offsets_stream));
  CUDA_CHECK(cudaEventRecord(offsets_ready, offsets_stream));
  CUDA_CHECK(cudaMemcpyAsync(d_columns, h_columns.data(),
                             h_columns.size() * sizeof(int),
                             cudaMemcpyHostToDevice, columns_stream));
  CUDA_CHECK(cudaEventRecord(columns_ready, columns_stream));
  CUDA_CHECK(cudaMemcpyAsync(d_out_degrees, h_out_degrees.data(),
                             h_out_degrees.size() * sizeof(int),
                             cudaMemcpyHostToDevice, degrees_stream));
  CUDA_CHECK(cudaEventRecord(degrees_ready, degrees_stream));
  CUDA_CHECK(cudaStreamWaitEvent(compute_stream, offsets_ready, 0));
  CUDA_CHECK(cudaStreamWaitEvent(compute_stream, columns_ready, 0));
  CUDA_CHECK(cudaStreamWaitEvent(compute_stream, degrees_ready, 0));

  const int blocks = (kVertices + 31) / 32;
  initialize_scores<<<blocks, 32, 0, compute_stream>>>(d_scores_a, kVertices);
  CUDA_CHECK(cudaGetLastError());

  float* previous = d_scores_a;
  float* next = d_scores_b;
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    pagerank_update<<<blocks, 32, 0, compute_stream>>>(
        d_row_offsets, d_columns, d_out_degrees, previous, next, kVertices,
        kDamping);
    CUDA_CHECK(cudaGetLastError());
    float* temporary = previous;
    previous = next;
    next = temporary;
  }

  // The final copy is ordered after all eight dependent update nodes.
  CUDA_CHECK(cudaMemcpyAsync(h_result.data(), previous, h_result.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, compute_stream));
  CUDA_CHECK(cudaStreamEndCapture(compute_stream, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  int graph_replays = 0;
  if (const char* value = std::getenv("GRAPH_REPLAYS"))
    graph_replays = std::atoi(value);
  if (graph_replays <= 0) {
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, compute_stream));
    CUDA_CHECK(cudaGraphLaunch(executable, compute_stream));
    CUDA_CHECK(cudaEventRecord(stop, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float replay_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&replay_ms, start, stop));
    graph_replays = replay_ms > 0.0f
                        ? static_cast<int>(10000.0f / replay_ms) + 1
                        : 1;
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
  }
  for (int replay = 0; replay < graph_replays; ++replay)
    CUDA_CHECK(cudaGraphLaunch(executable, compute_stream));
  CUDA_CHECK(cudaStreamSynchronize(compute_stream));

  std::printf("PageRank scores after %d iterations (replays=%d): %.6f %.6f %.6f %.6f\n",
              kIterations, graph_replays, h_result[0], h_result[1],
              h_result[2], h_result[3]);

  CUDA_CHECK(cudaGraphExecDestroy(executable));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaEventDestroy(capture_start));
  CUDA_CHECK(cudaEventDestroy(columns_ready));
  CUDA_CHECK(cudaEventDestroy(degrees_ready));
  CUDA_CHECK(cudaEventDestroy(offsets_ready));
  CUDA_CHECK(cudaStreamDestroy(columns_stream));
  CUDA_CHECK(cudaStreamDestroy(degrees_stream));
  CUDA_CHECK(cudaStreamDestroy(offsets_stream));
  CUDA_CHECK(cudaStreamDestroy(compute_stream));
  CUDA_CHECK(cudaFree(d_scores_b));
  CUDA_CHECK(cudaFree(d_scores_a));
  CUDA_CHECK(cudaFree(d_columns));
  CUDA_CHECK(cudaFree(d_out_degrees));
  CUDA_CHECK(cudaFree(d_row_offsets));
  return 0;
}

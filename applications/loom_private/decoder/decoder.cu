// loom-private/llama2c-decoder: batched greedy decoding through llama2.c's
// final rmsnorm, classifier matmul, softmax, and argmax on a synthetic model.
//
// Upstream: private Loom test/applications/llama2c-kernels-smoke.h around
// llama2.c run.c (rmsnorm, matmul, softmax, sample_argmax). This port runs
// `batch` independent sequences at once; each step forwards every sequence's
// current token, and the argmax is fed back on the GPU as the next token.
// See contract.json and README.md.
#include "lp_harness.cuh"
#include "oracle.h"

#include <climits>
#include <cmath>
#include <cstring>

namespace {

constexpr long long kLargeBatch = 1024;
constexpr long long kLargeWidth = 16384;
constexpr long long kLargeVocab = 8192;
constexpr long long kLargeSteps = 4;

constexpr int kRowThreads = 256;     // rmsnorm and softmax: one block per sequence
constexpr int kTile = 64;            // classifier tile: 64 sequences x 64 rows
constexpr int kTileK = 16;           // columns per shared-memory stage
constexpr int kTileThreads = 256;    // 16 x 16 threads, 4 x 4 outputs each

// Synthetic model (contract "data_generation"): embedding row t is 1 except 4
// at column t; the RMSNorm weight is 1; classifier row r is 0 except 1 at
// column (r + vocab - 1) mod vocab.
__global__ void init_model_kernel(float *embedding, float *classifier,
                                  float *norm_weight, int width, int vocab) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t first = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = (size_t)vocab * width;
  for (size_t i = first; i < total; i += stride) {
    const int row = (int)(i / width);
    const int column = (int)(i % width);
    embedding[i] = column == row ? 4.0f : 1.0f;
    classifier[i] = column == (row + vocab - 1) % vocab ? 1.0f : 0.0f;
  }
  for (size_t j = first; j < (size_t)width; j += stride) norm_weight[j] = 1.0f;
}

// llama2.c rmsnorm on the embedding row of each sequence's current token:
// ss = sum x^2; ss = ss / width; ss += 1e-5f; ss = 1 / sqrtf(ss);
// o[j] = weight[j] * (ss * x[j]). The block reduces ss in a fixed tree order;
// for this model every partial sum is an exact float integer, so the result
// equals llama2.c's sequential sum.
__global__ void __launch_bounds__(kRowThreads)
rmsnorm_kernel(float *normalized, const int *tokens, const float *embedding,
               const float *weight, int width, int vocab) {
  __shared__ float partial[kRowThreads];
  const int sequence = blockIdx.x;
  const int token = tokens[sequence];
  float *o = normalized + (size_t)sequence * width;
  if (token < 0 || token >= vocab) { // corrupt state: make the check fail
    for (int j = threadIdx.x; j < width; j += kRowThreads) o[j] = NAN;
    return;
  }
  const float *x = embedding + (size_t)token * width;
  float ss = 0.0f;
  for (int j = threadIdx.x; j < width; j += kRowThreads) ss += x[j] * x[j];
  partial[threadIdx.x] = ss;
  __syncthreads();
  for (int stride = kRowThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  ss = partial[0];
  ss /= (float)width;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  for (int j = threadIdx.x; j < width; j += kRowThreads)
    o[j] = weight[j] * (ss * x[j]);
}

// Dense classifier, llama2.c matmul for every sequence at once:
// logits[b][r] = sum over k = 0..width-1 of classifier[r][k] * normalized[b][k].
// Each output is accumulated by one thread in increasing k from 0.0f with a
// separate multiply and add (--fmad=false), the same order as llama2.c.
__global__ void __launch_bounds__(kTileThreads)
classifier_kernel(float *logits, const float *normalized,
                  const float *classifier, int batch, int width, int vocab) {
  __shared__ float a_tile[kTileK][kTile + 1]; // [k][sequence]
  __shared__ float w_tile[kTileK][kTile + 1]; // [k][row]
  const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
  const int row0 = blockIdx.x * kTile, sequence0 = blockIdx.y * kTile;
  float acc[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
  for (int k0 = 0; k0 < width; k0 += kTileK) {
    for (int e = threadIdx.x; e < kTile * kTileK; e += kTileThreads) {
      const int line = e / kTileK, kk = e % kTileK, k = k0 + kk;
      const int sequence = sequence0 + line, row = row0 + line;
      a_tile[kk][line] = sequence < batch && k < width
                             ? normalized[(size_t)sequence * width + k]
                             : 0.0f;
      w_tile[kk][line] =
          row < vocab && k < width ? classifier[(size_t)row * width + k] : 0.0f;
    }
    __syncthreads();
    const int count = min(kTileK, width - k0);
    for (int kk = 0; kk < count; ++kk) {
      float a[4], w[4];
      for (int i = 0; i < 4; ++i) a[i] = a_tile[kk][ty + 16 * i];
      for (int j = 0; j < 4; ++j) w[j] = w_tile[kk][tx + 16 * j];
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
          const float product = w[j] * a[i];
          acc[i][j] = acc[i][j] + product;
        }
    }
    __syncthreads();
  }
  for (int i = 0; i < 4; ++i) {
    const int sequence = sequence0 + ty + 16 * i;
    if (sequence >= batch) continue;
    for (int j = 0; j < 4; ++j) {
      const int row = row0 + tx + 16 * j;
      if (row < vocab) logits[(size_t)sequence * vocab + row] = acc[i][j];
    }
  }
}

// llama2.c softmax (max-subtracted expf, then divide by the sum) followed by
// sample_argmax (first maximum). The argmax becomes the sequence's next token
// on the device and is recorded in generated[sequence][step]. Reductions use
// a fixed tree order, so results are deterministic.
__global__ void __launch_bounds__(kRowThreads)
softmax_argmax_kernel(float *probabilities, const float *logits, int *tokens,
                      int *generated, int vocab, int steps, int step) {
  __shared__ float value[kRowThreads];
  __shared__ int index[kRowThreads];
  const int sequence = blockIdx.x;
  const float *x = logits + (size_t)sequence * vocab;
  float *p = probabilities + (size_t)sequence * vocab;

  float maximum = -INFINITY;
  for (int r = threadIdx.x; r < vocab; r += kRowThreads) maximum = fmaxf(maximum, x[r]);
  value[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = kRowThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      value[threadIdx.x] = fmaxf(value[threadIdx.x], value[threadIdx.x + stride]);
    __syncthreads();
  }
  maximum = value[0];
  __syncthreads();

  float sum = 0.0f;
  for (int r = threadIdx.x; r < vocab; r += kRowThreads) {
    const float e = expf(x[r] - maximum);
    p[r] = e;
    sum += e;
  }
  value[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = kRowThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) value[threadIdx.x] += value[threadIdx.x + stride];
    __syncthreads();
  }
  sum = value[0];
  __syncthreads();

  // Each thread normalizes the entries it wrote, keeping its first maximum.
  float best = -INFINITY;
  int best_index = INT_MAX;
  for (int r = threadIdx.x; r < vocab; r += kRowThreads) {
    const float q = p[r] / sum;
    p[r] = q;
    if (q > best) {
      best = q;
      best_index = r;
    }
  }
  value[threadIdx.x] = best;
  index[threadIdx.x] = best_index;
  __syncthreads();
  for (int stride = kRowThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other = value[threadIdx.x + stride];
      const int other_index = index[threadIdx.x + stride];
      if (other > value[threadIdx.x] ||
          (other == value[threadIdx.x] && other_index < index[threadIdx.x])) {
        value[threadIdx.x] = other;
        index[threadIdx.x] = other_index;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const int next = index[0] == INT_MAX ? -1 : index[0];
    tokens[sequence] = next;
    generated[(size_t)sequence * steps + step] = next;
  }
}

class Decoder final : public lp::Workload {
public:
  Decoder(long long batch, long long width, long long vocab, long long steps,
          long long fault)
      : B_(batch), W_(width), V_(vocab), S_(steps), fault_(fault) {}

  ~Decoder() override {
    for (void *p : {(void *)d_embedding_, (void *)d_classifier_,
                    (void *)d_norm_weight_, (void *)d_prompts_, (void *)d_tokens_,
                    (void *)d_generated_, (void *)d_normalized_,
                    (void *)d_logits_, (void *)d_probabilities_})
      if (p) cudaFree(p);
    if (h_prompts_) cudaFreeHost(h_prompts_);
    if (h_generated_) cudaFreeHost(h_generated_);
  }

  std::string identity() const override { return "loom-private/llama2c-decoder"; }

  lp::JsonObject dimensions() const override {
    using lp::Json;
    return {{"batch", Json::integer(B_)},
            {"width", Json::integer(W_)},
            {"vocab", Json::integer(V_)},
            {"steps", Json::integer(S_)},
            {"generated_tokens", Json::integer(B_ * S_)},
            {"classifier_macs_per_step", Json::integer(B_ * V_ * W_)},
            {"model_bytes", Json::integer((2 * V_ * W_ + W_) * 4)}};
  }

  std::string application_scope() const override {
    return "prompt upload (pinned H2D, batch int32), token reset from the "
           "prompts, every step (rmsnorm, dense classifier matmul, "
           "softmax+argmax with on-device token feedback), generated-token "
           "download (pinned D2H, batch x steps int32); model weights and "
           "working buffers stay resident";
  }
  std::string compute_scope() const override {
    return "token reset from the resident prompts and every step (rmsnorm, "
           "dense classifier matmul, softmax+argmax)";
  }

  void setup(cudaStream_t stream) override {
    const size_t model = (size_t)(V_ * W_);
    LP_CUDA(cudaMalloc(&d_embedding_, model * sizeof(float)));
    LP_CUDA(cudaMalloc(&d_classifier_, model * sizeof(float)));
    LP_CUDA(cudaMalloc(&d_norm_weight_, (size_t)W_ * sizeof(float)));
    LP_CUDA(cudaMalloc(&d_prompts_, (size_t)B_ * sizeof(int)));
    LP_CUDA(cudaMalloc(&d_tokens_, (size_t)B_ * sizeof(int)));
    LP_CUDA(cudaMalloc(&d_generated_, generated_count() * sizeof(int)));
    LP_CUDA(cudaMalloc(&d_normalized_, normalized_count() * sizeof(float)));
    LP_CUDA(cudaMalloc(&d_logits_, row_count() * sizeof(float)));
    LP_CUDA(cudaMalloc(&d_probabilities_, row_count() * sizeof(float)));
    LP_CUDA(cudaMallocHost(&h_prompts_, (size_t)B_ * sizeof(int)));
    LP_CUDA(cudaMallocHost(&h_generated_, generated_count() * sizeof(int)));
    h_normalized_.resize(normalized_count());
    h_logits_.resize(row_count());
    h_probabilities_.resize(row_count());

    // The model is generated in place on the device (no host copy of the
    // ~2 * vocab * width floats); the oracle regenerates it independently.
    init_model_kernel<<<1024, 256, 0, stream>>>(d_embedding_, d_classifier_,
                                                d_norm_weight_, (int)W_, (int)V_);
    LP_CUDA_LAUNCH();
    // Prompt of sequence b is b mod vocab.
    for (long long b = 0; b < B_; ++b) h_prompts_[b] = (int)(b % V_);
  }

  void enqueue_application(cudaStream_t stream) override {
    LP_CUDA(cudaMemcpyAsync(d_prompts_, h_prompts_, (size_t)B_ * sizeof(int),
                            cudaMemcpyHostToDevice, stream));
    enqueue_compute(stream);
    LP_CUDA(cudaMemcpyAsync(h_generated_, d_generated_,
                            generated_count() * sizeof(int),
                            cudaMemcpyDeviceToHost, stream));
  }

  void enqueue_compute(cudaStream_t stream) override {
    LP_CUDA(cudaMemcpyAsync(d_tokens_, d_prompts_, (size_t)B_ * sizeof(int),
                            cudaMemcpyDeviceToDevice, stream));
    const dim3 tiles((unsigned)((V_ + kTile - 1) / kTile),
                     (unsigned)((B_ + kTile - 1) / kTile));
    for (long long s = 0; s < S_; ++s) {
      float *normalized = d_normalized_ + (size_t)(s * B_ * W_);
      float *logits = d_logits_ + (size_t)(s * B_ * V_);
      float *probabilities = d_probabilities_ + (size_t)(s * B_ * V_);
      rmsnorm_kernel<<<(unsigned)B_, kRowThreads, 0, stream>>>(
          normalized, d_tokens_, d_embedding_, d_norm_weight_, (int)W_, (int)V_);
      LP_CUDA_LAUNCH();
      classifier_kernel<<<tiles, kTileThreads, 0, stream>>>(
          logits, normalized, d_classifier_, (int)B_, (int)W_, (int)V_);
      LP_CUDA_LAUNCH();
      softmax_argmax_kernel<<<(unsigned)B_, kRowThreads, 0, stream>>>(
          probabilities, logits, d_tokens_, d_generated_, (int)V_, (int)S_,
          (int)s);
      LP_CUDA_LAUNCH();
    }
  }

  void poison_outputs(cudaStream_t stream) override {
    // 0xFF bytes: int32 -1 (an invalid token) and float NaN.
    LP_CUDA(cudaMemsetAsync(d_tokens_, 0xFF, (size_t)B_ * sizeof(int), stream));
    LP_CUDA(cudaMemsetAsync(d_generated_, 0xFF, generated_count() * sizeof(int), stream));
    LP_CUDA(cudaMemsetAsync(d_normalized_, 0xFF, normalized_count() * sizeof(float), stream));
    LP_CUDA(cudaMemsetAsync(d_logits_, 0xFF, row_count() * sizeof(float), stream));
    LP_CUDA(cudaMemsetAsync(d_probabilities_, 0xFF, row_count() * sizeof(float), stream));
    std::memset(h_generated_, 0xFF, generated_count() * sizeof(int));
    std::memset(h_normalized_.data(), 0xFF, normalized_count() * sizeof(float));
    std::memset(h_logits_.data(), 0xFF, row_count() * sizeof(float));
    std::memset(h_probabilities_.data(), 0xFF, row_count() * sizeof(float));
  }

  void compute_reference() override {
    reference_ = decoder_oracle::compute(shape());
  }

  bool check_outputs() override {
    // Diagnostics for the check only, copied outside every timed scope.
    LP_CUDA(cudaMemcpy(h_normalized_.data(), d_normalized_,
                       normalized_count() * sizeof(float), cudaMemcpyDeviceToHost));
    LP_CUDA(cudaMemcpy(h_logits_.data(), d_logits_, row_count() * sizeof(float),
                       cudaMemcpyDeviceToHost));
    LP_CUDA(cudaMemcpy(h_probabilities_.data(), d_probabilities_,
                       row_count() * sizeof(float), cudaMemcpyDeviceToHost));
    inject_fault();
    const decoder_oracle::Observed observed{h_generated_, h_normalized_.data(),
                                            h_logits_.data(),
                                            h_probabilities_.data()};
    return decoder_oracle::check(shape(), reference_, observed);
  }

  uint64_t output_digest() const override {
    uint64_t digest = lp::fnv1a64(h_generated_, generated_count() * sizeof(int));
    digest = lp::fnv1a64(h_normalized_.data(), normalized_count() * sizeof(float), digest);
    digest = lp::fnv1a64(h_logits_.data(), row_count() * sizeof(float), digest);
    return lp::fnv1a64(h_probabilities_.data(), row_count() * sizeof(float), digest);
  }

private:
  size_t generated_count() const { return (size_t)(B_ * S_); }
  size_t normalized_count() const { return (size_t)(S_ * B_ * W_); }
  size_t row_count() const { return (size_t)(S_ * B_ * V_); }
  decoder_oracle::Shape shape() const { return {B_, W_, V_, S_}; }

  // Test hook (--set fault=N, check only): corrupts one host output after the
  // download so tests can confirm the check rejects it. 0 disables it.
  void inject_fault() {
    const long long last = S_ - 1;
    if (fault_ == 1) { // a generated token of the last sequence
      h_generated_[generated_count() - 1] ^= 1;
    } else if (fault_ == 2) { // final-step activation of a middle sequence
      const long long b = B_ / 2;
      h_normalized_[(size_t)((last * B_ + b) * W_)] *= 2.0f;
    } else if (fault_ == 3) { // one final-step probability of sequence 0
      h_probabilities_[(size_t)(last * B_ * V_ + V_ / 2)] *= 1.001f;
    }
  }

  long long B_, W_, V_, S_, fault_;
  float *d_embedding_ = nullptr, *d_classifier_ = nullptr, *d_norm_weight_ = nullptr;
  int *d_prompts_ = nullptr, *d_tokens_ = nullptr, *d_generated_ = nullptr;
  float *d_normalized_ = nullptr, *d_logits_ = nullptr, *d_probabilities_ = nullptr;
  int *h_prompts_ = nullptr, *h_generated_ = nullptr;
  std::vector<float> h_normalized_, h_logits_, h_probabilities_;
  decoder_oracle::Reference reference_;
};

std::unique_ptr<lp::Workload> make_decoder(lp::Options &options) {
  const long long batch = options.take_int("batch", kLargeBatch);
  const long long width = options.take_int("width", kLargeWidth);
  const long long vocab = options.take_int("vocab", kLargeVocab);
  const long long steps = options.take_int("steps", kLargeSteps);
  const long long fault = options.take_int("fault", 0);
  if (options.preset != "large") return nullptr;

  if (batch < 1 || batch > (1LL << 20))
    lp::fail(lp::kExitUsage, "batch must be in [1, 1048576] (got %lld)", batch);
  if (width < 2 || width > (1LL << 24))
    lp::fail(lp::kExitUsage, "width must be in [2, 16777216] (got %lld)", width);
  if (vocab < 2 || vocab > width)
    lp::fail(lp::kExitUsage,
             "vocab must satisfy 2 <= vocab <= width: every token owns one "
             "embedding position (got vocab=%lld, width=%lld)",
             vocab, width);
  if (steps < 2 || steps > 1024)
    lp::fail(lp::kExitUsage,
             "steps must be in [2, 1024]: generation repeats the per-token "
             "kernels (got %lld)",
             steps);
  if (fault < 0 || fault > 3)
    lp::fail(lp::kExitUsage, "fault must be 0, 1, 2 or 3 (got %lld)", fault);
  // Device: model plus per-step diagnostics; host: the diagnostics copy.
  const double diagnostics = 4.0 * (double)steps * (double)batch *
                             ((double)width + 2.0 * (double)vocab);
  const double device = 4.0 * (2.0 * (double)vocab * (double)width) + diagnostics;
  if (device > 16.0 * (1LL << 30) || diagnostics > 8.0 * (1LL << 30))
    lp::fail(lp::kExitUsage,
             "dimensions need %.1f GiB of device memory and %.1f GiB of host "
             "diagnostics (limits 16 and 8 GiB)",
             device / (1LL << 30), diagnostics / (1LL << 30));
  return std::make_unique<Decoder>(batch, width, vocab, steps, fault);
}

} // namespace

int main(int argc, char **argv) { return lp::run_main(argc, argv, make_decoder); }

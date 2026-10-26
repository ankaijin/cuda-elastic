// loom-private/vecadd-memory, optimized variant: the same contract as
// vecadd_memory.cu (see contract.json and the README's "Optimized variant").
// The application scope is split into chunks, each on its own stream, so the
// upload of later chunks overlaps the kernel and the download of earlier ones
// (the two copy directions use separate copy engines). One 2D copy uploads a
// chunk of lhs and rhs together. The kernel moves 16 bytes per access, with a
// per-GPU shape. The workload class (poison, oracle call, check, digest) is
// copied unchanged from the baseline.
#include "lp_harness.cuh"
#include "oracle.h"

namespace {

constexpr int kBlock = 256;
constexpr int32_t kPoisonPattern = (int32_t)0xA5A5A5A5u;
constexpr int kMaxChunks = 64;
constexpr int kKernelVariants = 3;

// Four elements per access through 16-byte loads and stores. A chunk's base
// address is 16-byte aligned, so only the final (length mod 4) elements of the
// last chunk take the scalar path. Each thread handles kVectors accesses
// spaced a block apart; kStreaming adds streaming cache hints.
template <int kVectors, bool kStreaming>
__global__ void vecadd_kernel(const int32_t *__restrict__ lhs,
                              const int32_t *__restrict__ rhs,
                              int32_t *__restrict__ result, long long length) {
#ifdef LP_VECADD_MUTANT_DROP_LAST
  // Test-only mutant: never writes the last element.
  length -= 1;
#endif
  const long long full = length >> 2;
  long long v = (long long)blockIdx.x * blockDim.x * kVectors + threadIdx.x;
#pragma unroll
  for (int j = 0; j < kVectors; ++j, v += blockDim.x) {
    if (v >= full) continue;
    const int4 *pa = reinterpret_cast<const int4 *>(lhs) + v;
    const int4 *pb = reinterpret_cast<const int4 *>(rhs) + v;
    int4 *pr = reinterpret_cast<int4 *>(result) + v;
    const int4 a = kStreaming ? __ldcs(pa) : *pa;
    const int4 b = kStreaming ? __ldcs(pb) : *pb;
    int4 sum;
    sum.x = a.x + b.x;
    sum.y = a.y + b.y;
    sum.z = a.z + b.z;
    sum.w = a.w + b.w;
    if (kStreaming) __stcs(pr, sum);
    else *pr = sum;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0)
    for (long long i = full * 4; i < length; ++i) result[i] = lhs[i] + rhs[i];
}

// Per-GPU choices, from measurements (see the README). `kernel` indexes
// launch(); LP_VECADD_CHUNKS and LP_VECADD_KERNEL override them for the tests
// and for tuning, and never change the result.
struct Tuning {
  int chunks;
  int kernel;
};

Tuning tuning_for(const cudaDeviceProp &properties) {
  Tuning tuning{8, 0};
  switch (properties.major * 10 + properties.minor) {
  case 86: tuning = {8, 0}; break;  // RTX 3090: DRAM bound, streaming hints
  case 89: tuning = {8, 1}; break;  // RTX 4090: L2 resident, 4 vectors per thread
  case 120: tuning = {4, 2}; break; // RTX 5090: L2 resident, streaming hints
  default: break;
  }
  if (const char *text = std::getenv("LP_VECADD_CHUNKS")) {
    const int value = std::atoi(text);
    if (value >= 1 && value <= kMaxChunks) tuning.chunks = value;
  }
  if (const char *text = std::getenv("LP_VECADD_KERNEL")) {
    const int value = std::atoi(text);
    if (value >= 0 && value < kKernelVariants) tuning.kernel = value;
  }
  return tuning;
}

class VecaddMemory final : public lp::Workload {
public:
  VecaddMemory(long long length, long long multiplier, long long bias)
      : length_(length), multiplier_(multiplier), bias_(bias) {}

  ~VecaddMemory() override {
    for (cudaStream_t s : streams_) cudaStreamDestroy(s);
    for (cudaEvent_t e : joins_) cudaEventDestroy(e);
    if (fork_) cudaEventDestroy(fork_);
    cudaFree(d_lhs_);
    cudaFree(d_result_);
    cudaFreeHost(h_lhs_);
    cudaFreeHost(h_result_);
  }

  std::string identity() const override { return "loom-private/vecadd-memory"; }

  lp::JsonObject dimensions() const override {
    return {{"length", lp::Json::integer(length_)},
            {"multiplier", lp::Json::integer(multiplier_)},
            {"bias", lp::Json::integer(bias_)}};
  }

  std::string application_scope() const override {
    return "upload of lhs and rhs from pinned host memory, the element-wise add, "
           "and download of result to pinned host memory";
  }
  std::string compute_scope() const override {
    return "the element-wise add with lhs and rhs resident on the device; no "
           "transfers";
  }

  void setup(cudaStream_t) override {
    const size_t bytes = (size_t)length_ * sizeof(int32_t);
    // lhs and rhs share one allocation on each side, rhs a pitch after lhs
    // (a multiple of 256 bytes), so one 2D copy moves a chunk of both.
    pitch_ = (size_t)((length_ + 63) & ~63ll) * sizeof(int32_t);
    LP_CUDA(cudaMallocHost((void **)&h_lhs_, 2 * pitch_));
    h_rhs_ = h_lhs_ + pitch_ / sizeof(int32_t);
    LP_CUDA(cudaMallocHost((void **)&h_result_, bytes));
    LP_CUDA(cudaMalloc((void **)&d_lhs_, 2 * pitch_));
    d_rhs_ = d_lhs_ + pitch_ / sizeof(int32_t);
    LP_CUDA(cudaMalloc((void **)&d_result_, bytes));
    // Validated up front, so the 64-bit products below fit int32 once cast.
    for (long long i = 0; i < length_; ++i) {
      h_lhs_[i] = (int32_t)i;
      h_rhs_[i] = (int32_t)(i * multiplier_ + bias_);
    }
    std::memset(h_result_, 0, bytes);

    int device = 0;
    cudaDeviceProp properties{};
    LP_CUDA(cudaGetDevice(&device));
    LP_CUDA(cudaGetDeviceProperties(&properties, device));
    tuning_ = tuning_for(properties);
    // Chunks are multiples of four elements; fewer chunks result when the
    // length is small.
    const long long quads = (length_ + 3) / 4;
    const long long per_chunk = (quads + tuning_.chunks - 1) / tuning_.chunks;
    chunk_elements_ = 4 * per_chunk;
    chunks_ = (int)((length_ + chunk_elements_ - 1) / chunk_elements_);
    // One stream per chunk, joined back through events, so Direct and the
    // captured Graph both fork and join the same pipeline.
    LP_CUDA(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming));
    for (int c = 0; c < chunks_; ++c) {
      cudaStream_t stream;
      cudaEvent_t join;
      LP_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      LP_CUDA(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
      streams_.push_back(stream);
      joins_.push_back(join);
    }
  }

  void enqueue_application(cudaStream_t stream) override {
    LP_CUDA(cudaEventRecord(fork_, stream));
    for (int c = 0; c < chunks_; ++c) {
      cudaStream_t side = streams_[(size_t)c];
      const long long begin = (long long)c * chunk_elements_;
      const long long count = std::min(chunk_elements_, length_ - begin);
      const size_t bytes = (size_t)count * sizeof(int32_t);
      LP_CUDA(cudaStreamWaitEvent(side, fork_, 0));
#ifdef LP_VECADD_MUTANT_SKIP_UPLOAD
      // Test-only mutant: the last chunk is never uploaded.
      if (c + 1 < chunks_)
#endif
        LP_CUDA(cudaMemcpy2DAsync(d_lhs_ + begin, pitch_, h_lhs_ + begin, pitch_,
                                  bytes, 2, cudaMemcpyHostToDevice, side));
      launch(d_lhs_ + begin, d_rhs_ + begin, d_result_ + begin, count, side);
#ifdef LP_VECADD_MUTANT_SKIP_DOWNLOAD
      // Test-only mutant: the last chunk is never downloaded.
      if (c + 1 < chunks_)
#endif
        LP_CUDA(cudaMemcpyAsync(h_result_ + begin, d_result_ + begin, bytes,
                                cudaMemcpyDeviceToHost, side));
      LP_CUDA(cudaEventRecord(joins_[(size_t)c], side));
    }
    for (int c = 0; c < chunks_; ++c)
      LP_CUDA(cudaStreamWaitEvent(stream, joins_[(size_t)c], 0));
  }

  void enqueue_compute(cudaStream_t stream) override {
    launch(d_lhs_, d_rhs_, d_result_, length_, stream);
  }

  // Host and device results both start as the bitwise complement of the
  // expected value when the oracle has run (so no element can already hold its
  // correct value), else as a fixed pattern. The inputs are poisoned first.
  void poison_outputs(cudaStream_t stream) override {
    const size_t count = (size_t)length_;
    // The device inputs hold the bitwise complement of their host sources
    // (staged through h_result_), so an invocation that skips either upload
    // gets a different sum at every element.
    for (size_t i = 0; i < count; ++i) h_result_[i] = ~h_lhs_[i];
    LP_CUDA(cudaMemcpyAsync(d_lhs_, h_result_, count * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < count; ++i) h_result_[i] = ~h_rhs_[i];
    LP_CUDA(cudaMemcpyAsync(d_rhs_, h_result_, count * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < count; ++i)
      h_result_[i] = have_reference_ ? ~expected_.result[i] : kPoisonPattern;
    LP_CUDA(cudaMemcpyAsync(d_result_, h_result_, count * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaStreamSynchronize(stream));
  }

  void compute_reference() override {
    expected_ = vecadd_oracle::compute(length_, multiplier_, bias_);
    have_reference_ = true;
  }

  bool check_outputs() override {
    long long mismatches = 0;
    int64_t checksum = 0;
    for (long long i = 0; i < length_; ++i) {
      checksum += h_result_[i];
      if (h_result_[i] != expected_.result[(size_t)i]) {
        if (mismatches < 8)
          std::fprintf(stderr, "%s: mismatch at index %lld: expected %d, actual %d\n",
                       LP_APP, i, expected_.result[(size_t)i], h_result_[i]);
        ++mismatches;
      }
    }
    const bool checksum_ok = checksum == expected_.checksum;
    if (!checksum_ok)
      std::fprintf(stderr, "%s: checksum mismatch: expected %lld, actual %lld\n", LP_APP,
                   (long long)expected_.checksum, (long long)checksum);
    std::fprintf(stderr, "%s: %lld of %lld results differ; checksum %lld (expected %lld)\n",
                 LP_APP, mismatches, length_, (long long)checksum,
                 (long long)expected_.checksum);
    return mismatches == 0 && checksum_ok;
  }

  uint64_t output_digest() const override {
    return lp::fnv1a64(h_result_, (size_t)length_ * sizeof(int32_t));
  }

private:
  void launch(const int32_t *lhs, const int32_t *rhs, int32_t *result,
              long long count, cudaStream_t stream) const {
    const long long vectors = count / 4;
    switch (tuning_.kernel) {
    case 0: run<1, true>(lhs, rhs, result, count, vectors, stream); break;
    case 1: run<4, false>(lhs, rhs, result, count, vectors, stream); break;
    default: run<2, true>(lhs, rhs, result, count, vectors, stream); break;
    }
  }

  template <int kVectors, bool kStreaming>
  static void run(const int32_t *lhs, const int32_t *rhs, int32_t *result,
                  long long count, long long vectors, cudaStream_t stream) {
    const long long per_block = (long long)kBlock * kVectors;
    // At least one block, which also runs the scalar tail when count < 4.
    const long long blocks = std::max(1ll, (vectors + per_block - 1) / per_block);
    vecadd_kernel<kVectors, kStreaming>
        <<<(unsigned)blocks, kBlock, 0, stream>>>(lhs, rhs, result, count);
    LP_CUDA_LAUNCH();
  }

  long long length_, multiplier_, bias_;
  int32_t *h_lhs_ = nullptr, *h_rhs_ = nullptr, *h_result_ = nullptr;
  int32_t *d_lhs_ = nullptr, *d_rhs_ = nullptr, *d_result_ = nullptr;
  size_t pitch_ = 0;
  Tuning tuning_{};
  long long chunk_elements_ = 0;
  int chunks_ = 0;
  std::vector<cudaStream_t> streams_;
  std::vector<cudaEvent_t> joins_;
  cudaEvent_t fork_ = nullptr;
  vecadd_oracle::Result expected_;
  bool have_reference_ = false;
};

std::unique_ptr<lp::Workload> make_workload(lp::Options &options) {
  if (options.preset != "large") return nullptr;
  const long long length = options.take_int("length", 1048576);
  const long long multiplier = options.take_int("multiplier", 2);
  const long long bias = options.take_int("bias", 0);
  const std::string problem = vecadd_oracle::validate(length, multiplier, bias);
  if (!problem.empty()) lp::fail(lp::kExitUsage, "%s", problem.c_str());
  return std::make_unique<VecaddMemory>(length, multiplier, bias);
}

} // namespace

int main(int argc, char **argv) { return lp::run_main(argc, argv, make_workload); }

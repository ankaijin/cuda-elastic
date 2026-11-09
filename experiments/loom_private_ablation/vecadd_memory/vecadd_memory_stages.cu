// Cumulative optimization stages of loom-private/vecadd-memory, between the
// baseline (applications/loom_private/vecadd_memory/vecadd_memory.cu) and the
// optimized variant (vecadd_memory_optimized.cu in the same directory). See
// README.md.
//
//   -DLP_STAGE=1  the application scope split into chunks, each on its own
//                 stream (upload, kernel, download), forked from and joined
//                 back to the harness stream; the baseline's separate lhs and
//                 rhs buffers (two upload copies per chunk) and its scalar
//                 kernel, one thread per element.
//   -DLP_STAGE=2  stage 1 with lhs and rhs in one pitched allocation on each
//                 side, so one 2D copy uploads a chunk of both.
//   -DLP_STAGE=3  stage 2 with the optimized variant's kernel: 16-byte loads
//                 and stores, one vector per thread, plain loads and stores.
//   -DLP_STAGE=4  stage 3 with streaming cache hints (__ldcs/__stcs). This is
//                 vecadd_memory_optimized.cu with its generic default tuning
//                 (8 chunks, kernel 0) instead of the per-GPU table.
//
// Stage 5, the per-GPU tuning, is the production optimized binary itself.
// Every stage uses the default chunk count, 8. Chunks are multiples of four
// elements in every stage (the vector kernel needs 16-byte aligned chunk
// starts), which changes no chunk at the large size.
//
// Every stage computes the baseline's int32 adds, so every output is bitwise
// equal to the baseline's. The workload class (poison, oracle call, check,
// digest) is copied from vecadd_memory_optimized.cu, which copies it from the
// baseline.
//
// Test builds only (-DLP_VECADD_TEST_TUNING): LP_VECADD_CHUNKS (1 to 64)
// overrides the chunk count, as in the optimized variant's test builds. The
// mutant macros are the optimized variant's. Timed builds read no variable.
#include "lp_harness.cuh"
#include "oracle.h"

#if !defined(LP_STAGE) || LP_STAGE < 1 || LP_STAGE > 4
#error "build with -DLP_STAGE=1, 2, 3 or 4"
#endif

namespace {

constexpr int kBlock = 256;
constexpr int32_t kPoisonPattern = (int32_t)0xA5A5A5A5u;
// The optimized variant's generic default chunk count (tuning_for's default).
constexpr int kDefaultChunks = 8;

#if LP_STAGE <= 2
constexpr const char *kKernelName = "scalar (1 element per thread)";

// The baseline's kernel: one thread per element; the guard is the tail
// handling.
__global__ void vecadd_kernel(const int32_t *__restrict__ lhs,
                              const int32_t *__restrict__ rhs,
                              int32_t *__restrict__ result, long long length) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < length) result[i] = lhs[i] + rhs[i];
}
#else
constexpr const char *kKernelName =
    LP_STAGE == 3 ? "1 vector per thread" : "1 vector per thread, streaming hints";

// The optimized variant's kernel, unchanged.
// Four elements per access through 16-byte loads and stores. A chunk's base
// address is 16-byte aligned, so only the final (length mod 4) elements of the
// last chunk take the scalar path. Each thread handles kVectors accesses
// spaced a block apart; kStreaming adds streaming cache hints.
template <int kVectors, bool kStreaming>
__global__ void vecadd_kernel(const int32_t *__restrict__ lhs,
                              const int32_t *__restrict__ rhs,
                              int32_t *__restrict__ result, long long length) {
  const long long full = length >> 2;
  long long v = (long long)blockIdx.x * blockDim.x * kVectors + threadIdx.x;
#pragma unroll
  for (int j = 0; j < kVectors; ++j, v += blockDim.x) {
    if (v >= full) continue;
    const int4 *pa = reinterpret_cast<const int4 *>(lhs) + v;
    const int4 *pb = reinterpret_cast<const int4 *>(rhs) + v;
    int4 *pr = reinterpret_cast<int4 *>(result) + v;
    int4 a, b;
    if constexpr (kStreaming) {
      a = __ldcs(pa);
      b = __ldcs(pb);
    } else {
      a = *pa;
      b = *pb;
    }
    int4 sum;
    sum.x = a.x + b.x;
    sum.y = a.y + b.y;
    sum.z = a.z + b.z;
    sum.w = a.w + b.w;
    if constexpr (kStreaming) __stcs(pr, sum);
    else *pr = sum;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0)
    for (long long i = full * 4; i < length; ++i) result[i] = lhs[i] + rhs[i];
}
#endif

#ifdef LP_VECADD_TEST_TUNING
// Test builds only: as in the optimized variant. A value out of range exits 2.
int override_from_env(const char *name, int low, int high, int fallback) {
  const char *text = std::getenv(name);
  if (!text) return fallback;
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end || value < low || value > high)
    lp::fail(lp::kExitUsage, "%s must be an integer from %d to %d, got '%s'", name, low,
             high, text);
  return (int)value;
}
#endif

int chunk_count() {
  int chunks = kDefaultChunks;
#ifdef LP_VECADD_TEST_TUNING
  chunks = override_from_env("LP_VECADD_CHUNKS", 1, 64, chunks);
#endif
  return chunks;
}

#ifdef LP_VECADD_MUTANT_NO_JOIN
// Test-only (no-join mutant): hold the last chunk for 20 ms, so that without
// the join the harness stream is certain to finish first.
__global__ void spin_20ms() {
  unsigned long long start, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  } while (now - start < 20000000ull);
}
#endif

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
#if LP_STAGE == 1
    cudaFree(d_rhs_);
    cudaFreeHost(h_rhs_);
#endif
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
#if LP_STAGE == 1
    // The baseline's buffers: lhs and rhs are separate allocations.
    LP_CUDA(cudaMallocHost((void **)&h_lhs_, bytes));
    LP_CUDA(cudaMallocHost((void **)&h_rhs_, bytes));
    LP_CUDA(cudaMallocHost((void **)&h_result_, bytes));
    LP_CUDA(cudaMalloc((void **)&d_lhs_, bytes));
    LP_CUDA(cudaMalloc((void **)&d_rhs_, bytes));
    LP_CUDA(cudaMalloc((void **)&d_result_, bytes));
#else
    // lhs and rhs share one allocation on each side, rhs a pitch after lhs
    // (a multiple of 256 bytes), so one 2D copy moves a chunk of both.
    pitch_ = (size_t)((length_ + 63) & ~63ll) * sizeof(int32_t);
    LP_CUDA(cudaMallocHost((void **)&h_lhs_, 2 * pitch_));
    h_rhs_ = h_lhs_ + pitch_ / sizeof(int32_t);
    LP_CUDA(cudaMallocHost((void **)&h_result_, bytes));
    LP_CUDA(cudaMalloc((void **)&d_lhs_, 2 * pitch_));
    d_rhs_ = d_lhs_ + pitch_ / sizeof(int32_t);
    LP_CUDA(cudaMalloc((void **)&d_result_, bytes));
#endif
    // Validated up front, so the 64-bit products below fit int32 once cast.
    for (long long i = 0; i < length_; ++i) {
      h_lhs_[i] = (int32_t)i;
      h_rhs_[i] = (int32_t)(i * multiplier_ + bias_);
    }
    std::memset(h_result_, 0, bytes);

    // Chunks are multiples of four elements; fewer chunks result when the
    // length is small.
    const int chunks = chunk_count();
    const long long quads = (length_ + 3) / 4;
    const long long per_chunk = (quads + chunks - 1) / chunks;
    chunk_elements_ = 4 * per_chunk;
    chunks_ = (int)((length_ + chunk_elements_ - 1) / chunk_elements_);
    std::fprintf(stderr, "%s: stage %d: %d chunks of %lld elements, %s kernel\n", LP_APP,
                 LP_STAGE, chunks_, chunk_elements_, kKernelName);
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
      {
#if LP_STAGE == 1
        LP_CUDA(cudaMemcpyAsync(d_lhs_ + begin, h_lhs_ + begin, bytes,
                                cudaMemcpyHostToDevice, side));
        LP_CUDA(cudaMemcpyAsync(d_rhs_ + begin, h_rhs_ + begin, bytes,
                                cudaMemcpyHostToDevice, side));
#else
        LP_CUDA(cudaMemcpy2DAsync(d_lhs_ + begin, pitch_, h_lhs_ + begin, pitch_,
                                  bytes, 2, cudaMemcpyHostToDevice, side));
#endif
      }
      launch(d_lhs_ + begin, d_rhs_ + begin, d_result_ + begin, count,
             c + 1 == chunks_, side);
#ifdef LP_VECADD_MUTANT_NO_JOIN
      if (c + 1 == chunks_) {
        spin_20ms<<<1, 1, 0, side>>>();
        LP_CUDA_LAUNCH();
      }
#endif
#ifdef LP_VECADD_MUTANT_SKIP_DOWNLOAD
      // Test-only mutant: the last chunk is never downloaded.
      if (c + 1 < chunks_)
#endif
        LP_CUDA(cudaMemcpyAsync(h_result_ + begin, d_result_ + begin, bytes,
                                cudaMemcpyDeviceToHost, side));
      LP_CUDA(cudaEventRecord(joins_[(size_t)c], side));
    }
#ifndef LP_VECADD_MUTANT_NO_JOIN // Test-only mutant: the join is dropped.
    for (int c = 0; c < chunks_; ++c)
      LP_CUDA(cudaStreamWaitEvent(stream, joins_[(size_t)c], 0));
#endif
  }

  void enqueue_compute(cudaStream_t stream) override {
    launch(d_lhs_, d_rhs_, d_result_, length_, true, stream);
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
  // `last` marks the launch that covers the end of the vectors.
  void launch(const int32_t *lhs, const int32_t *rhs, int32_t *result,
              long long count, bool last, cudaStream_t stream) const {
#ifdef LP_VECADD_MUTANT_DROP_LAST
    // Test-only mutant: never writes the last element.
    if (last) count -= 1;
#else
    (void)last;
#endif
#if LP_STAGE <= 2
#ifdef LP_VECADD_MUTANT_DROP_LAST
    if (count == 0) return; // the mutant's one-element chunk: nothing to write
#endif
    // The baseline's launch: one thread per element, 256-thread blocks.
    const long long blocks = (count + kBlock - 1) / kBlock;
    vecadd_kernel<<<(unsigned)blocks, kBlock, 0, stream>>>(lhs, rhs, result, count);
    LP_CUDA_LAUNCH();
#else
    // The optimized variant's kernel 0 shape (1 vector per thread), with
    // streaming hints from stage 4 on.
    run<1, LP_STAGE == 4>(lhs, rhs, result, count, count / 4, stream);
#endif
  }

#if LP_STAGE >= 3
  // The optimized variant's launch, unchanged.
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
#endif

  long long length_, multiplier_, bias_;
  int32_t *h_lhs_ = nullptr, *h_rhs_ = nullptr, *h_result_ = nullptr;
  int32_t *d_lhs_ = nullptr, *d_rhs_ = nullptr, *d_result_ = nullptr;
  size_t pitch_ = 0;
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

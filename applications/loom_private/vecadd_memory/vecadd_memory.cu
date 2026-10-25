// loom-private/vecadd-memory: result[i] = lhs[i] + rhs[i] over int32 vectors.
// CUDA port of the private Loom application of the same name (see
// contract.json). Correctness only: one thread per element, no tuning.
#include "lp_harness.cuh"
#include "oracle.h"

namespace {

constexpr int kBlock = 256;
constexpr int32_t kPoisonPattern = (int32_t)0xA5A5A5A5u;

// One thread per element; the guard is the tail handling.
__global__ void vecadd_kernel(const int32_t *__restrict__ lhs,
                              const int32_t *__restrict__ rhs,
                              int32_t *__restrict__ result, long long length) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
#ifdef LP_VECADD_MUTANT_DROP_LAST
  // Test-only mutant: never writes the last element.
  if (i + 1 < length) result[i] = lhs[i] + rhs[i];
#else
  if (i < length) result[i] = lhs[i] + rhs[i];
#endif
}

class VecaddMemory final : public lp::Workload {
public:
  VecaddMemory(long long length, long long multiplier, long long bias)
      : length_(length), multiplier_(multiplier), bias_(bias) {}

  ~VecaddMemory() override {
    cudaFree(d_lhs_);
    cudaFree(d_rhs_);
    cudaFree(d_result_);
    cudaFreeHost(h_lhs_);
    cudaFreeHost(h_rhs_);
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
    LP_CUDA(cudaMallocHost((void **)&h_lhs_, bytes));
    LP_CUDA(cudaMallocHost((void **)&h_rhs_, bytes));
    LP_CUDA(cudaMallocHost((void **)&h_result_, bytes));
    LP_CUDA(cudaMalloc((void **)&d_lhs_, bytes));
    LP_CUDA(cudaMalloc((void **)&d_rhs_, bytes));
    LP_CUDA(cudaMalloc((void **)&d_result_, bytes));
    // Validated up front, so the 64-bit products below fit int32 once cast.
    for (long long i = 0; i < length_; ++i) {
      h_lhs_[i] = (int32_t)i;
      h_rhs_[i] = (int32_t)(i * multiplier_ + bias_);
    }
    std::memset(h_result_, 0, bytes);
  }

  void enqueue_application(cudaStream_t stream) override {
    const size_t bytes = (size_t)length_ * sizeof(int32_t);
    LP_CUDA(cudaMemcpyAsync(d_lhs_, h_lhs_, bytes, cudaMemcpyHostToDevice, stream));
    LP_CUDA(cudaMemcpyAsync(d_rhs_, h_rhs_, bytes, cudaMemcpyHostToDevice, stream));
    enqueue_compute(stream);
    LP_CUDA(cudaMemcpyAsync(h_result_, d_result_, bytes, cudaMemcpyDeviceToHost, stream));
  }

  void enqueue_compute(cudaStream_t stream) override {
    const long long blocks = (length_ + kBlock - 1) / kBlock;
    vecadd_kernel<<<(unsigned)blocks, kBlock, 0, stream>>>(d_lhs_, d_rhs_, d_result_,
                                                          length_);
    LP_CUDA_LAUNCH();
  }

  // Host and device results both start as the bitwise complement of the
  // expected value when the oracle has run (so no element can already hold its
  // correct value), else as a fixed pattern.
  void poison_outputs(cudaStream_t stream) override {
    const size_t count = (size_t)length_;
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
  long long length_, multiplier_, bias_;
  int32_t *h_lhs_ = nullptr, *h_rhs_ = nullptr, *h_result_ = nullptr;
  int32_t *d_lhs_ = nullptr, *d_rhs_ = nullptr, *d_result_ = nullptr;
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

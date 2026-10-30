// Exhaustive lane test of one case's add_bytes (experiments/blackwell/packed).
//
// Compiled with -DPACKED_SOURCE=\"<case source>\", which is included whole, so
// the test runs the exact add_bytes the case's binary uses (its main is renamed
// and unused). Three input sets, each checked byte for byte against
// (a + b) mod 256 per lane on the host:
//   every pair: 65,536 words; lane l holds pair (i * m_l + c_l) mod 65,536 with
//     odd m_l, so every lane sees all 65,536 (a, b) byte pairs exactly once,
//     next to different pairs in the other lanes;
//   edge mix: every word whose four lanes each take a value from {0x00, 0x01,
//     0x7f, 0x80, 0x81, 0xfe, 0xff}, for a and b independently (5,764,801
//     words), which covers every combination of carries and top-bit overflows
//     across neighbouring lanes;
//   random: 2^20 words from a fixed splitmix64 stream.
// Exit 0 when every byte matches, 1 otherwise, 70 on a CUDA error.
#define main packed_source_main
#include PACKED_SOURCE
#undef main

#include <cstdio>
#include <vector>

namespace {

__global__ void add_words(const uint32_t *a, const uint32_t *b, uint32_t *sum,
                          size_t words) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < words) sum[i] = add_bytes(a[i], b[i]);
}

uint64_t splitmix64(uint64_t &state) {
  uint64_t z = (state += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

uint32_t reference(uint32_t a, uint32_t b) {
  uint32_t out = 0;
  for (int lane = 0; lane < 4; ++lane) {
    const unsigned x = (a >> (8 * lane)) & 255u, y = (b >> (8 * lane)) & 255u;
    out |= ((x + y) % 256u) << (8 * lane);
  }
  return out;
}

// Runs add_bytes on the device over (a, b) and returns the mismatching words.
long long run_set(const char *name, const std::vector<uint32_t> &a,
                  const std::vector<uint32_t> &b) {
  const size_t words = a.size(), bytes = words * sizeof(uint32_t);
  uint32_t *d_a, *d_b, *d_sum;
  LP_CUDA(cudaMalloc(&d_a, bytes));
  LP_CUDA(cudaMalloc(&d_b, bytes));
  LP_CUDA(cudaMalloc(&d_sum, bytes));
  LP_CUDA(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
  LP_CUDA(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
  LP_CUDA(cudaMemset(d_sum, 0xA5, bytes));
  add_words<<<(unsigned)((words + 255) / 256), 256>>>(d_a, d_b, d_sum, words);
  LP_CUDA_LAUNCH();
  std::vector<uint32_t> sum(words);
  LP_CUDA(cudaMemcpy(sum.data(), d_sum, bytes, cudaMemcpyDeviceToHost));
  LP_CUDA(cudaFree(d_a));
  LP_CUDA(cudaFree(d_b));
  LP_CUDA(cudaFree(d_sum));
  long long bad = 0;
  for (size_t i = 0; i < words; ++i) {
    const uint32_t want = reference(a[i], b[i]);
    if (sum[i] == want) continue;
    if (bad < 4)
      std::fprintf(stderr, "%s: word %zu: %08x + %08x = %08x, expected %08x\n", name, i,
                   a[i], b[i], sum[i], want);
    ++bad;
  }
  std::printf("%-10s %9zu words, %lld mismatching\n", name, words, bad);
  return bad;
}

} // namespace

int main() {
  long long bad = 0;
  {
    const uint32_t multiplier[4] = {1u, 40503u, 25033u, 52429u};
    const uint32_t offset[4] = {0u, 12345u, 777u, 65535u};
    std::vector<uint32_t> a(65536), b(65536);
    std::vector<std::vector<bool>> seen(4, std::vector<bool>(65536, false));
    for (uint32_t i = 0; i < 65536; ++i)
      for (int lane = 0; lane < 4; ++lane) {
        const uint32_t pair = (i * multiplier[lane] + offset[lane]) & 0xffffu;
        seen[lane][pair] = true;
        a[i] |= (pair & 255u) << (8 * lane);
        b[i] |= (pair >> 8) << (8 * lane);
      }
    for (int lane = 0; lane < 4; ++lane)
      for (bool covered : seen[lane])
        if (!covered) {
          std::fprintf(stderr, "lane %d does not cover every byte pair\n", lane);
          return 1;
        }
    bad += run_set("every-pair", a, b);
  }
  {
    const uint8_t edges[7] = {0x00, 0x01, 0x7f, 0x80, 0x81, 0xfe, 0xff};
    std::vector<uint32_t> words;
    for (int i = 0; i < 7 * 7 * 7 * 7; ++i) {
      int k = i;
      uint32_t w = 0;
      for (int lane = 0; lane < 4; ++lane, k /= 7) w |= (uint32_t)edges[k % 7] << (8 * lane);
      words.push_back(w);
    }
    std::vector<uint32_t> a, b;
    for (uint32_t x : words)
      for (uint32_t y : words) {
        a.push_back(x);
        b.push_back(y);
      }
    bad += run_set("edge-mix", a, b);
  }
  {
    uint64_t state = 20261030;
    std::vector<uint32_t> a(1 << 20), b(1 << 20);
    for (size_t i = 0; i < a.size(); ++i) {
      const uint64_t r = splitmix64(state);
      a[i] = (uint32_t)r;
      b[i] = (uint32_t)(r >> 32);
    }
    bad += run_set("random", a, b);
  }
  std::printf("%s: %lld mismatching words\n", bad ? "FAIL" : "PASS", bad);
  return bad ? 1 : 0;
}

#include "oracle.h"

#include <algorithm>
#include <thread>

namespace vecadd_oracle {

void record_output(const uint8_t *fixture, uint64_t record, uint8_t *slot) {
  const int shift = (int)(record % 256);
  for (int k = 0; k < 64; ++k) {
    const int left = (fixture[k] + shift) % 256;
    const int right = (fixture[64 + k] + shift) % 256;
    slot[k] = (uint8_t)((left + right) % 256);
  }
}

std::vector<uint8_t> all_outputs(const uint8_t *fixture, uint64_t records) {
  std::vector<uint8_t> out(records * 64);
  const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
  const uint64_t workers = std::min<uint64_t>(std::min<uint64_t>(hardware, 8), records);
  const uint64_t span = (records + workers - 1) / workers;
  std::vector<std::thread> threads;
  for (uint64_t w = 0; w < workers; ++w) {
    const uint64_t begin = w * span, end = std::min(records, begin + span);
    threads.emplace_back([&, begin, end] {
      for (uint64_t r = begin; r < end; ++r)
        record_output(fixture, r, out.data() + r * 64);
    });
  }
  for (auto &t : threads) t.join();
  return out;
}

} // namespace vecadd_oracle

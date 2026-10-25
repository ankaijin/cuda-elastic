// Independent CPU oracle; see oracle.h.
#include "oracle.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>

namespace attention_oracle {
namespace {

// Sensor sample at a flat ordinal (token * 3 + lane) for phase 0 (audio) or
// 1 (IMU): a value in (-1, 1) on a 1/128 grid, times the variant scale. The
// mixing is done in 32-bit unsigned arithmetic, as upstream does.
double sample(uint32_t ordinal, uint32_t phase, double scale) {
  const uint32_t mixed = (ordinal * 37u + phase * 101u) % 251u;
  return (static_cast<double>(static_cast<int>(mixed) - 125) / 128.0) * scale;
}

struct Feature {
  double lane[4];
};

// Audio token t: query (x, y, z, x + y).
Feature query_feature(long long token, double scale) {
  const uint32_t base = static_cast<uint32_t>(token * 3);
  const double x = sample(base + 0, 0, scale);
  const double y = sample(base + 1, 0, scale);
  const double z = sample(base + 2, 0, scale);
  return {{x, y, z, x + y}};
}

// IMU token t: key and value (x, y, z, x - y).
Feature context_feature(long long token, double scale) {
  const uint32_t base = static_cast<uint32_t>(token * 3);
  const double x = sample(base + 0, 1, scale);
  const double y = sample(base + 1, 1, scale);
  const double z = sample(base + 2, 1, scale);
  return {{x, y, z, x - y}};
}

double half_dot(const Feature &q, const Feature &k) {
  double dot = 0.0;
  for (int lane = 0; lane < 4; ++lane) dot += q.lane[lane] * k.lane[lane];
  return 0.5 * dot;
}

} // namespace

void compute(long long query_tokens, long long context_tokens, double scale,
             Reference &reference) {
  std::vector<Feature> context(static_cast<size_t>(context_tokens));
  for (long long c = 0; c < context_tokens; ++c)
    context[c] = context_feature(c, scale);

  // attention[t][lane], filled by the worker threads (each row by one thread).
  std::vector<double> attention(static_cast<size_t>(query_tokens) * 4);
  std::atomic<long long> next_row{0};
  auto worker = [&]() {
    for (;;) {
      const long long row = next_row.fetch_add(1);
      if (row >= query_tokens) return;
      const Feature q = query_feature(row, scale);
      double maximum = half_dot(q, context[0]);
      for (long long c = 1; c < context_tokens; ++c)
        maximum = std::max(maximum, half_dot(q, context[c]));
      double denominator = 0.0;
      double numerator[4] = {0.0, 0.0, 0.0, 0.0};
      for (long long c = 0; c < context_tokens; ++c) {
        const Feature &k = context[c];
        const double weight = std::exp(half_dot(q, k) - maximum);
        denominator += weight;
        for (int lane = 0; lane < 4; ++lane) numerator[lane] += weight * k.lane[lane];
      }
      for (int lane = 0; lane < 4; ++lane)
        attention[row * 4 + lane] = numerator[lane] / denominator;
    }
  };
  const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
  const unsigned count = static_cast<unsigned>(
      std::min<long long>(hardware, std::max<long long>(1, query_tokens)));
  std::vector<std::thread> threads;
  for (unsigned i = 0; i < count; ++i) threads.emplace_back(worker);
  for (std::thread &thread : threads) thread.join();

  // Residual and statistics, sequentially in token-major order.
  reference.output.assign(static_cast<size_t>(query_tokens) * 4, 0.0);
  double energy = 0.0, sum = 0.0, weighted = 0.0, largest = 0.0;
  for (long long t = 0; t < query_tokens; ++t) {
    const Feature q = query_feature(t, scale);
    for (int lane = 0; lane < 4; ++lane) {
      const double a = attention[t * 4 + lane];
      energy += q.lane[lane] * q.lane[lane];
      sum += a;
      weighted += static_cast<double>(4 * t + lane + 1) * a;
      largest = std::max(largest, std::fabs(a));
      reference.output[t * 4 + lane] = a + q.lane[lane] * 0.125;
    }
  }
  reference.statistics[0] = energy;
  reference.statistics[1] = sum;
  reference.statistics[2] = weighted;
  reference.statistics[3] = largest;
  reference.combined = weighted + 3.0 * energy + 7.0 * sum + 11.0 * largest;
}

} // namespace attention_oracle

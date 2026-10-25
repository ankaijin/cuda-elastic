// Independent CPU oracle for the llama2c decoder. See oracle.h and contract.json.
#include "oracle.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>

namespace decoder_oracle {
namespace {

// Synthetic model, from its definition: token t embeds as 1 everywhere except
// 4 at position t; the RMSNorm weight is 1; classifier row r holds a single 1
// at column (r + vocab - 1) mod vocab.
double embedding_value(long long token, long long column) {
  return column == token ? 4.0 : 1.0;
}
double norm_weight_value(long long) { return 1.0; }
double classifier_value(long long row, long long column, long long vocab) {
  return column == (row + vocab - 1) % vocab ? 1.0 : 0.0;
}

// Tolerances for the dense comparison (see contract.json "tolerances").
constexpr double kNormalizedRelative = 2.0e-6;
constexpr double kLogitRelative = 2.0e-6;
constexpr double kProbabilityRelative = 2.0e-5;
constexpr double kAbsoluteFloor = 1.0e-12;
constexpr int kMaxReported = 8;

// Dense double evaluation of one sequence, every step.
void evaluate_sequence(const Shape &shape, long long sequence, double *normalized,
                       double *logits, double *probabilities, bool *consistent) {
  const long long W = shape.width, V = shape.vocab;
  std::vector<double> activation(W);
  long long token = prompt_of(sequence, V);
  for (long long step = 0; step < shape.steps; ++step) {
    double *o = normalized + step * W;
    double *l = logits + step * V;
    double *p = probabilities + step * V;
    double square_sum = 0.0;
    for (long long j = 0; j < W; ++j) {
      activation[j] = embedding_value(token, j);
      square_sum += activation[j] * activation[j];
    }
    const double scale = 1.0 / std::sqrt(square_sum / (double)W + 1.0e-5);
    for (long long j = 0; j < W; ++j)
      o[j] = norm_weight_value(j) * (scale * activation[j]);
    // Dense classifier: every row against every column.
    for (long long r = 0; r < V; ++r) {
      double sum = 0.0;
      for (long long j = 0; j < W; ++j) sum += classifier_value(r, j, V) * o[j];
      l[r] = sum;
    }
    double maximum = l[0];
    for (long long r = 1; r < V; ++r) maximum = std::max(maximum, l[r]);
    double total = 0.0;
    for (long long r = 0; r < V; ++r) {
      p[r] = std::exp(l[r] - maximum);
      total += p[r];
    }
    long long best = 0;
    for (long long r = 0; r < V; ++r) {
      p[r] /= total;
      if (p[r] > p[best]) best = r;
    }
    // The dense path must reproduce the analytic progression t -> t + 1.
    if (best != (token + 1) % V) *consistent = false;
    token = best;
  }
}

bool within(double observed, double expected, double relative) {
  const double bound = relative * std::fabs(expected) + kAbsoluteFloor;
  // Written so that NaN (the poison value) fails.
  return std::fabs(observed - expected) <= bound;
}

} // namespace

long long prompt_of(long long sequence, long long vocab) {
  return sequence % vocab;
}

Reference compute(const Shape &shape) {
  const long long B = shape.batch, W = shape.width, V = shape.vocab,
                  S = shape.steps;
  Reference ref;
  ref.tokens.resize((size_t)(B * S));
  for (long long b = 0; b < B; ++b) {
    long long token = prompt_of(b, V);
    for (long long s = 0; s < S; ++s) {
      token = (token + 1) % V;
      ref.tokens[(size_t)(b * S + s)] = (int32_t)token;
    }
  }

  // Up to 8 sequences spread evenly over the batch, both ends included.
  const long long count = std::min<long long>(8, B);
  for (long long i = 0; i < count; ++i) {
    const long long b = count == 1 ? 0 : i * (B - 1) / (count - 1);
    if (ref.sampled.empty() || ref.sampled.back() != b) ref.sampled.push_back(b);
  }
  const size_t n = ref.sampled.size();
  ref.normalized.resize(n * S * W);
  ref.logits.resize(n * S * V);
  ref.probabilities.resize(n * S * V);
  std::vector<char> consistent(n, 1);
  std::vector<std::thread> workers;
  for (size_t i = 0; i < n; ++i)
    workers.emplace_back([&, i] {
      bool ok = true;
      evaluate_sequence(shape, ref.sampled[i], &ref.normalized[i * S * W],
                        &ref.logits[i * S * V], &ref.probabilities[i * S * V],
                        &ok);
      consistent[i] = ok;
    });
  for (auto &worker : workers) worker.join();
  for (char ok : consistent) ref.self_consistent = ref.self_consistent && ok;
  if (!ref.self_consistent)
    std::fprintf(stderr, "decoder oracle: dense evaluation disagrees with the "
                         "analytic token progression\n");
  return ref;
}

bool check(const Shape &shape, const Reference &ref, const Observed &obs) {
  const long long B = shape.batch, W = shape.width, V = shape.vocab,
                  S = shape.steps;
  bool pass = ref.self_consistent;

  // 1. Every generated token.
  long long token_errors = 0;
  for (long long b = 0; b < B; ++b)
    for (long long s = 0; s < S; ++s) {
      const size_t at = (size_t)(b * S + s);
      if (obs.tokens[at] != ref.tokens[at]) {
        if (token_errors < kMaxReported)
          std::fprintf(stderr,
                       "decoder: token mismatch sequence %lld step %lld: "
                       "expected %d, actual %d\n",
                       b, s, ref.tokens[at], obs.tokens[at]);
        ++token_errors;
      }
    }

  // 2. Upstream's normalization invariants, every sequence and every step:
  //    float mean of squares (sum in double) within 2e-5 + 1.2e-7 W of 1,
  //    float probability sum (sum in double) within 1e-6 + 1.2e-7 V of 1.
  const float square_bound = 2.0e-5f + 1.2e-7f * (float)W;
  const float probability_bound = 1.0e-6f + 1.2e-7f * (float)V;
  long long invariant_errors = 0;
  float worst_square = 0.0f, worst_probability = 0.0f;
  for (long long s = 0; s < S; ++s)
    for (long long b = 0; b < B; ++b) {
      const float *o = obs.normalized + (size_t)((s * B + b) * W);
      const float *p = obs.probabilities + (size_t)((s * B + b) * V);
      double square_sum = 0.0;
      for (long long j = 0; j < W; ++j) square_sum += (double)o[j] * (double)o[j];
      const float square_mean = (float)(square_sum / (double)W);
      double probability_sum = 0.0;
      for (long long r = 0; r < V; ++r) probability_sum += (double)p[r];
      const float square_error = std::fabs(square_mean - 1.0f);
      const float probability_error = std::fabs((float)probability_sum - 1.0f);
      const bool square_ok = square_error <= square_bound;
      const bool probability_ok = probability_error <= probability_bound;
      if (square_ok) worst_square = std::max(worst_square, square_error);
      if (probability_ok)
        worst_probability = std::max(worst_probability, probability_error);
      if (!square_ok || !probability_ok) {
        if (invariant_errors < kMaxReported)
          std::fprintf(stderr,
                       "decoder: invariant failure sequence %lld step %lld: "
                       "mean square expected 1 +- %.3g, actual %.9g; "
                       "probability sum expected 1 +- %.3g, actual %.9g\n",
                       b, s, square_bound, square_mean, probability_bound,
                       (float)probability_sum);
        ++invariant_errors;
      }
    }

  // 3. Dense double evaluation of the sampled sequences, every step.
  long long dense_errors = 0;
  double worst[3] = {0.0, 0.0, 0.0};
  auto compare = [&](const char *what, int kind, long long b, long long s,
                     long long index, float observed, double expected,
                     double relative) {
    if (within(observed, expected, relative)) {
      const double error = std::fabs(observed - expected) /
                           std::max(std::fabs(expected), kAbsoluteFloor);
      worst[kind] = std::max(worst[kind], error);
      return;
    }
    if (dense_errors < kMaxReported)
      std::fprintf(stderr,
                   "decoder: %s mismatch sequence %lld step %lld index %lld: "
                   "expected %.9g, actual %.9g\n",
                   what, b, s, index, expected, (double)observed);
    ++dense_errors;
  };
  for (size_t i = 0; i < ref.sampled.size(); ++i) {
    const long long b = ref.sampled[i];
    for (long long s = 0; s < S; ++s) {
      const double *eo = &ref.normalized[(i * S + s) * W];
      const double *el = &ref.logits[(i * S + s) * V];
      const double *ep = &ref.probabilities[(i * S + s) * V];
      const float *o = obs.normalized + (size_t)((s * B + b) * W);
      const float *l = obs.logits + (size_t)((s * B + b) * V);
      const float *p = obs.probabilities + (size_t)((s * B + b) * V);
      for (long long j = 0; j < W; ++j)
        compare("normalized", 0, b, s, j, o[j], eo[j], kNormalizedRelative);
      for (long long r = 0; r < V; ++r) {
        compare("logit", 1, b, s, r, l[r], el[r], kLogitRelative);
        compare("probability", 2, b, s, r, p[r], ep[r], kProbabilityRelative);
      }
    }
  }

  pass = pass && token_errors == 0 && invariant_errors == 0 && dense_errors == 0;
  std::fprintf(stderr,
               "decoder: %s: %lld/%lld tokens wrong; %lld/%lld (sequence, step) "
               "invariant failures (worst passing |mean square - 1| %.3g, "
               "|probability sum - 1| %.3g); dense check of %zu sequences x "
               "%lld steps: %lld mismatches (worst passing relative error "
               "normalized %.3g, logits %.3g, probabilities %.3g)\n",
               pass ? "PASS" : "FAIL", token_errors, B * S, invariant_errors,
               B * S, (double)worst_square, (double)worst_probability,
               ref.sampled.size(), S, dense_errors, worst[0], worst[1], worst[2]);
  return pass;
}

} // namespace decoder_oracle

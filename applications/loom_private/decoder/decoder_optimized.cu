// loom-private/llama2c-decoder, optimized variant: the same contract as
// decoder.cu (batched greedy decoding through llama2.c's final rmsnorm,
// classifier matmul, softmax, and argmax on a synthetic model).
//
// The classifier matmul is ~99% of the baseline's GPU time. This variant
// replaces it with a register-tiled SGEMM (16 x 8 outputs per thread, cp.async
// multi-stage shared-memory pipeline, float4 shared loads) and schedules the
// tiles with a data-parallel + stream-K hybrid so that the last wave fills
// every SM. Every logit is still accumulated by one thread in increasing
// column order from 0.0f with a separate multiply and add, exactly as
// llama2.c's matmul and the baseline do, so all outputs (and the digest) are
// bitwise identical to the baseline's. rmsnorm and softmax+argmax are the
// baseline kernels. See README.md ("Optimized variant").
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

constexpr int kRowThreads = 256; // rmsnorm and softmax: one block per sequence

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

// llama2.c rmsnorm on the embedding row of each sequence's current token
// (unchanged from the baseline; see decoder.cu for the exactness argument).
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

// ---------------------------------------------------------------------------
// Classifier: logits[b][r] = sum over k = 0..width-1 of
// classifier[r][k] * normalized[b][k], llama2.c's matmul for every sequence.
//
// A block computes a kBM (sequences) x kBN (classifier rows) tile; each warp
// owns 4 kTM x 64 outputs and each thread kTM x 8 of them: sequences
// 4 lm + {0..3} + 16 q (q < kTM / 4) and rows 4 ln + {0..3, 32..35} of its
// warp's tile, for lane = 8 lm + ln. Both operands are K-contiguous in global
// memory; 4-byte cp.async copies transpose kBK columns into k-major shared
// arrays ([k][line], padded by 4 floats so that the copies of a warp hit 32
// different banks), and each thread reads its sequences and rows of one
// column with conflict-free float4 loads. A kStages-deep pipeline keeps
// the copies in flight behind the arithmetic. Every output's adds happen in
// increasing k, one product at a time, starting from 0.0f: the baseline's
// (and llama2.c's) order, so logits are bitwise the baseline's.
//
// Edges: columns past width are copied as zeros. A padded column adds the
// product +0 * +0 = +0 to an accumulator, which leaves it unchanged (an
// accumulator that starts at +0.0f and only receives round-to-nearest
// additions is never -0), so the result is bitwise the same as stopping at
// width. Lines past batch or vocab only feed outputs that are never stored.
template <int WarpsM, int WarpsN, int TM, int BK, int Stages, int MinBlocks>
struct TileConfig {
  static constexpr int kThreads = 32 * WarpsM * WarpsN;
  static constexpr int kWarpsN = WarpsN;
  static constexpr int kTM = TM; // outputs per thread: kTM sequences x 8 rows
  static constexpr int kBM = 4 * TM * WarpsM, kBN = 64 * WarpsN, kBK = BK;
  static constexpr int kLdA = kBM + 4, kLdW = kBN + 4; // padded shared lines
  static constexpr int kStageFloats = BK * (kLdA + kLdW);
  static constexpr int kStages = Stages, kMinBlocks = MinBlocks;
  static constexpr size_t kSmemBytes = sizeof(float) * (size_t)Stages * kStageFloats;
  static_assert(kThreads % BK == 0 && (kBM * BK) % kThreads == 0 &&
                    (kBN * BK) % kThreads == 0 && TM % 4 == 0 && Stages >= 2,
                "bad tile configuration");
};

// Tile schedule (computed on the host for each shape and GPU). Tiles are
// numbered with the sequence tile fastest, so blocks running at the same time
// share classifier rows in L2. Block c first computes tiles c, c + grid, ...
// (dp_waves of them, data parallel); the k-stages of the remaining tiles are
// then split into `grid` equal contiguous ranges (stream-K), so every block
// does the same amount of work. See classifier_kernel for the ordering.
struct GemmPlan {
  int tiles_m;              // sequence tiles
  int k_tiles;              // kBK-column stages per tile
  int dp_waves;             // data-parallel tiles per block
  long long sk_first_tile;  // first stream-K tile (dp_waves * grid)
  long long sk_iterations;  // stream-K tiles * k_tiles (0: none)
};

__device__ __forceinline__ void cp_async4(float *shared, const float *global,
                                          int bytes) {
  const unsigned address = (unsigned)__cvta_generic_to_shared(shared);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(address),
               "l"(global), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N> __device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// One column of the thread's operands: sequences am + {0..3} + 16 q and rows
// wn + {0..3, 32..35} of column k of a stage.
template <class C> struct Fragment {
  float4 a[C::kTM / 4], w[2];
};

template <class C>
__device__ __forceinline__ void load_fragment(Fragment<C> &f, const float *stage,
                                              int k, int am, int wn) {
  const float *sa = stage + k * C::kLdA + am;
  const float *sw = stage + C::kBK * C::kLdA + k * C::kLdW + wn;
#pragma unroll
  for (int q = 0; q < C::kTM / 4; ++q)
    f.a[q] = *reinterpret_cast<const float4 *>(sa + 16 * q);
  f.w[0] = *reinterpret_cast<const float4 *>(sw);
  f.w[1] = *reinterpret_cast<const float4 *>(sw + 32);
}

// Adds one column's products to the thread's outputs: a separate multiply
// and add per output, as in llama2.c (--fmad=false keeps them unfused).
template <class C>
__device__ __forceinline__ void multiply_add(float (&acc)[C::kTM][8],
                                             const Fragment<C> &f) {
  const float w[8] = {f.w[0].x, f.w[0].y, f.w[0].z, f.w[0].w,
                      f.w[1].x, f.w[1].y, f.w[1].z, f.w[1].w};
#pragma unroll
  for (int i = 0; i < C::kTM; ++i) {
    const float4 &q = f.a[i / 4];
    const float a = i % 4 == 0 ? q.x : i % 4 == 1 ? q.y : i % 4 == 2 ? q.z : q.w;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float product = w[j] * a;
      acc[i][j] = acc[i][j] + product;
    }
  }
}

// Thread's line offsets inside a tile: output (i, j) is sequence line
// am + (i & 3) + 16 * (i >> 2) and row line wn + (j & 3) + 32 * (j >> 2).
template <class C> __device__ __forceinline__ int thread_am() {
  return (threadIdx.x / 32 / C::kWarpsN) * 4 * C::kTM + 4 * (threadIdx.x % 32 / 8);
}
template <class C> __device__ __forceinline__ int thread_wn() {
  return (threadIdx.x / 32 % C::kWarpsN) * 64 + 4 * (threadIdx.x % 8);
}

// Copies of one operand (kLines lines x kBK columns per stage) for one tile.
// Lane l of a warp copies column 8 g + l % 8 (g < kBK / 8) of line
// l / 8 + 4 warp + p * kLinesPerPass (p < kPasses): every copy instruction
// of a warp reads four 32-byte runs and its 32 transposed stores land in 32
// different banks. Lines past batch or vocab are clamped to line 0 (their
// products only feed outputs that are never stored); columns past width are
// zero-filled. Offsets are 32-bit: the device-memory limit in make_decoder
// keeps vocab * width and batch * width below 2^31.
template <class C, int kLines> struct OperandCopy {
  static constexpr int kLinesPerPass = C::kThreads / 8;
  static constexpr int kPasses = kLines / kLinesPerPass;
  static_assert(C::kThreads % 32 == 0 && C::kBK % 8 == 0 &&
                    kLines % kLinesPerPass == 0,
                "bad copy shape");
  const float *base;
  unsigned offset[kPasses]; // clamped line * width + column
  int line, col;

  __device__ __forceinline__ OperandCopy(const float *matrix, int first_line,
                                         int lines, int width) {
    base = matrix;
    line = threadIdx.x / 8;
    col = threadIdx.x % 8;
#pragma unroll
    for (int p = 0; p < kPasses; ++p) {
      const int row = first_line + line + p * kLinesPerPass;
      offset[p] = (unsigned)(row < lines ? row : 0) * (unsigned)width + col;
    }
  }

  // Copies columns [k0, k0 + kBK) into dst, a [k][kLd] operand array.
  template <int kLd>
  __device__ __forceinline__ void copy(float *dst, int k0, int width) const {
    dst += col * kLd + line;
    if (k0 + C::kBK <= width) {
      const float *start = base + k0;
#pragma unroll
      for (int p = 0; p < kPasses; ++p)
#pragma unroll
        for (int g = 0; g < C::kBK / 8; ++g)
          cp_async4(dst + p * kLinesPerPass + 8 * g * kLd, start + offset[p] + 8 * g, 4);
    } else {
#pragma unroll
      for (int p = 0; p < kPasses; ++p)
#pragma unroll
        for (int g = 0; g < C::kBK / 8; ++g) {
          const bool inside = k0 + col + 8 * g < width;
          cp_async4(dst + p * kLinesPerPass + 8 * g * kLd,
                    base + offset[p] + (inside ? k0 + 8 * g : 0), inside ? 4 : 0);
        }
    }
  }
};

// Accumulates k-stages [kt_begin, kt_end) of tile `tile` into acc through a
// kStages-deep cp.async pipeline, then drains it so the next call may reuse
// shared memory. Fragments are double-buffered in registers: column k + 1 is
// read from shared memory while column k is multiplied, and the barrier that
// publishes the next stage sits before the last column of the current one.
template <class C>
__device__ __forceinline__ void accumulate_tile(
    float *smem, float (&acc)[C::kTM][8], const float *normalized,
    const float *classifier, int batch, int width, int vocab,
    const GemmPlan &plan, long long tile, int kt_begin, int kt_end) {
#if DECODER_OPT_MUTANT == 2 // test mutant: drop the last stage of every tile
  if (kt_end == plan.k_tiles && kt_end - kt_begin > 1) --kt_end;
#endif
  const int k_first = kt_begin * C::kBK;
  const OperandCopy<C, C::kBM> a(normalized, (int)(tile % plan.tiles_m) * C::kBM,
                                 batch, width);
  const OperandCopy<C, C::kBN> w(classifier, (int)(tile / plan.tiles_m) * C::kBN,
                                 vocab, width);
  const int am = thread_am<C>(), wn = thread_wn<C>();
  const int count = kt_end - kt_begin;
  auto load_a = [&](int slot, int s) {
    a.template copy<C::kLdA>(smem + slot * C::kStageFloats, k_first + s * C::kBK, width);
  };
  auto load_w = [&](int slot, int s) {
    w.template copy<C::kLdW>(smem + slot * C::kStageFloats + C::kBK * C::kLdA,
                             k_first + s * C::kBK, width);
  };
#pragma unroll
  for (int s = 0; s < C::kStages - 1; ++s) {
    if (s < count) {
      load_a(s, s);
      load_w(s, s);
    }
    cp_async_commit();
  }
  cp_async_wait<C::kStages - 2>();
  __syncthreads();
  Fragment<C> f[2];
  load_fragment<C>(f[0], smem, 0, am, wn);
  int read = 0, write = C::kStages - 1;
  for (int it = 0; it < count; ++it) {
    // Every thread passed the barrier after its last read of slot `write`.
    // The copies of the stage kStages - 1 ahead are split around the
    // arithmetic: sequence lines first, classifier lines before the last two
    // columns. The column loop is unrolled twice only, which keeps the
    // kernel inside the instruction cache.
    const bool prefetch = it + C::kStages - 1 < count;
    const float *stage = smem + read * C::kStageFloats;
    if (prefetch) load_a(write, it + C::kStages - 1);
#pragma unroll 2
    for (int k = 0; k < C::kBK - 2; k += 2) {
      load_fragment<C>(f[1], stage, k + 1, am, wn);
      multiply_add<C>(acc, f[0]);
      load_fragment<C>(f[0], stage, k + 2, am, wn);
      multiply_add<C>(acc, f[1]);
    }
    if (prefetch) load_w(write, it + C::kStages - 1);
    cp_async_commit();
    load_fragment<C>(f[1], stage, C::kBK - 1, am, wn);
    multiply_add<C>(acc, f[0]);
    if (it + 1 < count) {
      cp_async_wait<C::kStages - 2>();
      __syncthreads();
      read = read == C::kStages - 1 ? 0 : read + 1;
      load_fragment<C>(f[0], smem + read * C::kStageFloats, 0, am, wn);
    }
    multiply_add<C>(acc, f[1]);
    write = write == C::kStages - 1 ? 0 : write + 1;
  }
  cp_async_wait<0>();
  __syncthreads();
}

// Stream-K blocks hand a split tile's running sums to the next block through
// `partial` (one kTM x 8 slice per thread, [block][element][thread]) and the
// block's `ready` flag. A block's work units run in this order:
//   1. its data-parallel tiles;
//   2. if its stream-K range ends inside a tile, that tile's first stages,
//      published to partial[block] and ready[block] (the head);
//   3. the whole tiles of its range;
//   4. if its range starts inside a tile, that tile's remaining stages,
//      started from partial[block - 1] once ready[block - 1] is set (the tail).
// Each split output's sum thus runs in exactly the order one block would use,
// and the previous block published its head before its other work, so the
// wait is short. The host sizes the grid so that every block is resident.
template <class C>
__global__ void __launch_bounds__(C::kThreads, C::kMinBlocks)
classifier_kernel(float *logits, const float *normalized,
                  const float *classifier, float *partial, int *ready,
                  int batch, int width, int vocab, GemmPlan plan) {
  extern __shared__ __align__(16) float smem[];
  const int block = blockIdx.x, grid = gridDim.x, K = plan.k_tiles;

  // Stream-K range of this block, in k-stages from the first stream-K tile.
  long long head_tile = 0, first_whole = 0, end_whole = 0, tail_tile = 0;
  int head_end = 0, tail_begin = 0;
  if (plan.sk_iterations > 0) {
    const long long begin = plan.sk_iterations * block / grid;
    const long long end = plan.sk_iterations * (block + 1) / grid;
    tail_tile = begin / K;
    tail_begin = (int)(begin - tail_tile * K); // > 0: range starts mid-tile
    head_tile = (end - 1) / K;
    head_end = (int)(end - head_tile * K);     // < K: range ends mid-tile
    first_whole = tail_begin > 0 ? tail_tile + 1 : tail_tile;
    end_whole = head_end < K ? head_tile : head_tile + 1;
  }
  const bool has_head = plan.sk_iterations > 0 && head_end < K;
  const bool has_tail = plan.sk_iterations > 0 && tail_begin > 0;
  const long long units = plan.dp_waves + (has_head ? 1 : 0) +
                          (end_whole - first_whole) + (has_tail ? 1 : 0);
  constexpr int kSlice = C::kTM * 8 * C::kThreads;
  const int am = thread_am<C>(), wn = thread_wn<C>();

  float acc[C::kTM][8];
  for (long long u = 0; u < units; ++u) {
    long long tile;
    int kt_begin = 0, kt_end = K;
    bool publish = false, resume = false;
    if (u < plan.dp_waves) {
      tile = u * grid + block;
    } else if (has_head && u == plan.dp_waves) {
      tile = plan.sk_first_tile + head_tile;
      kt_end = head_end;
      publish = true;
    } else if (has_tail && u == units - 1) {
      tile = plan.sk_first_tile + tail_tile;
      kt_begin = tail_begin;
      resume = true;
    } else {
      tile = plan.sk_first_tile + first_whole +
             (u - plan.dp_waves - (has_head ? 1 : 0));
    }

    if (resume) {
      if (threadIdx.x == 0) {
        volatile int *flag = ready + block - 1;
        while (*flag == 0) __nanosleep(256);
        __threadfence();
      }
      __syncthreads();
      const float *previous = partial + (size_t)(block - 1) * kSlice + threadIdx.x;
#pragma unroll
      for (int i = 0; i < C::kTM; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#if DECODER_OPT_MUTANT == 1 // test mutant: drop the previous block's sums
          acc[i][j] = 0.0f * __ldcg(previous + (i * 8 + j) * C::kThreads);
#else
          acc[i][j] = __ldcg(previous + (i * 8 + j) * C::kThreads);
#endif
    } else {
#pragma unroll
      for (int i = 0; i < C::kTM; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;
    }

    accumulate_tile<C>(smem, acc, normalized, classifier, batch, width, vocab,
                       plan, tile, kt_begin, kt_end);

    if (publish) {
      float *mine = partial + (size_t)block * kSlice + threadIdx.x;
#pragma unroll
      for (int i = 0; i < C::kTM; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) __stcg(mine + (i * 8 + j) * C::kThreads, acc[i][j]);
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0) atomicExch(ready + block, 1);
      continue;
    }
    const int m = (int)(tile % plan.tiles_m) * C::kBM + am;
    const int n = (int)(tile / plan.tiles_m) * C::kBN + wn;
#pragma unroll
    for (int i = 0; i < C::kTM; ++i) {
      const int sequence = m + (i & 3) + 16 * (i >> 2);
      if (sequence >= batch) continue;
      float *out = logits + (size_t)sequence * vocab;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int row = n + (j & 3) + 32 * (j >> 2);
        if (row < vocab) out[row] = acc[i][j];
      }
    }
  }
}

using ClassifierFn = void (*)(float *, const float *, const float *, float *,
                              int *, int, int, int, GemmPlan);

// One tile configuration.
struct TileVariant {
  const char *name;
  int bm, bn, bk, threads;
  size_t smem_bytes;
  ClassifierFn kernel;
};

template <class C> TileVariant make_variant(const char *name) {
  return {name, C::kBM, C::kBN, C::kBK, C::kThreads, C::kSmemBytes,
          classifier_kernel<C>};
}

// Tile configurations (README "Optimized variant" explains the choice per
// GPU). Names are sequences x rows x kBK columns per stage. Every one is
// compiled into every build, so the tests run each of them on any GPU
// (DECODER_OPT_TILE).
const std::vector<TileVariant> &tile_variants() {
  static const std::vector<TileVariant> variants = {
      // 16 x 8 outputs per thread, 8 warps, one block per SM.
      make_variant<TileConfig<2, 4, 16, 16, 3, 1>>("128x256x16"),
      make_variant<TileConfig<2, 4, 16, 8, 4, 1>>("128x256x8"),
      // 8 x 8 outputs per thread, 8 warps, two blocks per SM.
      make_variant<TileConfig<4, 2, 8, 8, 4, 2>>("128x128x8"),
  };
  return variants;
}

// llama2.c softmax (max-subtracted expf, then divide by the sum) followed by
// sample_argmax (first maximum), unchanged from the baseline.
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

// Reads an optional integer environment override (tests only); exits 2 on a
// malformed or out-of-range value.
long long env_int(const char *name, long long fallback, long long lo, long long hi) {
  const char *text = std::getenv(name);
  if (!text || !*text) return fallback;
  char *end = nullptr;
  errno = 0;
  const long long value = std::strtoll(text, &end, 10);
  if (errno || *end || value < lo || value > hi)
    lp::fail(lp::kExitUsage, "%s must be an integer in [%lld, %lld] (got '%s')",
             name, lo, hi, text);
  return value;
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
                    (void *)d_logits_, (void *)d_probabilities_,
                    (void *)d_partial_, (void *)d_ready_})
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
    plan_classifier();

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
    for (long long s = 0; s < S_; ++s) {
      float *normalized = d_normalized_ + (size_t)(s * B_ * W_);
      float *logits = d_logits_ + (size_t)(s * B_ * V_);
      float *probabilities = d_probabilities_ + (size_t)(s * B_ * V_);
      rmsnorm_kernel<<<(unsigned)B_, kRowThreads, 0, stream>>>(
          normalized, d_tokens_, d_embedding_, d_norm_weight_, (int)W_, (int)V_);
      LP_CUDA_LAUNCH();
      if (plan_.sk_iterations > 0)
        LP_CUDA(cudaMemsetAsync(d_ready_, 0, (size_t)grid_ * sizeof(int), stream));
      classifier_<<<(unsigned)grid_, (unsigned)tile_->threads,
                    tile_->smem_bytes, stream>>>(
          logits, normalized, d_classifier_, d_partial_, d_ready_, (int)B_,
          (int)W_, (int)V_, plan_);
      LP_CUDA_LAUNCH();
      softmax_argmax_kernel<<<(unsigned)B_, kRowThreads, 0, stream>>>(
          probabilities, logits, d_tokens_, d_generated_, (int)V_, (int)S_,
          (int)s);
      LP_CUDA_LAUNCH();
    }
  }

  void poison_outputs(cudaStream_t stream) override {
    // 0xFF bytes: int32 -1 (an invalid token) and float NaN. The prompts are
    // an input, poisoned too, so an invocation that skips their upload
    // starts from invalid tokens and fails the check. The stream-K partial
    // sums are intermediates and are poisoned as well.
    LP_CUDA(cudaMemsetAsync(d_prompts_, 0xFF, (size_t)B_ * sizeof(int), stream));
    LP_CUDA(cudaMemsetAsync(d_tokens_, 0xFF, (size_t)B_ * sizeof(int), stream));
    LP_CUDA(cudaMemsetAsync(d_generated_, 0xFF, generated_count() * sizeof(int), stream));
    LP_CUDA(cudaMemsetAsync(d_normalized_, 0xFF, normalized_count() * sizeof(float), stream));
    LP_CUDA(cudaMemsetAsync(d_logits_, 0xFF, row_count() * sizeof(float), stream));
    LP_CUDA(cudaMemsetAsync(d_probabilities_, 0xFF, row_count() * sizeof(float), stream));
    if (d_partial_)
      LP_CUDA(cudaMemsetAsync(d_partial_, 0xFF, partial_count_ * sizeof(float), stream));
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

  // Per-GPU tile configuration and schedule (README "Optimized variant").
  // DECODER_OPT_TILE (a configuration name), DECODER_OPT_SCHEDULE (dp or
  // streamk) and DECODER_OPT_CTAS (the stream-K grid) are test overrides.
  void plan_classifier() {
    int device = 0;
    LP_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop;
    LP_CUDA(cudaGetDeviceProperties(&prop, device));
    const int arch = prop.major * 10 + prop.minor;
    // 128 x 256 tiles with 16 x 8 outputs per thread were fastest on all
    // three GPUs; the RTX 5090 (sm_120) prefers 8-column stages (four of
    // them in flight), the RTX 3090 and 4090 16-column stages (three).
    std::string name = arch == 120 ? "128x256x8" : "128x256x16";
    bool streamk = true;
    if (const char *forced = std::getenv("DECODER_OPT_TILE"); forced && *forced)
      name = forced;
    if (const char *forced = std::getenv("DECODER_OPT_SCHEDULE"); forced && *forced) {
      if (std::strcmp(forced, "dp") && std::strcmp(forced, "streamk"))
        lp::fail(lp::kExitUsage, "DECODER_OPT_SCHEDULE must be dp or streamk (got '%s')", forced);
      streamk = std::strcmp(forced, "streamk") == 0;
    }
    for (const TileVariant &v : tile_variants())
      if (name == v.name) tile_ = &v;
    if (!tile_)
      lp::fail(lp::kExitUsage, "DECODER_OPT_TILE '%s' is not a tile configuration", name.c_str());

    classifier_ = tile_->kernel;
    LP_CUDA(cudaFuncSetAttribute(classifier_, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)tile_->smem_bytes));
    int per_sm = 0;
    LP_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, classifier_, tile_->threads, tile_->smem_bytes));
    if (per_sm < 1)
      lp::fail(lp::kExitUsage, "tile configuration %s does not fit on this GPU", tile_->name);
    const long long resident = (long long)per_sm * prop.multiProcessorCount;
    const long long ctas = env_int("DECODER_OPT_CTAS", resident, 1, resident);

    const long long tiles_m = (B_ + tile_->bm - 1) / tile_->bm;
    const long long tiles_n = (V_ + tile_->bn - 1) / tile_->bn;
    const long long tiles = tiles_m * tiles_n;
    const long long k_tiles = (W_ + tile_->bk - 1) / tile_->bk;
    plan_ = {(int)tiles_m, (int)k_tiles, 1, 0, 0};
    grid_ = tiles;
    if (streamk && tiles > ctas) {
      // Whole waves stay data parallel; the last one to two waves' worth of
      // tiles (at least `ctas`, so a split tile spans at most two blocks) is
      // shared evenly by stream-K.
      const long long waves = tiles % ctas == 0 ? tiles / ctas : tiles / ctas - 1;
      grid_ = ctas;
      plan_.dp_waves = (int)waves;
      plan_.sk_first_tile = waves * ctas;
      plan_.sk_iterations = (tiles - waves * ctas) * k_tiles;
    }
    if (plan_.sk_iterations > 0) {
      partial_count_ = (size_t)grid_ * tile_->bm * tile_->bn;
      LP_CUDA(cudaMalloc(&d_partial_, partial_count_ * sizeof(float)));
      LP_CUDA(cudaMalloc(&d_ready_, (size_t)grid_ * sizeof(int)));
    }
    if (std::getenv("DECODER_OPT_VERBOSE"))
      std::fprintf(stderr,
                   "decoder optimized: sm_%d, %d SMs, tile %s, %d blocks/SM, "
                   "%lld tiles x %lld stages, grid %lld, %d data-parallel "
                   "waves, stream-K tiles %lld\n",
                   arch, prop.multiProcessorCount, tile_->name, per_sm, tiles,
                   k_tiles, grid_,
                   plan_.dp_waves, plan_.sk_iterations / k_tiles);
  }

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
  float *d_partial_ = nullptr;
  int *d_ready_ = nullptr;
  size_t partial_count_ = 0;
  const TileVariant *tile_ = nullptr;
  ClassifierFn classifier_ = nullptr;
  GemmPlan plan_{};
  long long grid_ = 0;
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
  // The rmsnorm square sum is width + 15; keeping it below 2^24 keeps every
  // partial sum an exact float integer, so the tree order matches llama2.c.
  if (width < 2 || width > (1LL << 24) - 16)
    lp::fail(lp::kExitUsage,
             "width must be in [2, 16777200] so the rmsnorm square sum "
             "(width + 15) stays an exact float integer (got %lld)",
             width);
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

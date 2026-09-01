#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';    \
            return 1;                                                      \
        }                                                                  \
    } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    constexpr int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_a(n, 1.5f);
    std::vector<float> h_b(n, 2.5f);
    std::vector<float> h_c(n);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    constexpr int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        if (std::fabs(h_c[i] - 4.0f) > 1e-5f) {
            std::cerr << "Verification failed at index " << i << '\n';
            return 1;
        }
    }

    std::cout << "Success: " << n << " values added on the GPU\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
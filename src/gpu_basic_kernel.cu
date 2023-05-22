#pragma once
#include "cuda_runtime.h"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
namespace gpu_basic_test {

template <std::size_t numthreads>
__global__ void basic_kernel(std::size_t const* a,
                             std::size_t const* b,
                             std::size_t        n,
                             std::size_t*       r) {
    auto t = threadIdx.x + blockIdx.x * numthreads;
    if (t < n) {
        r[t] = a[t] + b[t];
    }
}

void run() {
    constexpr std::size_t n        = 1024;
    constexpr std::size_t nthreads = 32;
    constexpr std::size_t gpu_id   = 0;

    thrust::host_vector<std::size_t> a(n, 1);
    thrust::host_vector<std::size_t> b(n, 2);

    thrust::device_vector<std::size_t> a_gpu = a;
    thrust::device_vector<std::size_t> b_gpu = b;
    thrust::device_vector<std::size_t> r_gpu(n, 0);

    cudaStream_t stream;
    cudaSetDevice(gpu_id);
    cudaStreamCreate(&stream);

    cudaSetDevice(gpu_id);
    basic_kernel<nthreads><<<n / nthreads, nthreads, 0, stream>>>(
        thrust::raw_pointer_cast(&a_gpu[0]),
        thrust::raw_pointer_cast(&b_gpu[0]),
        n,
        thrust::raw_pointer_cast(&r_gpu[0]));

    cudaSetDevice(gpu_id);
    cudaStreamSynchronize(stream);
    thrust::host_vector<std::size_t> r = r_gpu;

    bool succes = true;
    for (std::size_t i = 0; i < n; ++i) {
        if (r[i] != 3) {
            succes = false;
            break;
        }
    }
    if (succes) {
        std::cout << "  SUCCES" << std::endl;
    } else {
        std::cout << "  FAILURE" << std::endl;
    }

    cudaSetDevice(gpu_id);
    cudaStreamSynchronize(stream);
    cudaSetDevice(gpu_id);
    cudaStreamDestroy(stream);
}

}    // namespace gpu_basic_test

#pragma once
#include "cuda_runtime.h"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace relaxed_constexpr_test {

template <std::size_t I = 0, typename... Ts>
__device__ inline void add(std::tuple<Ts const*...> a,
                           std::tuple<Ts const*...> b,
                           std::size_t              i,
                           std::tuple<Ts*...>       r) {
    if constexpr (I < sizeof...(Ts)) {
        std::get<I>(r)[i] = std::get<I>(a)[i] + std::get<I>(b)[i];
        add<I + 1, Ts...>(a, b, i, r);
    }
}

template <std::size_t numthreads, typename... Ts>
__global__ void basic_kernel(std::tuple<Ts const*...> a,
                             std::tuple<Ts const*...> b,
                             std::size_t              n,
                             std::tuple<Ts*...>       r) {
    auto t = threadIdx.x + blockIdx.x * numthreads;
    if (t < n) {
        add(a, b, t, r);
    }
}

void run() {
    constexpr std::size_t n        = 1024;
    constexpr std::size_t nthreads = 32;
    constexpr std::size_t gpu_id   = 0;

    thrust::host_vector<std::size_t> as(n, 1);
    thrust::host_vector<std::size_t> bs(n, 2);

    thrust::device_vector<std::size_t> as_gpu = as;
    thrust::device_vector<std::size_t> bs_gpu = bs;
    thrust::device_vector<std::size_t> rs_gpu(n, 0);

    thrust::host_vector<float> af(n, 1.0);
    thrust::host_vector<float> bf(n, 2.0);

    thrust::device_vector<float> af_gpu = af;
    thrust::device_vector<float> bf_gpu = bf;
    thrust::device_vector<float> rf_gpu(n, 0.0);

    cudaStream_t stream;
    cudaSetDevice(gpu_id);
    cudaStreamCreate(&stream);

    cudaSetDevice(gpu_id);
    basic_kernel<nthreads, std::size_t, float>
        <<<n / nthreads, nthreads, 0, stream>>>(
            {thrust::raw_pointer_cast(&as_gpu[0]),
             thrust::raw_pointer_cast(&af_gpu[0])},
            {thrust::raw_pointer_cast(&bs_gpu[0]),
             thrust::raw_pointer_cast(&bf_gpu[0])},
            n,
            {thrust::raw_pointer_cast(&rs_gpu[0]),
             thrust::raw_pointer_cast(&rf_gpu[0])});

    cudaSetDevice(gpu_id);
    cudaStreamSynchronize(stream);
    thrust::host_vector<std::size_t> rs = rs_gpu;
    thrust::host_vector<float>       rf = rf_gpu;

    bool succes = true;
    for (std::size_t i = 0; i < n; ++i) {
        if (rs[i] != 3) {
            succes = false;
            break;
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (fabs(rf[i] - 3.0) >= 1e-5) {
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

}    // namespace relaxed_constexpr_test

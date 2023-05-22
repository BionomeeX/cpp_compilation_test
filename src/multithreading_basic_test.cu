#pragma once
#include <iostream>
#include <thread>
#include <vector>
namespace multithreading_basic_test {

void run() {

    constexpr std::size_t n = 32;

    std::vector<std::thread> th;
    th.reserve(n);

    thrust::host_vector<std::size_t> a(n, 1);
    thrust::host_vector<std::size_t> b(n, 2);
    thrust::host_vector<std::size_t> r(n, 0);

    for (std::size_t i = 0; i < n; ++i) {
        th.emplace_back([&a, &b, &r, i]() { r[i] = a[i] + b[i]; });
    }
    for (std::size_t i = 0; i < n; ++i) {
        th[i].join();
    }

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
}
}    // namespace multithreading_basic_test

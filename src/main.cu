#include "cuda_runtime.h"
#include "filesystem_test.cu"
#include "gpu_basic_kernel.cu"
#include "multithreading_basic_test.cu"
#include "relaxed_constexpr_test.cu"
#include "structured_binding_basic_test.cu"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <tuple>

int main(int argc, char** argv) {
    // use GPU with stream & thrust
    {
        std::cout << "START GPU BASIC TEST" << std::endl;
        gpu_basic_test::run();
        std::cout << "STOP GPU BASIC TEST" << std::endl;
    }
    // use multithreading and lambda
    {
        std::cout << "START MULTITHREADING BASIC TEST" << std::endl;
        multithreading_basic_test::run();
        std::cout << "STOP MULTITHREADING BASIC TEST" << std::endl;
    }
    // use auto & unpacking
    {
        std::cout << "START STRUCTURED BINDING BASIC TEST" << std::endl;
        structured_binding_basic_test::run();
        std::cout << "STOP STRUCTURED BINDING BASIC TEST" << std::endl;
    }
    // use filesystem
    {
        std::cout << "START FILESYSTEM TEST" << std::endl;
        filesystem_test::run();
        std::cout << "STOP FILESYSTEM TEST" << std::endl;
    }
    // use relaxed constexpr on GPU
    {
        std::cout << "START RELAXED CONSTEXPR TEST" << std::endl;
        relaxed_constexpr_test::run();
        std::cout << "STOP RELAXED CONSTEXPR TEST" << std::endl;
    }
    // use GPU async copy
    // use multiple GPU with streams

    return 0;
}

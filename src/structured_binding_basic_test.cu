#pragma once
#include <iostream>

namespace structured_binding_basic_test {

void run() {

    auto [a, b] = []() { return std::pair<std::size_t, std::size_t>{1, 2}; }();

    bool succes = true;
    if (a != 1 || b != 2) {
        succes = false;
    }
    if (succes) {
        std::cout << "  SUCCES" << std::endl;
    } else {
        std::cout << "  FAILURE" << std::endl;
    }
}

}    // namespace structured_binding_basic_test

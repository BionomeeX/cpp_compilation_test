#pragma once
#include <filesystem>
#include <iostream>

namespace filesystem_test {
void run() {
    if (std::filesystem::exists("test")) {
        std::filesystem::remove_all("test");
    }
    std::filesystem::create_directory("test");
    if (std::filesystem::exists("test")) {
        std::cout << "  SUCCES" << std::endl;
    } else {
        std::cout << "  FAILLURE" << std::endl;
    }
    if (std::filesystem::exists("test")) {
        std::filesystem::remove_all("test");
    }
}
}    // namespace filesystem_test

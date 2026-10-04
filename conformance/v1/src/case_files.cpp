// SPDX-License-Identifier: MIT
#include <parallel_mater_conformance/case_registry.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace {

std::string read_file(const std::filesystem::path &path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

bool write_file(const std::filesystem::path &path, const std::string &contents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    return output.good();
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 3 || (std::string(argv[1]) != "--write" &&
                      std::string(argv[1]) != "--check")) {
        std::cerr << "usage: parallel-mater-conformance-cases "
                     "--write|--check <cases-directory>\n";
        return 2;
    }
    const bool write = std::string(argv[1]) == "--write";
    const std::filesystem::path directory = argv[2];
    if (write) std::filesystem::create_directories(directory);
    bool valid = true;
    for (const auto &definition : parallel_mater::conformance::case_registry()) {
        const auto path = parallel_mater::conformance::case_file_path(
            directory, definition);
        const std::string expected =
            parallel_mater::conformance::serialize_case(definition);
        if (write) {
            if (!write_file(path, expected)) {
                std::cerr << "failed to write " << path << '\n';
                valid = false;
            }
        } else if (!std::filesystem::exists(path) ||
                   read_file(path) != expected) {
            std::cerr << "non-canonical conformance input: " << path << '\n';
            valid = false;
        }
    }
    return valid ? 0 : 1;
}

// SPDX-License-Identifier: MIT
#include <parallel_mater_conformance/runner.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Arguments {
    std::string contract;
    std::filesystem::path root;
    std::filesystem::path output;
};

Arguments parse(int argc, char **argv) {
    Arguments result{};
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--contract" && index + 1 < argc)
            result.contract = argv[++index];
        else if (argument == "--root" && index + 1 < argc)
            result.root = argv[++index];
        else if (argument == "--output" && index + 1 < argc)
            result.output = argv[++index];
        else
            throw std::runtime_error("unknown or incomplete argument: " + argument);
    }
    if (result.contract.empty() || result.root.empty() || result.output.empty())
        throw std::runtime_error(
            "usage: pm-android-device-contracts --contract "
            "rigid-quality|fluid-quality|adverse-lifecycle --root DIR --output FILE");
    return result;
}

void write(const std::filesystem::path &path, const std::string &contents) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!stream) throw std::runtime_error("cannot write " + path.string());
}

} // namespace

int main(int argc, char **argv) {
    try {
        const auto arguments = parse(argc, argv);
        if (arguments.contract == "rigid-quality") {
            write(arguments.output,
                  parallel_mater::conformance::conformance_rigid_brick_quality(
                      arguments.root));
        } else if (arguments.contract == "fluid-quality") {
            const auto temporary = arguments.output.string() + ".temporary";
            write(arguments.output,
                  parallel_mater::conformance::conformance_fluid_pair_quality(
                      temporary));
            std::filesystem::remove(temporary);
        } else if (arguments.contract == "adverse-lifecycle") {
            write(arguments.output,
                  parallel_mater::conformance::conformance_adverse_lifecycle(
                      arguments.root));
        } else {
            throw std::runtime_error("unknown contract: " + arguments.contract);
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "android qualification: " << error.what() << '\n';
        return 2;
    }
}

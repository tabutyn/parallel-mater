// SPDX-License-Identifier: MIT
#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

int parallel_mater_fluid_pair_quality_fixture(int argc, char **argv);

#ifndef PM_CONFORMANCE_DRIVER_NAME
#define PM_CONFORMANCE_DRIVER_NAME "selected-loadable-driver"
#endif

namespace parallel_mater::conformance {
namespace {
std::string read_file(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read fixture output " + path.string());
    return {std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
}

} // namespace

std::string conformance_fluid_pair_quality(
    const std::filesystem::path &temporary_output) {
    std::filesystem::create_directories(temporary_output.parent_path());
    std::string output = temporary_output.string();
    char program[] = "mobile-quality";
    char output_option[] = "--output";
    char driver_option[] = "--driver-name";
    char driver_name[] = PM_CONFORMANCE_DRIVER_NAME;
    char *arguments[]{program, output_option, output.data(), driver_option,
                      driver_name};
    const int status = parallel_mater_fluid_pair_quality_fixture(5, arguments);
    if (status == 0) return read_file(temporary_output);
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = "core-conservation-stability-v1";
    result["profile_sha256"] =
        "a59a81c71b5ddf9ae1f0a47cd8e61e2074c24d929b15318852863b2c92cd9326";
    result["execution_status"] = status == 77 ? "unsupported" : "failed";
    result["reason"] = status == 77
        ? "The selected driver does not advertise separate-system equal-fluid coupling."
        : "The public-API equal-fluid quality fixture failed.";
    result["fixtures"] = Json::array();
    return result.serialize();
}

} // namespace parallel_mater::conformance

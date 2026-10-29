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

std::string fluid_quality_failure(const char *profile_id,
                                  const char *profile_sha256,
                                  int status) {
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-quality-result/v1";
    result["profile_id"] = profile_id;
    result["profile_sha256"] = profile_sha256;
    result["execution_status"] = status == 77 ? "unsupported" : "failed";
    result["reason"] = status == 77
        ? "The selected driver does not advertise the required fluid interaction."
        : "The public-API fluid quality fixture failed.";
    result["fixtures"] = Json::array();
    return result.serialize();
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
    return fluid_quality_failure(
        "core-conservation-stability-v2", PM_QUALITY_PROFILE_SHA256, status);
}

std::string conformance_fluid_self_quality(
    const std::filesystem::path &temporary_output,
    const std::string &profile_id,
    ExecutionStatus *execution_status) {
    const char *option{};
    const char *profile_sha256{};
    if (profile_id == "fluid-self-contact-v1") {
        option = "--self-contact";
        profile_sha256 = PM_FLUID_SELF_CONTACT_PROFILE_SHA256;
    } else if (profile_id == "fluid-self-pressure-v1") {
        option = "--self-pressure";
        profile_sha256 = PM_FLUID_SELF_PRESSURE_PROFILE_SHA256;
    } else if (profile_id == "fluid-self-viscosity-v1") {
        option = "--self-viscosity";
        profile_sha256 = PM_FLUID_SELF_VISCOSITY_PROFILE_SHA256;
    } else {
        throw std::runtime_error("unknown fluid self quality profile: " +
                                 profile_id);
    }

    std::filesystem::create_directories(temporary_output.parent_path());
    std::string output = temporary_output.string();
    char program[] = "mobile-quality";
    char output_option[] = "--output";
    char driver_option[] = "--driver-name";
    char driver_name[] = PM_CONFORMANCE_DRIVER_NAME;
    char *arguments[]{program, output_option, output.data(), driver_option,
                      driver_name, const_cast<char *>(option)};
    const int status = parallel_mater_fluid_pair_quality_fixture(6, arguments);
    if (execution_status != nullptr)
        *execution_status = status == 0 ? ExecutionStatus::completed
            : status == 77 ? ExecutionStatus::unsupported
                           : ExecutionStatus::failed;
    if (status == 0) return read_file(temporary_output);
    return fluid_quality_failure(profile_id.c_str(), profile_sha256, status);
}

} // namespace parallel_mater::conformance

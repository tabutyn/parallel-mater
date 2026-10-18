// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>
#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>
#include "../../../tests/api/lifecycle_fixture.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

int parallel_mater_resource_lifecycle_fixture();
int parallel_mater_fluid_pair_quality_fixture(int argc, char **argv);

namespace parallel_mater::conformance {
namespace {
using namespace parallel_mater::gallery;

Json driver_json(PmPhysicalDevice physical, const char *name = "slang-metal") {
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(physical, &properties);
    Json result = Json::object();
    result["name"] = name;
    result["device"] = properties.deviceName;
    return result;
}

Json selected_driver_json() {
    PmInstanceCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create.apiVersion = PM_API_VERSION_0_2;
    PmInstance instance{};
    if (pmCreateInstance(&create, nullptr, &instance) != PM_SUCCESS)
        throw std::runtime_error("create lifecycle provenance instance");
    std::uint32_t count{};
    if (pmEnumeratePhysicalDevices(instance, &count, nullptr) != PM_SUCCESS || count == 0U) {
        pmDestroyInstance(instance);
        throw std::runtime_error("no lifecycle provenance device");
    }
    std::vector<PmPhysicalDevice> devices(count);
    if (pmEnumeratePhysicalDevices(instance, &count, devices.data()) != PM_SUCCESS) {
        pmDestroyInstance(instance);
        throw std::runtime_error("enumerate lifecycle provenance device");
    }
    Json result = driver_json(devices.front());
    pmDestroyInstance(instance);
    return result;
}

std::string read_file(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot read fixture output " + path.string());
    return {std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
}

Json missing_lifecycle_case(std::string id, Json families, Json unrelated,
                            int status, const char *fixture) {
    Json attempt = Json::object();
    attempt["fixture"] = fixture;
    attempt["status"] = status == 0 ? "passed-console-only"
        : status == 77 ? "unsupported" : "failed";
    attempt["reason"] = status == 0
        ? "Public-API operations completed; this fixture does not yet emit the profile's checkpoint fingerprints."
        : status == 77 ? "Required advertised features are unavailable."
                       : "Public-API fixture failed.";
    Json result = Json::object();
    result["attempt"] = std::move(attempt);
    result["execution_status"] = status == 77 ? "unsupported" : "missing";
    result["families"] = std::move(families);
    result["id"] = std::move(id);
    result["observations"] = Json::array();
    result["unrelated_resources"] = std::move(unrelated);
    return result;
}

Json capacity_lifecycle_case(
    const test::CapacityLifecycleResult &fixture, int stress_status) {
    Json families = Json::array();
    families.push_back("liquid");
    Json unrelated = Json::array();
    unrelated.push_back("unrelated-fluid");
    if (fixture.status != 0 || stress_status != 0) {
        const int status = fixture.status == 77 || stress_status == 77 ? 77 : 2;
        return missing_lifecycle_case(
            "capacity-replacement-recovery", std::move(families),
            std::move(unrelated), status, "capacity-recovery");
    }

    Json observations = Json::array();
    for (const auto &checkpoint : fixture.observations) {
        Json resources = Json::array();
        for (const auto &resource : checkpoint.resources) {
            Json value = Json::object();
            value["id"] = resource.id;
            value["active"] = resource.active;
            value["generation"] = resource.generation;
            value["state_sha256"] = resource.state_sha256;
            resources.push_back(std::move(value));
        }
        Json value = Json::object();
        value["id"] = checkpoint.id;
        value["outcome"] = checkpoint.outcome;
        value["resources"] = std::move(resources);
        observations.push_back(std::move(value));
    }
    Json attempt = Json::object();
    attempt["fixture"] = "capacity-recovery";
    attempt["stress_fixture"] = "resource-lifecycle";
    attempt["status"] = "completed";
    Json result = Json::object();
    result["attempt"] = std::move(attempt);
    result["execution_status"] = "completed";
    result["families"] = std::move(families);
    result["id"] = "capacity-replacement-recovery";
    result["observations"] = std::move(observations);
    result["unrelated_resources"] = std::move(unrelated);
    return result;
}

Json graph_lifecycle_case(const test::GraphLifecycleResult &fixture) {
    Json families = Json::array();
    for (const char *family : {"liquid", "rope", "soft-body"})
        families.push_back(family);
    Json unrelated = Json::array();
    for (const char *family : {"smoke", "cloth"}) unrelated.push_back(family);
    if (fixture.status != 0)
        return missing_lifecycle_case(
            "three-family-partial-rollback", std::move(families),
            std::move(unrelated), fixture.status, "interaction-graph-lifecycle");

    Json observations = Json::array();
    for (const auto &checkpoint : fixture.observations) {
        Json resources = Json::array();
        for (const auto &resource : checkpoint.resources) {
            Json value = Json::object();
            value["id"] = resource.id;
            value["active"] = resource.active;
            value["generation"] = resource.generation;
            value["state_sha256"] = resource.state_sha256;
            resources.push_back(std::move(value));
        }
        Json value = Json::object();
        value["id"] = checkpoint.id;
        value["outcome"] = checkpoint.outcome;
        value["active_graph_sha256"] = checkpoint.active_graph_sha256;
        value["resources"] = std::move(resources);
        observations.push_back(std::move(value));
    }
    Json attempt = Json::object();
    attempt["fixture"] = "interaction-graph-lifecycle";
    attempt["status"] = "completed";
    Json result = Json::object();
    result["attempt"] = std::move(attempt);
    result["execution_status"] = "completed";
    result["families"] = std::move(families);
    result["id"] = "three-family-partial-rollback";
    result["observations"] = std::move(observations);
    result["unrelated_resources"] = std::move(unrelated);
    return result;
}
} // namespace

std::string conformance_fluid_pair_quality(
    const std::filesystem::path &temporary_output) {
    std::filesystem::create_directories(temporary_output.parent_path());
    std::string output = temporary_output.string();
    char program[] = "ipad-quality";
    char output_option[] = "--output";
    char driver_option[] = "--driver-name";
    char driver_name[] = "slang-metal";
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

std::string conformance_adverse_lifecycle(const std::filesystem::path &root) {
    int capacity_stress = 2;
    test::CapacityLifecycleResult capacity;
    test::GraphLifecycleResult graph;
    try {
        capacity_stress = parallel_mater_resource_lifecycle_fixture();
        (void)parallel_mater_capacity_recovery_fixture(&capacity);
    } catch (const std::exception &) {
        capacity.status = 2;
        capacity_stress = 2;
    }
    try { (void)parallel_mater_interaction_graph_lifecycle_fixture(&graph); }
    catch (const std::exception &) { graph.status = 2; }
    Json cases = Json::array();
    cases.push_back(capacity_lifecycle_case(capacity, capacity_stress));
    cases.push_back(graph_lifecycle_case(graph));
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = "adverse-capacity-three-family-v1";
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/adverse-capacity-three-family-v1.json");
    result["execution_status"] = "completed";
    result["driver"] = selected_driver_json();
    result["cases"] = std::move(cases);
    return result.serialize();
}

} // namespace parallel_mater::conformance

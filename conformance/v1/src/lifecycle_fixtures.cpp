// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.h>
#include <parallel_mater_conformance/json.hpp>
#include <parallel_mater_conformance/runner.hpp>
#include <parallel_mater_conformance/sha256.hpp>

#include "../../../tests/api/lifecycle_fixture.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int parallel_mater_resource_lifecycle_fixture();

namespace parallel_mater::conformance {
namespace {

std::string driver_name(PmDevice device) {
    PmNativeDeviceInfo native{};
    const PmResult result = pmGetNativeDevice(device, &native);
    if (result == PM_ERROR_FEATURE_NOT_PRESENT) return "slang-cpu";
    if (result != PM_SUCCESS) return {};
    switch (native.backend) {
    case PM_NATIVE_BACKEND_CUDA: return "slang-cuda";
    case PM_NATIVE_BACKEND_D3D12: return "slang-d3d12";
    case PM_NATIVE_BACKEND_VULKAN: return "slang-vulkan";
    case PM_NATIVE_BACKEND_METAL: return "slang-metal";
    default: return {};
    }
}

Json selected_driver_json() {
    PmInstanceCreateInfo create{};
    create.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create.apiVersion = PM_API_VERSION_0_2;
    PmInstance instance{};
    if (pmCreateInstance(&create, nullptr, &instance) != PM_SUCCESS)
        throw std::runtime_error("create lifecycle provenance instance");
    std::uint32_t count{};
    if (pmEnumeratePhysicalDevices(instance, &count, nullptr) != PM_SUCCESS ||
        count == 0U) {
        pmDestroyInstance(instance);
        throw std::runtime_error("no lifecycle provenance device");
    }
    std::vector<PmPhysicalDevice> devices(count);
    if (pmEnumeratePhysicalDevices(instance, &count, devices.data()) != PM_SUCCESS) {
        pmDestroyInstance(instance);
        throw std::runtime_error("enumerate lifecycle provenance device");
    }
    const float priority = 1.0F;
    const PmDeviceQueueCreateInfo queue_create{0U, 1U, &priority};
    PmDeviceCreateInfo device_create{};
    device_create.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create.queueCreateInfoCount = 1U;
    device_create.pQueueInfos = &queue_create;
    PmDevice device{};
    if (pmCreateDevice(devices.front(), &device_create, &device) != PM_SUCCESS) {
        pmDestroyInstance(instance);
        throw std::runtime_error("create lifecycle provenance device");
    }
    PmPhysicalDeviceProperties properties{};
    pmGetPhysicalDeviceProperties(devices.front(), &properties);
    const std::string name = driver_name(device);
    Json result = Json::object();
    result["name"] = name;
    result["device"] = properties.deviceName;
    pmDestroyDevice(device);
    pmDestroyInstance(instance);
    if (name.empty())
        throw std::runtime_error("query lifecycle native backend");
    return result;
}

Json missing_lifecycle_case(std::string id, Json families, Json unrelated,
                            int status, const char *fixture) {
    Json attempt = Json::object();
    attempt["fixture"] = fixture;
    attempt["status"] = status == 77 ? "unsupported" : "failed";
    attempt["reason"] = status == 77
        ? "Required advertised features are unavailable."
        : "Public-API fixture failed.";
    Json result = Json::object();
    result["attempt"] = std::move(attempt);
    result["execution_status"] = status == 77 ? "unsupported" : "failed";
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

std::string conformance_adverse_lifecycle(const std::filesystem::path &root,
                                          ExecutionStatus *status) {
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
    try {
        (void)parallel_mater_interaction_graph_lifecycle_fixture(&graph);
    } catch (const std::exception &) {
        graph.status = 2;
    }
    Json cases = Json::array();
    cases.push_back(capacity_lifecycle_case(capacity, capacity_stress));
    cases.push_back(graph_lifecycle_case(graph));
    const int capacity_status = capacity.status != 0
        ? capacity.status : capacity_stress;
    const bool failed = (capacity_status != 0 && capacity_status != 77) ||
        (graph.status != 0 && graph.status != 77);
    const bool unsupported = capacity_status == 77 || graph.status == 77;
    const ExecutionStatus execution = failed ? ExecutionStatus::failed
        : unsupported ? ExecutionStatus::unsupported
                      : ExecutionStatus::completed;
    if (status != nullptr) *status = execution;
    Json result = Json::object();
    result["schema"] = "parallel-mater-conformance-lifecycle-result/v1";
    result["profile_id"] = "adverse-capacity-three-family-v1";
    result["profile_sha256"] = sha256_contract_file(
        root / "conformance/v1/lifecycle/adverse-capacity-three-family-v1.json");
    result["execution_status"] = execution == ExecutionStatus::completed
        ? "completed" : execution == ExecutionStatus::unsupported
        ? "unsupported" : "failed";
    result["driver"] = selected_driver_json();
    result["cases"] = std::move(cases);
    return result.serialize();
}

} // namespace parallel_mater::conformance

// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/device_profiles.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

#define CHECK(x) do { if (!(x)) { std::cerr << __LINE__ << ": " << #x << '\n'; std::abort(); } } while (false)

int main() {
    using namespace parallel_mater::gallery;
    DeviceProfileCatalog catalog;
    std::string error;
    CHECK(load_device_profiles(PM_PROFILE_CATALOG, catalog, error));
    const auto *pixel = select_workload_limits(catalog, "Android", "Pixel 7", "slang-cpu");
    CHECK(pixel && pixel->fluid_particles == 128 && pixel->brick_count == 4);
    CHECK(pixel->coupled_bodies == 4);
    CHECK(pixel->render_width == 640 && pixel->render_height == 360);
    const auto *fallback = select_workload_limits(catalog, "Android", "Unknown phone", "vulkan");
    CHECK(fallback && fallback->fluid_particles <= pixel->fluid_particles);
    CHECK(fallback->coupled_bodies == 2);
    CHECK(!select_workload_limits(catalog, "Linux", "Pixel 7", "slang-cpu"));
    const auto *ipad = select_workload_limits(catalog, "iPadOS", "iPad13,8", "metal");
    CHECK(ipad && ipad->fluid_particles == 256 && ipad->coupled_bodies == 4);
    CHECK(ipad->render_width == 960 && ipad->render_height == 720 && ipad->target_fps == 30);
    CHECK(!select_workload_limits(catalog, "iPadOS", "iPad13,8", "slang-cpu"));
    auto specific = *pixel;
    specific.name = "pixel-cpu-test";
    specific.backend = "slang-cpu";
    specific.fluid_particles = 32;
    catalog.startup_limits.push_back(specific);
    CHECK(select_workload_limits(catalog, "Android", "Pixel 7", "slang-cpu")->fluid_particles == 32);
    CHECK(select_workload_limits(catalog, "Android", "Pixel 7", "vulkan")->fluid_particles == 128);

    // Fake snapshots keep admission policy independent of any specific phone,
    // GPU API or machine. Unknown values can never be read as usable memory.
    PmMemoryBudget budget{};
    budget.heapCount = 1U;
    budget.budget[0] = 1'000U;
    budget.usage[0] = 100U;
    auto admission = assess_workload_admission(budget, 1U);
    CHECK(!admission.accepted());
    CHECK(admission.status == WorkloadAdmissionStatus::budget_unknown);

    budget.confidence[0] = PM_MEMORY_BUDGET_CONFIDENCE_IMPLEMENTATION_ESTIMATE;
    admission = assess_workload_admission(budget, 600U);
    CHECK(admission.accepted());
    CHECK(admission.available_bytes == 900U && admission.limit_bytes == 630U);
    CHECK(admission.confidence ==
          PM_MEMORY_BUDGET_CONFIDENCE_IMPLEMENTATION_ESTIMATE);

    // Borrowed ranges are not assumed to be included in heap usage. Reserve
    // all of them, even when that conservative subtraction denies the work.
    budget.borrowedBufferBytes = 100U;
    admission = assess_workload_admission(budget, 600U);
    CHECK(!admission.accepted());
    CHECK(admission.status == WorkloadAdmissionStatus::insufficient_budget);
    CHECK(admission.available_bytes == 800U && admission.limit_bytes == 560U);
    budget.borrowedBufferBytes = 2'000U;
    admission = assess_workload_admission(budget, 1U);
    CHECK(!admission.accepted() && admission.available_bytes == 0U);

    // A new live observation replaces the preceding decision. Falling budget
    // deterministically denies; a later recovery is evaluated from that new
    // snapshot rather than stale cached headroom.
    budget.borrowedBufferBytes = 0U;
    budget.confidence[0] = PM_MEMORY_BUDGET_CONFIDENCE_DRIVER_REPORTED;
    budget.budget[0] = 1'000U;
    budget.usage[0] = 100U;
    CHECK(assess_workload_admission(budget, 500U).accepted());
    budget.budget[0] = 700U;
    budget.usage[0] = 100U;
    CHECK(!assess_workload_admission(budget, 500U).accepted());
    budget.budget[0] = 1'000U;
    CHECK(assess_workload_admission(budget, 500U).accepted());

    budget.usage[0] = 1'001U;
    admission = assess_workload_admission(budget, 1U);
    CHECK(!admission.accepted());
    CHECK(admission.status == WorkloadAdmissionStatus::invalid_budget);
    budget.usage[0] = 0U;
    budget.confidence[0] = static_cast<PmMemoryBudgetConfidence>(99);
    CHECK(assess_workload_admission(budget, 1U).status ==
          WorkloadAdmissionStatus::invalid_budget);
    budget.heapCount = 17U;
    CHECK(assess_workload_admission(budget, 1U).status ==
          WorkloadAdmissionStatus::invalid_budget);

    budget = {};
    budget.heapCount = 1U;
    budget.budget[0] = std::numeric_limits<std::uint64_t>::max();
    budget.confidence[0] = PM_MEMORY_BUDGET_CONFIDENCE_OS_REPORTED;
    admission = assess_workload_admission(
        budget, std::numeric_limits<std::uint64_t>::max());
    CHECK(!admission.accepted());
    CHECK(admission.limit_bytes ==
          (std::numeric_limits<std::uint64_t>::max() / 10U) * 7U +
              ((std::numeric_limits<std::uint64_t>::max() % 10U) * 7U) / 10U);

    // Existing verification must not erase the newly added workload records.
    const auto path = std::filesystem::current_path() / "workload-profile-roundtrip.json";
    std::filesystem::copy_file(PM_PROFILE_CATALOG, path,
        std::filesystem::copy_options::overwrite_existing);
    VerifiedBrickProfile verified;
    verified.hardware.gpu_model = "test-device";
    verified.hardware.backend = "test-backend";
    verified.scene = {1, 1, 1};
    verified.solver_version = "test-solver";
    verified.build_revision = "test-build";
    verified.verified_at = "2026-10-15";
    verified.verifier = "automated fixture";
    verified.metrics = {180, 1, 2, 1, 2, 1, true, true};
    CHECK(save_verified_profile(path, verified, error));
    CHECK(load_device_profiles(path, catalog, error));
    CHECK(select_workload_limits(catalog, "Android", "Pixel 7", "slang-cpu")->fluid_particles == 128);
    CHECK(select_workload_limits(catalog, "Android", "Pixel 7", "slang-cpu")->coupled_bodies == 4);
    CHECK(catalog.measurements.size() >= 1);
    CHECK(select_workload_limits(catalog, "iPadOS", "iPad13,8", "metal")->smoke_particles == 128);

    for (const char *bad : {"-1", "0", "1.5", "4294967296"}) {
        std::ofstream out(path);
        out << "{\"schema_version\":1,\"startup_limits\":[{\"name\":\"bad\","
               "\"platform\":\"Android\",\"machine_model\":\"*\",\"backend\":\"*\","
               "\"brick_count\":4,\"coupled_bodies\":" << bad << ",\"fluid_particles\":128,\"smoke_particles\":128,"
               "\"dump_spheres\":10,\"render_width\":640,\"render_height\":360,\"target_fps\":30}]}";
        out.close();
        CHECK(!load_device_profiles(path, catalog, error));
    }
    std::filesystem::remove(path);
}

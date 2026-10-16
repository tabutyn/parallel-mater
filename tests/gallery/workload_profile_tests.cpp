// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/device_profiles.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

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
    auto specific = *pixel;
    specific.name = "pixel-cpu-test";
    specific.backend = "slang-cpu";
    specific.fluid_particles = 32;
    catalog.startup_limits.push_back(specific);
    CHECK(select_workload_limits(catalog, "Android", "Pixel 7", "slang-cpu")->fluid_particles == 32);
    CHECK(select_workload_limits(catalog, "Android", "Pixel 7", "vulkan")->fluid_particles == 128);

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

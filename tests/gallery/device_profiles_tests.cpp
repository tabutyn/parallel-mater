// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/device_profiles.hpp>
#include <parallel_mater_gallery/scene.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace {
int failures = 0;

void check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Status>
bool succeeded(Status status, const char *message) {
    if (status) return true;
    std::cerr << "FAIL: " << message;
    if (status.message != nullptr) std::cerr << ": " << status.message;
    std::cerr << '\n';
    ++failures;
    return false;
}
}

int main() {
    using namespace parallel_mater::gallery;
    using parallel_mater::RigidBodyState;
    using parallel_mater::WorldOptions;
    using parallel_mater::WorldStatistics;
#if defined(PARALLEL_MATER_GALLERY_METAL)
    using namespace parallel_mater::metal::gallery;
    using World = parallel_mater::metal::World;
#else
    using namespace parallel_mater::gallery;
    using World = parallel_mater::World;
#endif

    std::string error;
    check(validate_brick_config({1U, 0.5F, 1U}, error),
          "minimum brick config must be valid");
    check(validate_brick_config({4'096U, 2.0F, 16U}, error),
          "maximum brick config must be valid");
    check(!validate_brick_config({8U, 1.0F, 9U}, error),
          "wall count cannot exceed brick count");
    check(!validate_brick_config({0U, 1.0F, 1U}, error) &&
              !validate_brick_config({4'097U, 1.0F, 1U}, error) &&
              !validate_brick_config({8U, 0.49F, 1U}, error) &&
              !validate_brick_config({8U, 2.01F, 1U}, error) &&
              !validate_brick_config({32U, 1.0F, 17U}, error),
          "brick configuration limits must reject every out-of-range field");
    check(brick_calibration_presets().front() == BrickSceneConfig{1U, 2.0F, 1U} &&
              brick_calibration_presets().back() ==
                  BrickSceneConfig{3'840U, 0.5F, 5U},
          "calibration ladder must retain its version-one endpoints");
    check(estimated_metal_contact_bytes(400U) >
              estimated_metal_contact_bytes(200U),
          "contact memory estimate must grow with body count");

    DeviceProfileCatalog catalog;
    check(load_device_profiles(PARALLEL_MATER_DEVICE_PROFILES_PATH,
                               catalog, error),
          error.empty() ? "load device profile catalog" : error.c_str());
    check(catalog.hardware.size() >= 26U,
          "catalog must seed at least 26 hardware variants");
    check(catalog.verified_profiles.empty(),
          "repository catalog starts without invented verified capacities");

    SceneDefinition authored, generated;
    check(load_glb_scene(PARALLEL_MATER_RIGID_BODY_SCENE_PATH,
                         authored, error),
          error.empty() ? "load authored brick scene" : error.c_str());
    check(make_brick_scene(authored, {7U, 0.5F, 3U}, generated, error),
          error.empty() ? "generate odd multi-wall scene" : error.c_str());
    std::size_t bricks = 0U;
    std::set<int> wall_depths;
    std::map<int, std::vector<const RigidBodyDefinition *>> walls;
    float authored_margin = 0.0F;
    for (const auto &body : authored.rigid_bodies)
        if (body.source_name == "Layer1") {
            authored_margin = body.options.collision_margin;
            break;
        }
    for (const auto &body : generated.rigid_bodies) {
        if (body.source_name != "Layer1" && body.source_name != "Layer2") continue;
        ++bricks;
        const int wall_depth = static_cast<int>(std::lround(
            body.options.initial_state.position.z * 1'000.0F));
        wall_depths.insert(wall_depth);
        walls[wall_depth].push_back(&body);
        check(std::fabs(body.options.mass - 0.125F) < 1.0e-5F,
              "brick mass must scale by volume");
        check(std::fabs(body.options.collision_margin - authored_margin * 0.5F) <
                  1.0e-6F,
              "brick collision margin must scale with geometry");
        check(body.options.inertia_diagonal.x == 0.0F &&
                  body.options.inertia_diagonal.y == 0.0F &&
                  body.options.inertia_diagonal.z == 0.0F,
              "scaled brick inertia must be derived from scaled triangles");
        check(body.mesh_indices.size() == 1U &&
                  generated.meshes[body.mesh_indices.front()].indices.size() % 3U == 0U,
              "every generated brick must use triangle geometry");
    }
    check(bricks == 7U && generated.rigid_bodies.size() == 9U,
          "generator must place exact count and retain ball and floor");
    check(wall_depths.size() == 3U,
          "generator must distribute odd counts across every requested wall");
    for (const auto &[depth, bodies] : walls) {
        (void)depth;
        const float top = (*std::max_element(bodies.begin(), bodies.end(),
            [](const auto *left, const auto *right) {
                return left->options.initial_state.position.y <
                       right->options.initial_state.position.y;
            }))->options.initial_state.position.y;
        float top_center = 0.0F;
        unsigned top_count = 0U;
        for (const auto *body : bodies)
            if (std::fabs(body->options.initial_state.position.y - top) < 1.0e-6F) {
                top_center += body->options.initial_state.position.x;
                ++top_count;
            }
        check(top_count != 0U && std::fabs(top_center / top_count) < 1.0e-6F,
              "every partial top row must be centered");
    }
    if (wall_depths.size() == 3U) {
        auto depth = wall_depths.begin();
        const int first_depth = *depth++;
        const int second_depth = *depth;
        check(std::abs(second_depth - first_depth) > 2'000,
              "wall surfaces must leave at least one ball diameter of travel");
    }
    for (const std::uint32_t count : {1U, 17U, 384U, 4'096U}) {
        SceneDefinition exact;
        const BrickSceneConfig config{count, count == 4'096U ? 0.5F : 1.0F,
                                      std::min(count, 5U)};
        check(make_brick_scene(authored, config, exact, error),
              "generate exact requested brick count");
        check(std::count_if(exact.rigid_bodies.begin(), exact.rigid_bodies.end(),
                  [](const auto &body) {
                      return body.source_name == "Layer1" ||
                             body.source_name == "Layer2";
                  }) == count,
              "procedural scene must never round its brick count");
    }

    SceneDefinition physical_scene;
    check(make_brick_scene(authored, {24U, 1.0F, 2U}, physical_scene, error),
          error.empty() ? "generate physics regression scene" : error.c_str());
    WorldOptions world_options{};
    if (succeeded(scene_world_options(physical_scene, world_options),
                  "derive generated wall capacities")) {
        world_options.rigid_sleeping = true;
        World world;
        SceneInstance instance;
        if (succeeded(World::create(world_options, world),
                      "create generated wall world") &&
            succeeded(instantiate_scene(physical_scene, world, instance),
                      "instantiate generated wall")) {
            bool stepped = true;
            for (unsigned frame = 0U; frame < 120U && stepped; ++frame)
                stepped = succeeded(world.step({}), "settle generated wall");
            WorldStatistics quiet_statistics{};
            if (stepped && succeeded(world.collect_statistics(quiet_statistics),
                                     "collect generated wall sleep state")) {
                check(quiet_statistics.rigid_body_count == 26U,
                      "generated wall world must retain every body");
            }

            std::size_t ball = physical_scene.rigid_bodies.size();
            float quiet_maximum_displacement = 0.0F;
            float quiet_maximum_speed = 0.0F;
            for (std::size_t index = 0U;
                 index < physical_scene.rigid_bodies.size(); ++index) {
                RigidBodyState state{};
                if (!succeeded(world.read_rigid_body_state(
                                   instance.rigid_bodies[index], state),
                               "read quiet generated body")) {
                    stepped = false;
                    break;
                }
                check(std::isfinite(state.position.x) &&
                          std::isfinite(state.position.y) &&
                          std::isfinite(state.position.z),
                      "generated support state must remain finite");
                const auto initial =
                    physical_scene.rigid_bodies[index].options.initial_state;
                const auto &definition = physical_scene.rigid_bodies[index];
                if (definition.source_name == "Layer1" ||
                    definition.source_name == "Layer2") {
                    quiet_maximum_displacement = std::max(
                        quiet_maximum_displacement,
                        std::hypot(std::hypot(state.position.x - initial.position.x,
                                             state.position.y - initial.position.y),
                                   state.position.z - initial.position.z));
                    quiet_maximum_speed = std::max(quiet_maximum_speed, std::sqrt(
                        state.linear_velocity.x * state.linear_velocity.x +
                        state.linear_velocity.y * state.linear_velocity.y +
                        state.linear_velocity.z * state.linear_velocity.z));
                }
                if (definition.source_name == "Icosphere")
                    ball = index;
            }
            check(quiet_maximum_displacement < 0.03F &&
                      quiet_maximum_speed < 0.03F,
                  "generated multi-wall support must remain stable at rest");
            if (stepped && ball < physical_scene.rigid_bodies.size()) {
                RigidBodyState launch =
                    physical_scene.rigid_bodies[ball].options.initial_state;
                launch.position = {0.0F, 1.05F, 3.0F};
                launch.linear_velocity = {0.0F, 0.0F, -10.0F};
                stepped = succeeded(world.set_rigid_body_state(
                    instance.rigid_bodies[ball], launch),
                    "launch ball into generated walls");
            }
            for (unsigned frame = 0U; frame < 240U && stepped; ++frame) {
                stepped = succeeded(world.step({}), "impact generated walls");
            }
            unsigned displaced = 0U;
            float peak_speed = 0.0F;
            for (std::size_t index = 0U;
                 index < physical_scene.rigid_bodies.size() && stepped; ++index) {
                RigidBodyState state{};
                if (!succeeded(world.read_rigid_body_state(
                                   instance.rigid_bodies[index], state),
                               "read generated impact body")) {
                    stepped = false;
                    break;
                }
                peak_speed = std::max(peak_speed, std::sqrt(
                    state.linear_velocity.x * state.linear_velocity.x +
                    state.linear_velocity.y * state.linear_velocity.y +
                    state.linear_velocity.z * state.linear_velocity.z));
                const auto &body = physical_scene.rigid_bodies[index];
                if (body.source_name != "Layer1" && body.source_name != "Layer2")
                    continue;
                const float dx = state.position.x - body.options.initial_state.position.x;
                const float dz = state.position.z - body.options.initial_state.position.z;
                displaced += std::hypot(dx, dz) > 0.05F;
            }
            check(displaced >= 4U,
                  "ball impact must move bricks in the generated walls");
            check(std::isfinite(peak_speed) && peak_speed < 100.0F,
                  "generated wall impact must not produce explosive velocity");
        }
    }

    SceneDefinition sleeping_scene;
    check(make_brick_scene(authored, {2U, 1.0F, 1U}, sleeping_scene, error),
          error.empty() ? "generate sleep/wake scene" : error.c_str());
    WorldOptions sleeping_options{};
    if (succeeded(scene_world_options(sleeping_scene, sleeping_options),
                  "derive sleep/wake capacities")) {
        sleeping_options.rigid_sleeping = true;
        World world;
        SceneInstance instance;
        if (succeeded(World::create(sleeping_options, world),
                      "create sleep/wake world") &&
            succeeded(instantiate_scene(sleeping_scene, world, instance),
                      "instantiate sleep/wake scene")) {
            bool stepped = true;
            for (unsigned frame = 0U; frame < 240U && stepped; ++frame)
                stepped = succeeded(world.step({}), "settle sleeping brick");
            std::size_t brick = sleeping_scene.rigid_bodies.size();
            for (std::size_t index = 0U;
                 index < sleeping_scene.rigid_bodies.size(); ++index) {
                const auto &body = sleeping_scene.rigid_bodies[index];
                if (body.source_name == "Layer1" || body.source_name == "Layer2")
                    brick = index;
            }
            WorldStatistics before_wake{};
            if (stepped && succeeded(world.collect_statistics(before_wake),
                                     "collect sleeping brick state"))
                check(before_wake.rigid_body_count == 4U,
                      "sleep-enabled generated scene must retain every body");
            if (stepped && brick < sleeping_scene.rigid_bodies.size()) {
                RigidBodyState before{};
                stepped = succeeded(world.read_rigid_body_state(
                    instance.rigid_bodies[brick], before),
                    "read sleeping brick");
                if (stepped)
                    stepped = succeeded(world.apply_impulse(
                        instance.rigid_bodies[brick], {10.0F, 0.0F, 0.0F},
                        before.position), "wake generated brick");
                for (unsigned frame = 0U; frame < 12U && stepped; ++frame)
                    stepped = succeeded(world.step({}), "step woken brick");
                RigidBodyState after{};
                if (stepped && succeeded(world.read_rigid_body_state(
                        instance.rigid_bodies[brick], after),
                        "read woken brick"))
                    check(after.position.x - before.position.x > 0.02F,
                          "an impulse must wake and move a generated brick");
            }
        }
    }

    const std::vector<CalibrationSample> samples{
        {10.0, false, true}, {14.0, false, true},
        {12.0, true, true}, {14.5, true, true}};
    const CalibrationMetrics passing =
        summarize_calibration(samples, 180.0, true, false);
    check(calibration_passes(passing),
          "stable real-time quiet and collision samples must qualify");
    CalibrationMetrics failing = passing;
    failing.collision_maximum_milliseconds = 31.0;
    check(!calibration_passes(failing),
          "a collision frame above the hard budget must fail");
    auto dropped_samples = samples;
    dropped_samples.back().simulation_advanced = false;
    check(!calibration_passes(summarize_calibration(
              dropped_samples, 180.0, true, true)),
          "dropped simulation progress must fail calibration");
    check(!calibration_passes(summarize_calibration(
              samples, 180.0, false, false)),
          "an unstable run must fail calibration");
    auto thermally_slow = samples;
    thermally_slow.insert(thermally_slow.end(), 20U,
                          CalibrationSample{31.0, true, true});
    check(!calibration_passes(summarize_calibration(
              thermally_slow, 180.0, true, false)),
          "sustained thermal slowdown must fail calibration");

    const auto unique = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto directory = std::filesystem::temp_directory_path() /
        ("parallel-mater-profile-test-" + unique);
    const auto path = directory / "device-profiles.json";
    std::filesystem::create_directories(directory);
    std::filesystem::copy_file(PARALLEL_MATER_DEVICE_PROFILES_PATH, path);
    VerifiedBrickProfile profile;
    profile.hardware = {.machine_model = "test-machine", .cpu_model = "test-cpu",
        .gpu_model = "test-gpu", .gpu_variant = "test-variant",
        .memory_bytes = 16U * 1'024U * 1'024U * 1'024U,
        .backend = "test", .operating_system = "test-os",
        .driver = "test-driver", .power_mode = "default"};
    profile.scene = {48U, 2.0F, 1U};
    profile.solver_version = "1";
    profile.build_revision = "test";
    profile.metrics = passing;
    profile.verified_at = "2026-10-09T00:00:00Z";
    profile.verifier = "test";
    check(save_verified_profile(path, profile, error),
          error.empty() ? "save verified profile" : error.c_str());
    DeviceProfileCatalog saved;
    check(load_device_profiles(path, saved, error) &&
              saved.verified_profiles.size() == 1U &&
              find_matching_profile(saved, profile.hardware) != nullptr,
          "saved verified profile must round-trip and match");
    check(find_matching_profile(saved, profile.hardware,
                                brick_render_width, brick_render_height,
                                "different-solver") == nullptr,
          "profile matching must reject incompatible solver versions");
    auto mismatched_hardware = profile.hardware;
    mismatched_hardware.power_mode = "low-power";
    check(find_matching_profile(saved, mismatched_hardware) == nullptr,
          "profile matching must revalidate a different power mode");
    saved.verified_profiles.front().scene_version = brick_scene_version + 1U;
    check(find_matching_profile(saved, profile.hardware) == nullptr,
          "profile matching must reject a stale scene version");
    saved.verified_profiles.front().scene_version = brick_scene_version;
    {
        std::ifstream persisted(path);
        std::ostringstream contents;
        contents << persisted.rdbuf();
        check(contents.str().find("\"compute_units\":0") == std::string::npos,
              "unknown hardware specifications must remain omitted");
    }
    VerifiedBrickProfile second_profile = profile;
    second_profile.hardware.gpu_model = "second-test-gpu";
    second_profile.hardware.gpu_variant = "second-test-variant";
    second_profile.scene = {80U, 1.5F, 1U};
    check(save_verified_profile(path, second_profile, error),
          "saving a second machine profile must succeed");
    profile.scene = {24U, 2.0F, 1U};
    const bool updated_profiles = save_verified_profile(path, profile, error) &&
        load_device_profiles(path, saved, error);
    const auto *updated_profile = updated_profiles
        ? find_matching_profile(saved, profile.hardware) : nullptr;
    check(updated_profiles && saved.verified_profiles.size() == 2U &&
              updated_profile != nullptr && updated_profile->scene == profile.scene &&
              find_matching_profile(saved, second_profile.hardware) != nullptr,
          "profile updates must preserve records for other devices");
    const auto export_path = directory / "missing-checkout" /
        "verified-profile-export.json";
    check(save_verified_profile(export_path, profile, error),
          "verified profiles must export without a repository checkout");
    DeviceProfileCatalog exported;
    check(load_device_profiles(export_path, exported, error) &&
              exported.verified_profiles.size() == 1U,
          "standalone verified profile export must round-trip");
    profile.metrics.collision_maximum_milliseconds = 31.0;
    check(!save_verified_profile(path, profile, error),
          "failing measurements must never enter the verified catalog");
    profile.metrics = passing;
    profile.metrics.duration_seconds = 179.0;
    check(!save_verified_profile(path, profile, error),
          "short calibration searches must never be published as verified");
    {
        std::ofstream malformed(path, std::ios::trunc);
        malformed << "{not-json}";
    }
    check(!load_device_profiles(path, saved, error),
          "malformed profile catalogs must fail without partial data");
    std::filesystem::remove_all(directory);

    if (failures != 0) {
        std::cerr << failures << " device profile test(s) failed\n";
        return 1;
    }
    std::cout << "Device profile and procedural brick tests passed\n";
    return 0;
}

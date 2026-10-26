// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/physics_debug.hpp>
#include <scene_builder.hpp>

#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace parallel_mater::gallery;

void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

std::filesystem::path temporary(std::string_view label) {
    const auto stamp = std::chrono::steady_clock::now()
                           .time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
        ("parallel-mater-" + std::string(label) + "-" +
         std::to_string(stamp) + ".log");
}

void write_text(const std::filesystem::path &path, std::string_view text) {
    std::ofstream stream(path);
    stream << text;
    check(bool(stream), "write compatibility fixture");
}

bool same(float left, float right) {
    return std::bit_cast<std::uint32_t>(left) ==
        std::bit_cast<std::uint32_t>(right);
}

PmRigidState state(float seed) {
    return {{seed, seed + 1.0F, seed + 2.0F},
            {seed + 3.0F, seed + 4.0F, seed + 5.0F, seed + 6.0F},
            {seed + 7.0F, seed + 8.0F, seed + 9.0F},
            {seed + 10.0F, seed + 11.0F, seed + 12.0F}};
}

void check_state(const PmRigidState &left, const PmRigidState &right) {
    for (std::size_t i = 0; i < 3U; ++i) {
        check(same(left.position[i], right.position[i]), "position round trip");
        check(same(left.linearVelocity[i], right.linearVelocity[i]),
              "linear velocity round trip");
        check(same(left.angularVelocity[i], right.angularVelocity[i]),
              "angular velocity round trip");
    }
    for (std::size_t i = 0; i < 4U; ++i)
        check(same(left.orientation[i], right.orientation[i]),
              "orientation round trip");
}

void check_legacy_readers() {
    const auto v3 = temporary("physics-v3");
    write_text(v3,
        "parallel_mater_physics_capture 3\n"
        "frame 7 rigid 1 constraints 0 fluid 0 cloth 0 soft_body 0 rope 0 smoke 0 contacts 0\n"
        "rigid 0 position 1 2 3 orientation 0 0 0 1 linear_velocity 4 5 6 angular_velocity 7 8 9\n"
        "end_frame\n");
    PhysicsDebugCapture capture;
    std::string error;
    check(read_physics_debug_capture(v3, capture, error),
          error.empty() ? "read v3 capture" : error.c_str());
    check(capture.format_version == 3U && capture.has_frame &&
              capture.frame.bodies.size() == 1U && capture.history.empty() &&
              capture.provenance.rigid_body_ids.empty(),
          "v3 compatibility result");

    const auto v4 = temporary("physics-v4");
    write_text(v4,
        "parallel_mater_physics_capture 4\n"
        "rigid_history 1\n"
        "history_frame 6 timestep 0.00416666688 substeps 1 gravity 0 -9.81 0\n"
        "rigid 0 position 1 2 3 orientation 0 0 0 1 linear_velocity 4 5 6 angular_velocity 7 8 9\n"
        "end_history_frame\n"
        "frame 7 rigid 1 constraints 0 fluid 0 cloth 0 soft_body 0 rope 0 smoke 0 contacts 0\n"
        "rigid 0 position 1 2 3 orientation 0 0 0 1 linear_velocity 4 5 6 angular_velocity 7 8 9\n"
        "end_frame\n");
    check(read_physics_debug_capture(v4, capture, error),
          error.empty() ? "read v4 capture" : error.c_str());
    check(capture.format_version == 4U && capture.history.size() == 1U &&
              !capture.history_complete &&
              capture.provenance.rigid_body_ids.empty(),
          "v4 compatibility result");

    CommandScene simulation;
    CommandSceneInstance instance;
    instance.rigid_bodies.push_back(0U);
    SceneDefinition scene;
    scene.rigid_bodies.push_back({.name = "Body", .source_name = "Body"});
    check(!replay_rigid_physics_sample(
              simulation, instance, scene, capture.history.front(), {},
              PhysicsReplayPolicy::allow_partial, error) &&
              error.find("stable identity") != std::string::npos,
          "identity-free legacy capture must not replay");

    std::error_code ignored;
    std::filesystem::remove(v3, ignored);
    std::filesystem::remove(v4, ignored);
}

void check_archived_fixture() {
    PhysicsDebugCapture capture;
    std::string error;
    check(read_physics_debug_capture(
              PARALLEL_MATER_HINGE_CAPTURE_PATH, capture, error),
          error.empty() ? "read archived v5 fixture" : error.c_str());
    check(capture.format_version == 5U && !capture.history_complete &&
              capture.history.size() == 1U &&
              capture.history.front().revision == 2073U &&
              capture.provenance.rigid_body_ids.size() == 4U,
          "archived v5 fixture identity and partial semantics");
}

void check_negative_cases() {
    const auto truncated = temporary("physics-truncated");
    write_text(truncated,
        "parallel_mater_physics_capture 5\n"
        "provenance scene \"x\" backend \"cuda\" device \"gpu\" build \"abc\"\n"
        "rigid_ids 0\nhistory_complete 1\nrigid_history 0\n"
        "frame 1 rigid 0 constraints 0 fluid 0 cloth 0 soft_body 0 rope 0 smoke 0 contacts 0\n");
    PhysicsDebugCapture capture;
    std::string error;
    check(!read_physics_debug_capture(truncated, capture, error) &&
              error.find("truncated") != std::string::npos,
          "truncated capture rejected");

    CommandScene simulation;
    CommandSceneInstance instance;
    instance.rigid_bodies = {0U, 1U};
    SceneDefinition scene;
    scene.rigid_bodies.resize(2U);
    scene.rigid_bodies[0].name = "generated-0";
    scene.rigid_bodies[1].name = "generated-1";
    scene.rigid_bodies[0].source_name = "prototype";
    scene.rigid_bodies[1].source_name = "prototype";
    check(physics_rigid_body_ids(scene) ==
              std::vector<std::string>{"generated-0", "generated-1"},
          "generated bodies sharing a source need stable unique identities");
    scene.rigid_bodies[0].source_name = "A";
    scene.rigid_bodies[1].source_name = "B";
    RigidPhysicsSample sample;
    sample.bodies.push_back(state(1.0F));
    const std::string a = "A";
    check(!replay_rigid_physics_sample(
              simulation, instance, scene, sample, {&a, 1U},
              PhysicsReplayPolicy::require_complete, error) &&
              error.find("partial") != std::string::npos,
          "complete replay rejects partial capture");
    const std::string unknown = "missing";
    check(!replay_rigid_physics_sample(
              simulation, instance, scene, sample, {&unknown, 1U},
              PhysicsReplayPolicy::allow_partial, error) &&
              error.find("unknown") != std::string::npos,
          "partial replay rejects unknown identity");

    std::error_code ignored;
    std::filesystem::remove(truncated, ignored);
}

} // namespace

int main() try {
    // State retained by F6-07 immediately before the archived over-top input.
    const std::vector<std::string> ids{
        "Gear", "Gear.001", "Gear.002", "Ground.002"};
    CommandSceneFrame frame;
    frame.revision = 2073U;
    frame.rigid_states = {
        {{-1.39690912F, 5.26486826F, 0.321522236F},
         {0.347802937F, -0.615656555F, -0.615656614F, 0.347802997F},
         {2.07317781F, 3.48106813F, 9.62050908e-5F},
         {1.19576362e-6F, 3.80805545e-6F, -2.49270654F}},
        {{-0.00123224803F, 1.39732695F, 1.87125552F},
         {0.63220185F, 0.316734195F, 0.316734165F, 0.632202268F},
         {2.38274811e-6F, 1.19136739e-5F, 3.69530085e-6F},
         {1.25820543e-5F, -2.43598333e-6F, 1.36825073F}},
        {{3.06544256F, 1.39732742F, 2.2055459F},
         {0.669637799F, -0.22712338F, -0.227123454F, 0.669638038F},
         {-4.87874095e-5F, 7.7194627e-6F, -7.69934559e-6F},
         {8.225491e-6F, -1.14934692e-5F, -0.605497777F}},
        {{5.19924974F, 3.47652507F, 2.20538616F},
         {-2.23634771e-7F, 5.45621504e-8F, 0.707106948F, 0.70710659F},
         {2.72171364e-5F, -1.2084924F, 8.29557102e-5F},
         {2.1786871e-5F, -1.2911597e-5F, 3.79184439e-6F}}};
    frame.constraint_states.push_back({PM_TRUE, PM_FALSE, 3.5F});

    PmStepInfo step{1.0F / 60.0F, 1U,
                    {4.88931274F, -8.49570942F, -0.391981125F},
                    PM_STEP_COLLECT_CONTACT_EVENTS_BIT};
    PhysicsDebugHistory history(2U);
    std::string error;
    check(history.record(frame, step, error), "record archived state");
    const PhysicsCaptureProvenance provenance{
        "ConstraintHinge.glb", "cuda", "reference-device", "test-revision", ids};
    const auto output = temporary("physics-v5");
    check(write_physics_debug_capture(
              frame, output, error, &history, &provenance),
          error.empty() ? "write v5 capture" : error.c_str());

    PhysicsDebugCapture capture;
    check(read_physics_debug_capture(output, capture, error),
          error.empty() ? "read v5 capture" : error.c_str());
    check(capture.format_version == 5U && capture.history_complete &&
              capture.history.size() == 1U && capture.has_frame &&
              capture.provenance.scene == provenance.scene &&
              capture.provenance.backend == provenance.backend &&
              capture.provenance.device == provenance.device &&
              capture.provenance.build_revision == provenance.build_revision &&
              capture.provenance.rigid_body_ids == ids,
          "v5 provenance and history round trip");
    check(capture.history.front().revision == 2073U &&
              same(capture.history.front().step.timestep, step.timestep) &&
              capture.history.front().step.substeps == 1U &&
              capture.history.front().step.flags == step.flags &&
              capture.history.front().bodies.size() == frame.rigid_states.size(),
          "v5 step round trip");
    for (std::size_t i = 0; i < frame.rigid_states.size(); ++i)
        check_state(capture.history.front().bodies[i], frame.rigid_states[i]);

    frame.revision = 2074U;
    check(history.record(frame, step, error), "record second state");
    frame.revision = 2075U;
    check(history.record(frame, step, error) && !history.complete() &&
              history.frames().size() == 2U &&
              history.frames().front().revision == 2074U,
          "bounded eviction marks capture partial");

    check_legacy_readers();
    check_archived_fixture();
    check_negative_cases();
    std::error_code ignored;
    std::filesystem::remove(output, ignored);
    return 0;
} catch (const std::exception &exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
}

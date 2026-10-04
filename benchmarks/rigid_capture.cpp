// SPDX-License-Identifier: MIT
#include "support.hpp"

#include <bit>
#include <charconv>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

namespace {
using namespace parallel_mater;

struct Snapshot {
    std::uint64_t frame{};
    float timestep{};
    Vec3 gravity{};
    std::vector<PhysicsDebugRigidSample> bodies{};
};

bool label(std::istream &input, const char *expected) {
    std::string value;
    return static_cast<bool>(input >> value) && value == expected;
}

bool read_vector(std::istream &input, Vec3 &value) {
    return static_cast<bool>(input >> value.x >> value.y >> value.z);
}

bool load_snapshots(const char *path, std::vector<Snapshot> &snapshots) {
    std::ifstream input(path);
    unsigned version{};
    std::size_t expected_frames{}, expected_bodies{};
    if (!label(input, "parallel_mater_physics_capture") ||
        !(input >> version) || version != 1U || !label(input, "frames") ||
        !(input >> expected_frames)) return false;
    bool in_frame = false;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::istringstream fields(line);
        std::string tag;
        fields >> tag;
        if (tag == "frame") {
            if (in_frame) return false;
            in_frame = true;
            snapshots.emplace_back();
            auto &snapshot = snapshots.back();
            if (!(fields >> snapshot.frame) || !label(fields, "timestep") ||
                !(fields >> snapshot.timestep) || !label(fields, "gravity") ||
                !read_vector(fields, snapshot.gravity)) return false;
            expected_bodies = 0U;
            std::size_t count{};
            while (fields >> tag >> count) {
                if (tag == "rigid") expected_bodies = count;
                if ((tag == "fluid" || tag == "cloth" || tag == "soft_body" ||
                     tag == "rope") && count != 0U) return false;
            }
            if (expected_bodies == 0U) return false;
        } else if (tag == "rigid") {
            if (!in_frame) return false;
            PhysicsDebugRigidSample body{};
            auto &state = body.state;
            if (!(fields >> body.id.index >> body.id.generation) ||
                !label(fields, "position") || !read_vector(fields, state.position) ||
                !label(fields, "orientation") ||
                !(fields >> state.orientation.x >> state.orientation.y >>
                  state.orientation.z >> state.orientation.w) ||
                !label(fields, "linear_velocity") ||
                !read_vector(fields, state.linear_velocity) ||
                !label(fields, "angular_velocity") ||
                !read_vector(fields, state.angular_velocity)) return false;
            snapshots.back().bodies.push_back(body);
        } else if (tag == "end_frame") {
            if (!in_frame || snapshots.back().bodies.size() != expected_bodies)
                return false;
            in_frame = false;
        }
    }
    return !in_frame && !snapshots.empty() && snapshots.size() == expected_frames;
}

bool parse_positive(const char *text, unsigned &value) {
    const std::string_view input(text);
    const auto parsed = std::from_chars(input.data(), input.data() + input.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == input.data() + input.size() &&
           value > 0U;
}
} // namespace

int main(int argc, char **argv) {
    using namespace parallel_mater::benchmark;
    using namespace parallel_mater::gallery;
    unsigned substeps = 8U, repetitions = 5U;
    if (argc < 3 || argc > 5 ||
        (argc > 3 && !parse_positive(argv[3], substeps)) || substeps > 1024U ||
        (argc > 4 && !parse_positive(argv[4], repetitions))) {
        std::cerr << "usage: parallel-mater-rigid-capture-benchmark scene.glb "
                     "capture.log [substeps=8] [repetitions=5]\n";
        return 2;
    }
    SceneDefinition scene;
    std::string error;
    std::vector<Snapshot> snapshots;
    if (!load_glb_scene(argv[1], scene, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    if (!load_snapshots(argv[2], snapshots)) {
        std::cerr << "capture must contain complete, rigid-only version 1 snapshots\n";
        return 1;
    }
    World world;
    SceneInstance instance;
    if (!prepare_world(scene, world, instance)) return 1;
    for (const auto &snapshot : snapshots) {
        std::vector<bool> seen(instance.rigid_bodies.size());
        for (const auto &body : snapshot.bodies) {
            if (body.id.index >= seen.size() || seen[body.id.index] ||
                !(body.id == instance.rigid_bodies[body.id.index])) {
                std::cerr << "capture body handles do not match scene\n";
                return 1;
            }
            seen[body.id.index] = true;
        }
        if (snapshot.bodies.size() != seen.size()) return 1;
    }
    // Captures hold post-step states. Benchmark one independent step from each
    // snapshot, with zero new input forces, rather than claim an input replay.
    // One complete unmeasured pass warms the GPU and managed allocations.
    TimingSamples samples{};
    std::uint64_t state_hash = 14695981039346656037ULL;
    std::uint64_t slowest_frame{};
    float maximum_gpu_ms{};
    for (std::uint64_t repetition = 0U; repetition <= repetitions; ++repetition) {
        for (const auto &snapshot : snapshots) {
            for (const auto &body : snapshot.bodies)
                if (!require(world.set_rigid_body_state(body.id, body.state),
                             "restore captured rigid body")) return 1;
            auto options = standard_step_options(true);
            options.substeps = substeps;
            options.timestep = snapshot.timestep;
            options.gravity = snapshot.gravity;
            WorldStepTimings timing{};
            double wall_ms{};
            if (!measure_step(world, options, "step captured snapshot", timing,
                              wall_ms)) return 1;
            if (repetition == 0U) continue;
            add_timing_samples(samples, timing, wall_ms);
            if (timing.total_gpu_milliseconds > maximum_gpu_ms) {
                maximum_gpu_ms = timing.total_gpu_milliseconds;
                slowest_frame = snapshot.frame;
            }
            for (auto id : instance.rigid_bodies) {
                RigidBodyState state{};
                if (!require(world.read_rigid_body_state(id, state), "read result"))
                    return 1;
                const float values[]{state.position.x, state.position.y, state.position.z,
                    state.orientation.x, state.orientation.y, state.orientation.z,
                    state.orientation.w, state.linear_velocity.x, state.linear_velocity.y,
                    state.linear_velocity.z, state.angular_velocity.x,
                    state.angular_velocity.y, state.angular_velocity.z};
                for (float value : values) {
                    if (!std::isfinite(value)) return 1;
                    state_hash ^= std::bit_cast<std::uint32_t>(value);
                    state_hash *= 1099511628211ULL;
                }
            }
        }
    }
    constexpr const char *names[]{"integration", "bounds", "pair_filter",
        "pair_compaction", "leaf_pairs", "triangle_contacts", "solve", "clear",
        "gpu_total", "wall"};
    for (std::size_t index = 0; index < samples.size(); ++index)
        samples[index].print_summary(std::cout, names[index]);
    std::cout << "maximum_gpu_ms=" << maximum_gpu_ms
              << " snapshot_frame=" << slowest_frame
              << " state_hash=" << std::hex << state_hash << '\n';
}

// SPDX-License-Identifier: MIT
#include "support.hpp"

#include <parallel_mater_gallery/gallery_context.hpp>

#include <charconv>
#include <fstream>
#include <iomanip>
#include <string_view>

namespace {
bool count(const char *text, unsigned &value) {
    const std::string_view input(text);
    const auto parsed = std::from_chars(input.data(), input.data() + input.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == input.data() + input.size() &&
           value <= 100000U;
}
}

int main(int argc, char **argv) {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;
    using namespace parallel_mater::benchmark;
    unsigned frames = 240U, settle = 120U;
    bool impact = false, tilt = false;
    std::string csv_path;
    for (int i = 2; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (option == "--frames" && i + 1 < argc && count(argv[++i], frames) && frames) {}
        else if (option == "--settle" && i + 1 < argc && count(argv[++i], settle)) {}
        else if (option == "--csv" && i + 1 < argc) csv_path = argv[++i];
        else if (option == "--impact") impact = true;
        else if (option == "--tilt") tilt = true;
        else {
            std::cerr << "Invalid benchmark option\n";
            return 2;
        }
    }
    if (argc < 2) {
        std::cerr << "usage: parallel-mater-rigid-frame-benchmark scene.glb "
                     "[--frames N] [--settle N] [--impact] [--tilt] [--csv file]\n";
        return 2;
    }
    SceneDefinition scene;
    SceneInstance instance;
    World world;
    OptixRenderer renderer;
    std::string error;
    // Match the interactive gallery's capture ring, but keep kernel profiling
    // and overlays disabled. Time one fixed physics step plus a complete
    // 960x720 render/readback; window upload/presentation is excluded.
    if (!load_glb_scene(argv[1], scene, error) ||
        !require(create_scene_world(scene, world, instance,
            {.frame_capacity = 30U, .frame_stride = 1U}), "create world") ||
        !OptixRenderer::create(scene, world, instance, PARALLEL_MATER_OPTIX_PTX_PATH,
                              960U, 720U, renderer, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    CameraController camera;
    camera.set_preset(gallery_entry(GalleryContext::rigid_body).camera);
    std::vector<std::uint32_t> pixels;
    auto options = standard_step_options(false);
    std::vector<RigidBodyId> vertical_gravity_bodies;
    for (std::size_t index = 0; index < scene.rigid_bodies.size(); ++index)
        if (scene.rigid_bodies[index].options.motion == MotionType::dynamic &&
            !scene.rigid_bodies[index].follows_gravity_tilt)
            vertical_gravity_bodies.push_back(instance.rigid_bodies[index]);
    RigidBodyId impact_body{};
    bool wall_hit = false;
    for (unsigned frame = 0; frame < settle + 10U; ++frame) {
        if (!require(world.step(options), "settle") ||
            !renderer.render(world, instance, camera.camera(), pixels, error)) return 1;
    }
    if (impact) {
        const auto sphere = std::find_if(scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
            [](const auto &body) { return body.source_name == "Icosphere"; });
        if (sphere == scene.rigid_bodies.end()) {
            std::cerr << "--impact needs the RigidBody scene's Icosphere\n";
            return 2;
        }
        const auto index = static_cast<std::size_t>(sphere - scene.rigid_bodies.begin());
        impact_body = instance.rigid_bodies[index];
        RigidBodyState state;
        if (!require(world.read_rigid_body_state(instance.rigid_bodies[index], state), "read sphere") ||
            !require(world.apply_impulse(instance.rigid_bodies[index],
                {0, 0, -sphere->options.mass * 10.0F}, state.position), "launch sphere")) return 1;
    }
    if (tilt) options.gravity = screen_space_gravity(camera.camera(), 0, 1, 9.81F, 20.0F);
    std::ofstream csv;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        if (!csv) return 1;
        csv << "frame,physics_ms,render_ms,total_ms\n" << std::setprecision(9);
    }
    Samples physics, render, total;
    using Clock = std::chrono::steady_clock;
    const auto milliseconds = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    for (unsigned frame = 0; frame < frames; ++frame) {
        const auto start = Clock::now();
        if (tilt && !require(world.apply_central_acceleration(
                {vertical_gravity_bodies.data(), vertical_gravity_bodies.size()},
                {-options.gravity.x, -9.81F - options.gravity.y,
                 -options.gravity.z}), "preserve wall gravity")) return 1;
        if (!require(world.step(options), "measure physics")) return 1;
        const auto simulated = Clock::now();
        if (!renderer.render(world, instance, camera.camera(), pixels, error)) {
            std::cerr << error << '\n';
            return 1;
        }
        const auto rendered = Clock::now();
        const double physics_ms = milliseconds(start, simulated);
        const double render_ms = milliseconds(simulated, rendered);
        const double total_ms = milliseconds(start, rendered);
        physics.add(physics_ms); render.add(render_ms); total.add(total_ms);
        if (csv) csv << frame << ',' << physics_ms << ',' << render_ms << ',' << total_ms << '\n';
        // Validate the workload outside the timed interval. Capture data is
        // already on the host, so this does not migrate live simulation pages.
        PhysicsDebugFrameView capture;
        if (!require(world.physics_debug_frame(capture), "read capture")) return 1;
        for (std::size_t i = 0; i < capture.rigid_bodies.size; ++i) {
            const auto &body = capture.rigid_bodies.data[i];
            const auto &s = body.state;
            for (float value : {s.position.x, s.position.y, s.position.z,
                                s.linear_velocity.x, s.linear_velocity.y, s.linear_velocity.z,
                                s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z,
                                s.orientation.x, s.orientation.y, s.orientation.z, s.orientation.w})
                if (!std::isfinite(value)) return 1;
        }
        if (impact) {
            for (std::size_t i = 0; i < capture.rigid_contacts.size; ++i) {
                const auto &contact = capture.rigid_contacts.data[i];
                if (contact.normal_impulse <= 0.0F) continue;
                const auto other = contact.body == impact_body ? contact.collider : contact.body;
                if (!(contact.body == impact_body) && !(contact.collider == impact_body)) continue;
                for (std::size_t body = 0; body < scene.rigid_bodies.size(); ++body)
                    if (instance.rigid_bodies[body] == other &&
                        (scene.rigid_bodies[body].source_name == "Layer1" ||
                         scene.rigid_bodies[body].source_name == "Layer2")) wall_hit = true;
            }
        }
    }
    const auto print = [](const char *label, Samples &samples) {
        std::cout << label << " median=" << samples.percentile(0.5)
                  << " p95=" << samples.percentile(0.95)
                  << " p99=" << samples.percentile(0.99)
                  << " max=" << samples.percentile(1.0) << " ms\n";
    };
    print("physics", physics); print("render", render); print("total", total);
    if (impact) {
        std::cout << "sphere_hit_wall=" << wall_hit << '\n';
        if (!wall_hit) return 1;
    }
    return csv_path.empty() || csv.good() ? 0 : 1;
}

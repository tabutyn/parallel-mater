// SPDX-License-Identifier: MIT
#include "support.hpp"

#include <parallel_mater_gallery/gallery_context.hpp>
#include <parallel_mater_gallery/renderer.hpp>
#include <render/render_scene.hpp>
#include <scene_builder.hpp>

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
    CommandSceneInstance instance;
    CommandScene simulation;
    CommandSceneFrame simulation_frame;
    RenderSceneFrame render_frame;
    OptixRenderer renderer;
    std::string error;
    // Match the interactive gallery's capture ring, but keep kernel profiling
    // and overlays disabled. Time one fixed physics step plus a complete
    // 960x720 render/readback; window upload/presentation is excluded.
    if (!load_glb_scene(argv[1], scene, error) ||
        !CommandScene::create(scene, 30U, simulation, instance, error) ||
        !OptixRenderer::create(scene, instance, PARALLEL_MATER_OPTIX_PTX_PATH,
                              960U, 720U, renderer, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    CameraController camera;
    camera.set_preset(gallery_entry(GalleryContext::rigid_body).camera);
    std::vector<std::uint32_t> pixels;
    auto options = standard_step_options(false);
    std::vector<std::uint32_t> vertical_gravity_bodies;
    for (std::size_t index = 0; index < scene.rigid_bodies.size(); ++index)
        if (scene.rigid_bodies[index].options.motion == MotionType::dynamic &&
            !scene.rigid_bodies[index].follows_gravity_tilt)
            vertical_gravity_bodies.push_back(instance.rigid_bodies[index]);
    std::uint32_t impact_body{PM_INVALID_INDEX};
    bool wall_hit = false;
    const auto step = [&]() {
        if (!simulation.step(options.timestep, options.substeps,
                             options.gravity, simulation_frame, error,
                             impact) ||
            !make_render_scene_frame(scene, simulation_frame, render_frame,
                                     error))
            return false;
        return true;
    };
    for (unsigned frame = 0; frame < settle + 10U; ++frame) {
        if (!step() ||
            !renderer.render(render_frame, camera.camera(), pixels, error)) {
            std::cerr << "settle: " << error << '\n';
            return 1;
        }
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
        PmRigidInput input{};
        input.bodyIndex = impact_body;
        input.linearImpulse[2] = -sphere->options.mass * 10.0F;
        if (!simulation.apply_rigid_inputs({input}, error)) {
            std::cerr << "launch sphere: " << error << '\n';
            return 1;
        }
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
        if (tilt) {
            std::vector<PmRigidInput> inputs;
            inputs.reserve(vertical_gravity_bodies.size());
            for (std::uint32_t body : vertical_gravity_bodies) {
                PmRigidInput input{};
                input.bodyIndex = body;
                const float mass = scene.rigid_bodies[body].options.mass;
                input.force[0] = -options.gravity.x * mass;
                input.force[1] = (-9.81F - options.gravity.y) * mass;
                input.force[2] = -options.gravity.z * mass;
                inputs.push_back(input);
            }
            if (!simulation.apply_rigid_inputs(inputs, error)) {
                std::cerr << "preserve wall gravity: " << error << '\n';
                return 1;
            }
        }
        if (!step()) {
            std::cerr << "measure physics: " << error << '\n';
            return 1;
        }
        const auto simulated = Clock::now();
        if (!renderer.render(render_frame, camera.camera(), pixels, error)) {
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
        for (const PmRigidState &state : simulation_frame.rigid_states) {
            for (float value : {state.position[0], state.position[1], state.position[2],
                                state.linearVelocity[0], state.linearVelocity[1], state.linearVelocity[2],
                                state.angularVelocity[0], state.angularVelocity[1], state.angularVelocity[2],
                                state.orientation[0], state.orientation[1], state.orientation[2], state.orientation[3]})
                if (!std::isfinite(value)) return 1;
        }
        if (impact) {
            for (const PmContactEvent &contact : simulation_frame.contacts) {
                if (contact.type != PM_CONTACT_RIGID_RIGID ||
                    contact.normalImpulse <= 0.0F) continue;
                const auto other = contact.first == impact_body
                    ? contact.second : contact.first;
                if (contact.first != impact_body &&
                    contact.second != impact_body) continue;
                if (other < scene.rigid_bodies.size() &&
                    (scene.rigid_bodies[other].source_name == "Layer1" ||
                     scene.rigid_bodies[other].source_name == "Layer2"))
                    wall_hit = true;
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

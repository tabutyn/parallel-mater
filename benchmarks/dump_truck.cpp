// SPDX-License-Identifier: MIT
#include "support.hpp"
#include <parallel_mater_gallery/dump_truck.hpp>
#include <parallel_mater_gallery/gallery_context.hpp>
#include <parallel_mater_gallery/renderer_adapter.hpp>
#include <render/render_scene.hpp>
#include <scene_builder.hpp>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <string_view>

int main(int argc, char **argv) {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;
    using namespace parallel_mater::benchmark;
    unsigned spheres = 100, frames = 180, warmup = 180, substeps = 8, pass_limit = 0;
    bool render = true, capture = true, profile = false, trace = false;
    std::string mode = "rest", csv_path;
    const auto number = [](const char *text, unsigned &value) {
        const std::string_view input(text);
        const auto result = std::from_chars(input.data(),input.data()+input.size(),value);
        return result.ec == std::errc{} && result.ptr == input.data()+input.size();
    };
    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (option == "--spheres" && i+1 < argc && number(argv[++i],spheres) &&
            (spheres == 0 || (spheres >= 10 && spheres <= 1000))) {}
        else if (option == "--frames" && i+1 < argc && number(argv[++i],frames) && frames && frames <= 10000) {}
        else if (option == "--warmup" && i+1 < argc && number(argv[++i],warmup) && warmup <= 10000) {}
        else if (option == "--substeps" && i+1 < argc && number(argv[++i],substeps) && substeps && substeps <= 32) {}
        else if (option == "--passes" && i+1 < argc && number(argv[++i],pass_limit) && pass_limit <= 64) {}
        else if (option == "--mode" && i+1 < argc) mode = argv[++i];
        else if (option == "--csv" && i+1 < argc) csv_path = argv[++i];
        else if (option == "--no-render") render = false;
        else if (option == "--no-capture") capture = false;
        else if (option == "--profile") profile = true;
        else if (option == "--trace") trace = true;
        else { std::cerr << "Invalid dump benchmark option: " << option << '\n'; return 2; }
    }
    if (mode != "rest" && mode != "drive" && mode != "tip") return 2;
    SceneDefinition scene;
    std::string error;
    if (!load_glb_scene(PARALLEL_MATER_DUMP_TRUCK_SCENE_PATH,scene,error) ||
        (spheres && !configure_dump_payload(scene,spheres,error))) {
        std::cerr << error << '\n'; return 1;
    }
    std::size_t triangles{};
    unsigned authored_max_iterations{};
    for (const auto &joint : scene.rigid_constraints)
        if (joint.options.enabled)
            authored_max_iterations = std::max(authored_max_iterations,joint.options.solver_iterations);
    for (const auto &body : scene.rigid_bodies)
        for (const auto mesh : body.mesh_indices) triangles += scene.meshes[mesh].indices.size()/3;
    cudaDeviceProp gpu{};
    if (cudaGetDeviceProperties(&gpu,0) != cudaSuccess) return 1;
    std::cout << "gpu=" << gpu.name << " spheres=" << spheres << " bodies=" << scene.rigid_bodies.size()
              << " constraints=" << scene.rigid_constraints.size() << " instanced_triangles=" << triangles
              << " mode=" << mode << " frames=" << frames << " warmup=" << warmup
              << " substeps=" << substeps << " capture=" << capture << " render=" << render
              << " pass_limit=" << pass_limit << " authored_max_iterations=" << authored_max_iterations
              << " kernel_profile=" << profile << std::endl;
    CommandScene simulation;
    CommandSceneInstance instance;
    CommandSceneFrame frame;
    RenderSceneFrame render_frame;
    if (!CommandScene::create(scene, capture ? 30U : 0U, simulation,
                              instance, error)) {
        std::cerr << "create simulation: " << error << '\n';
        return 1;
    }
    RendererHandle renderer;
    if (render && !RendererAdapterFactory::create_host(
                      scene, instance, 960U, 720U, renderer, error)) {
        std::cerr << error << '\n'; return 1;
    }
    CameraController camera;
    camera.set_preset(gallery_entry(GalleryContext::dump).camera);
    std::vector<std::uint32_t> pixels;
    const auto draw = [&]() {
        if (!render) return true;
        if (!make_render_scene_frame(scene, frame, render_frame, error))
            return false;
        if (renderer.render(render_frame, camera.camera(), pixels, error))
            return true;
        std::cerr << error << '\n'; return false;
    };
    DumpTruckBed bed;
    if (!bed.initialize(scene)) return 1;
    auto options = standard_step_options(false);
    options.substeps = substeps;
    options.rigid_contact_pass_limit = pass_limit;
    const auto step = [&]() {
        return simulation.step(options.timestep, options.substeps,
                               options.gravity, frame, error);
    };
    for (unsigned frame_index = 0; frame_index < warmup; ++frame_index)
        if (!step() || !draw()) {
            if (!error.empty()) std::cerr << "warmup: " << error << '\n';
            return 1;
        }
    if (!draw()) return 1;
    if (mode == "tip") bed.toggle();
    if (mode == "drive") {
        for (std::size_t i = 0; i < scene.rigid_constraints.size(); ++i) {
            const auto &joint = scene.rigid_constraints[i];
            if (joint.options.type != RigidConstraintType::motor) continue;
            auto motor = joint;
            motor.options.motor.angular_target_velocity = -8;
            if (!simulation.write_constraint(instance.rigid_constraints[i],
                                             motor, error)) {
                std::cerr << "drive: " << error << '\n';
                return 1;
            }
            scene.rigid_constraints[i].options = motor.options;
        }
    }
    std::ofstream csv;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        if (!csv) return 1;
        csv << "frame,controls_ms,physics_ms,render_ms,total_ms\n" << std::setprecision(9);
    }
    Samples controls, physics, rendering, total;
    options.collect_kernel_timings = profile;
    using Clock = std::chrono::steady_clock;
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
    if (trace && cudaProfilerStart() != cudaSuccess) return 1;
    for (unsigned frame_index = 0; frame_index < frames; ++frame_index) {
        const auto start = Clock::now();
        if (!bed.advance(simulation, scene, instance, options.timestep,
                         error)) {
            std::cerr << "actuate: " << error << '\n';
            return 1;
        }
        const auto controlled = Clock::now();
        if (!step()) {
            std::cerr << "step: " << error << '\n';
            return 1;
        }
        const auto stepped = Clock::now();
        if (!draw()) return 1;
        const auto drawn = Clock::now();
        controls.add(ms(start,controlled)); physics.add(ms(controlled,stepped));
        rendering.add(ms(stepped,drawn)); total.add(ms(start,drawn));
        if (csv) csv << frame_index << ',' << ms(start,controlled) << ',' << ms(controlled,stepped)
                     << ',' << ms(stepped,drawn) << ',' << ms(start,drawn) << '\n';
    }
    if (trace && cudaProfilerStop() != cudaSuccess) return 1;
    const auto print = [](const char *name, Samples &s) {
        std::cout << name << " median=" << s.percentile(.5) << " p95=" << s.percentile(.95)
                  << " p99=" << s.percentile(.99) << " max=" << s.percentile(1) << " ms\n";
    };
    print("controls",controls); print("physics",physics); print("render",rendering); print("total",total);
    if (profile)
        std::cout << "pose_iterations=" << frame.statistics.poseIterations
                  << " impact_iterations=" << frame.statistics.impactIterations
                  << " (public command statistics query)\n";
    for (const PmRigidState &state : frame.rigid_states) {
        for (const float value : {state.position[0],state.position[1],state.position[2],
             state.linearVelocity[0],state.linearVelocity[1],state.linearVelocity[2],
             state.orientation[0],state.orientation[1],state.orientation[2],state.orientation[3]})
            if (!std::isfinite(value)) return 1;
    }
    std::cout << "finite_states=1 contacts=" << frame.contacts.size()
              << " overflows=" << frame.statistics.overflowCount
              << " iterations="
              << frame.statistics.poseIterations +
                     frame.statistics.impactIterations << '\n';
    return csv_path.empty() || csv.good() ? 0 : 1;
}

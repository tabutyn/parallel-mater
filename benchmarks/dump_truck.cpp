// SPDX-License-Identifier: MIT
#include "support.hpp"
#include <parallel_mater_gallery/dump_truck.hpp>
#include <parallel_mater_gallery/gallery_context.hpp>
#include <cuda_profiler_api.h>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <string_view>

int main(int argc, char **argv) {
    using namespace parallel_mater;
    using namespace parallel_mater::gallery;
    using namespace parallel_mater::benchmark;
    unsigned spheres = 100, frames = 180, warmup = 180, substeps = 8;
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
    for (const auto &body : scene.rigid_bodies)
        for (const auto mesh : body.mesh_indices) triangles += scene.meshes[mesh].indices.size()/3;
    cudaDeviceProp gpu{};
    if (cudaGetDeviceProperties(&gpu,0) != cudaSuccess) return 1;
    std::cout << "gpu=" << gpu.name << " spheres=" << spheres << " bodies=" << scene.rigid_bodies.size()
              << " constraints=" << scene.rigid_constraints.size() << " instanced_triangles=" << triangles
              << " mode=" << mode << " frames=" << frames << " warmup=" << warmup
              << " substeps=" << substeps << " capture=" << capture << " render=" << render
              << " kernel_profile=" << profile << std::endl;
    World world;
    SceneInstance instance;
    if (!require(create_scene_world(scene,world,instance,
            {.frame_capacity = capture ? 30U : 0U, .frame_stride = 1U}),"create world")) return 1;
    OptixRenderer renderer;
    if (render && !OptixRenderer::create(scene,world,instance,PARALLEL_MATER_OPTIX_PTX_PATH,
                                         960,720,renderer,error)) {
        std::cerr << error << '\n'; return 1;
    }
    CameraController camera;
    camera.set_preset(gallery_entry(GalleryContext::dump).camera);
    std::vector<std::uint32_t> pixels;
    const auto draw = [&]() {
        if (!render || renderer.render(world,instance,camera.camera(),pixels,error)) return true;
        std::cerr << error << '\n'; return false;
    };
    DumpTruckBed bed;
    if (!bed.initialize(scene)) return 1;
    auto options = standard_step_options(false);
    options.substeps = substeps;
    for (unsigned frame = 0; frame < warmup; ++frame)
        if (!require(world.step(options),"warmup") || !draw()) return 1;
    if (!draw()) return 1;
    if (mode == "tip") bed.toggle();
    if (mode == "drive") {
        for (std::size_t i = 0; i < scene.rigid_constraints.size(); ++i) {
            const auto &joint = scene.rigid_constraints[i];
            if (joint.options.type != RigidConstraintType::motor) continue;
            auto motor = joint.options;
            motor.body_a = instance.rigid_bodies[joint.body_a];
            motor.body_b = instance.rigid_bodies[joint.body_b];
            motor.motor.angular_target_velocity = -8;
            if (!require(world.update_rigid_constraint(instance.rigid_constraints[i],motor),"drive")) return 1;
        }
    }
    std::ofstream csv;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        if (!csv) return 1;
        csv << "frame,controls_ms,physics_ms,render_ms,total_ms\n" << std::setprecision(9);
    }
    Samples controls, physics, rendering, total;
    TimingSamples stages;
    options.collect_kernel_timings = profile;
    using Clock = std::chrono::steady_clock;
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
    if (trace && cudaProfilerStart() != cudaSuccess) return 1;
    for (unsigned frame = 0; frame < frames; ++frame) {
        const auto start = Clock::now();
        if (!require(bed.advance(world,scene,instance,options.timestep),"actuate")) return 1;
        const auto controlled = Clock::now();
        if (!require(world.step(options),"step")) return 1;
        const auto stepped = Clock::now();
        if (!draw()) return 1;
        const auto drawn = Clock::now();
        controls.add(ms(start,controlled)); physics.add(ms(controlled,stepped));
        rendering.add(ms(stepped,drawn)); total.add(ms(start,drawn));
        if (csv) csv << frame << ',' << ms(start,controlled) << ',' << ms(controlled,stepped)
                     << ',' << ms(stepped,drawn) << ',' << ms(start,drawn) << '\n';
        if (profile) {
            WorldStepTimings timing;
            if (!require(world.collect_step_timings(timing),"timings") || !timing.available) return 1;
            add_timing_samples(stages,timing,ms(controlled,stepped));
        }
    }
    if (trace && cudaProfilerStop() != cudaSuccess) return 1;
    const auto print = [](const char *name, Samples &s) {
        std::cout << name << " median=" << s.percentile(.5) << " p95=" << s.percentile(.95)
                  << " p99=" << s.percentile(.99) << " max=" << s.percentile(1) << " ms\n";
    };
    print("controls",controls); print("physics",physics); print("render",rendering); print("total",total);
    if (profile) {
        constexpr std::array names{"integration","bounds","pair_filter","pair_compaction",
            "leaf_pairs","triangle_contacts","solve","clear","gpu_total","wall"};
        for (std::size_t i = 0; i < stages.size(); ++i) print(names[i],stages[i]);
    }
    WorldStatistics stats;
    if (!require(world.collect_statistics(stats),"statistics")) return 1;
    for (const auto id : instance.rigid_bodies) {
        RigidBodyState state;
        if (!require(world.read_rigid_body_state(id,state),"validate state")) return 1;
        for (const float value : {state.position.x,state.position.y,state.position.z,
             state.linear_velocity.x,state.linear_velocity.y,state.linear_velocity.z,
             state.orientation.x,state.orientation.y,state.orientation.z,state.orientation.w})
            if (!std::isfinite(value)) return 1;
    }
    std::cout << "finite_states=1 contacts=" << stats.contact_count
              << " contact_overflows=" << stats.contact_overflow_count
              << " allocated_MiB=" << stats.allocated_bytes/(1024.0*1024) << '\n';
    return csv_path.empty() || csv.good() ? 0 : 1;
}

// Device-profile measurement and explicit human verification.
struct DesktopGalleryState {
    GalleryRuntime runtime{};
    std::uint32_t dump_spheres{k_default_dump_spheres};
    std::uint32_t fluid_particles{k_default_fluid_particles};
    parallel_mater::Vec3 peg_gravity{};
    parallel_mater::Vec3 cloth_gravity{};
    std::uint64_t revision{};

    [[nodiscard]] bool rebuild(const Options &options, GalleryContext context,
                               std::uint32_t requested_dump_spheres,
                               std::uint32_t requested_fluid_particles,
                               std::string &error) {
        if (!build_runtime(options, context, requested_dump_spheres,
                           requested_fluid_particles, runtime, error)) {
            return false;
        }
        dump_spheres = requested_dump_spheres;
        fluid_particles = requested_fluid_particles;
        peg_gravity = initial_scene_gravity(context, runtime.scene.gravity_scale);
        cloth_gravity = peg_gravity;
        ++revision;
        std::cout << "Gallery scene ready index=" << gallery_context_index(context)
                  << " revision=" << revision << '\n' << std::flush;
        return true;
    }
};

struct CalibrationResult {
    BrickSceneConfig config{};
    parallel_mater::gallery::CalibrationMetrics metrics{};
    parallel_mater::gallery::HardwareIdentity hardware{};
};

std::filesystem::path device_profiles_path(const Options &options) {
    if (options.profiles_file_overridden) return options.profiles_file;
    return gallery_resource_path(options, options.profiles_file);
}

bool launch_calibration_ball(GalleryRuntime &runtime, bool off_center,
                             std::string &error) {
    for (std::size_t index = 0U; index < runtime.scene.rigid_bodies.size(); ++index) {
        if (runtime.scene.rigid_bodies[index].source_name != "Icosphere") continue;
        RigidBodyState state = runtime.scene.rigid_bodies[index].options.initial_state;
        state.position = {off_center ? 1.0F : 0.0F, 1.05F, 3.0F};
        state.linear_velocity = {0.0F, 0.0F, -10.0F};
        state.angular_velocity = {};
        return runtime.simulation.set_rigid_state(
            runtime.instance.rigid_bodies[index], state, error);
    }
    error = "brick calibration scene has no ball";
    return false;
}

bool stable_calibration_scene(GalleryRuntime &runtime, bool quiet,
                              float scale, std::string &error) {
    if (runtime.frame.rigid_states.size() < runtime.instance.rigid_bodies.size()) {
        error = "brick calibration rigid output is incomplete";
        return false;
    }
    unsigned displaced_bricks = 0U;
    for (std::size_t index = 0U; index < runtime.instance.rigid_bodies.size(); ++index) {
        const RigidBodyState state = rigid_state(runtime.frame.rigid_states[index]);
        if (!std::isfinite(state.position.x) || !std::isfinite(state.position.y) ||
            !std::isfinite(state.position.z) ||
            std::fabs(state.position.x) > 1'000.0F ||
            std::fabs(state.position.y) > 1'000.0F ||
            std::fabs(state.position.z) > 1'000.0F) {
            error = "brick calibration produced an unstable body";
            return false;
        }
        const float speed = std::sqrt(
            state.linear_velocity.x * state.linear_velocity.x +
            state.linear_velocity.y * state.linear_velocity.y +
            state.linear_velocity.z * state.linear_velocity.z);
        if (!std::isfinite(speed) || speed > 50.0F) {
            error = "brick calibration produced excessive kinetic energy";
            return false;
        }
        const auto &body = runtime.scene.rigid_bodies[index];
        if (body.source_name != "Layer1" && body.source_name != "Layer2")
            continue;
        const auto initial = body.options.initial_state.position;
        const float displacement = std::hypot(
            std::hypot(state.position.x - initial.x,
                       state.position.y - initial.y),
            state.position.z - initial.z);
        if (quiet && displacement > std::max(0.03F, 0.04F * scale)) {
            error = "brick wall lost quiet support";
            return false;
        }
        displaced_bricks += !quiet && displacement > 0.03F;
    }
    if (!quiet && displaced_bricks == 0U) {
        error = "calibration impact did not reach the brick wall";
        return false;
    }
    return true;
}

bool run_calibration_trial(const Options &base, const BrickSceneConfig &config,
                           GalleryWindow &window, std::uint32_t quiet_frames,
                           std::uint32_t collision_frames, bool off_center,
                           std::vector<parallel_mater::gallery::CalibrationSample> &samples,
                           bool &stable, std::string &error) {
    Options candidate = base;
    candidate.bricks = config;
    GalleryRuntime runtime;
    if (!build_runtime(candidate, GalleryContext::rigid_body,
                       candidate.dump_spheres, candidate.fluid_particles,
                       runtime, error)) return false;
    const StepOptions step{.timestep = k_timestep, .substeps = 4U,
                           .gravity = {0.0F, -k_gravity, 0.0F}};
    for (std::uint32_t frame = 0U; frame < 30U; ++frame) {
        if (!step_runtime(runtime, step, error)) {
            error = "calibration settle failed: " + error;
            return false;
        }
    }
    std::vector<std::uint32_t> pixels;
    CameraController camera;
    camera.set_preset(camera_preset_for(GalleryContext::rigid_body, config));
    parallel_mater::gallery::FrameTimingCollector timings;
    using parallel_mater::gallery::FramePhase;
    auto measure = [&](bool collision) {
        timings.begin_frame();
        timings.begin_phase(FramePhase::physics);
        if (!step_runtime(runtime, step, error)) {
            error = "calibration step failed: " + error;
            return false;
        }
        timings.end_phase();
        timings.begin_phase(FramePhase::render);
        if (!runtime.renderer.render(runtime.render_frame,
                                     camera.camera(), pixels, error)) return false;
        timings.end_phase();
        timings.begin_phase(FramePhase::present);
        if (!window.present(pixels, error, runtime.renderer.native_image())) return false;
        const auto presented = parallel_mater::gallery::FrameTimingCollector::Clock::now();
        timings.end_phase(presented);
        timings.finish_frame(presented);
        samples.push_back({timings.latest()->total_ms, collision, true});
        glfwPollEvents();
        return glfwWindowShouldClose(window.native_handle()) == GLFW_FALSE;
    };
    for (std::uint32_t frame = 0U; frame < quiet_frames; ++frame)
        if (!measure(false)) { if (error.empty()) error = "calibration cancelled"; return false; }
    std::string correctness_error;
    stable = stable && stable_calibration_scene(
        runtime, true, config.brick_scale, correctness_error);
    if (!launch_calibration_ball(runtime, off_center, error)) return false;
    for (std::uint32_t frame = 0U; frame < collision_frames; ++frame)
        if (!measure(true)) { if (error.empty()) error = "calibration cancelled"; return false; }
    correctness_error.clear();
    stable = stable && stable_calibration_scene(
        runtime, false, config.brick_scale, correctness_error);
    return true;
}

double measured_seconds(
    const std::vector<parallel_mater::gallery::CalibrationSample> &samples) {
    return std::accumulate(samples.begin(), samples.end(), 0.0,
        [](double total, const auto &sample) {
            return total + sample.milliseconds / 1'000.0;
        });
}

bool calibrate_bricks(Options &options, GalleryWindow &window,
                      CalibrationResult &output, std::string &error) {
    window.set_vsync(false);
    const auto &presets = parallel_mater::gallery::brick_calibration_presets();
    std::optional<std::size_t> best;
    for (std::size_t index = 0U; index < presets.size(); ++index) {
        std::vector<parallel_mater::gallery::CalibrationSample> samples;
        bool stable = true;
        std::string trial_error;
        const auto begin = std::chrono::steady_clock::now();
        if (!run_calibration_trial(options, presets[index], window, 60U, 120U,
                                   index % 2U != 0U, samples, stable,
                                   trial_error)) {
            if (trial_error.find("memory budget") != std::string::npos) break;
            window.set_vsync(true); error = trial_error; return false;
        }
        const double duration = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        auto metrics = parallel_mater::gallery::summarize_calibration(
            samples, duration, stable, false);
        const double work = measured_seconds(samples);
        metrics.simulation_progress_ratio = work > 0.0
            ? std::min(1.0, samples.size() * static_cast<double>(k_timestep) / work)
            : 0.0;
        std::cout << "Calibration " << presets[index].brick_count
                  << " bricks: collision p95 "
                  << metrics.collision_p95_milliseconds << " ms, max "
                  << metrics.collision_maximum_milliseconds << " ms\n";
        if (index == 0U) {
            output = {.config = presets[index], .metrics = metrics,
                      .hardware =
                          parallel_mater::gallery::platform_hardware_identity()};
        }
        if (!parallel_mater::gallery::calibration_passes(metrics)) {
            if (index == 0U) {
                window.set_vsync(true);
                error = "minimum one-brick preset failed the 1080p frame budget";
                return false;
            }
            if (best.has_value()) break;
            continue;
        }
        best = index;
    }
    if (!best.has_value()) {
        window.set_vsync(true);
        error = "no brick preset met the 1080p frame budget";
        return false;
    }
    for (std::size_t index = *best + 1U; index-- > 0U;) {
        std::vector<parallel_mater::gallery::CalibrationSample> samples;
        bool stable = true;
        bool off_center = false;
        const auto begin = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - begin).count() < 180.0) {
            if (!run_calibration_trial(options, presets[index], window, 60U, 300U,
                                       off_center, samples, stable, error)) {
                window.set_vsync(true); return false;
            }
            if (!stable) break;
            off_center = !off_center;
        }
        const double duration = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - begin).count();
        auto metrics = parallel_mater::gallery::summarize_calibration(
            samples, duration, stable, false);
        const double work = measured_seconds(samples);
        metrics.simulation_progress_ratio = work > 0.0
            ? std::min(1.0, samples.size() * static_cast<double>(k_timestep) / work)
            : 0.0;
        if (parallel_mater::gallery::calibration_passes(metrics)) {
            options.bricks = presets[index];
            output = {.config = presets[index], .metrics = metrics,
                      .hardware =
                          parallel_mater::gallery::platform_hardware_identity()};
            window.set_vsync(true);
            return true;
        }
        if (index == 0U) break;
    }
    window.set_vsync(true);
    error = "no brick preset passed sustained 1080p qualification";
    return false;
}

void save_local_calibration(const CalibrationResult &result) {
    auto path = parallel_mater::gallery::default_local_profiles_path();
    path = path.parent_path() / "latest-brick-calibration.json";
    std::error_code filesystem_error;
    std::filesystem::create_directories(path.parent_path(), filesystem_error);
    if (filesystem_error) return;
    const auto temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) return;
    output << "{\n  \"verified\": false,\n  \"backend\": \""
           << result.hardware.backend << "\",\n"
           << "  \"gpu_model\": \"" << result.hardware.gpu_model << "\",\n"
           << "  \"gpu_variant\": \"" << result.hardware.gpu_variant << "\",\n"
           << "  \"machine_model\": \"" << result.hardware.machine_model << "\",\n"
           << "  \"cpu_model\": \"" << result.hardware.cpu_model << "\",\n"
           << "  \"memory_bytes\": " << result.hardware.memory_bytes << ",\n"
           << "  \"operating_system\": \""
           << result.hardware.operating_system << "\",\n"
           << "  \"driver\": \"" << result.hardware.driver << "\",\n"
           << "  \"power_mode\": \"" << result.hardware.power_mode << "\",\n"
           << "  \"width\": " << brick_render_width << ",\n"
           << "  \"height\": " << brick_render_height << ",\n"
           << "  \"scene_version\": "
           << parallel_mater::gallery::brick_scene_version << ",\n"
           << "  \"solver_version\": \""
           << parallel_mater::gallery::slang_solver_version << "\",\n"
           << "  \"build_revision\": \""
           << PARALLEL_MATER_GALLERY_GIT_COMMIT << "\",\n"
           << "  \"brick_count\": " << result.config.brick_count << ",\n"
           << "  \"brick_scale\": " << result.config.brick_scale << ",\n"
           << "  \"wall_planes\": " << result.config.wall_planes << ",\n"
           << "  \"collision_p95_milliseconds\": "
           << result.metrics.collision_p95_milliseconds << ",\n"
           << "  \"collision_maximum_milliseconds\": "
           << result.metrics.collision_maximum_milliseconds << ",\n"
           << "  \"quiet_p95_milliseconds\": "
           << result.metrics.quiet_p95_milliseconds << ",\n"
           << "  \"quiet_maximum_milliseconds\": "
           << result.metrics.quiet_maximum_milliseconds << ",\n"
           << "  \"duration_seconds\": "
           << result.metrics.duration_seconds << ",\n"
           << "  \"simulation_progress_ratio\": "
           << result.metrics.simulation_progress_ratio << ",\n"
           << "  \"stable\": "
           << (result.metrics.stable ? "true" : "false") << ",\n"
           << "  \"no_dropped_steps\": "
           << (result.metrics.no_dropped_steps ? "true" : "false") << "\n}\n";
    output.close();
    if (!output) return;
    std::filesystem::rename(temporary, path, filesystem_error);
    if (filesystem_error) std::filesystem::remove(temporary);
}

bool verify_and_save_profile(const Options &options,
                             const CalibrationResult &result,
                             std::string &error) {
    parallel_mater::gallery::VerifiedBrickProfile profile;
    profile.hardware = result.hardware;
    profile.scene = result.config;
    profile.solver_version = parallel_mater::gallery::slang_solver_version;
    profile.build_revision = PARALLEL_MATER_GALLERY_GIT_COMMIT;
    profile.metrics = result.metrics;
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream timestamp;
    timestamp << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    profile.verified_at = timestamp.str();
    const char *user = std::getenv("USER");
    if (user == nullptr) user = std::getenv("USERNAME");
    profile.verifier = user != nullptr ? user : "human-verified";
    const std::filesystem::path destination = options.profiles_file_overridden
        ? options.profiles_file
        : parallel_mater::gallery::default_local_profiles_path().parent_path() /
              "verified-profile-export.json";
    const bool saved = parallel_mater::gallery::save_verified_profile(
        destination, profile, error);
    if (saved) std::cout << "Verified profile written to " << destination << '\n';
    return saved;
}

} // namespace

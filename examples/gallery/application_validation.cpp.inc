// Headless capture, scene construction, and output validation.
[[nodiscard]] bool write_ppm(const std::filesystem::path &path,
                             const std::vector<std::uint32_t> &pixels,
                             std::uint32_t width, std::uint32_t height) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }
    output << "P6\n" << width << ' ' << height << "\n255\n";
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(pixels.data());
    for (std::uint32_t row = 0; row < height; ++row) {
        const std::uint32_t source_row = height - row - 1U;
        for (std::uint32_t column = 0; column < width; ++column) {
            const std::size_t offset =
                (static_cast<std::size_t>(source_row) * width + column) * 4U;
            output.write(reinterpret_cast<const char *>(bytes + offset), 3);
        }
    }
    return static_cast<bool>(output);
}

[[nodiscard]] bool validate_render(
    const std::vector<std::uint32_t> &pixels, std::uint32_t width,
    std::uint32_t height, std::string &error) {
    const std::size_t expected =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (pixels.size() != expected || pixels.empty()) {
        error = "renderer returned an unexpected pixel count";
        return false;
    }
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(pixels.data());
    unsigned minimum_luminance = 3U * 255U;
    unsigned maximum_luminance = 0U;
    for (std::size_t index = 0; index < pixels.size(); ++index) {
        const unsigned luminance = static_cast<unsigned>(bytes[index * 4U]) +
                                   static_cast<unsigned>(bytes[index * 4U + 1U]) +
                                   static_cast<unsigned>(bytes[index * 4U + 2U]);
        minimum_luminance = std::min(minimum_luminance, luminance);
        maximum_luminance = std::max(maximum_luminance, luminance);
    }
    if (maximum_luminance < 48U || maximum_luminance - minimum_luminance < 96U) {
        error = "renderer produced a black or nearly uniform frame";
        return false;
    }
    return true;
}

[[nodiscard]] std::filesystem::path gallery_resource_path(
    const Options &options, const std::filesystem::path &configured) {
    if (configured.is_absolute() || options.executable_path.empty())
        return configured;
    std::error_code filesystem_error;
    const std::filesystem::path bundled =
        (options.executable_path.parent_path() / configured).lexically_normal();
    return std::filesystem::exists(bundled, filesystem_error)
        ? bundled : configured;
}

[[nodiscard]] GallerySessionConfig desktop_session_config(
    const Options &options, GalleryContext context) {
    GalleryWorkloadLimits limits = options.workload_limits.value_or(
        GalleryWorkloadLimits{
            .name = "desktop-explicit",
            .platform = "Desktop",
            .machine_model = "*",
            .backend = options.backend
                ? gallery_backend_name(*options.backend) : "*",
            .brick_count = options.bricks.brick_count,
            .fluid_particles = options.fluid_particles,
            .coupled_bodies = std::numeric_limits<std::uint32_t>::max(),
            .smoke_particles = std::numeric_limits<std::uint32_t>::max(),
            .dump_spheres = options.dump_spheres,
            .render_width = options.width,
            .render_height = options.height,
            .target_fps = GalleryFramePolicy::default_target_fps,
        });
    limits.fluid_particles = options.fluid_particles;
    limits.dump_spheres = options.dump_spheres;
    limits.render_width = options.width;
    limits.render_height = options.height;
    if (limits.target_fps == 0U)
        limits.target_fps = GalleryFramePolicy::default_target_fps;
    const bool create_native_renderer =
        (options.backend == PM_NATIVE_BACKEND_METAL ||
         options.backend == PM_NATIVE_BACKEND_VULKAN) &&
        !options.calibrate && options.headless_output.empty();
    GallerySessionConfig config{
        .assets = gallery_resource_path(
            options, PARALLEL_MATER_GALLERY_ASSET_DIRECTORY),
        .profiles = device_profiles_path(options),
        .platform = "Desktop",
        .model = "*",
        .backend = options.backend
            ? gallery_backend_name(*options.backend) : "parallel-mater-loader",
        .native_backend = options.backend,
        .gpu_rendering = create_native_renderer,
        .workload_limits = limits,
        .bricks = options.bricks,
        .dump_spheres = options.dump_spheres,
        .fluid_particles = options.fluid_particles,
        .initial_scene = static_cast<unsigned>(gallery_context_index(context)),
        .rigid_body_asset = gallery_resource_path(options, options.scene),
        .replacement = GalleryReplacementPolicy::preserve_until_ready,
        .create_host_renderer = true,
        // Direct renderers need one completed output generation before the
        // window can present its first frame. Headless/software desktop paths
        // retain their existing explicit stepping behavior.
        .prime_scene = create_native_renderer,
    };
    const auto backend = options.backend;
    config.validate_scene =
        [backend](GalleryContext selected, const SceneDefinition &scene,
                  std::string &failure) {
        if (selected != GalleryContext::rigid_body) return true;
        const auto admission =
            parallel_mater::gallery::assess_workload_admission(
                parallel_mater::gallery::platform_device_memory_budget(backend),
                parallel_mater::gallery::estimated_rigid_contact_bytes(
                    static_cast<std::uint32_t>(scene.rigid_bodies.size())));
        if (!admission.accepted()) {
            failure = parallel_mater::gallery::workload_admission_message(
                admission.status);
            return false;
        }
        return true;
    };
    return config;
}

[[nodiscard]] bool select_desktop_scene(
    GallerySession &session, const Options &options, GalleryContext context,
    std::uint32_t dump_spheres, std::uint32_t fluid_particles,
    std::string &error) {
    const GallerySceneParameters previous = session.scene_parameters();
    session.set_scene_parameters({options.bricks, dump_spheres,
                                  fluid_particles});
    try {
        session.select(
            static_cast<unsigned>(gallery_context_index(context)));
        std::cout << "Gallery scene ready index="
                  << gallery_context_index(context)
                  << " revision=" << session.revision() << '\n'
                  << std::flush;
        return true;
    } catch (const std::exception &failure) {
        session.set_scene_parameters(previous);
        error = failure.what();
        return false;
    }
}

[[nodiscard]] bool reset_desktop_scene(GallerySession &session,
                                       std::string &error) {
    try {
        session.reset();
        std::cout << "Gallery scene ready index=" << session.selection()
                  << " revision=" << session.revision() << '\n'
                  << std::flush;
        return true;
    } catch (const std::exception &failure) {
        error = failure.what();
        return false;
    }
}

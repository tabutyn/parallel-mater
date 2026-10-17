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

[[nodiscard]] bool build_runtime(const Options &options,
                                 GalleryContext context,
                                 std::uint32_t dump_spheres,
                                 std::uint32_t fluid_particles,
                                 GalleryRuntime &output,
                                 std::string &error) {
    const GalleryEntry &entry = gallery_entry(context);
    const std::array scene_paths{
        options.scene,
        std::filesystem::path(PARALLEL_MATER_CONSTRAINT_FIXED_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_CONSTRAINT_POINT_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_CONSTRAINT_HINGE_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_CONSTRAINT_PISTON_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_CONSTRAINT_GENERIC_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_CONSTRAINT_MOTOR_SPRING_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_DUMP_TRUCK_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_FLUID_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_FLUID_RIGID_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_PEGS_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_CLOTH_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_CLOTH_TEAR_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_CLOTH_PAINT_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_CLOTH_WATER_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_SOFT_BODY_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_SOFT_BODY_RIGID_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_SOFT_BODY_CLOTH_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_SOFT_BODY_FLUID_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_ROPE_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_ROPE_FLUID_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_ROPE_SOFT_BODY_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_ROPE_CLOTH_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_SMOKE_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_SMOKE_WATER_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_SMOKE_SOFT_BODY_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_SMOKE_CLOTH_SCENE_PATH),
        std::filesystem::path(PARALLEL_MATER_SMOKE_ROPE_SCENE_PATH)};
    GalleryRuntimeCreateInfo request;
    request.context = context;
    request.asset = gallery_resource_path(
        options, scene_paths[static_cast<std::size_t>(entry.source)]);
    request.backend = options.backend;
    request.debug_frame_capacity =
        context == GalleryContext::constraint_fixed ? 300U : 30U;
    request.width = options.width;
    request.height = options.height;
    request.replacement = GalleryReplacementPolicy::preserve_until_ready;
    const auto bricks = options.bricks;
    const auto workload_limits = options.workload_limits;
    const auto backend = options.backend;
    request.configure_scene =
        [context, bricks, workload_limits, backend, dump_spheres,
         fluid_particles](SceneDefinition &scene, std::string &failure) {
        const GalleryEntry &selected = gallery_entry(context);
        if (context == GalleryContext::rigid_body) {
            SceneDefinition generated;
            if (!parallel_mater::gallery::make_brick_scene(
                    scene, bricks, generated, failure)) {
                failure = "brick scene generation failed: " + failure;
                return false;
            }
            scene = std::move(generated);
        }
        if (selected.has_fluid) scene.fluid_options.capacity = fluid_particles;
        if (workload_limits &&
            (context == GalleryContext::fluid || context == GalleryContext::fluid_rigid)) {
            auto limits = *workload_limits;
            limits.fluid_particles = fluid_particles;
            if (!parallel_mater::gallery::apply_liquid_workload_limits(
                    scene, limits, failure)) return false;
        }
        if (context == GalleryContext::dump &&
            !parallel_mater::gallery::configure_dump_payload(
                scene, dump_spheres, failure)) return false;
        if (context == GalleryContext::rigid_body) {
            const std::uint64_t free_bytes =
                parallel_mater::gallery::platform_available_device_memory(backend);
            if (free_bytes != 0U &&
                parallel_mater::gallery::estimated_rigid_contact_bytes(
                    static_cast<std::uint32_t>(scene.rigid_bodies.size())) >
                    free_bytes * 7U / 10U) {
                failure = "brick scene exceeds the available GPU memory budget";
                return false;
            }
        }
        return true;
    };
    const bool create_native_renderer =
        options.backend == PM_NATIVE_BACKEND_METAL && !options.calibrate &&
        options.headless_output.empty();
    request.prepare_runtime =
        [context, create_native_renderer](GalleryRuntime &next,
                                          std::string &failure) {
        const GalleryEntry &selected = gallery_entry(context);
        if (context == GalleryContext::dump &&
            !next.dump_bed.initialize(next.scene)) {
            failure = "dump truck needs its authored DumpLift joint";
            return false;
        }
        if (selected.controls == GalleryControlPolicy::motor_drive) {
            for (std::size_t spring_index = 0U;
                 spring_index < next.scene.rigid_constraints.size();
                 ++spring_index) {
                const auto &spring =
                    next.scene.rigid_constraints[spring_index];
                if (spring.options.type !=
                    RigidConstraintType::generic_spring) {
                    continue;
                }
                const auto motor = std::find_if(
                    next.scene.rigid_constraints.begin(),
                    next.scene.rigid_constraints.end(),
                    [&](const auto &candidate) {
                        return candidate.options.type ==
                                   RigidConstraintType::motor &&
                               candidate.body_a == spring.body_b;
                    });
                if (motor == next.scene.rigid_constraints.end() ||
                    (motor->name.find("Front") == std::string::npos &&
                     motor->name.find("Rear") == std::string::npos)) {
                    failure = spring.name +
                        ": suspension hub needs a named Front or Rear motor";
                    return false;
                }
                next.suspension_steering_joints.push_back({
                    spring_index, spring.options.local_orientation_a,
                    motor->name.find("Front") != std::string::npos});
            }
            if (next.suspension_steering_joints.size() != 4U) {
                failure = "vehicle scene needs four steerable suspension hubs";
                return false;
            }
        }
        if (!next.arrow_forces.initialize(
                next.scene, next.instance, failure)) return false;
        next.gravity_tilt_bodies.reserve(next.scene.rigid_bodies.size());
        for (std::size_t index = 0U;
             index < next.scene.rigid_bodies.size(); ++index) {
            const auto &body = next.scene.rigid_bodies[index];
            if (body.options.motion == MotionType::dynamic &&
                !body.follows_gravity_tilt) {
                next.gravity_tilt_bodies.push_back(
                    next.instance.rigid_bodies[index]);
            }
        }
        if (context == GalleryContext::constraint_fixed &&
            !next.fixed_collector.initialize(
                next.scene, next.instance, failure)) return false;
        if (context == GalleryContext::rigid_body) {
            const auto dynamic_count = std::count_if(
                next.scene.rigid_bodies.begin(),
                next.scene.rigid_bodies.end(), [](const auto &body) {
                    return body.options.motion == MotionType::dynamic;
                });
            std::cout << "Rigid gravity tilt: "
                      << dynamic_count - next.gravity_tilt_bodies.size()
                      << " follow, " << next.gravity_tilt_bodies.size()
                      << " keep vertical\n";
        }
        if (next.arrow_forces.active()) {
            for (const auto &body : next.scene.rigid_bodies) {
                if (body.arrow_force > 0.0F) {
                    std::cout
                        << "Arrow force: " << body.source_name << " "
                        << body.arrow_force
                        << " N (camera-relative ground plane; gravity stays vertical)\n";
                }
            }
        }
#ifdef PM_GALLERY_METAL_RENDERER
        if (create_native_renderer) {
            try {
                next.native_renderer = std::make_shared<
                    parallel_mater::gallery::GpuGalleryRenderer>(
                    next.simulation.device(), next.scene, next.instance);
            } catch (const std::exception &exception) {
                failure = exception.what();
                return false;
            }
        }
#else
        (void)create_native_renderer;
#endif
        for (std::size_t index = 0;
             index < next.scene.rigid_bodies.size(); ++index) {
            if (next.scene.rigid_bodies[index].options.motion ==
                parallel_mater::gallery::MotionType::kinematic) {
                next.kinematic_index = index;
                next.kinematic_target =
                    next.scene.rigid_bodies[index].options.initial_state;
                break;
            }
        }
        return true;
    };
    return output.replace(std::move(request), error);
}

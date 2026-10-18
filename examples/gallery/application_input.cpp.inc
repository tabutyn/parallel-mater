// One platform-neutral parser and input callback path for every renderer.
[[nodiscard]] bool parse_positive(std::string_view value, int &output) {
    int parsed = 0;
    const auto result =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
        parsed <= 0 || parsed > 100'000) {
        return false;
    }
    output = parsed;
    return true;
}

[[nodiscard]] bool parse_count(std::string_view value, std::uint32_t minimum,
                               std::uint32_t maximum, std::uint32_t &output) {
    std::uint32_t parsed = 0U;
    const auto result =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
        parsed < minimum || parsed > maximum) {
        return false;
    }
    output = parsed;
    return true;
}

[[nodiscard]] bool parse_scale(std::string_view value, float &output) {
    float parsed = 0.0F;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
        !std::isfinite(parsed) || parsed < brick_minimum_scale ||
        parsed > brick_maximum_scale) return false;
    output = parsed;
    return true;
}

[[nodiscard]] bool parse_options(int argc, char **argv, Options &output) {
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--scene" && index + 1 < argc) {
            output.scene = argv[++index];
        } else if (argument == "--headless" && index + 1 < argc) {
            output.headless_output = argv[++index];
        } else if (argument == "--interactive-smoke" && index + 1 < argc) {
            output.interactive_smoke_output = argv[++index];
        } else if (argument == "--frame-timings" && index + 1 < argc) {
            output.frame_timings_output = argv[++index];
            if (output.frame_timings_output.empty()) return false;
        } else if (argument == "--frames" && index + 1 < argc) {
            if (!parse_positive(argv[++index], output.frames)) {
                return false;
            }
        } else if (argument == "--contact-passes" && index + 1 < argc) {
            if (!parse_count(argv[++index], 0U, 64U, output.contact_pass_limit)) return false;
        } else if (argument == "--dump-spheres" && index + 1 < argc) {
            output.dump_spheres_overridden = true;
            const GalleryEntry &entry = gallery_entry(GalleryContext::dump);
            if (!parse_count(argv[++index], entry.minimum_count,
                             entry.maximum_count, output.dump_spheres)) {
                return false;
            }
            output.initial_context = GalleryContext::dump;
        } else if (argument == "--brick-count" && index + 1 < argc) {
            if (!parse_count(argv[++index], brick_minimum_count,
                             brick_maximum_count, output.bricks.brick_count)) return false;
            output.initial_context = GalleryContext::rigid_body;
            output.bricks_overridden = true;
        } else if (argument == "--brick-scale" && index + 1 < argc) {
            if (!parse_scale(argv[++index], output.bricks.brick_scale)) return false;
            output.initial_context = GalleryContext::rigid_body;
            output.bricks_overridden = true;
        } else if (argument == "--brick-planes" && index + 1 < argc) {
            if (!parse_count(argv[++index], 1U, brick_maximum_planes,
                             output.bricks.wall_planes)) return false;
            output.initial_context = GalleryContext::rigid_body;
            output.bricks_overridden = true;
        } else if (argument == "--profiles-file" && index + 1 < argc) {
            output.profiles_file = argv[++index];
            output.profiles_file_overridden = true;
        } else if (argument == "--backend" && index + 1 < argc) {
            output.backend =
                parallel_mater::gallery::gallery_backend_from_name(argv[++index]);
            if (!output.backend) return false;
        } else if (argument == "--calibrate") {
            output.calibrate = true;
            output.initial_context = GalleryContext::rigid_body;
        } else if (argument == "--print-brick-profile") {
            output.print_brick_profile = true;
        } else if (argument == "--fluid-particles" && index + 1 < argc) {
            output.fluid_particles_overridden = true;
            const GalleryEntry &entry = gallery_entry(GalleryContext::fluid);
            if (!parse_count(argv[++index], entry.minimum_count,
                             entry.maximum_count,
                             output.fluid_particles)) return false;
            if (!is_fluid_context(output.initial_context))
                output.initial_context = GalleryContext::fluid;
        } else if (const GalleryEntry *entry = entry_for_option(argument)) {
            output.initial_context = entry->context;
        } else if (argument == "--cloth-tilt-degrees" && index + 1 < argc) {
            if (!parse_count(argv[++index], 1U, 45U,
                             output.headless_cloth_tilt_degrees)) return false;
        } else if (argument == "--gravity-tilt-degrees" && index + 1 < argc) {
            if (!parse_count(argv[++index], 1U, 80U,
                             output.headless_cloth_tilt_degrees)) return false;
        } else if (argument == "--cloth-tilt-after-frames" && index + 1 < argc) {
            if (!parse_count(argv[++index], 0U, 100000U,
                             output.headless_cloth_tilt_after_frames)) return false;
        } else if (argument == "--cloth-tilt-left") {
            output.headless_cloth_tilt_left = true;
        } else if ((argument == "--constraint-action-after-frames" || argument == "--dump-after-frames") &&
                   index + 1 < argc) {
            if (!parse_count(argv[++index], 1U, 100000U,
                             output.headless_constraint_action_after_frames))
                return false;
        } else if (argument == "--motor-forward") {
            output.headless_motor_forward = true;
        } else if (argument == "--motor-right") {
            output.headless_motor_right = true;
        } else if (argument == "--fluid-particle-view") {
            if (!is_fluid_context(output.initial_context))
                output.initial_context = GalleryContext::fluid;
            output.fluid_particle_view = true;
        } else if (argument == "--trace-fluid-escapes") {
            if (!is_fluid_context(output.initial_context))
                output.initial_context = GalleryContext::fluid;
            output.trace_fluid_escapes = true;
        } else if (argument == "--water-cloth-debug" || argument == "--cloth-debug") {
            if (argument == "--water-cloth-debug")
                output.initial_context = GalleryContext::water_cloth;
            output.cloth_debug = true;
        } else if (argument == "--physics-capture" && index + 1 < argc) {
            output.physics_capture_output = argv[++index];
        } else if (argument == "--version") {
            std::cout << "Parallel-Mater " PARALLEL_MATER_GALLERY_VERSION
                      << " (" << build_identity_label() << ")\n";
            std::exit(0);
        } else if (argument == "--help") {
            std::cout << "parallel-mater-gallery [--scene file.glb] "
                         "[--brick-count 1..4096] [--brick-scale 0.5..2] "
                         "[--brick-planes 1..16] [--calibrate] "
                         "[--profiles-file file.json] "
                         "[--backend cuda|d3d12|metal|vulkan] "
                         "[--interactive-smoke final.ppm] "
                         "[--print-brick-profile] "
                         "[--dump-spheres N] [";
            bool first = true;
            for (const GalleryEntry &entry : gallery_entries) {
                if (entry.command_line_option.empty()) continue;
                std::cout << (first ? "" : "|") << entry.command_line_option;
                first = false;
            }
            std::cout << "] [--fluid-particles N] [--contact-passes 0..64] "
                         "[--gravity-tilt-degrees 1..80 (headless)] "
                         "[--cloth-tilt-after-frames N (headless)] "
                         "[--cloth-tilt-left (headless)] "
                         "[--constraint-action-after-frames N (headless)] "
                         "[--dump-after-frames N (headless)] "
                         "[--motor-forward (headless)] "
                         "[--motor-right (headless)] "
                         "[--fluid-particle-view] [--trace-fluid-escapes] "
                         "[--cloth-debug | --water-cloth-debug] "
                         "[--physics-capture output.log] "
                         "[--frame-timings output.csv] "
                         "[--version] "
                         "[--headless output.ppm] "
                         "[--frames N]\n";
            std::exit(0);
        } else {
            return false;
        }
    }
    std::string brick_error;
    return (output.frame_timings_output.empty() ||
            (!output.calibrate && !output.print_brick_profile)) &&
        (!output.print_brick_profile || output.headless_output.empty()) &&
        (!output.trace_fluid_escapes || !output.headless_output.empty()) &&
        validate_brick_config(output.bricks, brick_error);
}

[[nodiscard]] parallel_mater::Vec3 initial_scene_gravity(
    GalleryContext context, float scale) {
    (void)context;
    return {0.0F, -k_gravity * scale, 0.0F};
}

void mouse_button(GLFWwindow *window, int button, int action, int modifiers) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) {
        return;
    }
    auto *input = static_cast<InputState *>(glfwGetWindowUserPointer(window));
    if (action == GLFW_RELEASE) {
        input->camera->end_drag();
        return;
    }
    if (action != GLFW_PRESS || input->shared.parameters_visible ||
        input->shared.catalog_visible) return;
    double x = 0.0, y = 0.0;
    glfwGetCursorPos(window, &x, &y);
    input->camera->begin_drag((modifiers & GLFW_MOD_SHIFT) != 0
                                 ? CameraDragMode::pan
                                 : CameraDragMode::orbit,
                             x, y);
}

void cursor_position(GLFWwindow *window, double x, double y) {
    auto *input = static_cast<InputState *>(glfwGetWindowUserPointer(window));
    int height = 0;
    glfwGetWindowSize(window, nullptr, &height);
    input->camera->move_cursor(
        x, y, height,
        !input->shared.parameters_visible && !input->shared.catalog_visible);
}

void scroll(GLFWwindow *window, double, double offset) {
    auto *input = static_cast<InputState *>(glfwGetWindowUserPointer(window));
    if (input->shared.parameters_visible || input->shared.catalog_visible) {
        return;
    }
    input->camera->zoom(offset);
}

void character_input(GLFWwindow *window, unsigned int codepoint) {
    auto *input = static_cast<InputState *>(glfwGetWindowUserPointer(window));
    const bool digit = codepoint >= '0' && codepoint <= '9';
    const bool scale_point = input->brick_dialog && input->brick_field == 1U &&
        codepoint == '.';
    if (!input->shared.parameters_visible || (!digit && !scale_point)) {
        return;
    }
    if (input->replace_count_value) {
        if (input->brick_dialog) input->brick_values[input->brick_field].clear();
        else input->count_value.clear();
        input->replace_count_value = false;
    }
    std::string &value = input->brick_dialog
        ? input->brick_values[input->brick_field] : input->count_value;
    if (value.size() < 6U && (!scale_point || value.find('.') == std::string::npos)) {
        value.push_back(static_cast<char>(codepoint));
        input->count_value_invalid = false;
    }
}

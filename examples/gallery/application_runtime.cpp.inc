[[nodiscard]] RigidBodyState rigid_state(const PmRigidState &source) noexcept {
    return {{source.position[0], source.position[1], source.position[2]},
            {source.orientation[0], source.orientation[1],
             source.orientation[2], source.orientation[3]},
            {source.linearVelocity[0], source.linearVelocity[1],
             source.linearVelocity[2]},
            {source.angularVelocity[0], source.angularVelocity[1],
             source.angularVelocity[2]}};
}

[[nodiscard]] const GalleryEntry *entry_for_option(
    std::string_view option) noexcept {
    const auto found = std::find_if(
        gallery_entries.begin(), gallery_entries.end(),
        [&](const GalleryEntry &entry) {
            return !entry.command_line_option.empty() &&
                   entry.command_line_option == option;
        });
    return found == gallery_entries.end() ? nullptr : &*found;
}

struct FluidEscapeTrace {
    int first_frame{-1};
    int first_local_frame{-1};
    std::uint32_t peak_below{};
    std::uint32_t final_below{};
    std::uint32_t peak_below_local{};
    std::uint32_t final_below_local{};
    float lowest_y{std::numeric_limits<float>::max()};
    std::vector<std::array<parallel_mater::Vec3, 64>> recent_positions{};
    std::vector<std::uint8_t> recent_counts{};
};

[[nodiscard]] bool trace_fluid_escapes(const CommandSceneFrame &simulation,
                                       const StaticTriangleSurface &floor,
                                       int frame, FluidEscapeTrace &trace) {
    std::vector<parallel_mater::Vec3> positions;
    positions.reserve(simulation.fluid_particles.size());
    for (const auto &particle : simulation.fluid_particles)
        positions.push_back({particle.position[0], particle.position[1],
                             particle.position[2]});
    const parallel_mater::Vec3 mesh_minimum = floor.minimum();
    const parallel_mater::Vec3 mesh_maximum = floor.maximum();
    const float threshold = mesh_minimum.y;
    std::uint32_t below = 0U;
    std::uint32_t below_local = 0U;
    std::size_t first = positions.size();
    std::size_t first_local = positions.size();
    float first_local_surface = 0.0F;
    for (std::size_t i = 0; i < positions.size(); ++i) {
        trace.lowest_y = std::min(trace.lowest_y, positions[i].y);
        const auto surface = floor.height(
            positions[i].x, positions[i].z, SurfaceSelection::lowest);
        if (surface && positions[i].y < *surface - 0.001F) {
            ++below_local;
            if (first_local == positions.size()) {
                first_local = i;
                first_local_surface = *surface;
            }
        }
        if (positions[i].y < threshold) {
            ++below;
            if (first == positions.size()) first = i;
        }
    }
    trace.peak_below = std::max(trace.peak_below, below);
    trace.final_below = below;
    trace.peak_below_local = std::max(trace.peak_below_local, below_local);
    trace.final_below_local = below_local;
    std::vector<std::uint32_t> stable_ids;
    if ((trace.first_frame < 0 || trace.first_local_frame < 0) &&
        !positions.empty()) {
        stable_ids.reserve(simulation.fluid_particles.size());
        for (const auto &particle : simulation.fluid_particles)
            stable_ids.push_back(particle.stableId);
        for (std::size_t i = 0; i < positions.size(); ++i) {
            const std::uint32_t id = stable_ids[i];
            if (id >= trace.recent_positions.size()) {
                trace.recent_positions.resize(static_cast<std::size_t>(id) + 1U);
                trace.recent_counts.resize(static_cast<std::size_t>(id) + 1U);
            }
            trace.recent_positions[id][frame % 64] = positions[i];
            trace.recent_counts[id] = std::min<std::uint8_t>(
                64U, static_cast<std::uint8_t>(trace.recent_counts[id] + 1U));
        }
    }
    if (below_local != 0U && trace.first_local_frame < 0) {
        trace.first_local_frame = frame;
        const auto p = positions[first_local];
        std::cout << "First local floor penetration frame=" << frame
                  << " id=" << stable_ids[first_local] << " position=("
                  << p.x << ',' << p.y << ',' << p.z << ") floor_y="
                  << first_local_surface << " below_local=" << below_local
                  << '\n';
    }
    if (below != 0U && trace.first_frame < 0) {
        trace.first_frame = frame;
        const std::uint32_t stable_id = stable_ids[first];
        const auto p = positions[first];
        const bool within_xz_bounds = p.x >= mesh_minimum.x &&
            p.x <= mesh_maximum.x && p.z >= mesh_minimum.z &&
            p.z <= mesh_maximum.z;
        std::cout << "First fluid escape frame=" << frame
                  << " id=" << stable_id << " position=(" << p.x << ','
                  << p.y << ',' << p.z << ") below=" << below
                  << " within_mesh_xz_bounds=" << within_xz_bounds << '\n';
        const int history_count = trace.recent_counts[stable_id];
        for (int sample_frame = frame - history_count + 1;
             sample_frame <= frame; ++sample_frame) {
            const auto sample =
                trace.recent_positions[stable_id][sample_frame % 64];
            float normal_y = 0.0F;
            const auto surface = floor.height(
                sample.x, sample.z, SurfaceSelection::highest, &normal_y);
            std::cout << "  particle " << stable_id << " frame=" << sample_frame
                      << " y=" << sample.y << " surface_y=";
            if (surface) std::cout << *surface << " normal_y=" << normal_y;
            else std::cout << "none";
            std::cout << " x=" << sample.x << " z=" << sample.z << '\n';
        }
    }
    return true;
}

[[nodiscard]] parallel_mater::gallery::GalleryControlInput directional_input(
    GLFWwindow *window, bool normalize_diagonal = true) {
    parallel_mater::gallery::GalleryControlInput input{
        .right =
            static_cast<float>(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) -
            static_cast<float>(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS),
        .forward =
            static_cast<float>(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) -
            static_cast<float>(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS),
    };
    const float length = std::hypot(input.right, input.forward);
    if (normalize_diagonal && length > 1.0F) {
        input.right /= length;
        input.forward /= length;
    }
    return input;
}

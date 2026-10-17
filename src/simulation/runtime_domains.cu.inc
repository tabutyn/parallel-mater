// SPDX-License-Identifier: MIT
// Included by runtime.cpp; shared backend-neutral runtime implementation.

Status SimulationRuntime::add_fluid(FluidOptions options,
                        DeviceSpan<const FluidParticle> initial_particles,
                        FluidId &output, device::ExecutionQueue queue) noexcept {
    const device::ExecutionQueue stream = queue;
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.capacity == 0U || options.capacity > INT_MAX ||
        initial_particles.size > options.capacity ||
        (initial_particles.size != 0U && initial_particles.data == nullptr) ||
        options.solver_iterations == 0U || options.solver_iterations > 16U ||
        options.maximum_neighbors == 0U ||
        !finite(options.particle_radius) || options.particle_radius <= 0.0F ||
        !finite(options.support_radius) ||
        options.support_radius < 2.0F * options.particle_radius ||
        !finite(options.rest_density) || options.rest_density <= 0.0F ||
        !finite(options.repulsion) || options.repulsion < 0.0F ||
        !finite(options.viscosity) || options.viscosity < 0.0F ||
        !finite(options.velocity_damping) || options.velocity_damping < 0.0F ||
        !finite(options.maximum_speed) || options.maximum_speed <= 0.0F ||
        !finite(options.normal_damping) || options.normal_damping < 0.0F ||
        !finite(options.rest_particle_volume) ||
        options.rest_particle_volume < 0.0F ||
        !finite(options.maximum_pair_acceleration) ||
        options.maximum_pair_acceleration < 0.0F) {
        return failure(StatusCode::invalid_argument, "fluid options or initial particles are invalid");
    }
    std::uint32_t slot = 0U;
    for (; slot < impl_->fluids.size(); ++slot) {
        if (!impl_->fluids[slot] || !impl_->fluids[slot]->alive) break;
    }
    if (slot == impl_->fluids.size())
        return failure(StatusCode::capacity_exceeded, "fluid capacity exhausted");
    std::unique_ptr<FluidStorage> fluid;
    try { fluid = std::make_unique<FluidStorage>(); }
    catch (...) { return failure(StatusCode::out_of_memory, "fluid owner allocation failed"); }
    fluid->generation = impl_->fluids[slot].generation();
    fluid->options = options;
    StorageAllocator allocator;
    status = simulation::visit_fluid_storage(*fluid, options.capacity, allocator);
    if (!status) return status;
    PmResult error = PM_SUCCESS;
    *fluid->count = static_cast<std::uint32_t>(initial_particles.size);
    fluid->next_id = static_cast<std::uint32_t>(initial_particles.size);
    fluid->initial_count = initial_particles.size;
    if (!initial_particles.empty()) {
        *impl_->fluid_neighbor_overflow = 0U;
        simulation::copy_initial_fluid(
            (initial_particles.size + 127U) / 128U, 128U, stream,
            initial_particles.data, static_cast<std::uint32_t>(initial_particles.size),
            fluid->positions, fluid->velocities, fluid->ids, fluid->foam,
            fluid->temperatures,
            impl_->fluid_neighbor_overflow);
        error = impl_->execution.services->last_error(true);
        if (error == PM_SUCCESS) error = wait_queue(stream);
        if (error != PM_SUCCESS)
            return backend_failure(error, "initial fluid copy failed");
        status = read_host_objects(impl_->execution, impl_->fluid_neighbor_overflow);
        if (!status) return status;
        if (*impl_->fluid_neighbor_overflow != 0U)
            return failure(StatusCode::invalid_argument,
                           "initial fluid particles must be finite");
    }
    fluid->alive = true;
    output = {slot, fluid->generation};
    impl_->fluids[slot] = std::move(fluid);
    ++impl_->fluid_count;
    ++impl_->revision;
    return success();
}
Status SimulationRuntime::add_fluid_geometry(FluidOptions options,
                                 FluidGeometrySource source, FluidId &output,
                                 device::ExecutionQueue queue) noexcept {
    std::vector<FluidParticle> sampled;
    Status status = sample_fluid_geometry(source, sampled);
    if (!status) return status;
    if (options.capacity == 0U)
        return failure(StatusCode::invalid_argument,
                       "fluid geometry requires a positive fluid capacity");
    if (sampled.size() > options.capacity) {
        std::vector<FluidParticle> selected;
        try {
            selected.reserve(options.capacity);
            for (std::size_t i = 0; i < options.capacity; ++i)
                selected.push_back(sampled[i * sampled.size() /
                                            options.capacity]);
        } catch (...) {
            return failure(StatusCode::out_of_memory,
                           "fluid geometry selection allocation failed");
        }
        sampled.swap(selected);
    }
    FluidParticle *device = nullptr;
    PmResult error = queue.services->allocate(
        sampled.size() * sizeof(FluidParticle), false,
        reinterpret_cast<void **>(&device));
    if (error != PM_SUCCESS)
        return backend_failure(error, "fluid geometry upload allocation failed");
    error = copy_bytes_wait(device, sampled.data(),
                       sampled.size() * sizeof(FluidParticle),
                       device::CopyDirection::host_to_device, queue);
    if (error == PM_SUCCESS)
        status = add_fluid(options, {device, sampled.size()}, output, queue);
    queue.services->free(device);
    if (error != PM_SUCCESS)
        return backend_failure(error, "fluid geometry upload failed");
    return status;
}

Status SimulationRuntime::remove_fluid(
    FluidId id, device::ExecutionQueue) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(id, fluid))) return status;
    for (const auto &coupling : impl_->fluid_soft_couplings)
        if (coupling && coupling->alive && coupling->options.fluid == id)
            return failure(StatusCode::invalid_argument,
                           "fluid is still referenced by a soft-body coupling");
    for (const auto &coupling : impl_->fluid_rope_couplings)
        if (coupling && coupling->alive && coupling->options.fluid == id)
            return failure(StatusCode::invalid_argument,
                           "fluid is still referenced by a rope coupling");
    for (const auto &coupling : impl_->fluid_smoke_couplings)
        if (coupling && coupling->alive && coupling->options.fluid == id)
            return failure(StatusCode::invalid_argument,
                           "fluid is still referenced by a smoke coupling");
    for (const auto &coupling : impl_->fluid_pair_couplings)
        if (coupling && coupling->alive &&
            (coupling->options.first == id || coupling->options.second == id))
            return failure(StatusCode::invalid_argument,
                           "fluid is still referenced by a fluid coupling");
    for (std::uint32_t i = 0; i < impl_->options.paint_rule_capacity; ++i)
        if (impl_->paint_rules[i].alive &&
            impl_->paint_rules[i].options.source == id)
            return failure(StatusCode::invalid_argument,
                           "fluid is still referenced by a paint rule");
    for (const FluidClothCouplingResource &coupling :
         impl_->fluid_cloth_couplings)
        if (coupling.alive && coupling.options.fluid == id)
            return failure(StatusCode::invalid_argument,
                "fluid is still referenced by a cloth coupling");
    status = read_host_objects(impl_->execution, fluid->count);
    if (!status) return status;
    impl_->destroyed_particle_count +=
        fluid->initial_count + fluid->emitted_count - *fluid->count;
    impl_->fluids[id.index].retire();
    for (ParticleSourceSlot &plane : impl_->particle_sources)
        if (plane.alive && plane.options.fluid == id) {
            plane.alive = false; ++plane.generation; plane.data.reset();
        }
    for (DestroyPlaneSlot &plane : impl_->destroy_planes)
        if (plane.alive && plane.options.fluid == id) {
            plane.alive = false; ++plane.generation;
        }
    --impl_->fluid_count;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::fluid_view(FluidId id, FluidDeviceView &output) const noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_current_device();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(id, fluid))) return status;
    if (impl_->frame && !impl_->frame->acknowledged)
        return failure(StatusCode::busy, "fluid view requires a completed frame");
    std::uint32_t count{};
    status = read_device_u32(impl_->execution, fluid->count, count,
                             "fluid view count readback failed");
    if (!status) return status;
    output = {{fluid->positions, count}, {fluid->velocities, count},
              {fluid->forces, count}, {fluid->ids, count},
              {fluid->foam, count}, {fluid->temperatures, count},
              count, fluid->options.particle_radius,
              fluid->options.support_radius, impl_->revision};
    return success();
}

namespace {
bool equal_fluid_material(const FluidOptions &first,
                          const FluidOptions &second) noexcept {
    return first.particle_radius == second.particle_radius &&
        first.rest_density == second.rest_density &&
        first.support_radius == second.support_radius &&
        first.rest_particle_volume == second.rest_particle_volume &&
        first.repulsion == second.repulsion &&
        first.viscosity == second.viscosity &&
        first.normal_damping == second.normal_damping &&
        first.velocity_damping == second.velocity_damping &&
        first.maximum_speed == second.maximum_speed &&
        first.maximum_pair_acceleration == second.maximum_pair_acceleration &&
        first.solver_iterations == second.solver_iterations;
}
} // namespace

Status SimulationRuntime::add_fluid_pair_coupling(
    FluidPairCouplingOptions options, FluidPairCouplingId &output) noexcept {
    output = {};
    if (!impl_)
        return failure(StatusCode::invalid_argument,
                       "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.first.index >= options.second.index)
        return failure(StatusCode::invalid_argument,
                       "fluid pair endpoints must be distinct and canonical");
    FluidStorage *first = nullptr;
    FluidStorage *second = nullptr;
    if (!(status = impl_->validate_handle(options.first, first)) ||
        !(status = impl_->validate_handle(options.second, second)))
        return status;
    if (!equal_fluid_material(first->options, second->options))
        return failure(StatusCode::not_supported,
                       "fluid pair requires equal material state");
    for (const auto &coupling : impl_->fluid_pair_couplings)
        if (coupling && coupling->alive &&
            coupling->options.first == options.first &&
            coupling->options.second == options.second)
            return failure(StatusCode::invalid_argument,
                           "fluid pair is already coupled");
    std::uint32_t slot = 0U;
    while (slot < impl_->fluid_pair_couplings.size() &&
           impl_->fluid_pair_couplings[slot] &&
           impl_->fluid_pair_couplings[slot]->alive)
        ++slot;
    if (slot == impl_->fluid_pair_couplings.size())
        return failure(StatusCode::capacity_exceeded,
                       "fluid pair coupling capacity exhausted");
    std::unique_ptr<FluidPairCouplingStorage> coupling(
        new (std::nothrow) FluidPairCouplingStorage());
    if (!coupling)
        return failure(StatusCode::out_of_memory,
                       "fluid pair coupling allocation failed");
    StorageAllocator allocator;
    status = simulation::visit_fluid_pair_neighbor_index_storage(
        coupling->index, first->options.capacity, second->options.capacity,
        allocator);
    if (!status) return status;
    for (std::uint32_t endpoint = 0U; endpoint < 2U; ++endpoint) {
        *coupling->index.neighbor_overflow[endpoint] = 0U;
        *coupling->index.maximum_neighbor_count[endpoint] = 0U;
    }
    coupling->options = options;
    coupling->generation = impl_->fluid_pair_couplings[slot].generation();
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->fluid_pair_couplings[slot] = std::move(coupling);
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_fluid_pair_coupling(
    FluidPairCouplingId id) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument,
                       "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_pair_couplings.size() ||
        !impl_->fluid_pair_couplings[id.index] ||
        !impl_->fluid_pair_couplings[id.index]->alive ||
        impl_->fluid_pair_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle,
                       "fluid pair coupling is stale");
    impl_->fluid_pair_couplings[id.index].retire();
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::add_smoke(SmokeOptions options, SmokeId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.capacity == 0U || options.capacity > 1'000'000U ||
        !finite(options.emitter_center) || !finite(options.initial_velocity) ||
        !finite(options.wind) || !finite(options.emitter_half_extents.x) ||
        !finite(options.emitter_half_extents.y) ||
        options.emitter_half_extents.x <= 0.0F ||
        options.emitter_half_extents.y <= 0.0F ||
        !finite(options.particles_per_second) || options.particles_per_second <= 0.0F ||
        options.particles_per_second > 1'000'000.0F ||
        !finite(options.lifetime) || options.lifetime <= 0.0F ||
        !finite(options.particle_radius) || options.particle_radius <= 0.0F ||
        !finite(options.buoyancy) || !finite(options.response) ||
        options.response < 0.0F || !finite(options.rest_number_density) ||
        options.rest_number_density <= 0.0F ||
        !finite(options.pressure_stiffness) ||
        options.pressure_stiffness < 0.0F ||
        !finite(options.viscosity) || options.viscosity < 0.0F ||
        !finite(options.vorticity_confinement) ||
        options.vorticity_confinement < 0.0F ||
        !finite(options.maximum_speed) ||
        options.maximum_speed <= 0.0F ||
        (options.grid_resolution != 0U &&
         (options.grid_resolution < 16U || options.grid_resolution > 256U ||
          options.grid_vertical_resolution < 8U ||
          options.grid_vertical_resolution > 256U ||
          options.grid_pressure_iterations < 4U ||
          options.grid_pressure_iterations > 128U ||
          !finite(options.grid_kinematic_viscosity) ||
          options.grid_kinematic_viscosity < 0.0F ||
          !finite(options.grid_les_coefficient) ||
          options.grid_les_coefficient < 0.0F ||
          !finite(options.grid_pressure_tolerance) ||
          options.grid_pressure_tolerance <= 0.0F ||
          options.grid_pressure_tolerance > 1.0F)) ||
        !finite(options.grid_minimum) ||
        !finite(options.grid_edge_length) || options.grid_edge_length < 0.0F)
        return failure(StatusCode::invalid_argument, "invalid smoke options");
    std::uint32_t slot = 0U;
    while (slot < impl_->smokes.size() && impl_->smokes[slot] &&
           impl_->smokes[slot]->alive) ++slot;
    if (slot == impl_->smokes.size())
        return failure(StatusCode::capacity_exceeded, "smoke capacity exhausted");
    std::unique_ptr<SmokeStorage> smoke;
    try { smoke = std::make_unique<SmokeStorage>(); }
    catch (...) { return failure(StatusCode::out_of_memory, "smoke owner allocation failed"); }
    smoke->generation = impl_->smokes[slot].generation();
    smoke->options = options;
    if (options.grid_resolution != 0U) {
        auto &grid = smoke->grid;
        grid.resolution = options.grid_resolution;
        grid.height = options.grid_vertical_resolution;
        grid.cell_count = options.grid_resolution * options.grid_resolution *
                          options.grid_vertical_resolution;
        if (options.grid_edge_length > 0.0F) {
            grid.minimum = options.grid_minimum;
            grid.spacing = options.grid_edge_length / float(grid.resolution);
        } else {
            const Vec3 end = add(options.emitter_center,
                multiply(options.wind, options.lifetime));
            const Vec3 low{fminf(options.emitter_center.x, end.x) - 1.5F,
                options.emitter_center.y,
                fminf(options.emitter_center.z, end.z) - 1.5F};
            const Vec3 high{fmaxf(options.emitter_center.x, end.x) + 1.5F,
                options.emitter_center.y,
                fmaxf(options.emitter_center.z, end.z) + 1.5F};
            const float edge = fmaxf(high.x-low.x, high.z-low.z);
            const Vec3 center = multiply(add(low, high), 0.5F);
            grid.spacing = edge / float(grid.resolution);
            grid.minimum = subtract(center, {edge*0.5F,
                grid.spacing * float(grid.height) * 0.5F, edge*0.5F});
        }
        if (!finite(grid.spacing) || grid.spacing < 1.0e-4F)
            return failure(StatusCode::invalid_argument,
                "invalid smoke grid cell spacing");
        for (int axis = 0; axis < 3; ++axis)
            grid.face_count[axis] = smoke_grid_face_count(axis,
                int(grid.resolution), int(grid.height));
        std::uint32_t level_n = grid.resolution;
        std::uint32_t level_h = grid.height;
        for (auto &level : grid.coarse) {
            level_n = std::max(2U, level_n / 2U);
            level_h = std::max(2U, level_h / 2U);
            level.n = level_n;
            level.height = level_h;
            level.cell_count = level_n * level_h * level_n;
            for (int axis = 0; axis < 3; ++axis)
                level.face_count[axis] = smoke_grid_face_count(axis, int(level_n), int(level_h));
        }
        StorageAllocator allocator;
        status = simulation::visit_smoke_grid_storage(grid, allocator);
        if (!status) return status;
        const auto blocks = (grid.cell_count + 255U) / 256U;
        const float wind[3]{options.wind.x, options.wind.y, options.wind.z};
        simulation::initialize_smoke_grid_cells(
            blocks, 256U, impl_->execution, grid.velocity,
            grid.density, grid.temperature, grid.pressure[0],
            grid.divergence, grid.vorticity, grid.strain,
            grid.subgrid_force, grid.solid, grid.cell_count, wind);
        clear_bytes(grid.pressure[1], 0, grid.cell_count * sizeof(float), impl_->execution);
        clear_bytes(grid.residual, 0, grid.cell_count * sizeof(float), impl_->execution);
        clear_bytes(grid.rhs_max, 0, sizeof(float), impl_->execution);
        clear_bytes(grid.residual_max, 0, sizeof(float), impl_->execution);
        clear_bytes(grid.pressure_converged, 0, sizeof(unsigned int), impl_->execution);
        for (int axis = 0; axis < 3; ++axis) {
            const float component = axis == 0 ? options.wind.x :
                                    axis == 1 ? options.wind.y : options.wind.z;
            simulation::initialize_smoke_grid_faces(
                (grid.face_count[axis] + 255U) / 256U, 256U, impl_->execution,
                grid.face_velocity[axis][0], grid.face_reverse[axis],
                grid.face_open[axis], grid.face_wall_velocity[axis],
                grid.face_normal[axis], grid.face_nearest_triangle[axis],
                grid.face_count[axis], component);
            copy_bytes_wait(grid.face_velocity[axis][1],
                grid.face_velocity[axis][0],
                grid.face_count[axis] * sizeof(float),
                device::CopyDirection::device_to_device, impl_->execution);
        }
        for (auto &level : grid.coarse) {
            clear_bytes(level.pressure[0], 0,
                level.cell_count * sizeof(float), impl_->execution);
            clear_bytes(level.pressure[1], 0,
                level.cell_count * sizeof(float), impl_->execution);
            clear_bytes(level.rhs, 0, level.cell_count * sizeof(float), impl_->execution);
            clear_bytes(level.residual, 0,
                level.cell_count * sizeof(float), impl_->execution);
        }
        PmResult grid_error = impl_->execution.services->last_error(false);
        if (grid_error != PM_SUCCESS)
            return backend_failure(grid_error, "smoke grid initialization failed");
        grid_error = wait_queue(impl_->execution);
        if (grid_error != PM_SUCCESS)
            return backend_failure(grid_error, "smoke grid initialization failed");
    }
    StorageAllocator allocator;
    status = simulation::visit_smoke_particles_storage(*smoke, options.capacity, allocator);
    if (!status) return status;
    smoke->alive = true;
    output = {slot, smoke->generation};
    impl_->smokes[slot] = std::move(smoke);
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_smoke(SmokeId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->smokes.size() || !impl_->smokes[id.index] ||
        !impl_->smokes[id.index]->alive ||
        impl_->smokes[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "smoke handle is stale");
    for (const auto &coupling : impl_->fluid_smoke_couplings)
        if (coupling && coupling->alive && coupling->options.smoke == id)
            return failure(StatusCode::invalid_argument,
                           "smoke is still referenced by a fluid coupling");
    for (const auto &coupling : impl_->smoke_soft_body_couplings)
        if (coupling && coupling->alive && coupling->options.smoke == id)
            return failure(StatusCode::invalid_argument,
                           "smoke is still referenced by a soft-body coupling");
    for (const auto &coupling : impl_->smoke_cloth_couplings)
        if (coupling && coupling->alive && coupling->options.smoke == id)
            return failure(StatusCode::invalid_argument,
                           "smoke is still referenced by a cloth coupling");
    for (const auto &coupling : impl_->smoke_rope_couplings)
        if (coupling && coupling->alive && coupling->options.smoke == id)
            return failure(StatusCode::invalid_argument,
                           "smoke is still referenced by a rope coupling");
    for (const auto &coupling : impl_->smoke_rigid_couplings)
        if (coupling && coupling->alive && coupling->options.smoke == id)
            return failure(StatusCode::invalid_argument,
                           "smoke is still referenced by a rigid coupling");
    impl_->smokes[id.index].retire();
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::smoke_view(SmokeId id, SmokeDeviceView &output) const noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (id.index >= impl_->smokes.size() || !impl_->smokes[id.index] ||
        !impl_->smokes[id.index]->alive ||
        impl_->smokes[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "smoke handle is stale");
    if (impl_->frame && !impl_->frame->acknowledged)
        return failure(StatusCode::busy, "smoke view requires a completed frame");
    const auto &smoke = *impl_->smokes[id.index];
    float pressure_residual = 0.0F;
    if (smoke.grid.resolution != 0U) {
        float residual_max = 0.0F;
        float rhs_max = 0.0F;
        PmResult error = copy_bytes_wait(&residual_max, smoke.grid.residual_max,
                                       sizeof(residual_max),
                                       device::CopyDirection::device_to_host, impl_->execution);
        if (error == PM_SUCCESS) {
            error = copy_bytes_wait(&rhs_max, smoke.grid.rhs_max, sizeof(rhs_max),
                               device::CopyDirection::device_to_host, impl_->execution);
        }
        if (error != PM_SUCCESS)
            return backend_failure(error, "smoke pressure residual readback failed");
        pressure_residual = residual_max / fmaxf(rhs_max, 1.0e-5F);
    }
    output = {{smoke.positions, smoke.count}, {smoke.velocities, smoke.count},
        {smoke.ages, smoke.count}, {smoke.number_densities, smoke.count},
        {smoke.pressures, smoke.count}, {smoke.vorticities, smoke.count},
        smoke.count,
        smoke.options.lifetime, smoke.options.particle_radius, impl_->revision,
        {smoke.grid.velocity, smoke.grid.cell_count},
        {smoke.grid.pressure[0], smoke.grid.cell_count},
        {smoke.grid.density, smoke.grid.cell_count},
        {smoke.grid.temperature, smoke.grid.cell_count},
        {smoke.grid.solid, smoke.grid.cell_count},
        {smoke.grid.vorticity, smoke.grid.cell_count},
        {smoke.grid.divergence, smoke.grid.cell_count},
        smoke.grid.resolution, smoke.grid.height,
        smoke.grid.minimum, smoke.grid.spacing,
        pressure_residual};
    return success();
}

Status SimulationRuntime::add_fluid_smoke_coupling(
    FluidSmokeCouplingOptions options, FluidSmokeCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (options.smoke.index >= impl_->smokes.size() ||
        !impl_->smokes[options.smoke.index] ||
        !impl_->smokes[options.smoke.index]->alive ||
        impl_->smokes[options.smoke.index]->generation != options.smoke.generation)
        return failure(StatusCode::invalid_handle, "smoke handle is invalid or stale");
    const float rotation_norm_squared =
        options.heater.orientation.x * options.heater.orientation.x +
        options.heater.orientation.y * options.heater.orientation.y +
        options.heater.orientation.z * options.heater.orientation.z +
        options.heater.orientation.w * options.heater.orientation.w;
    if (!finite(options.heater.center) || !finite(options.heater.orientation) ||
        rotation_norm_squared < 0.25F || rotation_norm_squared > 4.0F ||
        !finite(options.heater.half_extents.x) ||
        !finite(options.heater.half_extents.y) ||
        options.heater.half_extents.x <= 0.0F ||
        options.heater.half_extents.y <= 0.0F ||
        !finite(options.heater_temperature) ||
        !finite(options.boiling_temperature) ||
        !finite(options.heat_transfer_rate) || options.heat_transfer_rate < 0.0F ||
        !finite(options.wind_drag) || options.wind_drag < 0.0F ||
        !finite(options.steam_rise_speed) || options.steam_rise_speed < 0.0F)
        return failure(StatusCode::invalid_argument, "invalid fluid smoke coupling options");
    std::uint32_t slot = 0U;
    while (slot < impl_->fluid_smoke_couplings.size() &&
           impl_->fluid_smoke_couplings[slot] &&
           impl_->fluid_smoke_couplings[slot]->alive) ++slot;
    if (slot == impl_->fluid_smoke_couplings.size())
        return failure(StatusCode::capacity_exceeded, "fluid smoke coupling capacity exhausted");
    std::unique_ptr<FluidSmokeCouplingStorage> coupling;
    try { coupling = std::make_unique<FluidSmokeCouplingStorage>(); }
    catch (...) { return failure(StatusCode::out_of_memory, "fluid smoke coupling allocation failed"); }
    coupling->generation = impl_->fluid_smoke_couplings[slot].generation();
    coupling->options = options;
    if (!(status = allocate_managed(coupling->converted, 1U))) return status;
    *coupling->converted = 0U;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->fluid_smoke_couplings[slot] = std::move(coupling);
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_fluid_smoke_coupling(FluidSmokeCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_smoke_couplings.size() ||
        !impl_->fluid_smoke_couplings[id.index] ||
        !impl_->fluid_smoke_couplings[id.index]->alive ||
        impl_->fluid_smoke_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid smoke coupling handle is stale");
    impl_->fluid_smoke_couplings[id.index].retire();
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::add_smoke_soft_body_coupling(
    SmokeSoftBodyCouplingOptions options,
    SmokeSoftBodyCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.smoke.index >= impl_->smokes.size() ||
        !impl_->smokes[options.smoke.index] ||
        !impl_->smokes[options.smoke.index]->alive ||
        impl_->smokes[options.smoke.index]->generation != options.smoke.generation ||
        options.soft_body.index >= impl_->soft_bodies.size() ||
        !impl_->soft_bodies[options.soft_body.index] ||
        !impl_->soft_bodies[options.soft_body.index]->alive ||
        impl_->soft_bodies[options.soft_body.index]->generation != options.soft_body.generation)
        return failure(StatusCode::invalid_handle,
                       "smoke or soft-body coupling handle is stale");
    if (!finite(options.wind_drag) || options.wind_drag < 0.0F ||
        !finite(options.maximum_wind_acceleration) ||
        options.maximum_wind_acceleration < 0.0F ||
        !finite(options.contact_distance) || options.contact_distance < 0.0F)
        return failure(StatusCode::invalid_argument,
                       "invalid smoke soft-body coupling parameters");
    for (const auto &existing : impl_->smoke_soft_body_couplings)
        if (existing && existing->alive &&
            existing->options.smoke == options.smoke &&
            existing->options.soft_body == options.soft_body)
            return failure(StatusCode::invalid_argument,
                           "smoke soft-body coupling already exists");
    std::uint32_t slot = 0U;
    while (slot < impl_->smoke_soft_body_couplings.size() &&
           impl_->smoke_soft_body_couplings[slot] &&
           impl_->smoke_soft_body_couplings[slot]->alive) ++slot;
    if (slot == impl_->smoke_soft_body_couplings.size())
        return failure(StatusCode::capacity_exceeded,
                       "smoke soft-body coupling capacity exhausted");
    std::unique_ptr<SmokeSoftBodyCouplingSlot> coupling;
    try { coupling = std::make_unique<SmokeSoftBodyCouplingSlot>(); }
    catch (...) { return failure(StatusCode::out_of_memory,
                                "smoke soft-body coupling allocation failed"); }
    coupling->generation =
        impl_->smoke_soft_body_couplings[slot].generation();
    coupling->options = options;
    if (!(status = allocate_managed(coupling->minimum, 1U)) ||
        !(status = allocate_managed(coupling->maximum, 1U))) return status;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->smoke_soft_body_couplings[slot] = std::move(coupling);
    impl_->smokes[options.smoke.index]->grid.static_metadata_valid = false;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_smoke_soft_body_coupling(
    SmokeSoftBodyCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->smoke_soft_body_couplings.size() ||
        !impl_->smoke_soft_body_couplings[id.index] ||
        !impl_->smoke_soft_body_couplings[id.index]->alive ||
        impl_->smoke_soft_body_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle,
                       "smoke soft-body coupling handle is stale");
    const auto smoke_id = impl_->smoke_soft_body_couplings[id.index]->options.smoke;
    impl_->smoke_soft_body_couplings[id.index].retire();
    impl_->smokes[smoke_id.index]->grid.static_metadata_valid = false;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::add_smoke_cloth_coupling(
    SmokeClothCouplingOptions options, SmokeClothCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.smoke.index >= impl_->smokes.size() ||
        !impl_->smokes[options.smoke.index] ||
        !impl_->smokes[options.smoke.index]->alive ||
        impl_->smokes[options.smoke.index]->generation != options.smoke.generation ||
        options.cloth.index >= impl_->cloths.size() ||
        !impl_->cloths[options.cloth.index] ||
        !impl_->cloths[options.cloth.index]->alive ||
        impl_->cloths[options.cloth.index]->generation != options.cloth.generation)
        return failure(StatusCode::invalid_handle,
                       "smoke or cloth coupling handle is stale");
    if (!finite(options.wind_drag) || options.wind_drag < 0.0F ||
        !finite(options.maximum_wind_acceleration) ||
        options.maximum_wind_acceleration < 0.0F ||
        !finite(options.contact_distance) || options.contact_distance < 0.0F)
        return failure(StatusCode::invalid_argument,
                       "invalid smoke cloth coupling parameters");
    for (const auto &existing : impl_->smoke_cloth_couplings)
        if (existing && existing->alive &&
            existing->options.smoke == options.smoke &&
            existing->options.cloth == options.cloth)
            return failure(StatusCode::invalid_argument,
                           "smoke cloth coupling already exists");
    std::uint32_t slot = 0U;
    while (slot < impl_->smoke_cloth_couplings.size() &&
           impl_->smoke_cloth_couplings[slot] &&
           impl_->smoke_cloth_couplings[slot]->alive) ++slot;
    if (slot == impl_->smoke_cloth_couplings.size())
        return failure(StatusCode::capacity_exceeded,
                       "smoke cloth coupling capacity exhausted");
    std::unique_ptr<SmokeClothCouplingSlot> coupling;
    try { coupling = std::make_unique<SmokeClothCouplingSlot>(); }
    catch (...) { return failure(StatusCode::out_of_memory,
                                "smoke cloth coupling allocation failed"); }
    coupling->generation = impl_->smoke_cloth_couplings[slot].generation();
    coupling->options = options;
    if (!(status = allocate_managed(coupling->minimum, 1U)) ||
        !(status = allocate_managed(coupling->maximum, 1U))) return status;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->smoke_cloth_couplings[slot] = std::move(coupling);
    impl_->smokes[options.smoke.index]->grid.static_metadata_valid = false;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_smoke_cloth_coupling(SmokeClothCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->smoke_cloth_couplings.size() ||
        !impl_->smoke_cloth_couplings[id.index] ||
        !impl_->smoke_cloth_couplings[id.index]->alive ||
        impl_->smoke_cloth_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle,
                       "smoke cloth coupling handle is stale");
    const auto smoke_id = impl_->smoke_cloth_couplings[id.index]->options.smoke;
    impl_->smoke_cloth_couplings[id.index].retire();
    impl_->smokes[smoke_id.index]->grid.static_metadata_valid = false;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::add_smoke_rope_coupling(
    SmokeRopeCouplingOptions options, SmokeRopeCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.smoke.index >= impl_->smokes.size() ||
        !impl_->smokes[options.smoke.index] ||
        !impl_->smokes[options.smoke.index]->alive ||
        impl_->smokes[options.smoke.index]->generation != options.smoke.generation ||
        options.rope.index >= impl_->ropes.size() ||
        !impl_->ropes[options.rope.index] ||
        !impl_->ropes[options.rope.index]->alive ||
        impl_->ropes[options.rope.index]->generation != options.rope.generation)
        return failure(StatusCode::invalid_handle,
                       "smoke or rope coupling handle is stale");
    if (!finite(options.wind_drag) || options.wind_drag < 0.0F ||
        !finite(options.maximum_wind_acceleration) ||
        options.maximum_wind_acceleration < 0.0F ||
        !finite(options.contact_distance) || options.contact_distance < 0.0F)
        return failure(StatusCode::invalid_argument,
                       "invalid smoke rope coupling parameters");
    for (const auto &existing : impl_->smoke_rope_couplings)
        if (existing && existing->alive &&
            existing->options.smoke == options.smoke &&
            existing->options.rope == options.rope)
            return failure(StatusCode::invalid_argument,
                           "smoke rope coupling already exists");
    std::uint32_t slot = 0U;
    while (slot < impl_->smoke_rope_couplings.size() &&
           impl_->smoke_rope_couplings[slot] &&
           impl_->smoke_rope_couplings[slot]->alive) ++slot;
    if (slot == impl_->smoke_rope_couplings.size())
        return failure(StatusCode::capacity_exceeded,
                       "smoke rope coupling capacity exhausted");
    std::unique_ptr<SmokeRopeCouplingSlot> coupling;
    try { coupling = std::make_unique<SmokeRopeCouplingSlot>(); }
    catch (...) { return failure(StatusCode::out_of_memory,
                                "smoke rope coupling allocation failed"); }
    coupling->generation = impl_->smoke_rope_couplings[slot].generation();
    coupling->options = options;
    if (!(status = allocate_managed(coupling->minimum, 1U)) ||
        !(status = allocate_managed(coupling->maximum, 1U))) return status;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->smoke_rope_couplings[slot] = std::move(coupling);
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_smoke_rope_coupling(SmokeRopeCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->smoke_rope_couplings.size() ||
        !impl_->smoke_rope_couplings[id.index] ||
        !impl_->smoke_rope_couplings[id.index]->alive ||
        impl_->smoke_rope_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle,
                       "smoke rope coupling handle is stale");
    impl_->smoke_rope_couplings[id.index].retire();
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::add_smoke_rigid_coupling(
    SmokeRigidCouplingOptions options, SmokeRigidCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.smoke.index >= impl_->smokes.size() ||
        !impl_->smokes[options.smoke.index] ||
        !impl_->smokes[options.smoke.index]->alive ||
        impl_->smokes[options.smoke.index]->generation != options.smoke.generation)
        return failure(StatusCode::invalid_handle, "smoke coupling handle is stale");
    std::uint32_t dense = 0U;
    if (!(status = impl_->validate_handle(options.body, dense))) return status;
    if (!finite(options.air_density) || options.air_density < 0.0F ||
        !finite(options.drag_coefficient) || options.drag_coefficient < 0.0F ||
        !finite(options.contact_distance) || options.contact_distance < 0.0F)
        return failure(StatusCode::invalid_argument,
                       "invalid smoke rigid coupling parameters");
    for (const auto &existing : impl_->smoke_rigid_couplings)
        if (existing && existing->alive &&
            existing->options.smoke == options.smoke &&
            existing->options.body == options.body)
            return failure(StatusCode::invalid_argument,
                           "smoke rigid coupling already exists");
    std::uint32_t slot = 0U;
    while (slot < impl_->smoke_rigid_couplings.size() &&
           impl_->smoke_rigid_couplings[slot] &&
           impl_->smoke_rigid_couplings[slot]->alive) ++slot;
    if (slot == impl_->smoke_rigid_couplings.size())
        return failure(StatusCode::capacity_exceeded,
                       "smoke rigid coupling capacity exhausted");
    std::unique_ptr<SmokeRigidCouplingSlot> coupling;
    try { coupling = std::make_unique<SmokeRigidCouplingSlot>(); }
    catch (...) { return failure(StatusCode::out_of_memory,
                                "smoke rigid coupling allocation failed"); }
    coupling->generation = impl_->smoke_rigid_couplings[slot].generation();
    coupling->options = options;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->smoke_rigid_couplings[slot] = std::move(coupling);
    impl_->smokes[options.smoke.index]->grid.static_metadata_valid = false;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_smoke_rigid_coupling(SmokeRigidCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->smoke_rigid_couplings.size() ||
        !impl_->smoke_rigid_couplings[id.index] ||
        !impl_->smoke_rigid_couplings[id.index]->alive ||
        impl_->smoke_rigid_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle,
                       "smoke rigid coupling handle is stale");
    const auto smoke_id = impl_->smoke_rigid_couplings[id.index]->options.smoke;
    impl_->smoke_rigid_couplings[id.index].retire();
    impl_->smokes[smoke_id.index]->grid.static_metadata_valid = false;
    ++impl_->revision;
    return success();
}

namespace {
[[nodiscard]] bool valid_particle_plane(ParticlePlane plane) noexcept {
    const float squared = plane.orientation.x * plane.orientation.x +
        plane.orientation.y * plane.orientation.y +
        plane.orientation.z * plane.orientation.z +
        plane.orientation.w * plane.orientation.w;
    return finite(plane.center) && finite(plane.orientation) &&
        finite(plane.half_extents.x) && plane.half_extents.x > 0.0F &&
        finite(plane.half_extents.y) && plane.half_extents.y > 0.0F &&
        squared > 0.25F && squared < 4.0F;
}
} // namespace

Status SimulationRuntime::add_particle_source(ParticleSourceMesh mesh, ParticleSourceOptions options,
                                       ParticleSourceId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (mesh.spacing == 0.0F) mesh.spacing = fluid->options.support_radius;
    if (!finite(options.initial_velocity) || !finite(options.initial_temperature) ||
        !finite(mesh.spacing) ||
        mesh.spacing < 2.0F * fluid->options.particle_radius)
        return failure(StatusCode::invalid_argument, "fluid source spacing or velocity is invalid");
    for (std::uint32_t index = 0; index < impl_->particle_sources.size(); ++index) {
        ParticleSourceSlot &plane = impl_->particle_sources[index];
        if (plane.alive) continue;
        std::vector<Vec3> points;
        if (!(status = sample_fluid_source(mesh, points))) return status;
        std::unique_ptr<ParticleSourceData> data(new (std::nothrow) ParticleSourceData());
        if (!data) return failure(StatusCode::out_of_memory, "fluid source allocation failed");
        data->count = static_cast<std::uint32_t>(points.size());
        data->spacing = mesh.spacing;
        StorageAllocator allocator;
        status = simulation::visit_emitter_storage(*data, points.size(), allocator);
        if (!status) return status;
        std::copy(points.begin(), points.end(), data->points);
        *data->capacity_misses = 0;
        plane.options = options;
        plane.data = std::move(data);
        plane.alive = true;
        output = {index, plane.generation};
        return success();
    }
    return failure(StatusCode::capacity_exceeded, "fluid source capacity exhausted");
}

Status SimulationRuntime::update_particle_source(ParticleSourceId id,
                                          ParticleSourceOptions options) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->particle_sources.size() ||
        !impl_->particle_sources[id.index].alive ||
        impl_->particle_sources[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid source handle is stale");
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (!finite(options.initial_velocity) ||
        !finite(options.initial_temperature) ||
        !(options.fluid == impl_->particle_sources[id.index].options.fluid))
        return failure(StatusCode::invalid_argument, "fluid source destination is immutable or velocity is invalid");
    impl_->particle_sources[id.index].options = options;
    return success();
}

Status SimulationRuntime::remove_particle_source(ParticleSourceId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->particle_sources.size() ||
        !impl_->particle_sources[id.index].alive ||
        impl_->particle_sources[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid source handle is stale");
    impl_->particle_sources[id.index].alive = false;
    impl_->particle_sources[id.index].data.reset();
    ++impl_->particle_sources[id.index].generation;
    return success();
}

Status SimulationRuntime::add_particle_destroy_plane(ParticleDestroyPlaneOptions options,
                                         ParticleDestroyPlaneId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (!valid_particle_plane(options.plane))
        return failure(StatusCode::invalid_argument, "destroy plane options are invalid");
    for (std::uint32_t index = 0; index < impl_->destroy_planes.size(); ++index) {
        DestroyPlaneSlot &plane = impl_->destroy_planes[index];
        if (plane.alive) continue;
        plane.options = options;
        plane.alive = true;
        output = {index, plane.generation};
        return success();
    }
    return failure(StatusCode::capacity_exceeded, "destroy plane capacity exhausted");
}

Status SimulationRuntime::update_particle_destroy_plane(ParticleDestroyPlaneId id,
                                            ParticleDestroyPlaneOptions options) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->destroy_planes.size() ||
        !impl_->destroy_planes[id.index].alive ||
        impl_->destroy_planes[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "destroy plane handle is stale");
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (!valid_particle_plane(options.plane))
        return failure(StatusCode::invalid_argument, "destroy plane options are invalid");
    impl_->destroy_planes[id.index].options = options;
    return success();
}

Status SimulationRuntime::remove_particle_destroy_plane(ParticleDestroyPlaneId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->destroy_planes.size() ||
        !impl_->destroy_planes[id.index].alive ||
        impl_->destroy_planes[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "destroy plane handle is stale");
    impl_->destroy_planes[id.index].alive = false;
    ++impl_->destroy_planes[id.index].generation;
    return success();
}

// Rendering seams duplicate positions. Weld those positions only for the
// topology check; the original indexed triangle mesh remains unchanged.
static std::vector<CollisionPlane> closed_convex_planes(
    const Vec3 *vertices, std::uint32_t vertex_count,
    const std::vector<std::uint32_t> &indices) {
    std::map<std::array<float, 3>, std::uint32_t> welded;
    std::vector<std::uint32_t> remap(vertex_count);
    Vec3 center{};
    for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex) {
        const Vec3 p = vertices[vertex];
        const auto entry = welded.emplace(std::array<float, 3>{p.x, p.y, p.z},
            static_cast<std::uint32_t>(welded.size()));
        remap[vertex] = entry.first->second;
        if (entry.second) center = add(center, p);
    }
    if (welded.size() < 4U) return {};
    center = multiply(center, 1.0F / static_cast<float>(welded.size()));
    std::unordered_map<std::uint64_t, std::uint32_t> edges;
    for (std::size_t index = 0U; index < indices.size(); index += 3U)
        for (std::size_t edge = 0U; edge < 3U; ++edge) {
            const auto a = remap[indices[index + edge]];
            const auto b = remap[indices[index + (edge + 1U) % 3U]];
            ++edges[(static_cast<std::uint64_t>(std::min(a, b)) << 32U) |
                     std::max(a, b)];
        }
    for (const auto &edge : edges)
        if (edge.second != 2U) return {};
    float scale = 0.0F;
    for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex)
        scale = std::max(scale, vector_length(subtract(vertices[vertex], center)));
    const float tolerance = std::max(1.0e-7F, scale * 1.0e-5F);
    std::vector<CollisionPlane> planes;
    planes.reserve(indices.size() / 3U);
    for (std::size_t index = 0U; index < indices.size(); index += 3U) {
        const Vec3 a = vertices[indices[index]];
        Vec3 normal = normalized_or(cross(
            subtract(vertices[indices[index + 1U]], a),
            subtract(vertices[indices[index + 2U]], a)), {});
        float side = dot(normal, subtract(a, center));
        if (fabsf(side) <= tolerance) return {};
        if (side < 0.0F) normal = multiply(normal, -1.0F);
        const float offset = dot(normal, a);
        for (std::uint32_t vertex = 0U; vertex < vertex_count; ++vertex)
            if (dot(normal, vertices[vertex]) - offset > tolerance) return {};
        planes.push_back({normal, offset});
    }
    return planes;
}

// Certify outward normals once at upload. Concave closed shells need the same
// one-sided boundary metadata as convex components; this does not replace
// their triangles or treat a concave surface as its convex hull.
static std::vector<Vec3> closed_shell_normals(
    const Vec3 *vertices, std::uint32_t vertex_count,
    const std::vector<std::uint32_t> &indices) {
    std::map<std::array<float, 3>, std::uint32_t> welded;
    std::vector<std::uint32_t> remap(vertex_count), parent(vertex_count);
    for (std::uint32_t i = 0; i < vertex_count; ++i) {
        const auto p = vertices[i];
        remap[i] = welded.emplace(
            std::array<float, 3>{p.x, p.y, p.z}, i).first->second;
        parent[i] = i;
    }
    const auto root = [&](std::uint32_t i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    for (std::size_t i = 0; i < indices.size(); i += 3)
        for (std::size_t corner = 1; corner < 3; ++corner) {
            const auto a = root(remap[indices[i]]);
            const auto b = root(remap[indices[i + corner]]);
            parent[b] = a;
        }
    std::map<std::uint32_t, std::vector<std::uint32_t>> shells;
    for (std::uint32_t t = 0; t < indices.size() / 3U; ++t)
        shells[root(remap[indices[t * 3U]])].push_back(t);
    std::vector<Vec3> normals(indices.size() / 3U);
    bool found = false;
    for (const auto &[id, triangles] : shells) {
        (void)id;
        std::map<std::uint32_t, std::uint32_t> local;
        std::vector<Vec3> points;
        std::vector<std::uint32_t> faces;
        for (auto t : triangles) {
            for (std::uint32_t corner = 0; corner < 3U; ++corner) {
                const auto vertex = remap[indices[t * 3U + corner]];
                const auto [entry, inserted] = local.emplace(
                    vertex, static_cast<std::uint32_t>(points.size()));
                if (inserted) points.push_back(vertices[vertex]);
                faces.push_back(entry->second);
            }
        }
        const auto planes = closed_convex_planes(
            points.data(), static_cast<std::uint32_t>(points.size()), faces);
        if (!planes.empty()) {
            found = true;
            for (std::size_t i = 0; i < triangles.size(); ++i)
                normals[triangles[i]] = planes[i].normal;
            continue;
        }
        // A consistently wound, closed orientable component has two opposite
        // uses of every edge. Signed volume then determines its outward sign.
        // Open or inconsistent meshes retain two-sided triangle behavior.
        std::map<std::pair<std::uint32_t, std::uint32_t>,
            std::pair<unsigned, int>> edges;
        double volume = 0.0;
        for (std::size_t face = 0; face < faces.size(); face += 3U) {
            for (std::size_t corner = 0; corner < 3U; ++corner) {
                const auto a = faces[face + corner];
                const auto b = faces[face + (corner + 1U) % 3U];
                auto &edge = edges[{std::min(a, b), std::max(a, b)}];
                ++edge.first;
                edge.second += a < b ? 1 : -1;
            }
            std::array<std::array<double, 3>, 3> v{};
            for (std::size_t corner = 0; corner < 3U; ++corner) {
                const Vec3 p = points[faces[face + corner]];
                v[corner] = {double(p.x) - points[0].x,
                    double(p.y) - points[0].y, double(p.z) - points[0].z};
            }
            volume += v[0][0] * (v[1][1] * v[2][2] - v[1][2] * v[2][1]) +
                v[0][1] * (v[1][2] * v[2][0] - v[1][0] * v[2][2]) +
                v[0][2] * (v[1][0] * v[2][1] - v[1][1] * v[2][0]);
        }
        bool closed = volume != 0.0 && std::isfinite(volume);
        for (const auto &[edge, uses] : edges) {
            (void)edge;
            closed = closed && uses.first == 2U && uses.second == 0;
        }
        if (!closed) {
            // A support plane still has a certified exterior when a volumetric
            // component has open seams (for example, an uncapped gear hub).
            // Require extent off the plane; a flat open sheet stays two-sided.
            float scale = 0.0F;
            for (const auto point : points)
                scale = std::max(scale, vector_length(subtract(point, points[0])));
            const float tolerance = std::max(1.0e-7F, scale * 1.0e-5F);
            for (std::size_t i = 0; i < triangles.size(); ++i) {
                const Vec3 a = points[faces[i * 3U]];
                const Vec3 normal = normalized_or(cross(
                    subtract(points[faces[i * 3U + 1U]], a),
                    subtract(points[faces[i * 3U + 2U]], a)), {});
                float minimum = 0.0F, maximum = 0.0F;
                for (const auto point : points) {
                    const float distance = dot(normal, subtract(point, a));
                    minimum = std::min(minimum, distance);
                    maximum = std::max(maximum, distance);
                }
                if (maximum <= tolerance && minimum < -tolerance)
                    normals[triangles[i]] = normal;
                else if (minimum >= -tolerance && maximum > tolerance)
                    normals[triangles[i]] = multiply(normal, -1.0F);
                else {
                    // An interior triangle of an open surface still has a
                    // normal cone: tangent directions belong to neighboring
                    // faces, not to its tessellation edges. A quarter-length
                    // normal marks that two-sided feature without certifying
                    // an inside/outside. Boundary triangles stay unrestricted.
                    bool interior = true;
                    for (std::size_t corner = 0; corner < 3U; ++corner) {
                        const auto first = faces[i * 3U + corner];
                        const auto second = faces[i * 3U + (corner + 1U) % 3U];
                        const auto uses = edges.at({std::min(first, second),
                            std::max(first, second)});
                        interior = interior && uses.first == 2U && uses.second == 0;
                    }
                    if (!interior) continue;
                    normals[triangles[i]] = multiply(normal, 0.25F);
                }
                found = true;
            }
            continue;
        }
        found = true;
        for (std::size_t i = 0; i < triangles.size(); ++i) {
            const Vec3 a = points[faces[i * 3U]];
            const Vec3 b = points[faces[i * 3U + 1U]];
            const Vec3 c = points[faces[i * 3U + 2U]];
            normals[triangles[i]] = multiply(normalized_or(
                cross(subtract(b, a), subtract(c, a)), {}), volume > 0.0 ? 1.0F : -1.0F);
        }
    }
    return found ? normals : std::vector<Vec3>{};
}

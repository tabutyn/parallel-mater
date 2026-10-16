// SPDX-License-Identifier: MIT
#pragma once

#include "runtime.hpp"
#include "geometry_storage.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace parallel_mater::simulation {

inline constexpr std::uint32_t k_soft_contact_cleanup_passes = 8U;

enum class TimingStage : std::uint8_t {
    rigid_integration,
    rigid_world_bounds,
    rigid_pair_filter,
    rigid_pair_compaction,
    rigid_leaf_pair_generation,
    rigid_contact_evaluation,
    rigid_contact_solve,
    rigid_input_clear,
    cloth_prediction,
    cloth_constraints,
    cloth_contacts,
    soft_body_prediction,
    soft_body_constraints,
    soft_body_contacts,
    soft_body_contact_cleanup,
    soft_body_cloth_contacts,
    fluid_soft_body_contacts,
    fluid_rope_contacts,
    rope_soft_body_contacts,
    fluid_cloth_contacts,
    fluid_spawn,
    fluid_neighbor_sort,
    fluid_neighbor_forces,
    fluid_integration,
    fluid_static_contacts,
    fluid_body_index,
    fluid_moving_contacts,
    fluid_contact_events,
    fluid_outflow_compaction,
    fluid_smoke_exchange,
    rope_solve,
    smoke_grid,
    smoke_advection,
    smoke_emission,
};

inline Status schedule_failure(StatusCode code, const char *message) noexcept {
    return {code, 0, message};
}

// Shared reference schedule. State describes resources and host bookkeeping;
// Backend supplies typed Slang dispatch, copy/clear, readback, timing and waits.
// No native SDK, stream type, launch syntax or native error values occur here.
// Adapters must preserve ordered writes/reads on queue(), including barriers.
// Current emit/phase-transfer paths explicitly request readbacks; the public
// submission executor runs these on its worker, never the submitting thread.
template <typename State, typename Backend>
Status schedule_frame(State &state, Backend &backend, StepOptions options) noexcept {
    auto *impl_ = &state;
    const auto stream = backend.queue();
    auto error = backend.success_code();
    Status status{};
    const bool debug_enabled = impl_->options.physics_debug.frame_capacity != 0U;
    const bool collect_rigid_contacts = options.collect_rigid_contacts || debug_enabled;
    const bool collect_fluid_contacts = options.collect_fluid_contacts || debug_enabled;
    if (impl_->rigid_body_count != 0U) {
        error = backend.copy(
            impl_->render_previous_states,
            impl_->states[impl_->current_state],
            impl_->rigid_body_count * sizeof(RigidBodyState),
            device::CopyDirection::device_to_device, stream);
        if (error != backend.success_code())
            return backend.failure(error,
                                "rigid interpolation snapshot failed");
    }

    impl_->timing_available = false;
    impl_->timing_boundary_count = 0U;
    std::size_t timing_boundary = 0U;
    if (options.collect_kernel_timings) {
        std::size_t maximum_stages =
            static_cast<std::size_t>(options.substeps) * 8U + 1U;
        for (const auto &fluid : impl_->fluids) {
            if (fluid && fluid->alive) {
                maximum_stages += 2U +
                    static_cast<std::size_t>(options.substeps) *
                        fluid->options.solver_iterations *
                        (9U + 2U * impl_->fluid_soft_couplings.size() +
                         2U * impl_->fluid_rope_couplings.size());
            }
        }
        for (const auto &smoke : impl_->smokes)
            if (smoke && smoke->alive) maximum_stages += 3U;
        for (const auto &cloth : impl_->cloths) {
            if (cloth && cloth->alive)
                maximum_stages += static_cast<std::size_t>(options.substeps) *
                    (cloth->solver_iterations + 2U);
        }
        for (const auto &body : impl_->soft_bodies) {
            if (body && body->alive)
                maximum_stages += static_cast<std::size_t>(options.substeps) *
                    (body->solver_iterations + 3U +
                     (body->solver_iterations + 1U) / 2U);
        }
        status = impl_->prepare_timing_events(
            maximum_stages + options.substeps *
                (impl_->ropes.size()+impl_->rope_soft_body_couplings.size()) + 2U);
        if (!status) {
            return status;
        }
        error = backend.record_event(impl_->timing_events[timing_boundary++], stream);
        if (error != backend.success_code()) {
            return backend.failure(error, "failed to begin kernel timing");
        }
    }

    const auto record_timing_stage = [&](TimingStage stage,
                                         std::uint32_t launches = 0U) noexcept -> Status {
        if (!options.collect_kernel_timings) {
            return Status{};
        }
        if (timing_boundary >= impl_->timing_events.size()) {
            backend.wait(stream);
            return schedule_failure(StatusCode::internal_error,
                           "kernel timing stage budget exhausted");
        }
        impl_->timing_stages.push_back(stage);
        impl_->timing_launch_counts.push_back(launches);
        const auto timing_error =
            backend.record_event(impl_->timing_events[timing_boundary++], stream);
        return timing_error == backend.success_code()
                   ? Status{}
                   : backend.failure(timing_error,
                                  "failed to record kernel timing boundary");
    };

    constexpr std::uint32_t block_size = 128U;
    const std::uint32_t block_count =
        (impl_->rigid_body_count + block_size - 1U) / block_size;
    for (const auto &coupling : impl_->fluid_soft_couplings) {
        if (!coupling || !coupling->alive) continue;
        const auto &body = *impl_->soft_bodies[coupling->options.soft_body.index];
        error = backend.copy(coupling->previous_surface, body.surface_positions,
            body.surface_vertex_count * sizeof(Vec3), device::CopyDirection::device_to_device, stream);
        if (error != backend.success_code()) return backend.failure(error, "previous fluid soft-body skin copy failed");
    }
    for (auto &owner : impl_->smokes) {
        if (!owner || !owner->alive || owner->count == 0U ||
            owner->grid.resolution != 0U) continue;
        auto &smoke = *owner;
        backend.index_smoke_particles(
            (smoke.options.capacity + block_size - 1U) / block_size,
            block_size, stream, smoke.positions, smoke.ages, smoke.count,
            backend.smoke_domain_options(smoke.options), smoke.keys[0],
            smoke.indices[0]);
        error = backend.radix_sort_pairs_u64_u32(
            stream, smoke.keys[0], smoke.keys[1], smoke.indices[0],
            smoke.indices[1], smoke.options.capacity, smoke.sort_workspace,
            smoke.sort_workspace_size);
        if (error != backend.success_code())
            return backend.failure(error, "smoke neighbor index failed");
        smoke.index_dirty = false;
    }
    // The Eulerian air step precedes deformable integration, so every
    // coupled surface reads the same projected field for this frame.
    for (auto &owner : impl_->smokes) {
        if (!owner || !owner->alive || owner->grid.resolution == 0U) continue;
        auto &smoke = *owner;
        auto &grid = smoke.grid;
        const auto blocks = (grid.cell_count + 255U) / 256U;
        auto dispatch_field = backend.smoke_dispatch_grid(grid);
        const auto dispatch_options = backend.smoke_domain_options(smoke.options);
        const typename Backend::SmokeGridBoundaryDispatch dispatch_boundary{
            grid.solid,
            {grid.face_open[0], grid.face_open[1], grid.face_open[2]},
            {grid.face_wall_velocity[0], grid.face_wall_velocity[1],
             grid.face_wall_velocity[2]},
            {grid.face_normal[0], grid.face_normal[1], grid.face_normal[2]},
            {grid.face_nearest_triangle[0],
             grid.face_nearest_triangle[1],
             grid.face_nearest_triangle[2]}};
        const float smoke_gravity[3]{
            options.gravity.x, options.gravity.y, options.gravity.z};
        const auto largest = std::max({grid.cell_count, grid.face_count[0],
            grid.face_count[1], grid.face_count[2]});
        backend.clear_smoke_grid_obstacles(
            (largest + 255U) / 256U, 256U, stream,
            grid.solid, grid.cell_count,
            grid.face_open[0], grid.face_open[1], grid.face_open[2],
            grid.face_wall_velocity[0], grid.face_wall_velocity[1],
            grid.face_wall_velocity[2],
            grid.face_count[0],
            grid.face_count[1], grid.face_count[2]);
        backend.clear(grid.density_accumulator,0,
            grid.cell_count*sizeof(unsigned long long),stream);
        backend.clear(grid.temperature_accumulator,0,
            grid.cell_count*sizeof(unsigned long long),stream);
        if (smoke.count != 0U)
            backend.splat_smoke_grid_particles(
                (smoke.count + 127U) / 128U, 128U, stream,
                smoke.positions, smoke.ages, smoke.thermal_lift,
                smoke.count, dispatch_options, dispatch_field,
                reinterpret_cast<std::uint64_t *>(grid.density_accumulator),
                reinterpret_cast<std::uint64_t *>(
                    grid.temperature_accumulator));
        backend.resolve_smoke_grid_particle_fields(
            blocks, 256U, stream,
            reinterpret_cast<const std::uint64_t *>(grid.density_accumulator),
            reinterpret_cast<const std::uint64_t *>(
                grid.temperature_accumulator),
            grid.density,grid.temperature,grid.cell_count);
        bool moving_triangle_boundary = false;
        for (const auto &coupling : impl_->smoke_rigid_couplings) {
            if (!coupling || !coupling->alive || !coupling->options.enabled ||
                coupling->options.smoke.index >= impl_->smokes.size() ||
                impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                continue;
            std::uint32_t body{};
            status = impl_->validate_handle(coupling->options.body, body);
            if (!status) return status;
            // Rigid metadata lives in GPU-managed storage. Treat every active
            // coupling as moving rather than faulting on a host dereference.
            moving_triangle_boundary = true;
        }
        for (const auto &coupling : impl_->smoke_soft_body_couplings)
            moving_triangle_boundary |= coupling && coupling->alive &&
                coupling->options.enabled &&
                coupling->options.smoke.index < impl_->smokes.size() &&
                impl_->smokes[coupling->options.smoke.index].get() == &smoke;
        for (const auto &coupling : impl_->smoke_cloth_couplings)
            moving_triangle_boundary |= coupling && coupling->alive &&
                coupling->options.enabled &&
                coupling->options.smoke.index < impl_->smokes.size() &&
                impl_->smokes[coupling->options.smoke.index].get() == &smoke;
        const int raster_passes = moving_triangle_boundary ||
            !grid.static_metadata_valid ? 2 : 1;
        for (const auto &coupling : impl_->smoke_rigid_couplings) {
            if (!coupling || !coupling->alive || !coupling->options.enabled ||
                coupling->options.smoke.index >= impl_->smokes.size() ||
                impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                continue;
            std::uint32_t body{};
            status = impl_->validate_handle(coupling->options.body, body);
            if (!status) return status;
            BodyParameters parameters{};
            status = backend.read_value(stream, impl_->parameters + body,
                parameters, "failed to read smoke rigid-body parameters");
            if (!status) return status;
            const auto *mesh = impl_->meshes + parameters.mesh.index;
            TriangleMeshResource mesh_metadata{};
            status = backend.read_value(stream, mesh, mesh_metadata,
                "failed to read smoke rigid-body mesh metadata");
            if (!status) return status;
            const auto triangles = mesh_metadata.index_count / 3U;
            for (int pass = 0; pass < raster_passes; ++pass)
                backend.raster_smoke_grid_rigid_body(
                    (triangles + 127U) / 128U, 128U, stream,
                    &mesh_metadata, impl_->states[impl_->current_state], body,
                    dispatch_field, dispatch_boundary, pass != 0);
        }
        for (const auto &coupling : impl_->smoke_soft_body_couplings) {
            if (!coupling || !coupling->alive || !coupling->options.enabled ||
                coupling->options.smoke.index >= impl_->smokes.size() ||
                impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                continue;
            const auto &body = *impl_->soft_bodies[
                coupling->options.soft_body.index];
            const auto triangles = body.surface_index_count / 3U;
            for (int pass = 0; pass < raster_passes; ++pass)
                backend.raster_smoke_grid_soft_body(
                    (triangles + 127U) / 128U, 128U, stream,
                    body.surface_positions,
                    body.surface_bindings, body.velocities,
                    body.surface_indices, body.surface_index_count,
                    dispatch_field, dispatch_boundary, pass != 0);
        }
        for (const auto &coupling : impl_->smoke_cloth_couplings) {
            if (!coupling || !coupling->alive || !coupling->options.enabled ||
                coupling->options.smoke.index >= impl_->smokes.size() ||
                impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                continue;
            const auto &cloth = *impl_->cloths[coupling->options.cloth.index];
            const auto triangles = cloth.index_count / 3U;
            for (int pass = 0; pass < raster_passes; ++pass)
                backend.raster_smoke_grid_deformable(
                    (triangles + 127U) / 128U, 128U, stream,
                    cloth.positions, cloth.velocities,
                    cloth.indices, cloth.index_count, dispatch_field,
                    dispatch_boundary, pass != 0);
        }
        grid.static_metadata_valid = !moving_triangle_boundary;

        // Fixed inlet and far-field normal velocities are zero-aperture
        // pressure boundaries. Downstream faces stay open with p=0 outside.
        for (int axis = 0; axis < 3; ++axis)
            backend.mark_smoke_grid_domain_boundaries(
                (grid.face_count[axis] + 255U) / 256U, 256U, stream,
                axis, dispatch_field, grid.face_open[axis],
                grid.face_wall_velocity[axis], dispatch_options);

        // Coarsen the cut-face apertures once; all pressure V-cycles reuse
        // the same geometry for this frame.
        int fine_n = int(grid.resolution), fine_h = int(grid.height);
        const float *fine_open[3] = {grid.face_open[0], grid.face_open[1],
                                     grid.face_open[2]};
        for (auto &level : grid.coarse) {
            for (int axis = 0; axis < 3; ++axis) {
                const auto count = backend.smoke_grid_face_count(axis,
                    int(level.n), int(level.height));
                backend.restrict_smoke_grid_open_faces(
                    (count + 255U) / 256U, 256U, stream,
                    axis, fine_open[axis], fine_n, fine_h,
                    level.open[axis], int(level.n), int(level.height));
                fine_open[axis] = level.open[axis];
            }
            fine_n = int(level.n); fine_h = int(level.height);
        }

        // RK2 MacCormack self-advection on the staggered faces.
        for (int axis = 0; axis < 3; ++axis)
            backend.advect_smoke_grid_face(
                (grid.face_count[axis] + 255U) / 256U, 256U, stream,
                axis, dispatch_field, grid.face_velocity[axis][0],
                grid.face_velocity[axis][1], options.timestep);
        auto dispatch_predicted = dispatch_field;
        for (int axis = 0; axis < 3; ++axis)
            dispatch_predicted.face_velocity[axis] =
                grid.face_velocity[axis][1];
        for (int axis = 0; axis < 3; ++axis)
            backend.advect_smoke_grid_face(
                (grid.face_count[axis] + 255U) / 256U, 256U, stream,
                axis, dispatch_predicted, grid.face_velocity[axis][1],
                grid.face_reverse[axis], -options.timestep);
        for (int axis = 0; axis < 3; ++axis)
            backend.correct_smoke_grid_face(
                (grid.face_count[axis] + 255U) / 256U, 256U, stream,
                axis, dispatch_field, grid.face_velocity[axis][0],
                grid.face_velocity[axis][1], grid.face_reverse[axis],
                grid.face_velocity[axis][1], options.timestep);

        backend.calculate_smoke_grid_cell_diagnostics(
            blocks, 256U, stream, dispatch_predicted,
            nullptr, grid.vorticity, grid.strain, nullptr);
        backend.calculate_smoke_grid_subgrid_force(
            blocks, 256U, stream, dispatch_predicted,
            grid.vorticity, grid.subgrid_force,
            smoke.options.vorticity_confinement);
        for (int axis = 0; axis < 3; ++axis)
            backend.apply_smoke_grid_face_forces(
                (grid.face_count[axis] + 255U) / 256U, 256U, stream,
                axis, dispatch_predicted, grid.face_velocity[axis][1],
                grid.face_velocity[axis][0], grid.face_open[axis],
                grid.face_wall_velocity[axis], grid.strain,
                grid.subgrid_force, grid.temperature, dispatch_options,
                options.timestep, smoke_gravity);
        auto dispatch_forced = dispatch_field;
        for (int axis = 0; axis < 3; ++axis) {
            dispatch_forced.face_velocity[axis] =
                grid.face_velocity[axis][0];
            backend.apply_smoke_grid_face_boundaries(
                (grid.face_count[axis] + 255U) / 256U, 256U, stream,
                axis, dispatch_forced, grid.face_velocity[axis][0],
                grid.face_open[axis], grid.face_wall_velocity[axis],
                dispatch_options);
        }

        backend.clear(grid.rhs_max, 0, sizeof(float), stream);
        backend.clear(grid.pressure_converged, 0,
            sizeof(unsigned int), stream);
        backend.calculate_smoke_grid_divergence(
            blocks, 256U, stream, dispatch_forced,
            grid.divergence, grid.rhs_max, options.timestep);

        int level_n[4] = {int(grid.resolution), int(grid.coarse[0].n),
                          int(grid.coarse[1].n), int(grid.coarse[2].n)};
        int level_h[4] = {int(grid.height), int(grid.coarse[0].height),
                          int(grid.coarse[1].height), int(grid.coarse[2].height)};
        float level_spacing[4] = {grid.spacing, grid.spacing*2.0F,
                                  grid.spacing*4.0F, grid.spacing*8.0F};
        float *level_pressure[4][2] = {
            {grid.pressure[0], grid.pressure[1]},
            {grid.coarse[0].pressure[0], grid.coarse[0].pressure[1]},
            {grid.coarse[1].pressure[0], grid.coarse[1].pressure[1]},
            {grid.coarse[2].pressure[0], grid.coarse[2].pressure[1]}};
        float *level_rhs[4] = {grid.divergence, grid.coarse[0].rhs,
            grid.coarse[1].rhs, grid.coarse[2].rhs};
        float *level_residual[4] = {grid.residual, grid.coarse[0].residual,
            grid.coarse[1].residual, grid.coarse[2].residual};
        float *level_open[4][3] = {
            {grid.face_open[0],grid.face_open[1],grid.face_open[2]},
            {grid.coarse[0].open[0],grid.coarse[0].open[1],grid.coarse[0].open[2]},
            {grid.coarse[1].open[0],grid.coarse[1].open[1],grid.coarse[1].open[2]},
            {grid.coarse[2].open[0],grid.coarse[2].open[1],grid.coarse[2].open[2]}};
        const auto smooth = [&](int level, int iterations) {
            const auto count = std::uint32_t(level_n[level] * level_h[level] *
                                             level_n[level]);
            for (int iteration = 0; iteration < iterations; ++iteration) {
                backend.smooth_smoke_grid_pressure(
                    (count + 255U) / 256U, 256U, stream,
                    level_n[level], level_h[level],
                    level_spacing[level], level_rhs[level],
                    level_pressure[level][0], level_pressure[level][1],
                    level_open[level][0], level_open[level][1],
                    level_open[level][2], grid.pressure_converged);
                std::swap(level_pressure[level][0], level_pressure[level][1]);
            }
        };
        // One V-cycle costs about 4.6 fine-grid Jacobi sweeps after accounting
        // for the geometrically smaller levels. Keep the authored value as a
        // fine-grid-equivalent work ceiling and stop earlier on the GPU when
        // the relative residual reaches the requested tolerance.
        const auto cycles = std::max(1U,
            (smoke.options.grid_pressure_iterations + 4U) / 5U);
        for (std::uint32_t cycle = 0U; cycle < cycles; ++cycle) {
            for (int level = 0; level < 3; ++level) {
                smooth(level, 2);
                const auto count = std::uint32_t(level_n[level] *
                    level_h[level] * level_n[level]);
                backend.calculate_smoke_grid_pressure_residual(
                    (count + 255U) / 256U, 256U, stream,
                    level_n[level], level_h[level],
                    level_spacing[level], level_rhs[level],
                    level_pressure[level][0], level_residual[level],
                    level_open[level][0], level_open[level][1],
                    level_open[level][2], nullptr, grid.pressure_converged);
                const auto coarse_count = std::uint32_t(level_n[level+1] *
                    level_h[level+1] * level_n[level+1]);
                backend.restrict_smoke_grid_residual(
                    (coarse_count + 255U) / 256U, 256U, stream,
                    level_residual[level], level_n[level], level_h[level],
                    level_rhs[level+1], level_n[level+1], level_h[level+1]);
                backend.clear(level_pressure[level+1][0], 0,
                    coarse_count * sizeof(float), stream);
                backend.clear(level_pressure[level+1][1], 0,
                    coarse_count * sizeof(float), stream);
            }
            smooth(3, 12);
            for (int level = 2; level >= 0; --level) {
                const auto fine_count = std::uint32_t(level_n[level] *
                    level_h[level] * level_n[level]);
                backend.prolong_smoke_grid_pressure(
                    (fine_count + 255U) / 256U, 256U, stream,
                    level_pressure[level+1][0],
                    level_n[level+1], level_h[level+1],
                    level_pressure[level][0], level_n[level], level_h[level]);
                smooth(level, 2);
            }
            backend.reset_smoke_grid_residual(
                1U, 1U, stream,
                grid.residual_max,grid.pressure_converged);
            backend.calculate_smoke_grid_pressure_residual(
                blocks, 256U, stream,
                level_n[0], level_h[0], level_spacing[0], level_rhs[0],
                level_pressure[0][0], level_residual[0], level_open[0][0],
                level_open[0][1], level_open[0][2], grid.residual_max,
                grid.pressure_converged);
            backend.compare_smoke_grid_residual(
                1U, 1U, stream, grid.rhs_max,
                grid.residual_max, smoke.options.grid_pressure_tolerance,
                grid.pressure_converged);
        }
        grid.pressure[0] = level_pressure[0][0];
        grid.pressure[1] = level_pressure[0][1];
        for (int level = 0; level < 3; ++level) {
            grid.coarse[level].pressure[0] = level_pressure[level+1][0];
            grid.coarse[level].pressure[1] = level_pressure[level+1][1];
        }
        for (int axis = 0; axis < 3; ++axis) {
            backend.project_smoke_grid_face(
                (grid.face_count[axis] + 255U) / 256U, 256U, stream,
                axis, dispatch_forced, grid.pressure[0],
                grid.face_velocity[axis][0],
                grid.face_open[axis], grid.face_wall_velocity[axis],
                options.timestep);
        }
        auto dispatch_projected = dispatch_forced;
        dispatch_projected.pressure = grid.pressure[0];
        dispatch_projected.vorticity = grid.vorticity;
        backend.calculate_smoke_grid_cell_diagnostics(
            blocks, 256U, stream, dispatch_projected,
            grid.velocity, grid.vorticity, grid.strain, grid.divergence);
        error = backend.check_error();
        if (error != backend.success_code())
            return backend.failure(error, "smoke grid step failed");
    }
    if (std::any_of(impl_->smokes.begin(), impl_->smokes.end(),
        [](const auto &smoke) {
            return smoke && smoke->alive && smoke->grid.resolution != 0U;
        })) {
        status = record_timing_stage(TimingStage::smoke_grid);
        if (!status) return status;
    }
    if (debug_enabled && impl_->rigid_body_count != 0U) {
        error = backend.capture_rigid_inputs(
            block_count, block_size, stream,
            impl_->accumulators, impl_->debug_applied_forces,
            impl_->debug_applied_torques, impl_->rigid_body_count);
        if (error != backend.success_code())
            return backend.failure(error,
                                "rigid debug input capture launch failed");
    }
    const bool has_cloth = std::any_of(impl_->cloths.begin(), impl_->cloths.end(),
        [](const auto &cloth) { return cloth && cloth->alive; });
    const bool has_soft_body = std::any_of(
        impl_->soft_bodies.begin(), impl_->soft_bodies.end(),
        [](const auto &body) { return body && body->alive; });
    const bool has_rope_soft_body = std::any_of(
        impl_->rope_soft_body_couplings.begin(),
        impl_->rope_soft_body_couplings.end(),
        [](const auto &coupling) {
            return coupling && coupling->alive && coupling->options.enabled;
        });
    const bool has_smoke = std::any_of(impl_->smokes.begin(), impl_->smokes.end(),
        [](const auto &smoke) { return smoke && smoke->alive; });
    // Body parameters live in GPU-managed storage. Reading motion types here
    // can fault on Windows while the GPU owns those pages, so conservatively
    // run the rigid coupling path whenever rigid bodies exist.
    const bool has_rigid_bodies = impl_->rigid_body_count != 0U;
    if ((impl_->fluid_count != 0U || has_smoke) &&
        has_rigid_bodies) {
        error = backend.copy(impl_->fluid_previous_states,
            impl_->states[impl_->current_state],
            impl_->rigid_body_count * sizeof(RigidBodyState),
            device::CopyDirection::device_to_device, stream);
        if (error != backend.success_code())
            return backend.failure(error, "fluid body state copy failed");
    }
    const float substep_timestep =
        options.timestep / static_cast<float>(options.substeps);
    impl_->rigid_solve_kernels_per_substep = 6U;
    const auto coupled_cloth = [&](std::uint32_t index) {
        return std::any_of(impl_->soft_cloth_couplings.begin(),
            impl_->soft_cloth_couplings.end(), [&](const auto &coupling) {
                return coupling && coupling->alive && coupling->options.enabled &&
                       coupling->options.cloth.index == index;
            });
    };
    const auto coupled_soft = [&](std::uint32_t index) {
        return std::any_of(impl_->soft_cloth_couplings.begin(),
            impl_->soft_cloth_couplings.end(), [&](const auto &coupling) {
                return coupling && coupling->alive && coupling->options.enabled &&
                       coupling->options.soft_body.index == index;
            });
    };
    const auto advance_cloth = [&](const RigidBodyState *previous_states)
        noexcept -> Status {
        for (std::uint32_t cloth_index = 0U;
             cloth_index < impl_->cloths.size(); ++cloth_index) {
            const auto &cloth_pointer = impl_->cloths[cloth_index];
            if (!cloth_pointer || !cloth_pointer->alive) continue;
            auto &cloth = *cloth_pointer;
            const std::uint32_t blocks =
                (cloth.vertex_count + block_size - 1U) / block_size;
            for (const auto &owner : impl_->smoke_cloth_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled ||
                    owner->options.cloth.index != cloth_index ||
                    owner->options.wind_drag == 0.0F) continue;
                const auto &smoke = *impl_->smokes[owner->options.smoke.index];
                if (smoke.count == 0U) continue;
                if (smoke.grid.resolution != 0U) {
                    error = backend.clear(cloth.smoke_forces, 0,
                        cloth.vertex_count * sizeof(Vec3), stream);
                    if (error != backend.success_code())
                        return backend.failure(error,
                            "cloth smoke force clear failed");
                    const auto triangles = cloth.index_count / 3U;
                    backend.accumulate_smoke_grid_force_on_cloth(
                        (triangles + block_size - 1U) / block_size,
                        block_size, stream, cloth.positions,
                        cloth.velocities, cloth.inverse_masses, cloth.indices,
                        cloth.index_count, backend.smoke_dispatch_grid(smoke.grid),
                        owner->options.wind_drag,
                        smoke.options.rest_number_density,
                        smoke.options.grid_kinematic_viscosity,
                        smoke.options.grid_les_coefficient,
                        cloth.smoke_forces);
                    backend.apply_smoke_grid_force_to_cloth(
                        blocks, block_size, stream,
                        cloth.velocities, cloth.inverse_masses,
                        cloth.smoke_forces, cloth.vertex_count,
                        owner->options.maximum_wind_acceleration,
                        substep_timestep);
                } else {
                    backend.apply_smoke_wind_to_cloth(
                        blocks, block_size, stream,
                        cloth.positions, cloth.velocities, cloth.inverse_masses,
                        cloth.vertex_count,
                        backend.smoke_dispatch_options(smoke.options),
                        smoke.positions, smoke.velocities,
                        smoke.keys[1], smoke.indices[1], typename Backend::SmokeGridDispatchField{},
                        owner->options.wind_drag,
                        owner->options.maximum_wind_acceleration,
                        substep_timestep);
                }
            }
            backend.predict_deformable(blocks, block_size, stream,
                cloth.positions, cloth.previous, cloth.velocities,
                cloth.inverse_masses, cloth.vertex_count, &options.gravity.x,
                substep_timestep, cloth.velocity_damping);
            Status cloth_status = record_timing_stage(
                TimingStage::cloth_prediction);
            if (!cloth_status) return cloth_status;
            for (std::uint32_t iteration = 0U;
                 iteration < cloth.solver_iterations; ++iteration) {
                backend.project_deformable_links(blocks, block_size, stream,
                    cloth.positions, cloth.scratch, cloth.inverse_masses,
                    cloth.offsets, cloth.neighbors, cloth.bond_active,
                    cloth.vertex_count, substep_timestep, 0.0F);
                std::swap(cloth.positions, cloth.scratch);
                if (cloth.preserve_volume) {
                    backend.project_cloth_volume(stream,
                        cloth.positions, cloth.inverse_masses, cloth.indices,
                        cloth.vertex_count, cloth.index_count / 3U,
                        cloth.volume_corner_offsets, cloth.volume_corner_indices,
                        cloth.volume_gradients, cloth.target_volume,
                        cloth.orientation, cloth.volume_compliance,
                        substep_timestep, cloth.volume_lambda,
                        iteration == 0U);
                }
                cloth_status = record_timing_stage(
                    TimingStage::cloth_constraints);
                if (!cloth_status) return cloth_status;
            }
            if (impl_->rigid_body_count != 0U) {
                const auto clear_error = backend.clear(
                    impl_->fluid_body_contact_flags, 0,
                    impl_->rigid_body_count * sizeof(std::uint32_t), stream);
                if (clear_error != backend.success_code())
                    return backend.failure(clear_error,
                                        "cloth contact flags clear failed");
                backend.collide_deformable_with_rigid_geometry(
                    blocks, block_size, stream,
                    cloth.positions, cloth.velocities, cloth.previous,
                    cloth.inverse_masses, cloth.vertex_count,
                    cloth.thickness, substep_timestep,
                    impl_->parameters, previous_states,
                    impl_->states[impl_->current_state],
                    impl_->meshes, impl_->rigid_body_count,
                    cloth.body_impulses, cloth.rigid_contact_forces,
                    nullptr, nullptr, nullptr, nullptr, nullptr,
                    nullptr,
                    impl_->fluid_body_contact_flags,
                    false, false, false, true, 20.0F);
                backend.reduce_fluid_body_impulses(
                    impl_->rigid_body_count, block_size, stream,
                    cloth.body_impulses, cloth.count, impl_->parameters,
                    impl_->states[impl_->current_state],
                    impl_->fluid_body_contact_flags, impl_->rigid_body_count);
                if (cloth.surface_positions != nullptr) {
                    const std::uint32_t triangles = cloth.index_count / 3U;
                    backend.update_cloth_surface(
                        (triangles + block_size - 1U) / block_size,
                        block_size, stream, cloth.positions,
                        cloth.indices,
                        cloth.surface_positions, triangles);
                }
                backend.constrain_rigid_bodies_with_cloth(
                    block_count, block_size, stream,
                    cloth.positions, cloth.velocities, cloth.indices, cloth.vertex_sources,
                    cloth.surface_positions,
                    cloth.inverse_masses,
                    cloth.vertex_count, cloth.index_count / 3U,
                    cloth.thickness, substep_timestep,
                    cloth.contact_friction, impl_->parameters,
                    previous_states, impl_->states[impl_->current_state],
                    impl_->meshes, impl_->ids, impl_->rigid_body_count,
                    cloth_index, cloth.generation, impl_->paint_fields,
                    impl_->options.paint_field_capacity, impl_->paint_rules,
                    impl_->options.paint_rule_capacity,
                    cloth.body_corrections);
                backend.apply_cloth_body_corrections(
                    blocks, block_size, stream,
                    cloth.positions, cloth.velocities, cloth.inverse_masses,
                    cloth.vertex_count,
                    cloth.body_corrections, impl_->rigid_body_count);
            } else {
                backend.collide_deformable_with_rigid_geometry(
                    blocks, block_size, stream,
                    cloth.positions, cloth.velocities, cloth.previous,
                    cloth.inverse_masses, cloth.vertex_count,
                    cloth.thickness, substep_timestep,
                    impl_->parameters, previous_states,
                    impl_->states[impl_->current_state],
                    impl_->meshes, 0U, cloth.body_impulses,
                    cloth.rigid_contact_forces, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr,
                    impl_->fluid_body_contact_flags,
                    false, false, false, true, 20.0F);
            }
            if (cloth.free_triangle_nodes)
                backend.limit_cloth_strain(stream, cloth.positions,
                    cloth.scratch, cloth.inverse_masses, cloth.offsets,
                    cloth.neighbors, cloth.free_triangle_nodes, cloth.vertex_count);
            cloth_status = record_timing_stage(TimingStage::cloth_contacts);
            if (!cloth_status) return cloth_status;
            if (cloth.surface_positions != nullptr) {
                const std::uint32_t triangles = cloth.index_count / 3U;
                if (!coupled_cloth(cloth_index)) backend.break_cloth_bonds(
                    (cloth.bond_count + block_size - 1U) / block_size,
                    block_size, stream, cloth.positions, cloth.bonds,
                    cloth.bond_active, cloth.bond_damage, cloth.bond_count,
                    cloth.break_strain, cloth.fracture_persistence_substeps,
                    cloth.body_impulses, cloth.impact_break_impulse);
                backend.update_cloth_surface(
                    (triangles + block_size - 1U) / block_size,
                    block_size, stream, cloth.positions,
                    cloth.indices,
                    cloth.surface_positions, triangles);
            }
        }
        const auto cloth_error = backend.check_error();
        if (cloth_error != backend.success_code()) {
            backend.wait(stream);
            return backend.failure(cloth_error, "cloth kernel launch failed");
        }
        return Status{};
    };
    const auto advance_soft_bodies = [&](const RigidBodyState *previous_states)
        noexcept -> Status {
        for (const auto &body_pointer : impl_->soft_bodies) {
            if (!body_pointer || !body_pointer->alive) continue;
            auto &body = *body_pointer;
            const std::uint32_t blocks =
                (body.node_count + block_size - 1U) / block_size;
            bool grid_snapshot_ready = false;
            for (const auto &owner : impl_->smoke_soft_body_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled ||
                    owner->options.soft_body.index >= impl_->soft_bodies.size() ||
                    impl_->soft_bodies[owner->options.soft_body.index].get() != &body ||
                    owner->options.wind_drag == 0.0F) continue;
                const auto &smoke = *impl_->smokes[owner->options.smoke.index];
                if (smoke.count == 0U) continue;
                if (smoke.grid.resolution != 0U) {
                    if (!grid_snapshot_ready) {
                        const auto copy_error = backend.copy(
                            body.velocity_scratch, body.velocities,
                            body.node_count * sizeof(Vec3),
                            device::CopyDirection::device_to_device, stream);
                        if (copy_error != backend.success_code())
                            return backend.failure(copy_error,
                                "soft-body air velocity snapshot failed");
                        grid_snapshot_ready = true;
                    }
                    const auto triangles = body.surface_index_count / 3U;
                    backend.apply_smoke_grid_force_to_soft_body(
                        (triangles + block_size - 1U) / block_size,
                        block_size, stream, body.surface_positions,
                        body.surface_indices, body.surface_index_count,
                        body.surface_bindings, body.velocity_scratch,
                        body.velocities,
                        body.inverse_masses, backend.smoke_dispatch_grid(smoke.grid),
                        owner->options.wind_drag,
                        smoke.options.rest_number_density,
                        smoke.options.grid_kinematic_viscosity,
                        smoke.options.grid_les_coefficient,
                        owner->options.maximum_wind_acceleration,
                        substep_timestep);
                } else {
                    backend.apply_smoke_wind_to_soft_body(
                        blocks, block_size, stream,
                        body.positions, body.velocities, body.inverse_masses,
                        body.node_count,
                        backend.smoke_dispatch_options(smoke.options),
                        smoke.positions, smoke.velocities,
                        smoke.keys[1], smoke.indices[1], typename Backend::SmokeGridDispatchField{},
                        owner->options.wind_drag,
                        owner->options.maximum_wind_acceleration,
                        substep_timestep, body.maximum_speed);
                }
            }
            backend.predict_deformable(blocks, block_size, stream,
                body.positions, body.previous, body.velocities,
                body.inverse_masses, body.node_count, &options.gravity.x,
                substep_timestep, body.velocity_damping);
            auto friction_clear_error = backend.clear(
                body.contact_momentum_delta, 0,
                body.node_count * sizeof(Vec3), stream);
            if (friction_clear_error != backend.success_code())
                return backend.failure(friction_clear_error,
                    "soft-body momentum accumulator clear failed");
            friction_clear_error = backend.clear(
                body.contact_friction_delta, 0,
                body.node_count * sizeof(Vec3), stream);
            if (friction_clear_error != backend.success_code())
                return backend.failure(friction_clear_error,
                    "soft-body friction accumulator clear failed");
            friction_clear_error = backend.clear(
                body.contact_normal_delta, 0,
                body.node_count * sizeof(float), stream);
            if (friction_clear_error != backend.success_code())
                return backend.failure(friction_clear_error,
                    "soft-body normal accumulator clear failed");
            friction_clear_error = backend.clear(
                body.dynamic_contact_flag, 0,
                sizeof(std::uint32_t), stream);
            if (friction_clear_error != backend.success_code())
                return backend.failure(friction_clear_error,
                    "soft-body dynamic contact flag clear failed");
            backend.measure_soft_body_momentum(block_size, stream,
                body.velocities, body.inverse_masses, body.node_count,
                body.predicted_momentum);
            Status body_status = record_timing_stage(
                TimingStage::soft_body_prediction);
            if (!body_status) return body_status;
            const auto resolve_contacts = [&]() noexcept -> Status {
                auto clear_error = backend.clear(
                    body.contact_count, 0, sizeof(std::uint32_t), stream);
                if (clear_error != backend.success_code())
                    return backend.failure(clear_error,
                        "soft-body contact count clear failed");
                if (impl_->rigid_body_count != 0U) {
                    clear_error = backend.clear(
                        impl_->fluid_body_contact_flags, 0,
                        impl_->rigid_body_count * sizeof(std::uint32_t),
                        stream);
                    if (clear_error != backend.success_code())
                        return backend.failure(clear_error,
                            "soft-body contact flags clear failed");
                }
                backend.collide_deformable_with_rigid_geometry(
                    blocks, block_size, stream,
                    body.positions, body.velocities, body.previous,
                    body.inverse_masses, body.node_count, body.node_radius,
                    substep_timestep, impl_->parameters, previous_states,
                    impl_->states[impl_->current_state], impl_->meshes,
                    impl_->rigid_body_count, body.body_impulses,
                    body.rigid_contact_forces, body.contact_normals,
                    body.contact_arms,
                    body.contact_momentum_delta,
                    body.contact_normal_delta,
                    body.contact_count, body.dynamic_contact_flag,
                    impl_->fluid_body_contact_flags,
                    true, false, false, false, body.maximum_speed,
                    body.body_position_corrections);
                return Status{};
            };
            const auto apply_contact_traction = [&]() noexcept {
                backend.apply_soft_body_contact_friction(
                    blocks, block_size, stream,
                    body.positions, body.velocities, body.inverse_masses,
                    body.rigid_contact_forces, body.contact_normals,
                    body.contact_arms, body.body_impulses,
                    impl_->parameters,
                    body.contact_momentum_delta,
                    body.contact_friction_delta,
                    body.contact_normal_delta,
                    body.contact_count, body.node_count, body.movable_mass,
                    &options.gravity.x, substep_timestep, body.contact_friction,
                    body.maximum_speed);
            };
            const auto finish_contact_pass = [&]() noexcept -> Status {
                apply_contact_traction();
                if (impl_->rigid_body_count != 0U)
                    backend.reduce_fluid_body_impulses(
                        impl_->rigid_body_count, block_size, stream,
                        body.body_impulses, body.count, impl_->parameters,
                        impl_->states[impl_->current_state],
                        impl_->fluid_body_contact_flags,
                        impl_->rigid_body_count,
                        body.body_position_corrections);
                return record_timing_stage(TimingStage::soft_body_contacts);
            };
            // Contact and graph constraints form one position solve. Revisit
            // the triangle boundary after every two graph passes so spring
            // projection cannot strand nodes across a wall, while the next
            // passes distribute each contact correction through the volume.
            body_status = resolve_contacts();
            if (!body_status) return body_status;
            body_status = finish_contact_pass();
            if (!body_status) return body_status;
            for (std::uint32_t iteration = 0U;
                 iteration < body.solver_iterations; ++iteration) {
                if(body.warp_neighbors) {
                    constexpr std::uint32_t groups_per_block = 8U;
                    const std::uint32_t group_blocks =
                        (body.node_count + groups_per_block - 1U) /
                        groups_per_block;
                    backend.project_deformable_links_warp(
                        group_blocks, block_size, stream,
                        body.positions, body.scratch, body.inverse_masses,
                        body.offsets, body.warp_neighbors,
                        body.minimum_rest_lengths, body.node_count,
                        body.stretch_compliance, substep_timestep,
                        body.maximum_projection_fraction);
                } else
                    backend.project_deformable_links(
                        blocks, block_size, stream,
                        body.positions, body.scratch, body.inverse_masses,
                        body.offsets, body.neighbors, body.bond_active,
                        body.node_count, substep_timestep,
                        body.maximum_projection_fraction);
                std::swap(body.positions, body.scratch);
                if (iteration + 1U == body.solver_iterations &&
                    body.shape_matching_stiffness > 0.0F) {
                    backend.project_soft_body_rest_shape(stream,
                        body.positions, body.scratch, body.rest_positions,
                        body.inverse_masses, body.node_count,
                        body.movable_mass, &body.shape_rest_center.x,
                        &body.shape_inverse_rest, body.shape_orientation,
                        body.shape_matching_stiffness,
                        body.shape_maximum_projection,
                        body.dynamic_contact_flag);
                }
                body_status = record_timing_stage(
                    TimingStage::soft_body_constraints);
                if (!body_status) return body_status;
                if ((iteration + 1U) % 2U == 0U ||
                    iteration + 1U == body.solver_iterations) {
                    body_status = resolve_contacts();
                    if (!body_status) return body_status;
                    if (iteration + 1U != body.solver_iterations) {
                        body_status = finish_contact_pass();
                        if (!body_status) return body_status;
                    }
                }
            }
            backend.finalize_soft_body_velocities(
                blocks, block_size, stream,
                body.positions, body.previous, body.velocities,
                body.inverse_masses, body.node_count,
                1.0F / substep_timestep,
                body.constraint_velocity_response, body.maximum_speed);
            if (body.warp_neighbors) {
                constexpr std::uint32_t groups_per_block = 8U;
                const std::uint32_t group_blocks =
                    (body.node_count + groups_per_block - 1U) /
                    groups_per_block;
                backend.damp_soft_body_springs_warp(
                    group_blocks, block_size, stream,
                    body.positions, body.velocities, body.velocity_scratch,
                    body.inverse_masses, body.offsets, body.warp_neighbors,
                    body.node_count, body.spring_damping,
                    body.maximum_speed);
            } else {
                backend.damp_soft_body_springs(
                    blocks, block_size, stream,
                    body.positions, body.velocities, body.velocity_scratch,
                    body.inverse_masses, body.offsets, body.neighbors,
                    body.bond_active, body.node_count, body.spring_damping,
                    body.maximum_speed);
            }
            std::swap(body.velocities, body.velocity_scratch);
            body_status = finish_contact_pass();
            if (!body_status) return body_status;
            backend.restore_soft_body_momentum(block_size, stream,
                body.velocities, body.inverse_masses,
                body.contact_momentum_delta, body.node_count,
                body.movable_mass, body.predicted_momentum,
                body.dynamic_contact_flag,
                body.maximum_speed);
            // Traction changes positions too. Finish with nonpenetration so
            // neither friction nor a competing collider becomes next step's
            // already-invalid sweep origin. This pass adds no second impulse.
            // Recovery enforces geometry with a small skin. The full node
            // radius is the normal solver's contact target, but overlapping
            // safety margins in a pinch must not push a node through a solid.
            for (std::uint32_t pass = 0U;
                 impl_->rigid_body_count != 0U &&
                 pass < k_soft_contact_cleanup_passes; ++pass) {
                if (pass % 2U == 0U) {
                    backend.update_soft_body_surface(
                        (body.surface_vertex_count + block_size - 1U) / block_size,
                        block_size, stream, body.positions, body.rest_positions,
                        body.surface_rest_positions, body.surface_bindings,
                        body.surface_positions, body.surface_vertex_count);
                    const std::uint32_t triangles = body.surface_index_count / 3U;
                    backend.calculate_soft_body_surface_contacts(
                        (triangles + block_size - 1U) / block_size,
                        block_size, stream, body.surface_positions,
                        body.surface_indices, triangles, impl_->parameters,
                        impl_->states[impl_->current_state], impl_->meshes,
                        impl_->rigid_body_count, body.surface_corner_corrections);
                    backend.apply_soft_body_surface_contacts(
                        blocks, block_size, stream,
                        body.positions, body.surface_corner_corrections,
                        body.surface_node_offsets, body.surface_node_influences,
                        body.node_count);
                }
                backend.collide_deformable_with_rigid_geometry(
                    blocks, block_size, stream,
                    body.positions, body.velocities, body.previous,
                    body.inverse_masses, body.node_count,
                    body.node_radius * 0.01F,
                    substep_timestep, impl_->parameters, previous_states,
                    impl_->states[impl_->current_state], impl_->meshes,
                    impl_->rigid_body_count, body.body_impulses,
                    body.rigid_contact_forces, body.contact_normals,
                    body.contact_arms, body.contact_momentum_delta,
                    body.contact_normal_delta, body.contact_count,
                    body.dynamic_contact_flag, impl_->fluid_body_contact_flags,
                    true, true,
                    pass + 1U == k_soft_contact_cleanup_passes,
                    false, body.maximum_speed);
            }
            const std::uint32_t surface_blocks =
                (body.surface_vertex_count + block_size - 1U) / block_size;
            backend.update_soft_body_surface(
                surface_blocks, block_size, stream,
                body.positions, body.rest_positions,
                body.surface_rest_positions, body.surface_bindings,
                body.surface_positions, body.surface_vertex_count);
            body_status = record_timing_stage(TimingStage::soft_body_contact_cleanup);
            if (!body_status) return body_status;
        }
        const auto body_error = backend.check_error();
        if (body_error != backend.success_code()) {
            backend.wait(stream);
            return backend.failure(body_error, "soft-body kernel launch failed");
        }
        return Status{};
    };
    const auto advance_soft_cloth = [&](const RigidBodyState *previous_states)
        noexcept -> Status {
        std::uint32_t iterations = 0U;
        for (const auto &coupling : impl_->soft_cloth_couplings)
            if (coupling && coupling->alive && coupling->options.enabled)
                iterations = std::max(iterations, coupling->options.solver_iterations);
        for (const auto &body : impl_->soft_bodies) if (body && body->alive) {
            const auto clear = backend.clear(body->cloth_forces, 0,
                body->node_count * sizeof(Vec3), stream);
            if (clear != backend.success_code()) return backend.failure(clear, "soft-cloth force clear failed");
        }
        for (const auto &cloth : impl_->cloths) if (cloth && cloth->alive) {
            const auto clear = backend.clear(cloth->soft_body_forces, 0,
                cloth->vertex_count * sizeof(Vec3), stream);
            if (clear != backend.success_code()) return backend.failure(clear, "cloth-soft force clear failed");
        }
        if (iterations == 0U) return Status{};
        impl_->soft_cloth_kernels_per_substep = 0U;
        for (std::uint32_t pass = 0U; pass < iterations; ++pass) {
            for (const auto &owner : impl_->soft_cloth_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled) continue;
                auto &coupling = *owner;
                auto &body = *impl_->soft_bodies[coupling.options.soft_body.index];
                auto &cloth = *impl_->cloths[coupling.options.cloth.index];
                const auto clear = backend.clear(coupling.cloth_contact_counts, 0,
                    cloth.vertex_count * sizeof(std::uint32_t), stream);
                if (clear != backend.success_code()) return backend.failure(clear, "soft-cloth count clear failed");
                const float distance = coupling.options.contact_distance > 0.0F
                    ? coupling.options.contact_distance : body.node_radius + cloth.thickness;
                const auto blocks = (body.node_count + block_size - 1U) / block_size;
                if (cloth.surface_positions != nullptr) {
                    const auto triangles = cloth.index_count / 3U;
                    backend.update_cloth_surface(
                        (triangles + block_size - 1U) / block_size,
                        block_size, stream, cloth.positions,
                        cloth.indices,
                        cloth.surface_positions, triangles);
                    ++impl_->soft_cloth_kernels_per_substep;
                }
                backend.detect_soft_body_cloth_contacts(
                    blocks, block_size, stream,
                    body.positions, body.previous, body.velocities, body.inverse_masses,
                    body.node_count, cloth.positions, cloth.previous, cloth.velocities,
                    cloth.inverse_masses, cloth.indices, cloth.surface_positions,
                    cloth.index_count / 3U, distance, coupling.options.friction,
                    coupling.contacts, coupling.cloth_contact_counts);
                backend.apply_soft_body_cloth_contacts_to_soft_body(
                    blocks, block_size, stream,
                    body.positions, body.velocities, body.cloth_forces, body.inverse_masses,
                    body.node_count, coupling.contacts, coupling.cloth_contact_counts,
                    substep_timestep, body.maximum_speed);
                backend.apply_soft_body_cloth_contacts_to_cloth(
                    (cloth.vertex_count + block_size - 1U) / block_size,
                    block_size, stream, cloth.positions, cloth.velocities,
                    cloth.soft_body_forces, cloth.inverse_masses, cloth.vertex_count,
                    coupling.contacts, body.node_count, coupling.cloth_contact_counts,
                    substep_timestep);
                impl_->soft_cloth_kernels_per_substep += 3U;
            }
            for (std::uint32_t index = 0U; index < impl_->cloths.size(); ++index) {
                if (!coupled_cloth(index)) continue;
                auto &cloth = *impl_->cloths[index];
                // Sample once per substep, before projection erases impact strain.
                if (pass == 0U && cloth.surface_positions != nullptr) {
                    backend.break_cloth_bonds(
                        (cloth.bond_count + block_size - 1U) / block_size,
                        block_size, stream, cloth.positions, cloth.bonds,
                        cloth.bond_active, cloth.bond_damage, cloth.bond_count,
                        cloth.break_strain, cloth.fracture_persistence_substeps,
                        cloth.body_impulses, cloth.impact_break_impulse);
                    ++impl_->soft_cloth_kernels_per_substep;
                }
                if (pass + 1U < iterations) {
                    backend.project_deformable_links(
                        (cloth.vertex_count + block_size - 1U) / block_size,
                        block_size, stream, cloth.positions, cloth.scratch,
                        cloth.inverse_masses, cloth.offsets, cloth.neighbors,
                        cloth.bond_active, cloth.vertex_count, substep_timestep, 0.0F);
                    std::swap(cloth.positions, cloth.scratch);
                    ++impl_->soft_cloth_kernels_per_substep;
                    if (cloth.preserve_volume) {
                        backend.project_cloth_volume(stream,
                            cloth.positions, cloth.inverse_masses, cloth.indices,
                            cloth.vertex_count, cloth.index_count / 3U,
                            cloth.volume_corner_offsets, cloth.volume_corner_indices,
                            cloth.volume_gradients, cloth.target_volume,
                            cloth.orientation, cloth.volume_compliance,
                            substep_timestep, cloth.volume_lambda, false);
                        ++impl_->soft_cloth_kernels_per_substep;
                    }
                }
                if (cloth.surface_positions != nullptr) {
                    const auto triangles = cloth.index_count / 3U;
                    backend.limit_cloth_strain(stream, cloth.positions,
                        cloth.scratch, cloth.inverse_masses, cloth.offsets,
                        cloth.neighbors, cloth.free_triangle_nodes, cloth.vertex_count);
                    ++impl_->soft_cloth_kernels_per_substep;
                    backend.update_cloth_surface(
                        (triangles + block_size - 1U) / block_size,
                        block_size, stream, cloth.positions,
                        cloth.indices,
                        cloth.surface_positions, triangles);
                    ++impl_->soft_cloth_kernels_per_substep;
                }
            }
            if (pass + 1U < iterations)
                for (std::uint32_t index = 0U; index < impl_->soft_bodies.size(); ++index) {
                    if (!coupled_soft(index)) continue;
                    auto &body = *impl_->soft_bodies[index];
                    backend.project_deformable_links(
                        (body.node_count + block_size - 1U) / block_size,
                        block_size, stream, body.positions, body.scratch,
                        body.inverse_masses, body.offsets, body.neighbors,
                        body.bond_active, body.node_count, substep_timestep,
                        body.maximum_projection_fraction);
                    std::swap(body.positions, body.scratch);
                    ++impl_->soft_cloth_kernels_per_substep;
                }
        }
        for (std::uint32_t index = 0U; index < impl_->cloths.size(); ++index) {
            if (!coupled_cloth(index) || impl_->rigid_body_count == 0U) continue;
            auto &cloth = *impl_->cloths[index];
            backend.collide_deformable_with_rigid_geometry(
                (cloth.vertex_count + block_size - 1U) / block_size,
                block_size, stream,
                cloth.positions, cloth.velocities, cloth.previous, cloth.inverse_masses,
                cloth.vertex_count, cloth.thickness, substep_timestep,
                impl_->parameters, previous_states, impl_->states[impl_->current_state],
                impl_->meshes, impl_->rigid_body_count, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                false, true, true, false, 20.0F);
            ++impl_->soft_cloth_kernels_per_substep;
            if (cloth.surface_positions != nullptr) {
                const auto triangles = cloth.index_count / 3U;
                backend.update_cloth_surface(
                    (triangles + block_size - 1U) / block_size,
                    block_size, stream, cloth.positions,
                    cloth.indices,
                    cloth.surface_positions, triangles);
                ++impl_->soft_cloth_kernels_per_substep;
            }
        }
        for (std::uint32_t index = 0U; index < impl_->soft_bodies.size(); ++index) {
            if (!coupled_soft(index)) continue;
            auto &body = *impl_->soft_bodies[index];
            if (impl_->rigid_body_count != 0U) {
                backend.collide_deformable_with_rigid_geometry(
                    (body.node_count + block_size - 1U) / block_size,
                    block_size, stream,
                    body.positions, body.velocities, body.previous, body.inverse_masses,
                    body.node_count, body.node_radius * 0.01F, substep_timestep,
                    impl_->parameters, previous_states, impl_->states[impl_->current_state],
                    impl_->meshes, impl_->rigid_body_count, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    true, true, true, false,
                    body.maximum_speed);
                ++impl_->soft_cloth_kernels_per_substep;
            }
            backend.update_soft_body_surface(
                (body.surface_vertex_count + block_size - 1U) / block_size,
                block_size, stream, body.positions, body.rest_positions,
                body.surface_rest_positions, body.surface_bindings,
                body.surface_positions, body.surface_vertex_count);
            ++impl_->soft_cloth_kernels_per_substep;
        }
        const auto launch_error = backend.check_error();
        if (launch_error != backend.success_code())
            return backend.failure(launch_error, "soft-body cloth contact launch failed");
        return record_timing_stage(TimingStage::soft_body_cloth_contacts);
    };
    const auto advance_ropes = [&](const RigidBodyState *previous_states, bool first_substep) -> Status {
        if(first_substep)for(const auto &body:impl_->soft_bodies) {
            if(!body || !body->alive)continue;
            const auto clear=backend.clear(body->rope_forces,0,
                body->node_count*sizeof(Vec3),stream);
            if(clear!=backend.success_code())return backend.failure(clear,"soft-body rope force clear failed");
        }
        if(first_substep)for(const auto &body:impl_->cloths) {
            if(!body || !body->alive || !body->rope_forces)continue;
            const auto clear=backend.clear(body->rope_forces,0,
                body->vertex_count*sizeof(Vec3),stream);
            if(clear!=backend.success_code())return backend.failure(clear,"cloth rope force clear failed");
        }
        for (unsigned rope_index=0;rope_index<impl_->ropes.size();++rope_index) {
            const auto &rope=impl_->ropes[rope_index];
            if(!rope || !rope->alive)continue;
            typename Backend::RopeSoftTarget soft_targets[2]{};
            typename State::RopeSoftCoupling *couplings[2]{};
            unsigned coupled=0;
            for(const auto &owner:impl_->rope_soft_body_couplings) {
                if(!owner || !owner->alive || !owner->options.enabled ||
                   !(owner->options.rope==RopeId{rope_index,rope->generation}))continue;
                if(coupled>=2U)return schedule_failure(StatusCode::capacity_exceeded,
                    "a rope supports at most two soft-body contact targets");
                auto &item=*owner;
                auto &body=*impl_->soft_bodies[item.options.soft_body.index];
                const auto clear=backend.clear(item.node_impulses,0,
                    body.node_count*sizeof(Vec3),stream);
                if(clear!=backend.success_code())return backend.failure(clear,"rope soft-body impulse clear failed");
                if(first_substep) {
                    auto e=backend.clear(item.contact_count,0,sizeof(std::uint32_t),stream);
                    if(e!=backend.success_code())return backend.failure(e,"rope soft-body count clear failed");
                    e=backend.clear(item.maximum_penetration,0,sizeof(float),stream);
                    if(e!=backend.success_code())return backend.failure(e,"rope soft-body depth clear failed");
                }
                auto refit_error=backend.clear(item.ready,0,
                    item.tree_count*sizeof(std::uint32_t),stream);
                if(refit_error!=backend.success_code())
                    return backend.failure(refit_error,"rope soft-body BVH clear failed");
                backend.refit_fluid_soft_surface(
                    (item.tree_count + block_size - 1U) / block_size,
                    block_size, stream, body.surface_positions,
                    item.previous_surface,body.surface_indices,item.order,
                    item.tree,item.parents,item.ready,item.tree_count,item.bounds);
                for(unsigned end=0;end<2;++end)if(item.anchor_triangle[end]!=~0U)
                    backend.sample_rope_soft_body_anchor(stream,
                        rope->data.soft_anchor_positions,
                        rope->data.soft_anchor_velocities,
                        rope->data.soft_anchor_impulses,
                        rope->data.soft_anchor_inverse_masses,end,
                        body.surface_positions,body.velocities,body.surface_indices,
                        body.surface_bindings,item.anchor_triangle[end],
                        &item.anchor_weights[end].x,&item.anchor_offset[end].x);
                soft_targets[coupled]={body.surface_positions,item.previous_surface,
                    body.velocities,item.bounds,
                    item.tree,item.order,
                    body.surface_indices,body.surface_bindings,body.inverse_masses,
                    item.node_impulses,item.contact_count,item.maximum_penetration,
                    body.surface_index_count/3U,body.surface_vertex_count,
                    item.options.contact_distance>0?item.options.contact_distance:rope->data.options.radius,
                    item.options.friction,item.orientation,
                    item.options.attach_first,item.options.attach_last};
                couplings[coupled++]=&item;
            }
            for(const auto &item:impl_->rope_cloth_couplings) {
                if(!item.alive || !item.options.enabled ||
                   !(item.options.rope==RopeId{rope_index,rope->generation}))continue;
                auto &cloth=*impl_->cloths[item.options.cloth.index];
                for(unsigned end=0;end<2;++end) {
                    const unsigned vertex=end?item.options.last_vertex:
                        item.options.first_vertex;
                    if(vertex==UINT32_MAX)continue;
                    backend.sample_rope_cloth_anchor(stream,
                        rope->data.soft_anchor_positions,
                        rope->data.soft_anchor_velocities,
                        rope->data.soft_anchor_impulses,
                        rope->data.soft_anchor_inverse_masses,end,
                        cloth.positions,cloth.velocities,cloth.inverse_masses,
                        vertex,item.options.anchor_effective_mass);
                }
            }
            int first=-1,last=-1;unsigned dense;
            if(rope->data.options.first.enabled) {
                auto status=impl_->validate_handle(rope->data.options.first.body,dense);
                if (!status) return status;
                first = int(dense);
            }
            if(rope->data.options.last.enabled) {
                auto status=impl_->validate_handle(rope->data.options.last.body,dense);
                if (!status) return status;
                last = int(dense);
            }
            for (const auto &owner : impl_->smoke_rope_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled ||
                    !(owner->options.rope == RopeId{rope_index, rope->generation}) ||
                    owner->options.wind_drag == 0.0F) continue;
                const auto &smoke = *impl_->smokes[owner->options.smoke.index];
                if (smoke.count == 0U) continue;
                backend.apply_smoke_wind_to_rope(
                    (rope->data.count + block_size - 1U) / block_size,
                    block_size, stream, rope->data.positions,
                    rope->data.velocities, rope->data.count,
                    rope->data.options.maximum_speed,
                    first >= 0, last >= 0,
                    rope->data.soft_first, rope->data.soft_last,
                    backend.smoke_dispatch_options(smoke.options),
                    smoke.positions, smoke.velocities,
                    smoke.keys[1], smoke.indices[1],
                    backend.smoke_dispatch_grid(smoke.grid),
                    owner->options.wind_drag,
                    owner->options.maximum_wind_acceleration,
                    substep_timestep);
            }
            backend.advance_rope(stream,rope->data,substep_timestep,
                options.gravity,first,last,impl_->parameters,
                impl_->states[impl_->current_state],previous_states,
                impl_->meshes,impl_->rigid_body_count,first_substep,
                soft_targets[0],soft_targets[1]);
            auto error=backend.check_error();
            if(error!=backend.success_code())return backend.failure(error,"rope solver launch failed");
            for(const auto &item:impl_->rope_cloth_couplings) {
                if(!item.alive || !item.options.enabled ||
                   !(item.options.rope==RopeId{rope_index,rope->generation}))continue;
                auto &cloth=*impl_->cloths[item.options.cloth.index];
                for(unsigned end=0;end<2;++end) {
                    const unsigned vertex=end?item.options.last_vertex:
                        item.options.first_vertex;
                    if(vertex==UINT32_MAX)continue;
                    backend.apply_rope_cloth_anchor(stream,
                        rope->data.positions,rope->data.velocities,
                        rope->data.count,rope->data.soft_anchor_positions,
                        rope->data.soft_anchor_velocities,end,
                        cloth.positions,cloth.velocities,cloth.rope_forces,
                        cloth.inverse_masses,vertex,
                        item.options.maximum_cloth_acceleration,
                        substep_timestep,1.0F/options.timestep);
                }
                if(cloth.surface_positions) {
                    const unsigned triangles=cloth.index_count/3U;
                    backend.update_cloth_surface(
                        (triangles+block_size-1U)/block_size,
                        block_size,stream,cloth.positions,cloth.indices,
                        cloth.surface_positions,triangles);
                }
            }
            auto status=record_timing_stage(TimingStage::rope_solve);
            if(!status)return status;
            for(unsigned index=0;index<coupled;++index) {
                auto &item=*couplings[index];
                auto &body=*impl_->soft_bodies[item.options.soft_body.index];
                for(unsigned end=0;end<2;++end)if(item.anchor_triangle[end]!=~0U)
                    backend.scatter_rope_soft_body_anchor(1U,256U,stream,
                        rope->data.soft_anchor_positions,
                        rope->data.soft_anchor_impulses,end,
                        body.positions,body.node_count,
                        item.options.anchor_support_radius_scale*body.node_radius,
                        item.options.anchor_contact_support_radius_scale*
                            body.node_radius,item.contact_count,
                        body.surface_indices,body.surface_bindings,
                        item.anchor_triangle[end],&item.anchor_weights[end].x,
                        item.node_impulses);
                backend.apply_rope_soft_body_impulses(
                    (body.node_count+block_size-1U)/block_size,
                    block_size,stream,body.positions,body.velocities,
                    body.rope_forces,body.inverse_masses,item.node_impulses,
                    body.node_count,item.options.maximum_soft_body_acceleration,
                    body.maximum_speed,substep_timestep,1.0F/options.timestep);
                backend.update_soft_body_surface(
                    (body.surface_vertex_count+block_size-1U)/block_size,
                    block_size,stream,body.positions,body.rest_positions,
                    body.surface_rest_positions,body.surface_bindings,
                    body.surface_positions,body.surface_vertex_count);
                error=backend.copy(item.previous_surface,body.surface_positions,
                    body.surface_vertex_count*sizeof(Vec3),device::CopyDirection::device_to_device,stream);
                if(error!=backend.success_code())
                    return backend.failure(error,"rope soft-body skin snapshot failed");
                error=backend.check_error();
                if(error!=backend.success_code())return backend.failure(error,"rope soft-body coupling launch failed");
                status=record_timing_stage(TimingStage::rope_soft_body_contacts,
                    2U+unsigned(item.anchor_triangle[0]!=~0U)+
                    unsigned(item.anchor_triangle[1]!=~0U));
                if(!status)return status;
            }
        }
        return Status{};
    };
    for (std::uint32_t substep = 0; substep < options.substeps; ++substep) {
        if (impl_->rigid_body_count == 0U) {
            break;
        }
        const std::uint32_t output_state = 1U - impl_->current_state;
        error = backend.integrate_rigid_bodies(
            block_count, block_size, stream,
            impl_->parameters, impl_->accumulators, impl_->targets,
            impl_->states[impl_->current_state], impl_->states[output_state],
            impl_->rigid_body_count, &options.gravity.x, substep_timestep,
            options.substeps - substep, substep == 0U);
        if (error != backend.success_code()) {
            backend.wait(stream);
            return backend.failure(error, "rigid integration kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_integration);
        if (!status) {
            backend.wait(stream);
            return status;
        }
        const std::uint32_t pair_count =
            impl_->rigid_body_count * impl_->rigid_body_count;
        const std::uint32_t pair_block_count =
            (pair_count + block_size - 1U) / block_size;
        const std::uint32_t contact_block_count =
            std::min(pair_count, 128U);
        error = backend.compute_rigid_world_bounds(
            block_count, block_size, stream,
            impl_->parameters, impl_->states[impl_->current_state],
            impl_->states[output_state],
            impl_->rigid_body_count, impl_->meshes,
            impl_->rigid_world_bounds);
        if (error != backend.success_code()) {
            backend.wait(stream);
            return backend.failure(error,
                                "rigid world-bounds kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_world_bounds);
        if (!status) {
            backend.wait(stream);
            return status;
        }
        if (impl_->rigid_constraint_count != 0U) {
            error = backend.build_fixed_collision_groups(
                1U, 1U, stream,
                impl_->rigid_constraints, impl_->options.rigid_constraint_capacity,
                impl_->ids, impl_->rigid_body_count,
                impl_->fixed_contact_projection);
            if (error != backend.success_code())
                return backend.failure(error,
                    "fixed collision group launch failed");
        }
        error = backend.broad_phase_rigid_pairs(
            pair_block_count, block_size, stream,
            impl_->parameters, impl_->rigid_world_bounds,
            impl_->states[impl_->current_state], impl_->states[output_state],
            impl_->meshes, impl_->ids, impl_->rigid_constraints,
            impl_->options.rigid_constraint_capacity,
            impl_->rigid_constraint_count != 0U
                ? impl_->fixed_contact_projection : nullptr,
            impl_->rigid_body_count, impl_->rigid_active_pair_flags);
        if (error != backend.success_code()) {
            backend.wait(stream);
            return backend.failure(error,
                                "rigid broad-phase kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_pair_filter,
                                    impl_->rigid_constraint_count != 0U ? 2U : 1U);
        if (!status) {
            backend.wait(stream);
            return status;
        }
        error = backend.compact_flagged_indices(
            stream, impl_->rigid_active_pair_flags,
            static_cast<std::uint32_t>(pair_count),
            impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
            impl_->rigid_broad_phase_workspace,
            impl_->rigid_broad_phase_workspace_size);
        if (error != backend.success_code()) {
            backend.wait(stream);
            return backend.failure(error,
                                "rigid broad-phase compaction failed");
        }
        status = record_timing_stage(TimingStage::rigid_pair_compaction);
        if (!status) {
            backend.wait(stream);
            return status;
        }
        const std::uint32_t shared_leaf_capacity =
            impl_->rigid_leaf_manifolds != nullptr ? k_shared_rigid_leaf_capacity : 0U;
        constexpr std::uint32_t leaf_pair_generation_threads = 128U;
        error = backend.generate_rigid_leaf_pairs(
            contact_block_count, leaf_pair_generation_threads,
            0U, stream,
            impl_->parameters, impl_->states[impl_->current_state],
            impl_->states[output_state],
            impl_->rigid_body_count, impl_->meshes,
            impl_->options.triangle_mesh_capacity,
            impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
            impl_->rigid_leaf_pairs,
            impl_->rigid_leaf_pair_counts,
            impl_->rigid_leaf_pair_slot_capacity,
            impl_->rigid_leaf_pairs_per_slot, shared_leaf_capacity);
        if (error != backend.success_code()) {
            backend.wait(stream);
            return backend.failure(error,
                                "rigid leaf-pair kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_leaf_pair_generation);
        if (!status) {
            backend.wait(stream);
            return status;
        }
        // Many-body worlds parallelize across pairs. Few-body worlds spread
        // each dense pair across blocks and reduce in the same candidate order.
        // One shared reduction manifold accompanies the per-thread scratch.
        constexpr std::uint32_t contact_evaluation_threads = 16U;
        const std::uint32_t blocks_per_pair =
            impl_->rigid_leaf_manifolds != nullptr
                ? k_rigid_leaf_blocks_per_pair
                : 1U;
        error = backend.finalize_rigid_contact_manifolds(
            contact_block_count, block_size, stream,
            impl_->parameters, impl_->states[impl_->current_state], impl_->states[output_state],
            impl_->rigid_body_count, impl_->meshes, impl_->ids, impl_->rigid_constraints,
            impl_->rigid_constraint_count ? impl_->options.rigid_constraint_capacity : 0U,
            impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
            impl_->rigid_leaf_pair_counts, substep_timestep, impl_->rigid_manifolds, true);
        if (error != backend.success_code()) return backend.failure(
            error, "rigid face contact preparation failed");
        error = backend.evaluate_rigid_leaf_pairs(
            contact_block_count * blocks_per_pair,
            contact_evaluation_threads, 0U, stream,
                      impl_->parameters,
                      impl_->states[impl_->current_state],
                      impl_->states[output_state],
                      impl_->rigid_body_count, impl_->meshes, impl_->ids,
                      impl_->rigid_constraints,
                      impl_->rigid_constraint_count ? impl_->options.rigid_constraint_capacity : 0U,
                      impl_->rigid_active_pairs,
                      impl_->rigid_active_pair_count,
                      impl_->rigid_leaf_pairs, impl_->rigid_leaf_pair_counts,
                      impl_->rigid_leaf_pairs_per_slot,
                      impl_->rigid_leaf_manifolds, blocks_per_pair,
                      substep_timestep, impl_->rigid_manifolds);
        if (error != backend.success_code()) return backend.failure(
            error, "rigid leaf contact evaluation failed");
        if (impl_->rigid_leaf_manifolds != nullptr) {
            error = backend.reduce_rigid_leaf_manifolds(
                contact_block_count, block_size, stream,
                impl_->parameters, impl_->rigid_body_count,
                impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
                impl_->rigid_leaf_pair_counts,
                impl_->rigid_leaf_pairs_per_slot,
                impl_->rigid_leaf_manifolds, impl_->rigid_manifolds);
            if (error != backend.success_code()) return backend.failure(
                error, "rigid leaf contact reduction failed");
        }
        error = backend.finalize_rigid_contact_manifolds(
            contact_block_count, block_size, stream,
            impl_->parameters, impl_->states[impl_->current_state],
                impl_->states[output_state], impl_->rigid_body_count,
                impl_->meshes, impl_->ids, impl_->rigid_constraints,
                impl_->rigid_constraint_count ? impl_->options.rigid_constraint_capacity : 0U,
                impl_->rigid_active_pairs,
                impl_->rigid_active_pair_count,
                impl_->rigid_leaf_pair_counts, substep_timestep,
                impl_->rigid_manifolds, false);
        if (error != backend.success_code()) {
            backend.wait(stream);
            return backend.failure(error,
                                "rigid leaf contact kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_contact_evaluation);
        if (!status) {
            backend.wait(stream);
            return status;
        }
        if (impl_->rigid_contact_revision != impl_->revision) {
            impl_->rigid_contact_epoch += 2U;
            error = backend.clear(impl_->rigid_contact_cache_slots, 0xff,
                pair_count * sizeof(*impl_->rigid_contact_cache_slots), stream);
            if (error != backend.success_code())
                return backend.failure(error,
                    "rigid contact cache invalidation failed");
        }
        impl_->rigid_contact_revision = impl_->revision;
        ++impl_->rigid_contact_epoch;
        error = backend.prepare_parallel_contact_events(
            1U, 1U, stream,
            impl_->rigid_body_count, impl_->rigid_manifolds,
            impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
            impl_->ids, impl_->rigid_contact_event_offsets,
            impl_->rigid_contact_events, impl_->rigid_contact_capacity,
            impl_->rigid_contact_count,
            true, substep == 0U);
        if (error != backend.success_code())
            return backend.failure(error,
                "rigid contact event preparation launch failed");
        error = backend.load_rigid_contact_cache(
            contact_block_count, block_size, stream,
            impl_->rigid_manifolds, impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
            impl_->states[output_state], impl_->ids, impl_->rigid_body_count,
            impl_->rigid_contact_cache, impl_->rigid_contact_cache_slots,
            impl_->rigid_leaf_pair_slot_capacity, impl_->rigid_contact_epoch, substep_timestep);
        if (error != backend.success_code())
            return backend.failure(error, "rigid contact cache load launch failed");
        const unsigned avbd_grid = std::min(impl_->rigid_contact_grid_limit,
            std::max(1U, (impl_->rigid_body_count + 31U) / 32U));
        error = backend.prepare_geometry_constraints(stream,
            impl_->parameters, impl_->states[impl_->current_state], impl_->states[output_state],
            impl_->rigid_body_count, impl_->rigid_manifolds, impl_->rigid_active_pairs,
            impl_->rigid_active_pair_count, impl_->rigid_constraints,
            impl_->rigid_constraint_count ? impl_->options.rigid_constraint_capacity : 0U,
            impl_->ids, impl_->avbd_bodies, impl_->rigid_contact_schedule,
            substep_timestep, options.gravity, options.rigid_contact_iterations,
            options.rigid_contact_pass_limit, avbd_grid);
        if (error != backend.success_code())
            return backend.failure(error, "launch AVBD graph preparation");
        error = backend.solve_geometry_constraints(stream,
            impl_->parameters, impl_->states[impl_->current_state], impl_->states[output_state],
            impl_->rigid_body_count, impl_->rigid_manifolds, impl_->rigid_active_pairs,
            impl_->rigid_active_pair_count, impl_->rigid_constraints,
            impl_->rigid_constraint_count ? impl_->options.rigid_constraint_capacity : 0U,
            impl_->avbd_bodies, impl_->rigid_contact_schedule, substep_timestep,
            avbd_grid);
        if (error != backend.success_code()) return backend.failure(error, "launch AVBD vertex solve");
        error = backend.clamp_rigid_speeds(
            block_count, block_size, stream,
            impl_->parameters, impl_->states[output_state],
            impl_->rigid_body_count);
        if (error != backend.success_code())
            return backend.failure(error, "rigid speed clamp launch failed");
        error = backend.save_rigid_contact_cache(
            contact_block_count, block_size, stream,
            impl_->rigid_manifolds, impl_->rigid_active_pairs, impl_->rigid_active_pair_count,
            impl_->states[output_state], impl_->ids, impl_->rigid_body_count,
            impl_->rigid_contact_cache, impl_->rigid_contact_cache_slots,
            impl_->rigid_leaf_pair_slot_capacity, impl_->rigid_contact_epoch, substep_timestep,
            impl_->rigid_contact_event_offsets, impl_->rigid_contact_events,
            impl_->rigid_contact_capacity);
        if (error != backend.success_code()) {
            backend.wait(stream);
            return backend.failure(error,
                                "rigid contact solve kernel launch failed");
        }
        if (!collect_rigid_contacts) {
            error = backend.clear(impl_->rigid_contact_count, 0,
                                    sizeof(*impl_->rigid_contact_count), stream);
            if (error != backend.success_code())
                return backend.failure(error,
                    "rigid contact event clear failed");
        }
        status = record_timing_stage(TimingStage::rigid_contact_solve);
        if (!status) {
            backend.wait(stream);
            return status;
        }
        impl_->current_state = output_state;
        // Couple the sheet to the rigid state from this same substep.
        if (has_cloth) {
            status = advance_cloth(impl_->states[1U - impl_->current_state]);
            if (!status) return status;
        }
        if (has_rope_soft_body) {
            status = advance_ropes(impl_->states[1U - impl_->current_state], substep == 0);
            if (!status) return status;
        }
        if (has_soft_body) {
            status = advance_soft_bodies(
                impl_->states[1U - impl_->current_state]);
            if (!status) return status;
        }
        status = advance_soft_cloth(impl_->states[1U - impl_->current_state]);
        if (!status) return status;
        if (!has_rope_soft_body) {
            status = advance_ropes(impl_->states[1U - impl_->current_state], substep == 0);
            if (!status) return status;
        }
    }

    if (impl_->rigid_body_count > 0U) {
        error = backend.clear_rigid_inputs(
            block_count, block_size, stream,
            impl_->accumulators, impl_->targets, impl_->rigid_body_count);
        if (error != backend.success_code()) {
            backend.wait(stream);
            return backend.failure(error, "rigid input-clear kernel launch failed");
        }
        status = record_timing_stage(TimingStage::rigid_input_clear);
        if (!status) {
            backend.wait(stream);
            return status;
        }
    }

    if (impl_->rigid_body_count == 0U) {
        for (std::uint32_t substep = 0U; substep < options.substeps; ++substep) {
            if (has_cloth) {
                status = advance_cloth(nullptr);
                if (!status) return status;
            }
            if (has_rope_soft_body) {
                status = advance_ropes(nullptr, substep == 0);
                if (!status) return status;
            }
            if (has_soft_body) {
                status = advance_soft_bodies(nullptr);
                if (!status) return status;
            }
            status = advance_soft_cloth(nullptr);
            if (!status) return status;
            if (!has_rope_soft_body) {
                status = advance_ropes(nullptr, substep == 0);
                if (!status) return status;
            }
        }
    }

    if (impl_->rigid_body_count != 0U && impl_->fluid_count != 0U) {
        backend.calculate_fluid_body_bounds(
            block_count, block_size, stream,
            impl_->parameters, impl_->fluid_previous_states,
            impl_->states[impl_->current_state], impl_->meshes,
            impl_->rigid_body_count, impl_->fluid_body_bounds);
        error = backend.check_error();
        if (error != backend.success_code())
            return backend.failure(error, "fluid body bounds launch failed");
    }

    // Fluid storage is capacity-sized and all sort/selection workspaces were
    // reserved at creation. No CUDA allocations occur during a frame.
    for (const auto &body : impl_->soft_bodies) {
        if (!body || !body->alive) continue;
        error = backend.clear(body->fluid_forces, 0, body->node_count * sizeof(Vec3), stream);
        if (error != backend.success_code()) return backend.failure(error, "soft-body fluid force clear failed");
    }
    for (const auto &coupling : impl_->fluid_soft_couplings) {
        if (!coupling || !coupling->alive) continue;
        error = backend.clear(coupling->contact_count, 0, sizeof(std::uint32_t), stream);
        if (error == backend.success_code()) error = backend.clear(coupling->maximum_penetration, 0,
            sizeof(float), stream);
        if (error != backend.success_code()) return backend.failure(error, "fluid soft-body diagnostics clear failed");
    }
    for (const auto &coupling : impl_->fluid_rope_couplings) {
        if (!coupling || !coupling->alive) continue;
        error = backend.clear(coupling->contact_count, 0, sizeof(std::uint32_t), stream);
        if (error == backend.success_code()) error = backend.clear(coupling->maximum_penetration, 0,
            sizeof(float), stream);
        if (error != backend.success_code()) return backend.failure(error, "fluid rope diagnostics clear failed");
    }
    for (std::uint32_t fluid_index = 0U;
         fluid_index < impl_->fluids.size(); ++fluid_index) {
        if (!impl_->fluids[fluid_index] || !impl_->fluids[fluid_index]->alive)
            continue;
        auto &fluid = *impl_->fluids[fluid_index];
        const FluidId fluid_id{fluid_index, fluid.generation};
        std::uint32_t live{};
        status = backend.read_value(stream, fluid.count, live,
                                 "fluid count readback failed");
        if (!status) return status;
        const std::uint32_t first_spawned = live;
        bool spawned = false;
        const std::uint32_t blocks =
            (fluid.options.capacity + block_size - 1U) / block_size;
        float source_cell_size = fluid.options.support_radius;
        bool has_sources = false;
        for (const auto &slot : impl_->particle_sources) {
            if (!slot.alive || !slot.options.enabled || !(slot.options.fluid == fluid_id)) continue;
            has_sources = true;
            source_cell_size = std::max(source_cell_size, slot.data->spacing);
        }
        if (has_sources) {
            if (fluid.next_id > UINT32_MAX - (fluid.options.capacity - live))
                return schedule_failure(StatusCode::capacity_exceeded, "stable fluid particle ID range exhausted");
            backend.fluid_cell_keys(blocks, block_size, stream,
                fluid.positions,
                fluid.count, fluid.options.capacity, 1.0F / source_cell_size,
                fluid.keys[0],
                fluid.indices[0]);
            error = backend.radix_sort_pairs_u64_u32(
                stream, fluid.keys[0], fluid.keys[1], fluid.indices[0],
                fluid.indices[1], fluid.options.capacity,
                fluid.sort_workspace, fluid.sort_workspace_size);
            if (error != backend.success_code()) return backend.failure(error, "fluid source spatial index failed");
        }
        for (auto &slot : impl_->particle_sources) {
            if (!slot.alive || !slot.options.enabled ||
                !(slot.options.fluid == fluid_id)) continue;
            const auto &data = *slot.data;
            backend.find_fluid_source_vacancies(
                (data.count+block_size-1U)/block_size,block_size,stream,
                data.points, data.count, data.spacing, fluid.positions,
                fluid.keys[1], fluid.indices[1], fluid.options.capacity,
                1.0F/source_cell_size, data.vacant);
            backend.emit_fluid_source(stream,
                data.points, data.vacant, data.count, data.spacing,
                &slot.options.initial_velocity.x, fluid.positions, fluid.velocities,
                fluid.ids, fluid.foam, fluid.temperatures,
                slot.options.initial_temperature, fluid.count,
                fluid.options.capacity,
                first_spawned, fluid.next_id, data.capacity_misses);
        }
        if (has_sources) {
            // One readback per fluid, never one per sample. No frame allocation.
            status = backend.read_value(stream, fluid.count, live,
                                     "fluid source emission failed");
            if (!status) return status;
            const auto amount = live - first_spawned;
            spawned = amount != 0;
            fluid.next_id += amount;
            fluid.emitted_count += amount;
            impl_->emitted_particle_count += amount;
            for (const auto &slot : impl_->particle_sources) {
                if (!slot.alive || !slot.options.enabled || !(slot.options.fluid == fluid_id)) continue;
                std::uint32_t misses{};
                status = backend.read_value(stream, slot.data->capacity_misses, misses,
                    "fluid source capacity readback failed");
                if (!status) return status;
                impl_->spawn_capacity_miss_count += misses;
            }
            status = record_timing_stage(TimingStage::fluid_spawn);
            if (!status) return status;
        }
        if (live == 0U) continue;
        const std::uint32_t iterations =
            options.substeps * fluid.options.solver_iterations;
        const float dt = options.timestep / static_cast<float>(iterations);
        const typename Backend::FluidDispatchConstants fluid_constants = backend.fluid_constants(
            fluid.options, options.gravity, dt, live);
        const Vec3 fluid_up =
            backend.normalized_or(backend.multiply(options.gravity, -1.0F),
                          {0.0F, 1.0F, 0.0F});
        const float diameter = 2.0F * fluid.options.particle_radius;
        const float particle_mass = fluid.options.rest_density *
            (fluid.options.rest_particle_volume > 0.0F
                ? fluid.options.rest_particle_volume
                : diameter * diameter * diameter);
        const std::uint32_t body_words =
            (impl_->options.rigid_body_capacity + 63U) / 64U;
        if (has_rigid_bodies) {
            error = backend.clear(impl_->fluid_body_masks, 0,
                Backend::fluid_body_buckets * body_words * sizeof(unsigned long long),
                stream);
            if (error == backend.success_code())
                error = backend.clear(impl_->fluid_global_body_masks, 0,
                    body_words * sizeof(unsigned long long), stream);
            if (error != backend.success_code())
                return backend.failure(error, "fluid body index clear failed");
            backend.index_fluid_body_cells(
                block_count, block_size, stream,
                impl_->parameters, impl_->fluid_body_bounds,
                impl_->rigid_body_count, body_words,
                fluid.options.particle_radius + fluid.options.maximum_speed * dt,
                impl_->fluid_body_masks, impl_->fluid_global_body_masks);
            status = record_timing_stage(TimingStage::fluid_body_index);
            if (!status) return status;
        }
        if (collect_fluid_contacts) {
            error = backend.clear(fluid.contact_flags, 0,
                fluid.options.capacity * sizeof(std::uint8_t), stream);
            if (error != backend.success_code())
                return backend.failure(error, "fluid event flags clear failed");
        }
        for (std::uint32_t iteration = 0U; iteration < iterations;
             ++iteration) {
            backend.fluid_cell_keys(blocks, block_size, stream,
                fluid.positions,
                fluid.count, fluid.options.capacity,
                1.0F / fluid.options.support_radius,
                fluid.keys[0],
                fluid.indices[0]);
            error = backend.radix_sort_pairs_u64_u32(
                stream, fluid.keys[0], fluid.keys[1], fluid.indices[0],
                fluid.indices[1], fluid.options.capacity,
                fluid.sort_workspace, fluid.sort_workspace_size);
            if (error != backend.success_code()) {
                backend.wait(stream);
                return backend.failure(error, "fluid neighbor cell sort failed");
            }
            status = record_timing_stage(TimingStage::fluid_neighbor_sort);
            if (!status) return status;
            backend.fluid_forces(blocks, block_size, stream,
                fluid.positions, fluid.velocities, fluid.forces, fluid.foam,
                fluid.count, fluid_constants, &fluid_up.x, fluid.foam_source,
                fluid.keys[1],
                fluid.indices[1],
                impl_->fluid_neighbor_overflow,
                impl_->fluid_maximum_neighbor_count);
            status = record_timing_stage(TimingStage::fluid_neighbor_forces);
            if (!status) return status;
            for (auto &coupling :
                 impl_->fluid_cloth_couplings) {
                if (!coupling.alive || !coupling.options.enabled ||
                    !(coupling.options.fluid == fluid_id)) continue;
                if (coupling.options.cloth.index >= impl_->cloths.size())
                    continue;
                const auto &cloth_pointer =
                    impl_->cloths[coupling.options.cloth.index];
                if (!cloth_pointer || !cloth_pointer->alive ||
                    cloth_pointer->generation !=
                        coupling.options.cloth.generation)
                    continue;
                auto &cloth = *cloth_pointer;
                FluidClothCouplingOptions resolved = coupling.options;
                if (resolved.contact_distance == 0.0F)
                    resolved.contact_distance =
                        fluid.options.particle_radius + cloth.thickness;
                if (resolved.interaction_radius == 0.0F)
                    resolved.interaction_radius = fluid.options.support_radius;
                error = backend.clear(cloth.fluid_forces, 0,
                    cloth.vertex_count * sizeof(Vec3), stream);
                if (error != backend.success_code())
                    return backend.failure(error,
                        "fluid-cloth reaction clear failed");
                backend.apply_fluid_cloth_forces(
                        blocks, block_size, stream,
                        fluid.positions, fluid.velocities, fluid.count,
                        fluid.forces, fluid.foam_source, particle_mass,
                        cloth.positions, cloth.velocities, cloth.indices,
                        cloth.index_count / 3U, cloth.orientation,
                        typename Backend::FluidClothDispatchOptions{
                            resolved.contact_distance,
                            resolved.interaction_radius,
                            resolved.stiffness,
                            resolved.damping,
                            resolved.tangential_drag,
                            resolved.maximum_force},
                        cloth.fluid_forces);
                const std::uint32_t cloth_blocks =
                    (cloth.vertex_count + block_size - 1U) / block_size;
                backend.apply_fluid_forces_to_cloth(
                        cloth_blocks, block_size, stream,
                        cloth.positions, cloth.velocities,
                        cloth.inverse_masses, cloth.fluid_forces,
                        cloth.vertex_count, dt);
                backend.project_deformable_links(
                    cloth_blocks, block_size, stream,
                    cloth.positions, cloth.scratch, cloth.inverse_masses,
                    cloth.offsets, cloth.neighbors, cloth.bond_active,
                    cloth.vertex_count, dt, 0.0F);
                std::swap(cloth.positions, cloth.scratch);
                if (cloth.preserve_volume) {
                    backend.project_cloth_volume(stream,
                        cloth.positions, cloth.inverse_masses, cloth.indices,
                        cloth.vertex_count, cloth.index_count / 3U,
                        cloth.volume_corner_offsets, cloth.volume_corner_indices,
                        cloth.volume_gradients, cloth.target_volume,
                        cloth.orientation, cloth.volume_compliance, dt,
                        cloth.volume_lambda, true);
                }
                status = record_timing_stage(
                    TimingStage::fluid_cloth_contacts);
                if (!status) return status;
            }
            backend.integrate_fluid(blocks, block_size, stream,
                fluid.positions, fluid.velocities, fluid.forces,
                fluid.foam, fluid.count, fluid_constants,
                fluid.previous,
                fluid.foam_source);
            status = record_timing_stage(TimingStage::fluid_integration);
            if (!status) return status;
            for (const auto &owner : impl_->fluid_rope_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled ||
                    !(owner->options.fluid == fluid_id)) continue;
                auto &rope = impl_->ropes[owner->options.rope.index]->data;
                const float distance = owner->options.contact_distance > 0.0F
                    ? owner->options.contact_distance
                    : fluid.options.particle_radius + rope.options.radius;
                error = backend.clear(owner->node_impulses, 0,
                    rope.count * sizeof(Vec3), stream);
                if (error != backend.success_code()) return backend.failure(error, "fluid rope impulse clear failed");
                backend.detect_fluid_rope_contacts(
                    blocks, block_size, stream,
                    fluid.positions, fluid.velocities, fluid.count, particle_mass,
                    rope.positions, rope.velocities, rope.count, distance,
                    owner->options.friction, owner->node_impulses,
                    owner->contact_count, owner->maximum_penetration, true);
                backend.apply_fluid_rope_impulses(
                    (rope.count + block_size - 1U) / block_size,
                    block_size, stream, rope.velocities,
                    rope.fluid_contact_forces, rope.count, rope.options.mass,
                    rope.options.maximum_speed, rope.options.first.enabled,
                    rope.options.last.enabled,
                    owner->node_impulses, owner->options.maximum_rope_acceleration,
                    dt, 1.0F / options.timestep);
                status = record_timing_stage(TimingStage::fluid_rope_contacts, 2U);
                if (!status) return status;
            }
            for (const auto &owner : impl_->fluid_soft_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled ||
                    !(owner->options.fluid == fluid_id)) continue;
                auto &coupling = *owner;
                auto &body = *impl_->soft_bodies[coupling.options.soft_body.index];
                const auto node_blocks = (body.node_count + block_size - 1U) / block_size;
                const auto skin_blocks = (body.surface_vertex_count + block_size - 1U) / block_size;
                const float distance = coupling.options.contact_distance > 0
                    ? coupling.options.contact_distance : fluid.options.particle_radius;
                const auto refit_surface = [&]() -> Status {
                    const auto clear = backend.clear(coupling.ready, 0,
                        coupling.tree_count * sizeof(std::uint32_t), stream);
                    if (clear != backend.success_code()) return backend.failure(clear, "soft surface BVH clear failed");
                    backend.refit_fluid_soft_surface(
                        (coupling.tree_count + block_size - 1U) / block_size,
                        block_size, stream, body.surface_positions,
                        coupling.previous_surface,
                        body.surface_indices, coupling.triangle_order, coupling.tree, coupling.parents,
                        coupling.ready, coupling.tree_count, coupling.bounds);
                    return Status{};
                };
                for (std::uint32_t pass = 0; pass < coupling.options.solver_iterations; ++pass) {
                    error = backend.clear(coupling.counts, 0,
                        body.node_count * sizeof(std::uint32_t), stream);
                    if (error == backend.success_code()) error = backend.clear(coupling.position_deltas, 0,
                        body.node_count * sizeof(Vec3), stream);
                    if (error == backend.success_code()) error = backend.clear(coupling.impulses, 0,
                        body.node_count * sizeof(Vec3), stream);
                    if (error != backend.success_code()) return backend.failure(error, "fluid soft-body scratch clear failed");
                    status = refit_surface();
                    if (!status) return status;
                    backend.detect_fluid_soft_contacts(
                        blocks, block_size, stream,
                        fluid.positions, fluid.previous, fluid.velocities, fluid.count,
                        body.surface_positions, coupling.previous_surface, body.surface_indices,
                        body.surface_index_count, body.surface_bindings, body.velocities,
                        coupling.orientation, distance, coupling.bounds, pass == 0,
                        coupling.tree, coupling.triangle_order,
                        coupling.contacts, coupling.counts, coupling.contact_count,
                        coupling.maximum_penetration);
                    backend.solve_fluid_soft_contacts(
                        blocks, block_size, stream,
                        fluid.positions, fluid.velocities, fluid.forces, fluid.foam, fluid.count,
                        coupling.contacts, coupling.counts, body.inverse_masses,
                        body.velocities, body.maximum_speed,
                        1.0F / particle_mass, coupling.options.friction,
                        0.2F * body.node_radius, dt, coupling.position_deltas, coupling.impulses);
                    backend.apply_fluid_soft_contacts(
                        node_blocks, block_size, stream,
                        body.positions, body.velocities, body.inverse_masses,
                        coupling.position_deltas, coupling.impulses, body.fluid_forces,
                        body.node_count, 1.0F / options.timestep);
                    backend.project_deformable_links(
                        node_blocks, block_size, stream,
                        body.positions, body.scratch, body.inverse_masses, body.offsets,
                        body.neighbors, body.bond_active, body.node_count, dt,
                        body.maximum_projection_fraction);
                    std::swap(body.positions, body.scratch);
                    // Keep fluid reaction corrections on the valid side of
                    // passive/dynamic rigid geometry before updating the skin.
                    if (impl_->rigid_body_count != 0U)
                        backend.collide_deformable_with_rigid_geometry(
                            node_blocks, block_size, stream,
                            body.positions, body.velocities, body.previous, body.inverse_masses,
                            body.node_count, body.node_radius * 0.01F, dt, impl_->parameters,
                            nullptr, impl_->states[impl_->current_state], impl_->meshes,
                            impl_->rigid_body_count, body.body_impulses, body.rigid_contact_forces,
                            body.contact_normals, body.contact_arms, body.contact_momentum_delta,
                            body.contact_normal_delta, body.contact_count, body.dynamic_contact_flag,
                            impl_->fluid_body_contact_flags,
                            true, true, false, false, body.maximum_speed);
                    backend.update_soft_body_surface(
                        skin_blocks, block_size, stream,
                        body.positions, body.rest_positions, body.surface_rest_positions,
                        body.surface_bindings, body.surface_positions, body.surface_vertex_count);
                }
                status = refit_surface();
                if (!status) return status;
                backend.recover_fluid_soft_contacts(
                    blocks, block_size, stream,
                    fluid.positions, fluid.previous, fluid.count, body.surface_positions,
                    coupling.previous_surface, body.surface_indices, body.surface_index_count,
                    coupling.orientation, distance, coupling.bounds, true,
                    coupling.tree, coupling.triangle_order);
                error = backend.copy(coupling.previous_surface, body.surface_positions,
                    body.surface_vertex_count * sizeof(Vec3), device::CopyDirection::device_to_device, stream);
                if (error != backend.success_code()) return backend.failure(error, "fluid soft-body skin copy failed");
                status = record_timing_stage(TimingStage::fluid_soft_body_contacts,
                    (6U + (impl_->rigid_body_count != 0U ? 1U : 0U)) *
                    coupling.options.solver_iterations + 2U);
                if (!status) return status;
            }
            for (const auto &coupling :
                 impl_->fluid_cloth_couplings) {
                if (!coupling.alive || !coupling.options.enabled ||
                    !(coupling.options.fluid == fluid_id) ||
                    coupling.options.cloth.index >= impl_->cloths.size())
                    continue;
                const auto &cloth_pointer =
                    impl_->cloths[coupling.options.cloth.index];
                if (!cloth_pointer || !cloth_pointer->alive ||
                    cloth_pointer->generation !=
                        coupling.options.cloth.generation)
                    continue;
                const auto &cloth = *cloth_pointer;
                const float contact_distance =
                    coupling.options.contact_distance > 0.0F
                        ? coupling.options.contact_distance
                        : fluid.options.particle_radius + cloth.thickness;
                backend.project_fluid_inside_cloth(
                    blocks, block_size, stream,
                    fluid.positions, fluid.velocities, fluid.count,
                    contact_distance, cloth.positions, cloth.velocities,
                    cloth.indices, cloth.index_count / 3U,
                    cloth.orientation);
            }
            if (has_rigid_bodies) {
                error = backend.clear(impl_->fluid_body_contact_flags, 0,
                    impl_->rigid_body_count * sizeof(std::uint32_t), stream);
                if (error != backend.success_code())
                    return backend.failure(error, "fluid contact flags clear failed");
                backend.detect_fluid_moving_contacts(
                    blocks, block_size, stream,
                    fluid.positions, fluid.previous, fluid.count,
                    fluid.options.particle_radius,
                    impl_->parameters,
                    impl_->fluid_previous_states,
                    impl_->states[impl_->current_state], impl_->meshes,
                    impl_->fluid_body_bounds, impl_->rigid_body_count,
                    impl_->fluid_body_masks, impl_->fluid_global_body_masks,
                    body_words, iteration == 0U, fluid.moving_contacts,
                    impl_->fluid_body_contact_flags);
                backend.resolve_fluid_moving_contacts(
                    blocks, block_size, stream,
                    fluid.positions, fluid.velocities, fluid.foam,
                    fluid.count, fluid.options.particle_radius,
                    particle_mass, dt, fluid.options.maximum_speed,
                    impl_->parameters, impl_->states[impl_->current_state],
                    impl_->meshes, fluid.moving_contacts,
                    impl_->fluid_body_contact_flags, fluid.body_impulses,
                    collect_fluid_contacts, fluid.contact_samples,
                    fluid.contact_flags, fluid_id.index, fluid_id.generation,
                    impl_->ids,
                    impl_->paint_rule_count != 0U,
                    impl_->paint_fields, impl_->options.paint_field_capacity,
                    impl_->paint_rules, impl_->options.paint_rule_capacity);
                backend.reduce_fluid_body_impulses(
                    impl_->rigid_body_count, block_size, stream,
                    fluid.body_impulses, fluid.count, impl_->parameters,
                    impl_->states[impl_->current_state],
                    impl_->fluid_body_contact_flags,
                    impl_->rigid_body_count);
                status = record_timing_stage(TimingStage::fluid_moving_contacts, 3U);
                if (!status) return status;
            }
            const Vec3 static_contact_up = backend.normalized_or(
                backend.multiply(options.gravity, -1.0F), {0.0F, 1.0F, 0.0F});
            for (std::uint32_t body = 0U; body < impl_->rigid_body_count;
                 ++body) {
                backend.collide_fluid_with_static_body(
                    blocks, block_size, stream,
                    fluid.positions, fluid.velocities, fluid.previous,
                    fluid.foam, fluid.count, fluid.options.particle_radius,
                    fluid.options.support_radius, first_spawned,
                    spawned && iteration == 0U, &static_contact_up,
                    impl_->parameters, impl_->states[impl_->current_state],
                    impl_->meshes, impl_->ids, body, particle_mass,
                    collect_fluid_contacts, fluid.contact_samples,
                    fluid.contact_flags, fluid_id.index, fluid_id.generation,
                    impl_->paint_rule_count != 0U,
                    impl_->paint_fields, impl_->options.paint_field_capacity,
                    impl_->paint_rules, impl_->options.paint_rule_capacity);
            }
            if (has_rigid_bodies) {
                status = record_timing_stage(TimingStage::fluid_static_contacts);
                if (!status) return status;
            }
            // A rigid boundary can push water back into a neighboring soft
            // face. Finish with current-skin recovery, not the old surface or
            // an infinite triangle plane. No extra kinetic impulse is added.
            for (const auto &coupling : impl_->fluid_soft_couplings) {
                if (!coupling || !coupling->alive || !coupling->options.enabled ||
                    !(coupling->options.fluid == fluid_id)) continue;
                const auto &body = *impl_->soft_bodies[coupling->options.soft_body.index];
                const float distance = coupling->options.contact_distance > 0
                    ? coupling->options.contact_distance : fluid.options.particle_radius;
                for (std::uint32_t recovery = 0; recovery < 3; ++recovery)
                    backend.recover_fluid_soft_contacts(
                        blocks, block_size, stream,
                        fluid.positions, fluid.positions, fluid.count, body.surface_positions,
                        body.surface_positions, body.surface_indices, body.surface_index_count,
                        coupling->orientation, distance, coupling->bounds, false,
                        coupling->tree, coupling->triangle_order);
                status = record_timing_stage(TimingStage::fluid_soft_body_contacts, 3U);
                if (!status) return status;
            }
            bool any_destroy_plane = false;
            for (const auto &slot : impl_->destroy_planes) {
                if (!slot.alive || !slot.options.enabled ||
                    !(slot.options.fluid == fluid_id)) continue;
                backend.mark_fluid_destroy_flags(
                    blocks, block_size, stream,
                    fluid.positions, fluid.previous, fluid.count,
                    fluid.options.capacity, &slot.options,
                    any_destroy_plane, fluid.keep);
                any_destroy_plane = true;
            }
            if (any_destroy_plane) {
                error = backend.compact_flagged_indices(
                    stream, fluid.keep, fluid.options.capacity, fluid.selected,
                    fluid.count, fluid.select_workspace,
                    fluid.select_workspace_size);
                if (error != backend.success_code()) {
                    backend.wait(stream);
                    return backend.failure(error, "fluid particle compaction failed");
                }
                backend.gather_fluid(blocks, block_size, stream,
                    fluid.positions, fluid.velocities, fluid.ids, fluid.foam,
                    fluid.temperatures,
                    fluid.contact_samples, fluid.contact_flags,
                    collect_fluid_contacts,
                    fluid.selected, fluid.count, fluid.options.capacity,
                    fluid.next_positions, fluid.next_velocities,
                    fluid.next_ids, fluid.next_foam,
                    fluid.next_temperatures,
                    fluid.next_contact_samples, fluid.next_contact_flags);
                std::swap(fluid.positions, fluid.next_positions);
                std::swap(fluid.velocities, fluid.next_velocities);
                std::swap(fluid.ids, fluid.next_ids);
                std::swap(fluid.foam, fluid.next_foam);
                std::swap(fluid.temperatures, fluid.next_temperatures);
                std::swap(fluid.contact_samples, fluid.next_contact_samples);
                std::swap(fluid.contact_flags, fluid.next_contact_flags);
                status = record_timing_stage(
                    TimingStage::fluid_outflow_compaction);
                if (!status) return status;
            }
        }
        if (collect_fluid_contacts) {
            error = backend.compact_flagged_indices(
                stream, fluid.contact_flags, fluid.options.capacity, fluid.selected,
                fluid.contact_count, fluid.select_workspace,
                fluid.select_workspace_size);
            if (error != backend.success_code())
                return backend.failure(error, "fluid contact selection failed");
            backend.reserve_fluid_contact_events(stream,
                fluid.contact_count, fluid.contact_offset,
                impl_->fluid_contact_count, impl_->fluid_contact_overflow,
                impl_->options.contact_capacity);
            backend.gather_fluid_contact_events(
                blocks, block_size, stream,
                fluid.selected, fluid.contact_count, fluid.contact_offset,
                fluid.contact_samples, fluid.ids, impl_->ids, &fluid_id,
                impl_->fluid_contact_events, impl_->options.contact_capacity);
            status = record_timing_stage(TimingStage::fluid_contact_events);
            if (!status) return status;
        }
        error = backend.check_error();
        if (error != backend.success_code()) {
            backend.wait(stream);
            return backend.failure(error, "fluid kernel launch failed");
        }
    }
    for (auto &owner : impl_->fluid_smoke_couplings) {
        if (!owner || !owner->alive) continue;
        auto &coupling = *owner;
        auto &fluid = *impl_->fluids[coupling.options.fluid.index];
        auto &smoke = *impl_->smokes[coupling.options.smoke.index];
        std::uint32_t live{};
        status = backend.read_value(stream, fluid.count, live,
                                 "fluid-smoke count readback failed");
        if (!status) return status;
        if (live == 0U) continue;
        if (smoke.grid.resolution == 0U && smoke.count != 0U &&
            smoke.index_dirty) {
            backend.index_smoke_particles(
                (smoke.options.capacity + block_size - 1U) / block_size,
                block_size, stream,
                smoke.positions, smoke.ages, smoke.count,
                backend.smoke_domain_options(smoke.options),
                smoke.keys[0], smoke.indices[0]);
            error = backend.radix_sort_pairs_u64_u32(
                stream, smoke.keys[0], smoke.keys[1], smoke.indices[0],
                smoke.indices[1], smoke.options.capacity,
                smoke.sort_workspace, smoke.sort_workspace_size);
            if (error != backend.success_code())
                return backend.failure(error, "smoke phase-transfer index failed");
            smoke.index_dirty = false;
        }
        error = backend.clear(coupling.converted, 0, sizeof(std::uint32_t), stream);
        if (error != backend.success_code())
            return backend.failure(error, "phase transfer counter clear failed");
        const auto blocks = (fluid.options.capacity + block_size - 1U) / block_size;
        typename Backend::FluidSmokeDispatchOptions dispatch{};
        dispatch.smoke = backend.smoke_dispatch_options(smoke.options);
        dispatch.heater_center[0] = coupling.options.heater.center.x;
        dispatch.heater_center[1] = coupling.options.heater.center.y;
        dispatch.heater_center[2] = coupling.options.heater.center.z;
        dispatch.heater_orientation[0] = coupling.options.heater.orientation.x;
        dispatch.heater_orientation[1] = coupling.options.heater.orientation.y;
        dispatch.heater_orientation[2] = coupling.options.heater.orientation.z;
        dispatch.heater_orientation[3] = coupling.options.heater.orientation.w;
        dispatch.heater_half_extents[0] =
            coupling.options.heater.half_extents.x;
        dispatch.heater_half_extents[1] =
            coupling.options.heater.half_extents.y;
        dispatch.heater_temperature = coupling.options.heater_temperature;
        dispatch.boiling_temperature = coupling.options.boiling_temperature;
        dispatch.heat_transfer_rate = coupling.options.heat_transfer_rate;
        dispatch.wind_drag = coupling.options.wind_drag;
        dispatch.steam_rise_speed = coupling.options.steam_rise_speed;
        if (smoke.count != 0U && coupling.options.wind_drag > 0.0F) {
            if (smoke.grid.resolution != 0U) {
                const auto grid = backend.smoke_dispatch_grid(smoke.grid);
                backend.drag_fluid_with_smoke_grid(
                    blocks, block_size, stream,
                    fluid.positions, fluid.velocities, fluid.count,
                    dispatch, grid, options.timestep);
            } else {
                backend.drag_fluid_with_smoke_particles(
                    blocks, block_size, stream,
                    fluid.positions, fluid.velocities, fluid.count,
                    dispatch, smoke.positions,
                    smoke.velocities, smoke.keys[1], smoke.indices[1],
                    options.timestep);
            }
        }
        const float gravity[3]{
            options.gravity.x, options.gravity.y, options.gravity.z};
        backend.exchange_fluid_with_smoke(
            blocks, block_size, stream,
            fluid.positions, fluid.velocities, fluid.temperatures,
            fluid.count, fluid.options.particle_radius, dispatch,
            smoke.positions, smoke.velocities,
            smoke.ages, smoke.thermal_lift,
            smoke.number_densities, smoke.pressures, smoke.vorticities,
            smoke.next_slot,
            coupling.converted, fluid.keep, options.timestep,
            gravity);
        error = backend.take_error();
        if (error == backend.success_code()) error = backend.wait(stream);
        if (error != backend.success_code())
            return backend.failure(error, "fluid smoke exchange failed");
        std::uint32_t converted_count{};
        status = backend.read_value(stream, coupling.converted, converted_count,
            "phase transfer count readback failed");
        if (!status) return status;
        const auto converted = std::min(converted_count, smoke.options.capacity);
        if (converted == 0U) {
            status = record_timing_stage(TimingStage::fluid_smoke_exchange);
            if (!status) return status;
            continue;
        }
        smoke.next_slot = (smoke.next_slot + converted) % smoke.options.capacity;
        smoke.count = std::min(smoke.options.capacity, smoke.count + converted);
        smoke.emitted += converted;
        smoke.index_dirty = true;
        impl_->boiled_particle_count += converted;
        backend.clear_inactive_fluid_keep(
            blocks, block_size, stream,
            fluid.keep, fluid.count, fluid.options.capacity);
        error = backend.compact_flagged_indices(
            stream, fluid.keep, fluid.options.capacity, fluid.selected, fluid.count,
            fluid.select_workspace, fluid.select_workspace_size);
        if (error != backend.success_code())
            return backend.failure(error, "boiled fluid compaction failed");
        backend.gather_fluid(blocks, block_size, stream,
            fluid.positions, fluid.velocities, fluid.ids, fluid.foam,
            fluid.temperatures, fluid.contact_samples, fluid.contact_flags,
            false, fluid.selected, fluid.count, fluid.options.capacity,
            fluid.next_positions, fluid.next_velocities, fluid.next_ids,
            fluid.next_foam, fluid.next_temperatures,
            fluid.next_contact_samples, fluid.next_contact_flags);
        std::swap(fluid.positions, fluid.next_positions);
        std::swap(fluid.velocities, fluid.next_velocities);
        std::swap(fluid.ids, fluid.next_ids);
        std::swap(fluid.foam, fluid.next_foam);
        std::swap(fluid.temperatures, fluid.next_temperatures);
        status = record_timing_stage(TimingStage::fluid_smoke_exchange, 3U);
        if (!status) return status;
    }
    for (auto &owner : impl_->smokes) {
        if (!owner || !owner->alive) continue;
        auto &smoke = *owner;
        if (smoke.count != 0U) {
            const auto blocks = (smoke.count + 127U) / 128U;
            if (smoke.grid.resolution != 0U) {
                const float smoke_gravity[3]{
                    options.gravity.x, options.gravity.y, options.gravity.z};
                backend.trace_smoke_grid_particles(
                    blocks, 128U, stream,
                    smoke.positions, smoke.previous_positions,
                    smoke.velocities, smoke.ages, smoke.thermal_lift,
                    smoke.number_densities, smoke.pressures,
                    smoke.vorticities, smoke.count,
                    backend.smoke_domain_options(smoke.options),
                    backend.smoke_dispatch_grid(smoke.grid),
                    options.timestep, smoke_gravity);
            } else {
                backend.index_smoke_particles(
                    (smoke.options.capacity + 127U) / 128U, 128U, stream,
                    smoke.positions, smoke.ages, smoke.count,
                    backend.smoke_domain_options(smoke.options), smoke.keys[0],
                    smoke.indices[0]);
                error = backend.radix_sort_pairs_u64_u32(
                    stream, smoke.keys[0], smoke.keys[1], smoke.indices[0],
                    smoke.indices[1], smoke.options.capacity,
                    smoke.sort_workspace, smoke.sort_workspace_size);
                if (error != backend.success_code())
                    return backend.failure(error,
                        "smoke pressure neighbor sort failed");
                smoke.index_dirty = false;
                backend.calculate_smoke_density_pressure(
                    blocks, 128U, stream,
                    smoke.positions, smoke.ages, smoke.count,
                    backend.smoke_domain_options(smoke.options),
                    smoke.keys[1], smoke.indices[1], smoke.number_densities,
                    smoke.pressures, options.timestep);
                backend.calculate_smoke_vorticity(
                    blocks, 128U, stream,
                    smoke.positions, smoke.velocities, smoke.ages, smoke.count,
                    backend.smoke_domain_options(smoke.options), smoke.keys[1],
                    smoke.indices[1],
                    smoke.number_densities, smoke.vorticities,
                    smoke.vorticity_magnitudes);
                backend.calculate_smoke_pair_forces(
                    blocks, 128U, stream,
                    smoke.positions, smoke.velocities, smoke.ages, smoke.count,
                    backend.smoke_domain_options(smoke.options), smoke.keys[1],
                    smoke.indices[1],
                    smoke.number_densities, smoke.pressures,
                    smoke.vorticities, smoke.vorticity_magnitudes,
                    smoke.accelerations);
                backend.advect_smoke_particles(
                    blocks, 128U, stream,
                    smoke.positions, smoke.previous_positions,
                    smoke.velocities, smoke.ages,
                    smoke.thermal_lift, smoke.count,
                    backend.smoke_domain_options(smoke.options), smoke.accelerations,
                    options.timestep, &options.gravity.x);
            }
            for (const auto &coupling : impl_->smoke_soft_body_couplings) {
                if (!coupling || !coupling->alive || !coupling->options.enabled ||
                    coupling->options.smoke.index >= impl_->smokes.size() ||
                    impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                    continue;
                const auto &body = *impl_->soft_bodies[
                    coupling->options.soft_body.index];
                const float clearance = coupling->options.contact_distance > 0.0F
                    ? coupling->options.contact_distance
                    : smoke.options.particle_radius + body.node_radius;
                    backend.calculate_smoke_deformable_bounds(
                        1U, 1U, stream,
                        body.surface_positions, body.surface_vertex_count,
                        coupling->minimum, coupling->maximum);
                    backend.collide_smoke_with_soft_body(
                        (smoke.count + 127U) / 128U, 128U, stream,
                        smoke.positions, smoke.previous_positions,
                        smoke.velocities, smoke.ages,
                        smoke.count, backend.smoke_dispatch_options(smoke.options),
                        body.surface_positions, body.surface_bindings,
                        body.surface_indices, body.surface_index_count,
                        body.velocities, coupling->minimum,
                        coupling->maximum, clearance,
                        backend.smoke_dispatch_grid(smoke.grid));
            }
            for (const auto &coupling : impl_->smoke_cloth_couplings) {
                if (!coupling || !coupling->alive || !coupling->options.enabled ||
                    coupling->options.smoke.index >= impl_->smokes.size() ||
                    impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                    continue;
                const auto &cloth = *impl_->cloths[coupling->options.cloth.index];
                const float clearance = coupling->options.contact_distance > 0.0F
                    ? coupling->options.contact_distance
                    : smoke.options.particle_radius + cloth.thickness;
                backend.calculate_smoke_deformable_bounds(
                    1U, 1U, stream,
                    cloth.positions, cloth.vertex_count,
                    coupling->minimum, coupling->maximum);
                backend.collide_smoke_with_cloth(
                    (smoke.count + 127U) / 128U, 128U, stream,
                    smoke.positions, smoke.previous_positions,
                    smoke.velocities, smoke.ages, smoke.pressures,
                    smoke.count, backend.smoke_dispatch_options(smoke.options),
                    cloth.positions, cloth.velocities,
                    cloth.indices, cloth.index_count,
                    coupling->minimum, coupling->maximum, clearance,
                    backend.smoke_dispatch_grid(smoke.grid));
            }
            for (const auto &coupling : impl_->smoke_rigid_couplings) {
                if (!coupling || !coupling->alive || !coupling->options.enabled ||
                    coupling->options.smoke.index >= impl_->smokes.size() ||
                    impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                    continue;
                std::uint32_t body = 0U;
                status = impl_->validate_handle(coupling->options.body, body);
                if (!status) return status;
                BodyParameters parameters{};
                status = backend.read_value(stream, impl_->parameters + body,
                    parameters, "failed to read smoke rigid-body parameters");
                if (!status) return status;
                const auto *mesh = impl_->meshes + parameters.mesh.index;
                TriangleMeshResource mesh_metadata{};
                status = backend.read_value(stream, mesh, mesh_metadata,
                    "failed to read smoke rigid-body mesh metadata");
                if (!status) return status;
                const float clearance = coupling->options.contact_distance > 0.0F
                    ? coupling->options.contact_distance
                    : smoke.options.particle_radius;
                if (smoke.grid.resolution != 0U &&
                    coupling->options.air_density > 0.0F)
                    backend.apply_smoke_grid_force_to_rigid_body(
                        1U, 128U, stream, &mesh_metadata,
                        body, impl_->parameters,
                        impl_->states[impl_->current_state],
                        backend.smoke_dispatch_grid(smoke.grid),
                        coupling->options.air_density,
                        coupling->options.drag_coefficient,
                        smoke.options.rest_number_density,
                        smoke.options.grid_kinematic_viscosity,
                        smoke.options.grid_les_coefficient,
                        options.timestep);
                if (coupling->options.tracer_contact)
                    backend.collide_smoke_with_rigid_body(
                        (smoke.count + 127U) / 128U, 128U, stream,
                        smoke.positions, smoke.previous_positions,
                        smoke.velocities, smoke.ages, smoke.pressures, smoke.count,
                        smoke.options.lifetime, &mesh_metadata,
                        impl_->states[impl_->current_state],
                        has_rigid_bodies ? impl_->fluid_previous_states
                                        : impl_->states[impl_->current_state],
                        body, clearance,
                        (smoke.grid.resolution == 0U ?
                         coupling->options.air_density : 0.0F) *
                            std::pow(2.0F * smoke.options.particle_radius, 3.0F),
                        coupling->options.drag_coefficient,
                        smoke.options.pressure_stiffness, options.timestep,
                        smoke.options.maximum_speed,
                        backend.smoke_dispatch_grid(smoke.grid),
                        smoke.rigid_impulses);
                if (smoke.grid.resolution == 0U &&
                    coupling->options.air_density > 0.0F &&
                    coupling->options.tracer_contact)
                    backend.apply_smoke_rigid_body_impulses(
                        1U, 128U, stream,
                        smoke.rigid_impulses, smoke.count, body,
                        impl_->parameters,
                        impl_->states[impl_->current_state]);
            }
            for (const auto &coupling : impl_->smoke_rope_couplings) {
                if (!coupling || !coupling->alive || !coupling->options.enabled ||
                    coupling->options.smoke.index >= impl_->smokes.size() ||
                    impl_->smokes[coupling->options.smoke.index].get() != &smoke)
                    continue;
                const auto &rope = impl_->ropes[coupling->options.rope.index]->data;
                const float clearance = coupling->options.contact_distance > 0.0F
                    ? coupling->options.contact_distance
                    : smoke.options.particle_radius + rope.options.radius;
                backend.calculate_smoke_deformable_bounds(
                    1U, 1U, stream,
                    rope.positions, rope.count,
                    coupling->minimum, coupling->maximum);
                backend.collide_smoke_with_rope(
                    (smoke.count + 127U) / 128U, 128U, stream,
                    smoke.positions, smoke.previous_positions,
                    smoke.velocities, smoke.ages,
                    smoke.count, backend.smoke_dispatch_options(smoke.options),
                    rope.positions, rope.velocities, rope.count,
                    coupling->minimum, coupling->maximum, clearance);
            }
            status = record_timing_stage(TimingStage::smoke_advection, 1U);
            if (!status) return status;
        }
        const double exact = double(smoke.emission_fraction) +
            double(smoke.options.particles_per_second) * options.timestep;
        const auto requested = static_cast<std::uint32_t>(std::min(
            std::floor(exact), double(smoke.options.capacity)));
        smoke.emission_fraction = float(exact - std::floor(exact));
        if (requested != 0U) {
            backend.emit_smoke_particles(
                (requested + 127U) / 128U, 128U, stream,
                smoke.positions, smoke.previous_positions,
                smoke.velocities, smoke.ages,
                smoke.thermal_lift, smoke.number_densities,
                smoke.pressures, smoke.vorticities,
                backend.smoke_domain_options(smoke.options),
                smoke.next_slot, requested, smoke.emitted);
            smoke.next_slot = (smoke.next_slot + requested) % smoke.options.capacity;
            smoke.count = std::min(smoke.options.capacity, smoke.count + requested);
            smoke.emitted += requested;
            status = record_timing_stage(TimingStage::smoke_emission, 1U);
            if (!status) return status;
        }
        smoke.time = float(std::fmod(
            double(smoke.time) + double(options.timestep), 1000.0));
        error = backend.check_error();
        if (error != backend.success_code()) return backend.failure(error, "smoke kernel launch failed");
    }
    if (options.collect_kernel_timings && timing_boundary == 1U) {
        error = backend.record_event(impl_->timing_events[timing_boundary++], stream);
        if (error != backend.success_code())
            return backend.failure(error, "failed to finish empty kernel timing");
    }
    ++impl_->frame_index;
    ++impl_->revision;
    impl_->rigid_contact_revision = impl_->revision;
    if (options.collect_kernel_timings) {
        impl_->timing_available = true;
        impl_->timing_boundary_count = timing_boundary;
        impl_->timing_frame_index = impl_->frame_index;
    }
    return {};
}

} // namespace parallel_mater::simulation

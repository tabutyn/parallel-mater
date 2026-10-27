// SPDX-License-Identifier: MIT
// Included after runtime resource helpers; private to the native adapter.
namespace {
struct NativeScheduleBackend {
    device::ExecutionQueue execution;
    explicit NativeScheduleBackend(device::ExecutionQueue value) : execution(value) {}
#if defined(__CUDACC__) || defined(PARALLEL_MATER_METAL_DRIVER)
    static constexpr bool supports_masked_subgroup_barriers = true;
#else
    static constexpr bool supports_masked_subgroup_barriers = false;
#endif
    static constexpr bool geometry_constraints_inline_speed_clamp(
        std::uint32_t body_count, std::uint32_t constraint_capacity,
        std::uint32_t grid_blocks) noexcept {
#if defined(PARALLEL_MATER_VULKAN_DRIVER)
        return constraint_capacity == 0U && grid_blocks == 1U &&
            body_count <= k_small_rigid_leaf_body_capacity &&
            body_count != 4U;
#else
        (void)body_count;
        (void)constraint_capacity;
        (void)grid_blocks;
        return false;
#endif
    }
    static constexpr std::uint64_t realtime_geometry_prepare_workgroups(
        std::uint32_t grid_blocks) noexcept {
#if defined(PARALLEL_MATER_METAL_DRIVER)
        (void)grid_blocks;
        return 1U;
#else
        return grid_blocks;
#endif
    }
    static constexpr std::uint64_t realtime_geometry_solve_dispatches(
        std::uint32_t body_count) noexcept {
#if defined(PARALLEL_MATER_METAL_DRIVER)
        return simulation::rigid_realtime_cooperative_solve_dispatches(
            body_count);
#else
        (void)body_count;
        return 1U;
#endif
    }
    static constexpr std::uint64_t realtime_geometry_solve_workgroups(
        std::uint32_t body_count, std::uint32_t grid_blocks) noexcept {
#if defined(PARALLEL_MATER_METAL_DRIVER)
        (void)grid_blocks;
        return simulation::rigid_realtime_cooperative_solve_workgroups(
            body_count);
#else
        (void)body_count;
        return grid_blocks;
#endif
    }
    auto queue() const noexcept { return execution; }
    static auto success_code() noexcept { return PM_SUCCESS; }
    static Status failure(PmResult error, const char *message) noexcept {
        return backend_failure(error, message);
    }
    template <typename T> static decltype(auto) unwrap(T &&value) noexcept {
        return std::forward<T>(value);
    }
    auto copy(void *destination, const void *source, std::size_t bytes,
              device::CopyDirection direction, device::ExecutionQueue queue) noexcept {
        return queue.services->copy(destination, source, bytes, direction);
    }
    auto clear(void *destination, int value, std::size_t bytes,
               device::ExecutionQueue queue) noexcept {
        return queue.services->clear(destination, value, bytes);
    }
    auto check_error() noexcept { return execution.services->last_error(false); }
    auto take_error() noexcept { return execution.services->last_error(true); }
    bool schedule_profile_enabled() const noexcept {
        return execution.services->schedule_profile_enabled();
    }
    auto begin_schedule_profile() noexcept {
        return execution.services->begin_schedule_profile();
    }
    auto record_schedule_profile_stage(
        std::string_view name, std::uint32_t launches) noexcept {
        return execution.services->record_schedule_profile_stage(name, launches);
    }
    auto end_schedule_profile() noexcept {
        return execution.services->end_schedule_profile();
    }
    auto wait(device::ExecutionQueue queue) noexcept {
        return queue.services->wait_idle();
    }
    auto record_event(device::Event event, device::ExecutionQueue queue) noexcept {
        return queue.services->record_event(event);
    }
    template <typename T>
    Status read_value(device::ExecutionQueue queue, const T *source, T &destination,
                      const char *message) noexcept {
        return read_device_value(queue, source, destination, message);
    }
    static auto multiply(Vec3 value, float scale) noexcept {
        return parallel_mater::multiply(value, scale);
    }
    static auto normalized_or(Vec3 value, Vec3 fallback) noexcept {
        return parallel_mater::normalized_or(value, fallback);
    }
    static constexpr auto fluid_body_buckets = k_fluid_body_buckets;
    using SmokeGridDispatchField = simulation::SmokeGridDispatchField;
    using RopeSoftTarget = simulation::RopeSoftTargetDispatch;
    using FluidDispatchConstants = simulation::FluidDispatchConstants;
    using FluidClothDispatchOptions = simulation::FluidClothDispatchOptions;
    using FluidSmokeDispatchOptions = simulation::FluidSmokeDispatchOptions;
    using SmokeGridBoundaryDispatch = simulation::SmokeGridBoundaryDispatch;
#define PM_SCHEDULE_DISPATCH(name) \
    template <typename... Args> \
    decltype(auto) name(Args &&...args) noexcept { \
        return simulation::name(unwrap(std::forward<Args>(args))...); \
    }
    PM_SCHEDULE_DISPATCH(accumulate_smoke_grid_force_on_cloth)
    PM_SCHEDULE_DISPATCH(advance_rope)
    PM_SCHEDULE_DISPATCH(advect_smoke_grid_face)
    PM_SCHEDULE_DISPATCH(advect_smoke_grid_realtime_face)
    PM_SCHEDULE_DISPATCH(advect_smoke_grid_realtime_scalar)
    PM_SCHEDULE_DISPATCH(advect_smoke_particles)
    PM_SCHEDULE_DISPATCH(apply_cloth_body_corrections)
    PM_SCHEDULE_DISPATCH(apply_deformable_smoke_coupling_first)
    PM_SCHEDULE_DISPATCH(apply_deformable_smoke_coupling_second)
    PM_SCHEDULE_DISPATCH(apply_cloth_realtime_position_delta)
    PM_SCHEDULE_DISPATCH(apply_rope_realtime_position_delta)
    PM_SCHEDULE_DISPATCH(apply_rigid_coupling_bodies)
    PM_SCHEDULE_DISPATCH(apply_rigid_coupling_endpoints)
    PM_SCHEDULE_DISPATCH(apply_soft_body_realtime_position_delta)
    PM_SCHEDULE_DISPATCH(apply_soft_body_realtime_position_delta_batch)
    PM_SCHEDULE_DISPATCH(apply_fluid_cloth_forces)
    PM_SCHEDULE_DISPATCH(apply_fluid_forces_to_cloth)
    PM_SCHEDULE_DISPATCH(apply_fluid_rope_impulses)
    PM_SCHEDULE_DISPATCH(apply_fluid_soft_contacts)
    PM_SCHEDULE_DISPATCH(apply_rope_cloth_anchor)
    PM_SCHEDULE_DISPATCH(apply_rope_soft_body_impulses)
    PM_SCHEDULE_DISPATCH(apply_smoke_grid_face_boundaries)
    PM_SCHEDULE_DISPATCH(apply_smoke_grid_face_forces)
    PM_SCHEDULE_DISPATCH(apply_smoke_grid_realtime_boundaries)
    PM_SCHEDULE_DISPATCH(apply_smoke_grid_realtime_face_forces)
    PM_SCHEDULE_DISPATCH(apply_smoke_grid_realtime_source)
    PM_SCHEDULE_DISPATCH(apply_smoke_grid_force_to_cloth)
    PM_SCHEDULE_DISPATCH(apply_smoke_grid_force_to_rigid_body)
    PM_SCHEDULE_DISPATCH(apply_smoke_grid_force_to_soft_body)
    PM_SCHEDULE_DISPATCH(apply_smoke_grid_force_to_soft_body_batch)
    PM_SCHEDULE_DISPATCH(apply_smoke_rigid_body_impulses)
    PM_SCHEDULE_DISPATCH(apply_smoke_wind_to_cloth)
    PM_SCHEDULE_DISPATCH(apply_smoke_wind_to_rope)
    PM_SCHEDULE_DISPATCH(apply_smoke_wind_to_soft_body)
    PM_SCHEDULE_DISPATCH(apply_soft_body_cloth_contacts_to_cloth)
    PM_SCHEDULE_DISPATCH(apply_soft_body_cloth_contacts_to_soft_body)
    PM_SCHEDULE_DISPATCH(apply_soft_body_contact_friction)
    PM_SCHEDULE_DISPATCH(apply_soft_body_contact_friction_batch)
    PM_SCHEDULE_DISPATCH(apply_soft_body_surface_contacts)
    PM_SCHEDULE_DISPATCH(break_cloth_bonds)
    PM_SCHEDULE_DISPATCH(broad_phase_rigid_pairs)
    PM_SCHEDULE_DISPATCH(build_fixed_collision_groups)
    PM_SCHEDULE_DISPATCH(calculate_fluid_body_bounds)
    PM_SCHEDULE_DISPATCH(calculate_smoke_deformable_bounds)
    PM_SCHEDULE_DISPATCH(calculate_smoke_density_pressure)
    PM_SCHEDULE_DISPATCH(calculate_smoke_grid_cell_diagnostics)
    PM_SCHEDULE_DISPATCH(calculate_smoke_grid_divergence)
    PM_SCHEDULE_DISPATCH(calculate_smoke_grid_realtime_divergence)
    PM_SCHEDULE_DISPATCH(calculate_smoke_grid_pressure_residual)
    PM_SCHEDULE_DISPATCH(calculate_smoke_grid_subgrid_force)
    PM_SCHEDULE_DISPATCH(calculate_smoke_pair_forces)
    PM_SCHEDULE_DISPATCH(calculate_smoke_vorticity)
    PM_SCHEDULE_DISPATCH(calculate_soft_body_surface_contacts)
    PM_SCHEDULE_DISPATCH(capture_rigid_inputs)
    PM_SCHEDULE_DISPATCH(clamp_rigid_speeds)
    PM_SCHEDULE_DISPATCH(clear_inactive_fluid_keep)
    PM_SCHEDULE_DISPATCH(clear_rigid_inputs)
    PM_SCHEDULE_DISPATCH(clear_smoke_grid_obstacles)
    PM_SCHEDULE_DISPATCH(collide_deformable_with_rigid_geometry)
    PM_SCHEDULE_DISPATCH(collide_soft_bodies_with_rigid_geometry_batch)
    PM_SCHEDULE_DISPATCH(collide_fluid_with_static_body)
    PM_SCHEDULE_DISPATCH(collide_smoke_with_cloth)
    PM_SCHEDULE_DISPATCH(collide_smoke_with_rigid_body)
    PM_SCHEDULE_DISPATCH(collide_smoke_with_rope)
    PM_SCHEDULE_DISPATCH(collide_smoke_with_soft_body)
    PM_SCHEDULE_DISPATCH(collide_smoke_with_soft_body_batch)
    PM_SCHEDULE_DISPATCH(compact_flagged_indices)
    PM_SCHEDULE_DISPATCH(compare_smoke_grid_residual)
    PM_SCHEDULE_DISPATCH(compute_rigid_world_bounds)
    PM_SCHEDULE_DISPATCH(constrain_rigid_bodies_with_cloth)
    PM_SCHEDULE_DISPATCH(correct_smoke_grid_face)
    PM_SCHEDULE_DISPATCH(damp_soft_body_springs)
    PM_SCHEDULE_DISPATCH(damp_soft_body_springs_warp)
    PM_SCHEDULE_DISPATCH(detect_fluid_moving_contacts)
    PM_SCHEDULE_DISPATCH(detect_fluid_rope_contacts)
    PM_SCHEDULE_DISPATCH(detect_fluid_soft_contacts)
    PM_SCHEDULE_DISPATCH(detect_soft_body_cloth_contacts)
    PM_SCHEDULE_DISPATCH(drag_fluid_with_smoke_grid)
    PM_SCHEDULE_DISPATCH(drag_fluid_with_smoke_particles)
    PM_SCHEDULE_DISPATCH(emit_fluid_source)
    PM_SCHEDULE_DISPATCH(emit_fluid_spatial_particles)
    PM_SCHEDULE_DISPATCH(emit_fluid_spatial_source_proposals)
    PM_SCHEDULE_DISPATCH(apply_fluid_spatial_source_admission)
    PM_SCHEDULE_DISPATCH(emit_deformable_spatial_primitives)
    PM_SCHEDULE_DISPATCH(emit_rigid_spatial_proxies)
    PM_SCHEDULE_DISPATCH(emit_smoke_particles)
    PM_SCHEDULE_DISPATCH(emit_smoke_realtime_particles)
    PM_SCHEDULE_DISPATCH(evaluate_rigid_leaf_pairs)
    PM_SCHEDULE_DISPATCH(exchange_fluid_with_smoke)
    PM_SCHEDULE_DISPATCH(finalize_rigid_contact_manifolds)
    PM_SCHEDULE_DISPATCH(finalize_soft_body_velocities)
    PM_SCHEDULE_DISPATCH(find_fluid_source_vacancies)
    PM_SCHEDULE_DISPATCH(filter_rigid_spatial_candidates)
    PM_SCHEDULE_DISPATCH(filter_fluid_spatial_candidates)
    PM_SCHEDULE_DISPATCH(fluid_cell_keys)
    PM_SCHEDULE_DISPATCH(fluid_pair_cell_keys)
    PM_SCHEDULE_DISPATCH(fluid_miscible_pair_pressure_contact)
    PM_SCHEDULE_DISPATCH(fluid_miscible_pair_viscosity)
    PM_SCHEDULE_DISPATCH(fluid_pair_neighbor_counts)
    PM_SCHEDULE_DISPATCH(fluid_pair_pressure_contact)
    PM_SCHEDULE_DISPATCH(fluid_pair_viscosity)
    PM_SCHEDULE_DISPATCH(fluid_forces)
    PM_SCHEDULE_DISPATCH(predict_fluid_realtime)
    PM_SCHEDULE_DISPATCH(predict_smoke_realtime_particles)
    PM_SCHEDULE_DISPATCH(solve_fluid_realtime_density)
    PM_SCHEDULE_DISPATCH(solve_fluid_realtime_position_delta)
    PM_SCHEDULE_DISPATCH(apply_fluid_realtime_position_delta)
    PM_SCHEDULE_DISPATCH(solve_fluid_realtime_viscosity)
    PM_SCHEDULE_DISPATCH(solve_smoke_grid_realtime_pressure_cycle)
    PM_SCHEDULE_DISPATCH(solve_smoke_realtime_density_pressure)
    PM_SCHEDULE_DISPATCH(solve_smoke_realtime_position_delta)
    PM_SCHEDULE_DISPATCH(apply_smoke_realtime_position_delta)
    PM_SCHEDULE_DISPATCH(commit_fluid_realtime)
    PM_SCHEDULE_DISPATCH(commit_cloth_realtime)
    PM_SCHEDULE_DISPATCH(commit_rope_realtime)
    PM_SCHEDULE_DISPATCH(commit_smoke_realtime_particles)
    PM_SCHEDULE_DISPATCH(commit_soft_body_realtime)
    PM_SCHEDULE_DISPATCH(commit_soft_body_realtime_batch)
    PM_SCHEDULE_DISPATCH(gather_fluid)
    PM_SCHEDULE_DISPATCH(gather_rigid_spatial_pairs)
    PM_SCHEDULE_DISPATCH(gather_fluid_contact_events)
    PM_SCHEDULE_DISPATCH(generate_rigid_leaf_pairs)
    PM_SCHEDULE_DISPATCH(index_fluid_body_cells)
    PM_SCHEDULE_DISPATCH(index_smoke_particles)
    PM_SCHEDULE_DISPATCH(integrate_fluid)
    PM_SCHEDULE_DISPATCH(integrate_rigid_bodies)
    PM_SCHEDULE_DISPATCH(limit_cloth_strain)
    PM_SCHEDULE_DISPATCH(load_rigid_contact_cache)
    PM_SCHEDULE_DISPATCH(mark_fluid_destroy_flags)
    PM_SCHEDULE_DISPATCH(mark_smoke_grid_domain_boundaries)
    PM_SCHEDULE_DISPATCH(measure_soft_body_momentum)
    PM_SCHEDULE_DISPATCH(predict_deformable)
    PM_SCHEDULE_DISPATCH(predict_cloth_realtime)
    PM_SCHEDULE_DISPATCH(predict_rope_realtime)
    PM_SCHEDULE_DISPATCH(predict_soft_body_realtime)
    PM_SCHEDULE_DISPATCH(predict_soft_body_realtime_batch)
    PM_SCHEDULE_DISPATCH(prepare_geometry_constraints)
    PM_SCHEDULE_DISPATCH(prepare_parallel_contact_events)
    PM_SCHEDULE_DISPATCH(project_cloth_volume)
    PM_SCHEDULE_DISPATCH(project_deformable_smoke_couplings)
    PM_SCHEDULE_DISPATCH(project_cloth_realtime_links)
    PM_SCHEDULE_DISPATCH(project_cloth_realtime_volume)
    PM_SCHEDULE_DISPATCH(project_cloth_realtime_contacts)
    PM_SCHEDULE_DISPATCH(project_rope_realtime_constraints)
    PM_SCHEDULE_DISPATCH(project_rope_realtime_contacts)
    PM_SCHEDULE_DISPATCH(project_rigid_couplings)
    PM_SCHEDULE_DISPATCH(project_deformable_links)
    PM_SCHEDULE_DISPATCH(project_deformable_links_warp)
    PM_SCHEDULE_DISPATCH(project_fluid_inside_cloth)
    PM_SCHEDULE_DISPATCH(project_smoke_grid_face)
    PM_SCHEDULE_DISPATCH(project_smoke_grid_realtime_face)
    PM_SCHEDULE_DISPATCH(project_soft_body_rest_shape)
    PM_SCHEDULE_DISPATCH(project_soft_body_realtime_contacts)
    PM_SCHEDULE_DISPATCH(project_soft_body_realtime_contacts_batch)
    PM_SCHEDULE_DISPATCH(project_soft_body_realtime_links)
    PM_SCHEDULE_DISPATCH(project_soft_body_realtime_links_batch)
    PM_SCHEDULE_DISPATCH(project_soft_body_realtime_shape)
    PM_SCHEDULE_DISPATCH(project_soft_body_realtime_shape_batch)
    PM_SCHEDULE_DISPATCH(prolong_smoke_grid_pressure)
    PM_SCHEDULE_DISPATCH(radix_sort_pairs_u64_u32)
    PM_SCHEDULE_DISPATCH(raster_smoke_grid_deformable)
    PM_SCHEDULE_DISPATCH(raster_smoke_grid_rigid_body)
    PM_SCHEDULE_DISPATCH(raster_smoke_grid_soft_body)
    PM_SCHEDULE_DISPATCH(recover_fluid_soft_contacts)
    PM_SCHEDULE_DISPATCH(reduce_fluid_body_impulses)
    PM_SCHEDULE_DISPATCH(reduce_point_body_impulses_batch)
    PM_SCHEDULE_DISPATCH(reduce_rigid_leaf_manifolds)
    PM_SCHEDULE_DISPATCH(refit_fluid_soft_surface)
    PM_SCHEDULE_DISPATCH(reserve_fluid_contact_events)
    PM_SCHEDULE_DISPATCH(reset_smoke_grid_residual)
    PM_SCHEDULE_DISPATCH(resolve_fluid_moving_contacts)
    PM_SCHEDULE_DISPATCH(resolve_smoke_grid_particle_fields)
    PM_SCHEDULE_DISPATCH(restore_soft_body_momentum)
    PM_SCHEDULE_DISPATCH(restrict_smoke_grid_open_faces)
    PM_SCHEDULE_DISPATCH(restrict_smoke_grid_residual)
    PM_SCHEDULE_DISPATCH(sample_rope_cloth_anchor)
    PM_SCHEDULE_DISPATCH(sample_rope_soft_body_anchor)
    PM_SCHEDULE_DISPATCH(save_rigid_contact_cache)
    PM_SCHEDULE_DISPATCH(scatter_rope_soft_body_anchor)
    PM_SCHEDULE_DISPATCH(smooth_smoke_grid_pressure)
    PM_SCHEDULE_DISPATCH(solve_fluid_soft_contacts)
    PM_SCHEDULE_DISPATCH(solve_geometry_constraints)
    PM_SCHEDULE_DISPATCH(solve_geometry_constraints_fixed_realtime)
    PM_SCHEDULE_DISPATCH(splat_smoke_grid_particles)
    PM_SCHEDULE_DISPATCH(trace_smoke_grid_particles)
    PM_SCHEDULE_DISPATCH(trace_smoke_grid_realtime_particles)
    PM_SCHEDULE_DISPATCH(begin_smoke_grid_realtime)
    PM_SCHEDULE_DISPATCH(update_cloth_surface)
    PM_SCHEDULE_DISPATCH(update_soft_body_surface)
    PM_SCHEDULE_DISPATCH(update_soft_body_surface_batch)
#undef PM_SCHEDULE_DISPATCH
#define PM_SCHEDULE_PACK(name, helper) \
    template <typename... Args> \
    decltype(auto) name(Args &&...args) noexcept { \
        return helper(std::forward<Args>(args)...); \
    }
    PM_SCHEDULE_PACK(fluid_constants, parallel_mater::slang_fluid_constants)
    PM_SCHEDULE_PACK(smoke_domain_options, parallel_mater::smoke_domain_options)
    PM_SCHEDULE_PACK(smoke_dispatch_options, parallel_mater::smoke_dispatch_options)
    PM_SCHEDULE_PACK(smoke_dispatch_grid, parallel_mater::smoke_dispatch_grid)
    PM_SCHEDULE_PACK(smoke_grid_face_count, parallel_mater::smoke_grid_face_count)
#undef PM_SCHEDULE_PACK
};
} // namespace

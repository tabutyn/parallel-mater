// SPDX-License-Identifier: MIT
// Included by runtime.cpp after NativeScheduleBackend is defined.
namespace {

template <typename T>
T *realtime_range(void *base,
                  simulation::SpatialBufferRange range) noexcept {
    return reinterpret_cast<T *>(reinterpret_cast<std::byte *>(base) +
                                 range.offset);
}

template <typename State>
simulation::RealtimeScheduleScene native_realtime_scene(
    const State &state, const StepOptions &options) noexcept {
    simulation::RealtimeScheduleScene scene{};
    scene.domains[simulation::realtime_index(
        simulation::RealtimeDomain::rigid)] = state.rigid_body_count != 0U;
    scene.domains[simulation::realtime_index(
        simulation::RealtimeDomain::fluid)] = state.fluid_count != 0U;
    scene.domains[simulation::realtime_index(
        simulation::RealtimeDomain::cloth)] = std::any_of(
            state.cloths.begin(), state.cloths.end(),
            [](const auto &owner) { return owner && owner->alive; });
    scene.domains[simulation::realtime_index(
        simulation::RealtimeDomain::soft_body)] = std::any_of(
            state.soft_bodies.begin(), state.soft_bodies.end(),
            [](const auto &owner) { return owner && owner->alive; });
    scene.domains[simulation::realtime_index(
        simulation::RealtimeDomain::rope)] = std::any_of(
            state.ropes.begin(), state.ropes.end(),
            [](const auto &owner) { return owner && owner->alive; });
    scene.domains[simulation::realtime_index(
        simulation::RealtimeDomain::smoke)] = std::any_of(
            state.smokes.begin(), state.smokes.end(),
            [](const auto &owner) { return owner && owner->alive; });
    scene.spatial_update = std::any_of(
        scene.domains.begin(), scene.domains.end(),
        [](bool present) { return present; })
        ? simulation::RealtimeSpatialUpdate::build
        : simulation::RealtimeSpatialUpdate::none;
    scene.timestep = options.timestep;
    scene.substeps = options.substeps;
    scene.tick = state.frame_index + 1U;
    scene.graph_revision = state.revision + 1U;
    scene.resource_revision = state.revision + 1U;
    scene.input_snapshot = static_cast<std::uint32_t>(
        state.frame_index % (std::numeric_limits<std::uint32_t>::max() - 1U)) +
        1U;
    scene.render_snapshot = scene.input_snapshot;
    return scene;
}

[[nodiscard]] constexpr std::uint64_t realtime_identity_mix(
    std::uint64_t hash, std::uint64_t value) noexcept {
    constexpr std::uint64_t prime = 1099511628211ULL;
    for (std::uint32_t byte = 0U; byte < 8U; ++byte) {
        hash ^= (value >> (byte * 8U)) & 0xffU;
        hash *= prime;
    }
    return hash;
}

[[nodiscard]] PmResult add_realtime_registration(
    simulation::RealtimeResourceRegistration &registration,
    std::uint32_t slot, std::uint32_t generation,
    std::uint64_t elements, std::uint32_t operator_phases) noexcept {
    if (elements == 0U ||
        registration.live_slots == std::numeric_limits<std::uint32_t>::max() ||
        elements > std::numeric_limits<std::uint64_t>::max() -
            registration.reserved_elements)
        return PM_ERROR_CAPACITY_EXCEEDED;
    ++registration.live_slots;
    registration.reserved_elements += elements;
    auto hash = registration.identity_hash;
    if (hash == 0U) hash = 1469598103934665603ULL;
    hash = realtime_identity_mix(hash, slot);
    hash = realtime_identity_mix(hash, generation);
    hash = realtime_identity_mix(hash, elements);
    registration.identity_hash = hash == 0U ? 1U : hash;
    registration.operator_phases |= operator_phases;
    return PM_SUCCESS;
}

template <typename State>
PmResult native_realtime_registrations(
    const State &state,
    std::array<simulation::RealtimeResourceRegistration,
               simulation::realtime_resource_family_count> &result) noexcept {
    result = {};
    const std::uint32_t common_phases =
        simulation::realtime_phase_bit(
            simulation::RealtimeResourcePhase::boundary) |
        simulation::realtime_phase_bit(
            simulation::RealtimeResourcePhase::commit);
    auto &rigid = result[simulation::realtime_index(
        simulation::RealtimeResourceFamily::rigid)];
    for (std::uint32_t slot = 0U; slot < state.slots.size(); ++slot) {
        const auto &owner = state.slots[slot];
        if (!owner.alive) continue;
        const auto status = add_realtime_registration(
            rigid, slot, owner.generation, 1U,
            common_phases |
                simulation::realtime_domain_execution_phase_mask);
        if (status != PM_SUCCESS) return status;
    }

    auto add_owner = [&](simulation::RealtimeResourceFamily family,
                         std::uint32_t slot, std::uint32_t generation,
                         std::uint64_t elements) noexcept {
        return add_realtime_registration(
            result[simulation::realtime_index(family)], slot, generation,
            elements, common_phases);
    };
    for (std::uint32_t slot = 0U; slot < state.fluids.size(); ++slot) {
        const auto &owner = state.fluids[slot];
        if (!owner || !owner->alive) continue;
        const auto status = add_owner(
            simulation::RealtimeResourceFamily::liquid, slot,
            owner->generation, owner->options.capacity);
        if (status != PM_SUCCESS) return status;
    }
    for (std::uint32_t slot = 0U; slot < state.cloths.size(); ++slot) {
        const auto &owner = state.cloths[slot];
        if (!owner || !owner->alive) continue;
        const std::uint64_t elements = owner->vertex_capacity +
            std::uint64_t{owner->index_count} / 3U;
        const auto status = add_owner(
            simulation::RealtimeResourceFamily::cloth, slot,
            owner->generation, elements);
        if (status != PM_SUCCESS) return status;
    }
    for (std::uint32_t slot = 0U; slot < state.soft_bodies.size(); ++slot) {
        const auto &owner = state.soft_bodies[slot];
        if (!owner || !owner->alive) continue;
        const std::uint64_t elements = owner->node_count +
            std::uint64_t{owner->surface_vertex_count} +
            std::uint64_t{owner->surface_index_count} / 3U;
        const auto status = add_owner(
            simulation::RealtimeResourceFamily::soft_body, slot,
            owner->generation, elements);
        if (status != PM_SUCCESS) return status;
    }
    for (std::uint32_t slot = 0U; slot < state.ropes.size(); ++slot) {
        const auto &owner = state.ropes[slot];
        if (!owner || !owner->alive) continue;
        const std::uint64_t elements = owner->data.count +
            std::uint64_t{owner->data.count - 1U};
        const auto status = add_owner(
            simulation::RealtimeResourceFamily::rope, slot,
            owner->generation, elements);
        if (status != PM_SUCCESS) return status;
    }
    for (std::uint32_t slot = 0U; slot < state.smokes.size(); ++slot) {
        const auto &owner = state.smokes[slot];
        if (!owner || !owner->alive) continue;
        const bool grid = owner->options.grid_resolution != 0U;
        const std::uint64_t elements = owner->options.capacity +
            (grid ? std::uint64_t{owner->grid.cell_count} : 0U);
        const auto status = add_owner(
            grid ? simulation::RealtimeResourceFamily::grid_smoke
                 : simulation::RealtimeResourceFamily::particle_smoke,
            slot, owner->generation, elements);
        if (status != PM_SUCCESS) return status;
    }
    return PM_SUCCESS;
}

template <typename State>
simulation::RealtimeTelemetryLimits native_realtime_limits(
    const State &state) noexcept {
    simulation::RealtimeTelemetryLimits limits{};
    auto &boundary = limits.phase_maximum[simulation::realtime_index(
        simulation::RealtimeTelemetryPhase::boundary)];
    auto &predict = limits.phase_maximum[simulation::realtime_index(
        simulation::RealtimeTelemetryPhase::predict)];
    auto &index = limits.phase_maximum[simulation::realtime_index(
        simulation::RealtimeTelemetryPhase::index)];
    auto &candidates = limits.phase_maximum[simulation::realtime_index(
        simulation::RealtimeTelemetryPhase::candidates)];
    auto &domain = limits.phase_maximum[simulation::realtime_index(
        simulation::RealtimeTelemetryPhase::domain)];
    auto &commit = limits.phase_maximum[simulation::realtime_index(
        simulation::RealtimeTelemetryPhase::commit)];
    const auto dispatch = [](simulation::RealtimeWorkCounters &work,
                             std::uint64_t count,
                             std::uint64_t workgroups) noexcept {
        work.dispatches += count;
        work.workgroups += workgroups;
        work.barriers += static_cast<std::uint32_t>(count);
    };
    const std::uint32_t rigid = state.rigid_body_count;
    commit.submissions = 1U;
    index.index_publications = 1U;
    if (rigid == 0U) {
#if !defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
        dispatch(boundary, 1U, 1U);
        index.bytes_moved = sizeof(std::uint32_t);
        ++index.barriers;
        dispatch(index, 1U, 1U);
#endif
        commit.bytes_moved = sizeof(std::uint32_t);
        ++commit.barriers;
    } else {
        const std::uint64_t body_groups = (rigid + 127U) / 128U;
        boundary.bytes_moved = sizeof(std::uint32_t) +
            std::uint64_t{rigid} * sizeof(RigidBodyState);
        boundary.barriers = 2U;
#if !defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
        dispatch(boundary, 1U, 1U);
#endif
        dispatch(predict, 1U, body_groups);
        dispatch(index, 1U, body_groups);
        index.index_builds = 1U;
#if !defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
        const auto spatial = simulation::spatial_build_dispatch_budget(
            state.realtime_spatial_plan.leaf_capacity, rigid);
        dispatch(index, 1U + spatial.total_dispatches,
            state.realtime_spatial_plan.build.block_capacity +
                spatial.total_workgroups);
        if (state.realtime_spatial_plan.build.arrivals.size != 0U) {
            index.bytes_moved +=
                state.realtime_spatial_plan.build.arrivals.size;
            ++index.barriers;
        }
        dispatch(index, 1U, 1U);
        dispatch(candidates, 1U, (rigid + 63U) / 64U);
#endif
        if (state.rigid_constraint_count != 0U)
            dispatch(candidates, 1U, 1U);
        const auto candidate_groups =
            (std::uint64_t{state.realtime_rigid_spatial_plan
                               .candidate_slot_capacity} + 127U) / 128U;
        dispatch(candidates, 1U, candidate_groups);
        dispatch(candidates, 3U,
            2ULL * state.realtime_rigid_spatial_plan.compaction_blocks + 1U);
        dispatch(candidates, 1U, candidate_groups);

        const auto contact_groups = std::min<std::uint64_t>(
            128U, std::uint64_t{rigid} * rigid);
        const bool reduces_leaf_manifolds =
            state.rigid_leaf_manifolds != nullptr;
        const std::uint32_t leaf_blocks = reduces_leaf_manifolds
            ? k_rigid_leaf_blocks_per_pair : 1U;
        const std::uint32_t avbd_grid = std::min(
            state.rigid_contact_grid_limit,
            std::max(1U, (rigid + 31U) / 32U));
        const std::uint32_t constraint_capacity =
            state.rigid_constraint_count != 0U
                ? state.options.rigid_constraint_capacity : 0U;
        dispatch(domain, 1U, contact_groups); // generate leaf pairs
        dispatch(domain, 1U, contact_groups); // finalize leaf contacts
        dispatch(domain, 1U, contact_groups * leaf_blocks);
        if (reduces_leaf_manifolds)
            dispatch(domain, 1U, contact_groups);
        dispatch(domain, 1U, contact_groups); // finalize body contacts
        dispatch(domain, 1U, 1U); // prepare public contact events
        dispatch(domain, 1U, contact_groups); // load contact cache
        dispatch(domain, 1U,
            NativeScheduleBackend::realtime_geometry_prepare_workgroups(
                avbd_grid));
        dispatch(domain,
            NativeScheduleBackend::realtime_geometry_solve_dispatches(rigid),
            NativeScheduleBackend::realtime_geometry_solve_workgroups(
                rigid, avbd_grid));
        if (!NativeScheduleBackend::geometry_constraints_inline_speed_clamp(
                rigid, constraint_capacity, avbd_grid))
            dispatch(domain, 1U, body_groups);
        dispatch(domain, 1U, contact_groups); // save contact cache
        domain.numerical_iterations = simulation::rigid_realtime_sweeps;
        if (state.rigid_contact_revision != state.revision) {
            domain.bytes_moved = std::uint64_t{rigid} * rigid *
                sizeof(*state.rigid_contact_cache_slots);
            ++domain.barriers;
        }
        dispatch(commit, 1U, body_groups);
    }
    for (const auto &phase : limits.phase_maximum)
        (void)simulation::realtime_telemetry_detail::add(
            limits.total_maximum, phase);
#if defined(PARALLEL_MATER_TEST_FAILURE_DRIVER)
    if (const char *field = std::getenv("PM_TEST_REALTIME_ZERO_LIMIT")) {
#define PM_ZERO_LIMIT(name) if (std::strcmp(field, #name) == 0) \
        limits.total_maximum.name = 0U
        PM_ZERO_LIMIT(dispatches);
        else PM_ZERO_LIMIT(workgroups);
        else PM_ZERO_LIMIT(bytes_moved);
        else PM_ZERO_LIMIT(index_builds);
        else PM_ZERO_LIMIT(index_publications);
        else PM_ZERO_LIMIT(numerical_iterations);
        else PM_ZERO_LIMIT(barriers);
        else PM_ZERO_LIMIT(submissions);
#undef PM_ZERO_LIMIT
    }
    if (std::getenv("PM_TEST_REALTIME_ZERO_DOMAIN_DISPATCHES") != nullptr)
        limits.phase_maximum[simulation::realtime_index(
            simulation::RealtimeTelemetryPhase::domain)].dispatches = 0U;
    if (std::getenv("PM_TEST_REALTIME_ZERO_COMMIT_SUBMISSIONS") != nullptr)
        limits.phase_maximum[simulation::realtime_index(
            simulation::RealtimeTelemetryPhase::commit)].submissions = 0U;
    if (const char *field = std::getenv(
            "PM_TEST_REALTIME_LIMIT_MINUS_ONE")) {
#define PM_DECREMENT_LIMIT(name) if (std::strcmp(field, #name) == 0 && \
        limits.total_maximum.name != 0U) --limits.total_maximum.name
        PM_DECREMENT_LIMIT(dispatches);
        else PM_DECREMENT_LIMIT(workgroups);
#undef PM_DECREMENT_LIMIT
    }
#endif
    return limits;
}

template <typename State>
simulation::RealtimeWorkCounters native_realtime_minimum(
    const State &state) noexcept {
    const bool rigid = state.rigid_body_count != 0U;
    return {
        .dispatches = rigid ? 1U : 0U,
        .workgroups = rigid ? 1U : 0U,
        .bytes_moved = rigid
            ? std::uint64_t{state.rigid_body_count} * sizeof(RigidBodyState)
            : 0U,
        .index_builds = rigid ? 1U : 0U,
        .index_refits = 0U,
        .index_publications = 1U,
        .numerical_iterations = rigid
            ? simulation::rigid_realtime_sweeps : 0U,
        .barriers = rigid ? 1U : 0U,
        .submissions = 1U,
        .readbacks = 0U,
        .host_waits = 0U,
    };
}

template <typename State>
std::array<simulation::RealtimeWorkCounters,
           simulation::realtime_telemetry_phase_count>
native_realtime_phase_minimum(const State &state) noexcept {
    std::array<simulation::RealtimeWorkCounters,
               simulation::realtime_telemetry_phase_count> minimum{};
    const bool rigid = state.rigid_body_count != 0U;
    auto &index = minimum[simulation::realtime_index(
        simulation::RealtimeTelemetryPhase::index)];
    index.index_publications = 1U;
    if (rigid) index.index_builds = 1U;
    auto &domain = minimum[simulation::realtime_index(
        simulation::RealtimeTelemetryPhase::domain)];
    if (rigid) {
        domain.dispatches = 1U;
        domain.workgroups = 1U;
        domain.numerical_iterations = simulation::rigid_realtime_sweeps;
        domain.barriers = 1U;
    }
    auto &commit = minimum[simulation::realtime_index(
        simulation::RealtimeTelemetryPhase::commit)];
    commit.submissions = 1U;
    if (rigid) {
        commit.dispatches = 1U;
        commit.workgroups = 1U;
    } else {
        commit.bytes_moved = sizeof(std::uint32_t);
    }
    commit.barriers = 1U;
    return minimum;
}

template <typename State>
bool native_realtime_limits_admit_minimum(
    const State &state,
    const simulation::RealtimeTelemetryLimits &limits) noexcept {
    if (!simulation::realtime_telemetry_detail::within(
            native_realtime_minimum(state), limits.total_maximum))
        return false;
    const auto phase_minimum = native_realtime_phase_minimum(state);
    for (std::size_t index = 0U; index < phase_minimum.size(); ++index)
        if (!simulation::realtime_telemetry_detail::within(
                phase_minimum[index], limits.phase_maximum[index]))
            return false;
    return true;
}

template <typename State>
class NativeRealtimeBackend {
public:
    NativeRealtimeBackend(State &state, NativeScheduleBackend &native,
                          StepOptions options) noexcept
        : state_(state), native_(native), options_(options) {}

    PmResult begin_realtime_telemetry(
        const simulation::RealtimeScheduleScene &) noexcept {
        pending_ = {};
        cpu_encode_ns_.fill(0U);
        measurement_started_ = std::chrono::steady_clock::now();
        return PM_SUCCESS;
    }

    PmResult take_realtime_measurement(
        const simulation::RealtimeTelemetryOperation &operation,
        simulation::RealtimeOperationMeasurement &measurement) noexcept {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                now - measurement_started_).count());
        cpu_encode_ns_[simulation::realtime_index(
            telemetry_phase(operation.schedule_phase))] += elapsed;
        measurement_started_ = now;
        measurement = pending_;
        pending_ = {};
        return PM_SUCCESS;
    }

    PmResult end_realtime_telemetry(
        const simulation::RealtimeScheduleScene &,
        simulation::RealtimeTickTelemetry &telemetry) noexcept {
        for (std::size_t index = 0U; index < telemetry.phases.size();
             ++index) {
            telemetry.phases[index].timing.cpu_encode_ns =
                cpu_encode_ns_[index];
            telemetry.phases[index].timing.gpu_status =
                simulation::RealtimeGpuTimingStatus::unsupported;
        }
        return PM_SUCCESS;
    }

    PmResult abort_realtime_telemetry(
        const simulation::RealtimeScheduleScene &) noexcept {
        pending_ = {};
        return PM_SUCCESS;
    }

    PmResult begin_tick(
        const simulation::RealtimeScheduleScene &scene) noexcept {
        if (!state_.options.realtime_spatial_enabled ||
            !state_.realtime_spatial.configured() ||
            state_.rigid_body_count >
                 state_.options.realtime_spatial_leaf_capacity ||
            state_.rigid_body_count >
                 state_.options.realtime_spatial_query_capacity ||
            state_.options.realtime_spatial_candidates_per_query == 0U)
            return PM_ERROR_FEATURE_NOT_PRESENT;
        std::array<simulation::RealtimeResourceRegistration,
                   simulation::realtime_resource_family_count>
            registrations{};
        auto result = native_realtime_registrations(state_, registrations);
        if (result != PM_SUCCESS) return result;
        result = state_.realtime_resources.stage(
            scene.tick, scene.graph_revision, scene.resource_revision,
            scene.input_snapshot, scene.render_snapshot, registrations);
        if (result != PM_SUCCESS) return result;
        const auto fail_begin = [&](PmResult failure) noexcept {
            (void)state_.realtime_spatial.abort();
            if (state_.realtime_snapshots.pending())
                (void)state_.realtime_snapshots.abort();
            if (state_.realtime_resources.pending())
                (void)state_.realtime_resources.abort();
            return failure;
        };
        result = state_.realtime_snapshots.begin(
            scene.input_snapshot, scene.render_snapshot);
        if (result != PM_SUCCESS) return fail_begin(result);
        result = state_.realtime_resources.execution_status();
        if (result != PM_SUCCESS) return fail_begin(result);
#if defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
        result = state_.realtime_spatial.begin_tick(
            scene.tick, simulation::rigid_realtime_profile_version,
            scene.graph_revision, scene.resource_revision,
            state_.rigid_body_count);
#else
        result = state_.realtime_spatial.begin_device_tick(
            scene.tick, simulation::rigid_realtime_profile_version,
            scene.graph_revision, scene.resource_revision,
            state_.rigid_body_count);
#endif
        if (result != PM_SUCCESS) return fail_begin(result);
        snapshot_ = state_.realtime_spatial.staging_snapshot();
        if (!snapshot_) return fail_begin(PM_ERROR_INVALID_STATE);
#if defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
        if (state_.rigid_body_count == 0U) {
            result = state_.realtime_spatial.update(
                simulation::SpatialUpdateKind::empty);
            if (result != PM_SUCCESS) return fail_begin(result);
        }
#else
        const simulation::SpatialDeviceBuffers buffers{
            snapshot_.header, snapshot_.metadata, snapshot_.bounds,
            snapshot_.build_flags, snapshot_.admission,
            snapshot_.query_counts, snapshot_.query_candidates,
            snapshot_.query_visited_nodes, snapshot_.query_counter,
            snapshot_.query_status, snapshot_.build_output,
            snapshot_.build_scratch};
        result = simulation::initialize_spatial_epoch_device(
            native_.queue(), state_.realtime_spatial_plan.build, buffers,
            {snapshot_.token.tick, snapshot_.token.graph_revision,
             snapshot_.token.resource_revision,
             snapshot_.token.profile_version, snapshot_.token.generation,
             state_.rigid_body_count,
             static_cast<std::uint32_t>(state_.rigid_body_count == 0U
                 ? simulation::SpatialUpdateKind::empty
                 : simulation::SpatialUpdateKind::build)});
        if (result != PM_SUCCESS) return fail_begin(result);
        dispatch(simulation::RealtimeTelemetryPhase::boundary, 1U);
        if (state_.rigid_body_count == 0U) {
            auto *active_count = realtime_range<std::uint32_t>(
                snapshot_.build_scratch,
                state_.realtime_spatial_plan.build.active_count);
            result = native_.clear(
                active_count, 0, sizeof(*active_count), native_.queue());
            if (result == PM_SUCCESS)
                result = state_.realtime_spatial.record_device_update(
                    simulation::SpatialUpdateKind::empty);
            if (result != PM_SUCCESS) return fail_begin(result);
            auto &index = work(simulation::RealtimeTelemetryPhase::index);
            index.bytes_moved += sizeof(*active_count);
            ++index.barriers;
        }
#endif
        if (state_.rigid_body_count == 0U) {
            stage_commit_measurement();
            return PM_SUCCESS;
        }
        auto *status = rigid<std::uint32_t>(
            state_.realtime_rigid_spatial_plan.status);
        result = native_.clear(status, 0, sizeof(*status), native_.queue());
        if (result != PM_SUCCESS) return fail_begin(result);
        auto &boundary_work = work(
            simulation::RealtimeTelemetryPhase::boundary);
        boundary_work.bytes_moved += sizeof(*status);
        ++boundary_work.barriers;
        output_state_ = 1U - state_.current_state;
        pending_contact_epoch_ = state_.rigid_contact_epoch +
            (state_.rigid_contact_revision != state_.revision ? 3U : 1U);
        stage_commit_measurement();
        return PM_SUCCESS;
    }

    PmResult boundary(
        const simulation::RealtimeScheduleScene &) noexcept {
        if (state_.rigid_body_count == 0U) return PM_SUCCESS;
        const std::size_t bytes = std::size_t(state_.rigid_body_count) *
            sizeof(RigidBodyState);
        const auto result = native_.copy(
            state_.render_previous_states,
            state_.states[state_.current_state], bytes,
            device::CopyDirection::device_to_device, native_.queue());
        if (result == PM_SUCCESS) {
            auto &boundary_work = work(
                simulation::RealtimeTelemetryPhase::boundary);
            boundary_work.bytes_moved += bytes;
            ++boundary_work.barriers;
        }
        return result;
    }

    PmResult predict(simulation::RealtimeDomain domain,
                     const simulation::RealtimeScheduleScene &scene) noexcept {
        if (domain != simulation::RealtimeDomain::rigid)
            return PM_ERROR_FEATURE_NOT_PRESENT;
        constexpr std::uint32_t block = 128U;
        const auto groups = (state_.rigid_body_count + block - 1U) / block;
        const auto result = native_.integrate_rigid_bodies(
            groups, block, native_.queue(), state_.parameters,
            state_.accumulators, state_.targets,
            state_.states[state_.current_state], state_.states[output_state_],
            state_.rigid_body_count, &options_.gravity.x, scene.timestep, 1U,
            true);
        if (result == PM_SUCCESS)
            dispatch(simulation::RealtimeTelemetryPhase::predict, groups);
        return result;
    }

    PmResult update_spatial(
        simulation::RealtimeSpatialUpdate update,
        const simulation::RealtimeScheduleScene &scene) noexcept {
        if (update != simulation::RealtimeSpatialUpdate::build || !snapshot_)
            return PM_ERROR_FEATURE_NOT_PRESENT;
        constexpr std::uint32_t block = 128U;
        const auto groups = (state_.rigid_body_count + block - 1U) / block;
        const float path_deviation[3]{};
        const float correction_budget[3]{1.0F, 1.0F, 1.0F};
        auto *query_bounds = rigid<simulation::SpatialAabb>(
            state_.realtime_rigid_spatial_plan.query_bounds);
        auto *query_masks = rigid<std::uint32_t>(
            state_.realtime_rigid_spatial_plan.query_masks);
        auto *excluded = rigid<std::uint32_t>(
            state_.realtime_rigid_spatial_plan.excluded_sort_keys);
        auto result = native_.emit_rigid_spatial_proxies(
            groups, block, native_.queue(), state_.parameters,
            state_.states[state_.current_state], state_.states[output_state_],
            state_.meshes, state_.options.triangle_mesh_capacity, state_.ids,
            state_.rigid_body_count, 0U,
            state_.realtime_spatial_plan.leaf_capacity, 0U,
            static_cast<std::uint32_t>(
                scene.graph_revision %
                (std::numeric_limits<std::uint32_t>::max() - 1U)) + 1U,
            simulation::rigid_spatial_query_mask, path_deviation,
            correction_budget, 1.0e-5F, snapshot_.metadata,
            snapshot_.bounds, snapshot_.build_flags, query_bounds,
            query_masks, excluded, snapshot_.query_status);
        if (result != PM_SUCCESS) return result;
        dispatch(simulation::RealtimeTelemetryPhase::index, groups);
#if defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
        result = state_.realtime_spatial.update(
            simulation::SpatialUpdateKind::build);
        if (result != PM_SUCCESS) return result;
#else
        const simulation::SpatialDeviceBuffers buffers{
            snapshot_.header, snapshot_.metadata, snapshot_.bounds,
            snapshot_.build_flags, snapshot_.admission,
            snapshot_.query_counts, snapshot_.query_candidates,
            snapshot_.query_visited_nodes, snapshot_.query_counter,
            snapshot_.query_status, snapshot_.build_output,
            snapshot_.build_scratch};
        result = simulation::build_spatial_index_device(
            native_.queue(), state_.realtime_spatial_plan.build, buffers,
            excluded, state_.rigid_body_count);
        if (result != PM_SUCCESS) return result;
        result = state_.realtime_spatial.record_device_update(
            simulation::SpatialUpdateKind::build);
        if (result != PM_SUCCESS) return result;
        const auto budget = simulation::spatial_build_dispatch_budget(
            state_.realtime_spatial_plan.leaf_capacity,
            state_.rigid_body_count);
        dispatches(simulation::RealtimeTelemetryPhase::index,
                   1U + budget.total_dispatches,
                   state_.realtime_spatial_plan.build.block_capacity +
                       budget.total_workgroups);
        if (state_.realtime_spatial_plan.build.arrivals.size != 0U) {
            auto &index_work = work(
                simulation::RealtimeTelemetryPhase::index);
            index_work.bytes_moved +=
                state_.realtime_spatial_plan.build.arrivals.size;
            ++index_work.barriers;
        }
#endif
        ++work(simulation::RealtimeTelemetryPhase::index).index_builds;
        return PM_SUCCESS;
    }

    PmResult publish_spatial(
        const simulation::RealtimeScheduleScene &) noexcept {
        auto result = PM_SUCCESS;
#if defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
        result = state_.realtime_spatial.publish(epoch_);
#else
        const simulation::SpatialDeviceBuffers buffers{
            snapshot_.header, snapshot_.metadata, snapshot_.bounds,
            snapshot_.build_flags, snapshot_.admission,
            snapshot_.query_counts, snapshot_.query_candidates,
            snapshot_.query_visited_nodes, snapshot_.query_counter,
            snapshot_.query_status, snapshot_.build_output,
            snapshot_.build_scratch};
        result = simulation::publish_spatial_epoch_device(
            native_.queue(), state_.realtime_spatial_plan.build, buffers);
        if (result == PM_SUCCESS)
            result = state_.realtime_spatial.publish_device(epoch_);
#endif
        if (result != PM_SUCCESS) return result;
        result = state_.realtime_resources.publish_spatial();
        if (result != PM_SUCCESS) return result;
        ++work(simulation::RealtimeTelemetryPhase::index)
              .index_publications;
#if !defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
        dispatch(simulation::RealtimeTelemetryPhase::index, 1U);
#endif
        auto *query_bounds = rigid<simulation::SpatialAabb>(
            state_.realtime_rigid_spatial_plan.query_bounds);
        auto *query_masks = rigid<std::uint32_t>(
            state_.realtime_rigid_spatial_plan.query_masks);
        auto *excluded = rigid<std::uint32_t>(
            state_.realtime_rigid_spatial_plan.excluded_sort_keys);
#if defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
        for (std::uint32_t index = 0U; index < state_.rigid_body_count;
             ++index) {
            result = state_.realtime_spatial.query(
                epoch_, index, query_bounds[index], query_masks[index],
                excluded[index]);
            if (result != PM_SUCCESS) return result;
        }
#else
        const simulation::SpatialDeviceBuffers query_buffers{
            snapshot_.header, snapshot_.metadata, snapshot_.bounds,
            snapshot_.build_flags, snapshot_.admission,
            snapshot_.query_counts, snapshot_.query_candidates,
            snapshot_.query_visited_nodes, snapshot_.query_counter,
            snapshot_.query_status, snapshot_.build_output,
            snapshot_.build_scratch};
        result = simulation::query_spatial_epoch_device(
            native_.queue(), state_.realtime_spatial_plan.build,
            query_buffers, query_bounds, query_masks, excluded,
            state_.rigid_body_count,
            state_.options.realtime_spatial_candidates_per_query);
        if (result == PM_SUCCESS)
            result = state_.realtime_spatial.record_device_queries(
                epoch_, state_.rigid_body_count);
        if (result != PM_SUCCESS) return result;
        if (state_.rigid_body_count != 0U)
            dispatch(simulation::RealtimeTelemetryPhase::candidates,
                     (state_.rigid_body_count + 63U) / 64U);
#endif
        if (state_.rigid_constraint_count != 0U) {
            result = native_.build_fixed_collision_groups(
                1U, 1U, native_.queue(), state_.rigid_constraints,
                state_.options.rigid_constraint_capacity, state_.ids,
                state_.rigid_body_count, state_.fixed_contact_projection);
            if (result != PM_SUCCESS) return result;
            dispatch(simulation::RealtimeTelemetryPhase::candidates, 1U);
        }
        return PM_SUCCESS;
    }

    PmResult solve_domain(
        simulation::RealtimeDomain domain, std::uint32_t iteration,
        const simulation::RealtimeScheduleScene &scene) noexcept {
        if (domain != simulation::RealtimeDomain::rigid)
            return PM_ERROR_FEATURE_NOT_PRESENT;
        ++work(simulation::RealtimeTelemetryPhase::domain)
              .numerical_iterations;
        const auto *counters = state_.realtime_spatial.counters(epoch_);
        if (counters == nullptr) return PM_ERROR_INVALID_STATE;
        if (iteration == 0U) {
            const auto budget = simulation::rigid_realtime_work_budget(
                state_.rigid_body_count, 0U,
                state_.rigid_constraint_count, state_.rigid_body_count,
                std::max(1U, state_.rigid_contact_grid_limit));
            auto result = rigid_contract_.begin(epoch_, *counters, budget);
            if (result != PM_SUCCESS) return result;
            result = emit_rigid_solve(scene);
            if (result != PM_SUCCESS) return result;
        }
        const auto result = rigid_contract_.record_sweep(epoch_, *counters);
        if (result != PM_SUCCESS) return result;
        return iteration + 1U == simulation::rigid_realtime_sweeps
            ? rigid_contract_.finish() : PM_SUCCESS;
    }

    PmResult solve_couplings(
        simulation::RealtimeCouplingFamily, std::uint32_t, std::uint32_t,
        const simulation::RealtimeScheduleScene &) noexcept {
        return PM_ERROR_FEATURE_NOT_PRESENT;
    }

    PmResult commit_tick(
        const simulation::RealtimeScheduleScene &) noexcept {
        auto result = state_.realtime_resources.prepare_commit();
        if (result != PM_SUCCESS) return result;
        if (state_.rigid_body_count != 0U) {
            constexpr std::uint32_t block = 128U;
            const auto groups =
                (state_.rigid_body_count + block - 1U) / block;
            result = native_.clear_rigid_inputs(
                groups, block, native_.queue(), state_.accumulators,
                state_.targets, state_.rigid_body_count);
            if (result != PM_SUCCESS) return result;
        } else {
            result = native_.clear(
                state_.rigid_contact_count, 0,
                sizeof(*state_.rigid_contact_count), native_.queue());
            if (result != PM_SUCCESS) return result;
        }
        const auto spatial_result = state_.realtime_spatial.commit(epoch_);
        if (spatial_result != PM_SUCCESS) return spatial_result;
        const auto snapshot_result = state_.realtime_snapshots.commit();
        if (snapshot_result != PM_SUCCESS) return snapshot_result;
        const auto resource_result = state_.realtime_resources.commit();
        if (resource_result != PM_SUCCESS) return resource_result;
        if (state_.rigid_body_count != 0U) {
            state_.current_state = output_state_;
            state_.rigid_contact_epoch = pending_contact_epoch_;
        }
        ++state_.frame_index;
        ++state_.revision;
        state_.rigid_contact_revision = state_.revision;
        return PM_SUCCESS;
    }

    PmResult abort_tick(
        const simulation::RealtimeScheduleScene &) noexcept {
        state_.realtime_spatial.abort();
        if (state_.realtime_snapshots.pending())
            (void)state_.realtime_snapshots.abort();
        if (state_.realtime_resources.pending())
            (void)state_.realtime_resources.abort();
        return PM_SUCCESS;
    }

private:
    static constexpr simulation::RealtimeTelemetryPhase telemetry_phase(
        simulation::RealtimeSchedulePhase phase) noexcept {
        switch (phase) {
        case simulation::RealtimeSchedulePhase::boundary:
            return simulation::RealtimeTelemetryPhase::boundary;
        case simulation::RealtimeSchedulePhase::predict:
            return simulation::RealtimeTelemetryPhase::predict;
        case simulation::RealtimeSchedulePhase::spatial_update:
        case simulation::RealtimeSchedulePhase::spatial_publish:
            return simulation::RealtimeTelemetryPhase::index;
        case simulation::RealtimeSchedulePhase::domain_solve:
            return simulation::RealtimeTelemetryPhase::domain;
        case simulation::RealtimeSchedulePhase::coupling_solve:
            return simulation::RealtimeTelemetryPhase::coupling;
        case simulation::RealtimeSchedulePhase::commit:
            return simulation::RealtimeTelemetryPhase::commit;
        }
        return simulation::RealtimeTelemetryPhase::boundary;
    }

    template <typename T>
    T *rigid(simulation::SpatialBufferRange range) const noexcept {
        return realtime_range<T>(state_.realtime_rigid_spatial_scratch,
                                 range);
    }

    simulation::RealtimeWorkCounters &work(
        simulation::RealtimeTelemetryPhase phase) noexcept {
        return pending_.phases[simulation::realtime_index(phase)];
    }

    void dispatch(simulation::RealtimeTelemetryPhase phase,
                  std::uint64_t groups) noexcept {
        dispatches(phase, 1U, groups);
    }

    void dispatches(simulation::RealtimeTelemetryPhase phase,
                    std::uint64_t count,
                    std::uint64_t groups) noexcept {
        auto &entry = work(phase);
        entry.dispatches += count;
        entry.workgroups += groups;
        entry.barriers += count;
    }

    void stage_commit_measurement() noexcept {
        auto &entry = work(simulation::RealtimeTelemetryPhase::commit);
        entry.submissions = 1U;
        if (state_.rigid_body_count != 0U) {
            const auto groups =
                (state_.rigid_body_count + 127U) / 128U;
            dispatch(simulation::RealtimeTelemetryPhase::commit, groups);
        } else {
            entry.bytes_moved += sizeof(*state_.rigid_contact_count);
            ++entry.barriers;
        }
    }

    PmResult emit_rigid_solve(
        const simulation::RealtimeScheduleScene &scene) noexcept {
        const auto &plan = state_.realtime_rigid_spatial_plan;
        auto *pair_flags = rigid<std::uint8_t>(plan.candidate_pair_flags);
        auto *pair_codes = rigid<std::uint32_t>(plan.candidate_pair_codes);
        auto *compacted = rigid<std::uint32_t>(plan.compacted_slots);
        auto *status = rigid<std::uint32_t>(plan.status);
        auto *active_pairs = realtime_range<std::uint32_t>(
            state_.realtime_rigid_spatial_output, plan.active_pairs);
        auto *active_count = realtime_range<std::uint32_t>(
            state_.realtime_rigid_spatial_output, plan.active_pair_count);
        constexpr std::uint32_t block = 128U;
        const auto candidate_groups =
            (plan.candidate_slot_capacity + block - 1U) / block;
        auto result = native_.filter_rigid_spatial_candidates(
            candidate_groups, block, native_.queue(), snapshot_.metadata,
            state_.realtime_spatial_plan.leaf_capacity, 0U,
            snapshot_.query_counts, snapshot_.query_candidates,
            plan.candidates_per_body, plan.candidate_slot_capacity,
            state_.parameters, state_.ids, state_.rigid_constraints,
            state_.rigid_constraint_count != 0U
                ? state_.options.rigid_constraint_capacity : 0U,
            state_.rigid_constraint_count != 0U
                ? state_.fixed_contact_projection : nullptr,
            state_.rigid_body_count, pair_codes, pair_flags, status);
        if (result != PM_SUCCESS) return result;
        dispatch(simulation::RealtimeTelemetryPhase::candidates,
                 candidate_groups);
        auto *compaction = rigid<std::uint32_t>(plan.compaction_counts);
        result = native_.compact_flagged_indices(
            native_.queue(), pair_flags, plan.candidate_slot_capacity,
            compacted, active_count, compaction,
            plan.compaction_offsets.offset + plan.compaction_offsets.size -
                plan.compaction_counts.offset);
        if (result != PM_SUCCESS) return result;
        dispatches(simulation::RealtimeTelemetryPhase::candidates, 3U,
                   2U * plan.compaction_blocks + 1U);
        result = native_.gather_rigid_spatial_pairs(
            candidate_groups, block, native_.queue(), compacted,
            active_count, pair_codes, plan.candidate_slot_capacity,
            active_pairs, status);
        if (result != PM_SUCCESS) return result;
        dispatch(simulation::RealtimeTelemetryPhase::candidates,
                 candidate_groups);

        const std::uint32_t pair_capacity =
            state_.rigid_body_count * state_.rigid_body_count;
        const std::uint32_t contact_groups = std::min(pair_capacity, 128U);
        const std::uint32_t shared_leaf_capacity =
            state_.rigid_leaf_manifolds != nullptr
                ? k_shared_rigid_leaf_capacity : 0U;
        result = native_.generate_rigid_leaf_pairs(
            contact_groups, 128U, 0U, native_.queue(), state_.parameters,
            state_.states[state_.current_state], state_.states[output_state_],
            state_.rigid_body_count, state_.meshes,
            state_.options.triangle_mesh_capacity, active_pairs,
            active_count, state_.rigid_leaf_pairs,
            state_.rigid_leaf_pair_counts,
            state_.rigid_leaf_pair_slot_capacity,
            state_.rigid_leaf_pairs_per_slot, shared_leaf_capacity);
        if (result != PM_SUCCESS) return result;
        dispatch(simulation::RealtimeTelemetryPhase::domain, contact_groups);
        const std::uint32_t blocks_per_pair =
            state_.rigid_leaf_manifolds != nullptr
                ? k_rigid_leaf_blocks_per_pair : 1U;
        auto *leaf_manifolds = state_.rigid_leaf_manifolds;
        result = native_.finalize_rigid_contact_manifolds(
            contact_groups, block, native_.queue(), state_.parameters,
            state_.states[state_.current_state], state_.states[output_state_],
            state_.rigid_body_count, state_.meshes, state_.ids,
            state_.rigid_constraints,
            state_.rigid_constraint_count != 0U
                ? state_.options.rigid_constraint_capacity : 0U,
            active_pairs, active_count, state_.rigid_leaf_pair_counts,
            scene.timestep, state_.rigid_manifolds, true);
        if (result != PM_SUCCESS) return result;
        dispatch(simulation::RealtimeTelemetryPhase::domain, contact_groups);
        result = native_.evaluate_rigid_leaf_pairs(
            contact_groups * blocks_per_pair, 16U, 0U, native_.queue(),
            state_.parameters, state_.states[state_.current_state],
            state_.states[output_state_], state_.rigid_body_count,
            state_.meshes, state_.ids, state_.rigid_constraints,
            state_.rigid_constraint_count != 0U
                ? state_.options.rigid_constraint_capacity : 0U,
            active_pairs, active_count, state_.rigid_leaf_pairs,
            state_.rigid_leaf_pair_counts, state_.rigid_leaf_pairs_per_slot,
            leaf_manifolds, blocks_per_pair, scene.timestep, false,
            state_.rigid_manifolds);
        if (result != PM_SUCCESS) return result;
        dispatch(simulation::RealtimeTelemetryPhase::domain,
                 std::uint64_t(contact_groups) * blocks_per_pair);
        if (leaf_manifolds != nullptr) {
            result = native_.reduce_rigid_leaf_manifolds(
                contact_groups, block, native_.queue(), state_.parameters,
                state_.rigid_body_count, active_pairs, active_count,
                state_.rigid_leaf_pair_counts,
                state_.rigid_leaf_pairs_per_slot, leaf_manifolds, false,
                state_.rigid_manifolds);
            if (result != PM_SUCCESS) return result;
            dispatch(simulation::RealtimeTelemetryPhase::domain,
                     contact_groups);
        }
        result = native_.finalize_rigid_contact_manifolds(
            contact_groups, block, native_.queue(), state_.parameters,
            state_.states[state_.current_state], state_.states[output_state_],
            state_.rigid_body_count, state_.meshes, state_.ids,
            state_.rigid_constraints,
            state_.rigid_constraint_count != 0U
                ? state_.options.rigid_constraint_capacity : 0U,
            active_pairs, active_count, state_.rigid_leaf_pair_counts,
            scene.timestep, state_.rigid_manifolds, false);
        if (result != PM_SUCCESS) return result;
        dispatch(simulation::RealtimeTelemetryPhase::domain, contact_groups);

        if (state_.rigid_contact_revision != state_.revision) {
            result = native_.clear(
                state_.rigid_contact_cache_slots, 0xff,
                std::size_t(pair_capacity) *
                    sizeof(*state_.rigid_contact_cache_slots),
                native_.queue());
            if (result != PM_SUCCESS) return result;
            work(simulation::RealtimeTelemetryPhase::domain).bytes_moved +=
                std::size_t(pair_capacity) *
                sizeof(*state_.rigid_contact_cache_slots);
        }
        result = native_.prepare_parallel_contact_events(
            1U, 1U, native_.queue(), state_.rigid_body_count,
            state_.rigid_manifolds, active_pairs, active_count, state_.ids,
            state_.rigid_contact_event_offsets,
            state_.rigid_contact_events, state_.rigid_contact_capacity,
            state_.rigid_contact_count, options_.collect_rigid_contacts,
            true);
        if (result != PM_SUCCESS) return result;
        dispatch(simulation::RealtimeTelemetryPhase::domain, 1U);
        result = native_.load_rigid_contact_cache(
            contact_groups, block, native_.queue(), state_.rigid_manifolds,
            active_pairs, active_count, state_.states[output_state_],
            state_.ids, state_.rigid_body_count, state_.rigid_contact_cache,
            state_.rigid_contact_cache_slots,
            state_.rigid_leaf_pair_slot_capacity, pending_contact_epoch_,
            scene.timestep);
        if (result != PM_SUCCESS) return result;
        dispatch(simulation::RealtimeTelemetryPhase::domain, contact_groups);
        const auto avbd_grid = std::min(
            state_.rigid_contact_grid_limit,
            std::max(1U, (state_.rigid_body_count + 31U) / 32U));
        const auto constraint_capacity = state_.rigid_constraint_count != 0U
            ? state_.options.rigid_constraint_capacity : 0U;
        result = native_.prepare_geometry_constraints(
            native_.queue(), state_.parameters,
            state_.states[state_.current_state], state_.states[output_state_],
            state_.rigid_body_count, state_.rigid_manifolds, active_pairs,
            active_count, state_.rigid_constraints, constraint_capacity,
            state_.ids, state_.avbd_bodies, state_.rigid_contact_schedule,
            scene.timestep, options_.gravity,
            options_.rigid_contact_iterations,
            options_.rigid_contact_pass_limit,
            simulation::rigid_realtime_profile_version, avbd_grid);
        if (result != PM_SUCCESS) return result;
        dispatch(simulation::RealtimeTelemetryPhase::domain,
            native_.realtime_geometry_prepare_workgroups(avbd_grid));
        result = native_.solve_geometry_constraints_fixed_realtime(
            native_.queue(), state_.parameters,
            state_.states[state_.current_state], state_.states[output_state_],
            state_.rigid_body_count, state_.rigid_manifolds, active_pairs,
            active_count, state_.rigid_constraints, constraint_capacity,
            state_.avbd_bodies, state_.rigid_contact_schedule,
            scene.timestep, avbd_grid);
        if (result != PM_SUCCESS) return result;
        dispatches(simulation::RealtimeTelemetryPhase::domain,
            native_.realtime_geometry_solve_dispatches(
                state_.rigid_body_count),
            native_.realtime_geometry_solve_workgroups(
                state_.rigid_body_count, avbd_grid));
        if (!native_.geometry_constraints_inline_speed_clamp(
                state_.rigid_body_count, constraint_capacity, avbd_grid)) {
            const auto body_groups =
                (state_.rigid_body_count + block - 1U) / block;
            result = native_.clamp_rigid_speeds(
                body_groups, block, native_.queue(), state_.parameters,
                state_.states[output_state_], state_.rigid_body_count);
            if (result != PM_SUCCESS) return result;
            dispatch(simulation::RealtimeTelemetryPhase::domain,
                     body_groups);
        }
        result = native_.save_rigid_contact_cache(
            contact_groups, block, native_.queue(), state_.rigid_manifolds,
            active_pairs, active_count, state_.states[output_state_],
            state_.ids, state_.rigid_body_count, state_.rigid_contact_cache,
            state_.rigid_contact_cache_slots,
            state_.rigid_leaf_pair_slot_capacity, pending_contact_epoch_,
            scene.timestep, state_.rigid_contact_event_offsets,
            state_.rigid_contact_events, state_.rigid_contact_capacity);
        if (result == PM_SUCCESS)
            dispatch(simulation::RealtimeTelemetryPhase::domain,
                     contact_groups);
        return result;
    }

    State &state_;
    NativeScheduleBackend &native_;
    StepOptions options_{};
    simulation::RealtimeOperationMeasurement pending_{};
    std::array<std::uint64_t, simulation::realtime_telemetry_phase_count>
        cpu_encode_ns_{};
    std::chrono::steady_clock::time_point measurement_started_{};
    simulation::ProductionSpatialSnapshot snapshot_{};
    simulation::SpatialEpochToken epoch_{};
    simulation::RigidRealtimeTickContract rigid_contract_{};
    std::uint32_t output_state_{};
    std::uint64_t pending_contact_epoch_{};
};

template <typename State>
Status schedule_native_realtime(State &state, NativeScheduleBackend &native,
                                StepOptions options) noexcept {
    const auto scene = native_realtime_scene(state, options);
    const auto defaults = native_realtime_limits(state);
    const auto &limits = options.realtime_telemetry_limits != nullptr
        ? *options.realtime_telemetry_limits : defaults;
    if (!native_realtime_limits_admit_minimum(state, limits)) {
        state.realtime_ledger = {};
        state.realtime_ledger.failed = true;
        state.realtime_telemetry = {};
        state.realtime_telemetry.tick = scene.tick;
        if (options.realtime_telemetry_output != nullptr)
            *options.realtime_telemetry_output = state.realtime_telemetry;
        return backend_failure(PM_ERROR_CAPACITY_EXCEEDED,
            "native realtime telemetry limit is below mandatory work");
    }
    NativeRealtimeBackend<State> backend(state, native, options);
    const auto result = simulation::assemble_realtime_tick_measured(
        backend, scene, state.realtime_ledger, limits,
        state.realtime_telemetry);
    if (options.realtime_telemetry_output != nullptr)
        *options.realtime_telemetry_output = state.realtime_telemetry;
    return result == PM_SUCCESS ? success()
        : backend_failure(result, "native realtime schedule failed");
}

} // namespace

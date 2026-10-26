// SPDX-License-Identifier: MIT
// Included by runtime.cpp after NativeScheduleBackend is defined.
namespace {

inline constexpr std::uint32_t smoke_realtime_owner_tag = 0x80000000U;
inline constexpr std::uint32_t smoke_realtime_spatial_mask = 1U << 4U;
inline constexpr std::uint32_t smoke_realtime_candidates_per_particle = 64U;

template <typename T>
T *realtime_range(void *base,
                  simulation::SpatialBufferRange range) noexcept {
    return reinterpret_cast<T *>(reinterpret_cast<std::byte *>(base) +
                                 range.offset);
}

template <typename State>
PmResult native_realtime_spatial_counts(
    const State &state, std::uint32_t &leaves,
    std::uint32_t &queries) noexcept {
    std::uint64_t total = state.rigid_body_count;
    for (const auto &owner : state.fluids) {
        if (!owner || !owner->alive) continue;
        total += owner->options.capacity;
        if (total > std::numeric_limits<std::uint32_t>::max())
            return PM_ERROR_CAPACITY_EXCEEDED;
    }
    for (const auto &owner : state.cloths) {
        if (!owner || !owner->alive) continue;
        total += owner->index_count / 3U;
        if (total > std::numeric_limits<std::uint32_t>::max())
            return PM_ERROR_CAPACITY_EXCEEDED;
    }
    for (const auto &owner : state.soft_bodies) {
        if (!owner || !owner->alive) continue;
        total += owner->surface_index_count / 3U;
        if (total > std::numeric_limits<std::uint32_t>::max())
            return PM_ERROR_CAPACITY_EXCEEDED;
    }
    for (const auto &owner : state.ropes) {
        if (!owner || !owner->alive) continue;
        total += owner->data.count - 1U;
        if (total > std::numeric_limits<std::uint32_t>::max())
            return PM_ERROR_CAPACITY_EXCEEDED;
    }
    for (const auto &owner : state.smokes) {
        if (!owner || !owner->alive) continue;
        total += owner->options.capacity;
        if (total > std::numeric_limits<std::uint32_t>::max())
            return PM_ERROR_CAPACITY_EXCEEDED;
    }
    leaves = static_cast<std::uint32_t>(total);
    queries = leaves;
    return PM_SUCCESS;
}

template <typename State>
PmResult native_realtime_coupling_counts(
    const State &state,
    std::array<std::uint32_t,
               simulation::realtime_coupling_family_count> &counts) noexcept {
    counts = {};
    const auto add = [&](simulation::RealtimeCouplingFamily family,
                         std::uint32_t amount = 1U) noexcept -> PmResult {
        auto &count = counts[simulation::realtime_index(family)];
        if (amount > simulation::rigid_coupling_realtime_maximum_edges - count)
            return PM_ERROR_CAPACITY_EXCEEDED;
        count += amount;
        return PM_SUCCESS;
    };
    const auto add_liquid = [&](std::uint32_t amount = 1U) noexcept {
        return add(simulation::RealtimeCouplingFamily::liquid, amount);
    };
    for (const auto &owner : state.fluid_pair_couplings) {
        if (!owner || !owner->alive || !owner->options.enabled) continue;
        if (owner->options.material_profile !=
            FluidPairMaterialProfile::equal)
            return PM_ERROR_FEATURE_NOT_PRESENT;
        if (add_liquid(3U) != PM_SUCCESS)
            return PM_ERROR_CAPACITY_EXCEEDED;
    }
    if (state.rigid_body_count != 0U) {
        for (const auto &owner : state.fluids) {
            if (!owner || !owner->alive) continue;
            if (add_liquid(2U) != PM_SUCCESS)
                return PM_ERROR_CAPACITY_EXCEEDED;
        }
    }
    for (const auto &coupling : state.fluid_cloth_couplings) {
        if (!coupling.alive || !coupling.options.enabled) continue;
        const auto modes = std::uint32_t(coupling.options.contact) +
            std::uint32_t(coupling.options.containment);
        if (modes != 0U && add_liquid(modes) != PM_SUCCESS)
            return PM_ERROR_CAPACITY_EXCEEDED;
    }
    for (const auto &owner : state.fluid_soft_couplings)
        if (owner && owner->alive && owner->options.enabled &&
            add_liquid() != PM_SUCCESS)
            return PM_ERROR_CAPACITY_EXCEEDED;
    for (const auto &owner : state.fluid_rope_couplings)
        if (owner && owner->alive && owner->options.enabled &&
            add_liquid() != PM_SUCCESS)
            return PM_ERROR_CAPACITY_EXCEEDED;
    for (const auto &owner : state.fluid_smoke_couplings)
        if (owner && owner->alive && add_liquid(2U) != PM_SUCCESS)
            return PM_ERROR_CAPACITY_EXCEEDED;
    for (const auto &owner : state.smoke_rigid_couplings)
        if (owner && owner->alive && owner->options.enabled &&
            add(simulation::RealtimeCouplingFamily::rigid, 2U) != PM_SUCCESS)
            return PM_ERROR_CAPACITY_EXCEEDED;
    if (state.rigid_body_count != 0U) {
        for (const auto &owner : state.cloths)
            if (owner && owner->alive &&
                add(simulation::RealtimeCouplingFamily::rigid) != PM_SUCCESS)
                return PM_ERROR_CAPACITY_EXCEEDED;
        for (const auto &owner : state.soft_bodies)
            if (owner && owner->alive &&
                add(simulation::RealtimeCouplingFamily::rigid) != PM_SUCCESS)
                return PM_ERROR_CAPACITY_EXCEEDED;
        for (const auto &owner : state.ropes) {
            if (!owner || !owner->alive) continue;
            if (add(simulation::RealtimeCouplingFamily::rigid) != PM_SUCCESS)
                return PM_ERROR_CAPACITY_EXCEEDED;
            const auto &options = owner->data.options;
            const std::uint32_t attachments =
                std::uint32_t(options.first.enabled) +
                std::uint32_t(options.last.enabled);
            if (attachments != 0U &&
                add(simulation::RealtimeCouplingFamily::rigid,
                    attachments) != PM_SUCCESS)
                return PM_ERROR_CAPACITY_EXCEEDED;
        }
    }
    for (const auto &owner : state.soft_cloth_couplings)
        if (owner && owner->alive && owner->options.enabled &&
            add(simulation::RealtimeCouplingFamily::deformable_smoke) !=
                PM_SUCCESS)
            return PM_ERROR_CAPACITY_EXCEEDED;
    for (const auto &coupling : state.rope_cloth_couplings)
        if (coupling.alive && coupling.options.enabled &&
            add(simulation::RealtimeCouplingFamily::deformable_smoke) !=
                PM_SUCCESS)
            return PM_ERROR_CAPACITY_EXCEEDED;
    for (const auto &owner : state.rope_soft_body_couplings) {
        if (!owner || !owner->alive || !owner->options.enabled) continue;
        if (add(simulation::RealtimeCouplingFamily::deformable_smoke) !=
            PM_SUCCESS)
            return PM_ERROR_CAPACITY_EXCEEDED;
        if ((owner->options.attach_first || owner->options.attach_last) &&
            add(simulation::RealtimeCouplingFamily::deformable_smoke) !=
                PM_SUCCESS)
            return PM_ERROR_CAPACITY_EXCEEDED;
    }
    const auto add_smoke_deformable = [&](const auto &couplings) noexcept {
        for (const auto &owner : couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            if (add(simulation::RealtimeCouplingFamily::deformable_smoke,
                    2U) != PM_SUCCESS)
                return PM_ERROR_CAPACITY_EXCEEDED;
        }
        return PM_SUCCESS;
    };
    if (add_smoke_deformable(state.smoke_cloth_couplings) != PM_SUCCESS ||
        add_smoke_deformable(state.smoke_soft_body_couplings) != PM_SUCCESS ||
        add_smoke_deformable(state.smoke_rope_couplings) != PM_SUCCESS)
        return PM_ERROR_CAPACITY_EXCEEDED;
    return PM_SUCCESS;
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
    for (const auto &owner : state.smokes) {
        if (!owner || !owner->alive) continue;
        scene.smoke_representation = owner->options.grid_resolution == 0U
            ? simulation::SmokeRealtimeRepresentation::particles
            : simulation::SmokeRealtimeRepresentation::grid;
        break;
    }
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
    std::array<std::uint32_t,
               simulation::realtime_coupling_family_count> coupling_counts{};
    (void)native_realtime_coupling_counts(state, coupling_counts);
    scene.coupling_edges = coupling_counts;
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
    const std::uint32_t coupling_phase = simulation::realtime_phase_bit(
        simulation::RealtimeResourcePhase::coupling);
    auto &rigid = result[simulation::realtime_index(
        simulation::RealtimeResourceFamily::rigid)];
    for (std::uint32_t slot = 0U; slot < state.slots.size(); ++slot) {
        const auto &owner = state.slots[slot];
        if (!owner.alive) continue;
        const auto status = add_realtime_registration(
            rigid, slot, owner.generation, 1U,
            common_phases |
                simulation::realtime_domain_execution_phase_mask |
                coupling_phase);
        if (status != PM_SUCCESS) return status;
    }

    auto add_owner = [&](simulation::RealtimeResourceFamily family,
                         std::uint32_t slot, std::uint32_t generation,
                         std::uint64_t elements) noexcept {
        return add_realtime_registration(
            result[simulation::realtime_index(family)], slot, generation,
            elements, common_phases |
                simulation::realtime_domain_execution_phase_mask);
    };
    for (std::uint32_t slot = 0U; slot < state.fluids.size(); ++slot) {
        const auto &owner = state.fluids[slot];
        if (!owner || !owner->alive) continue;
        const auto status = add_realtime_registration(
            result[simulation::realtime_index(
                simulation::RealtimeResourceFamily::liquid)],
            slot, owner->generation, owner->options.capacity,
            common_phases |
                simulation::realtime_domain_execution_phase_mask);
        if (status != PM_SUCCESS) return status;
    }
    for (std::uint32_t slot = 0U; slot < state.cloths.size(); ++slot) {
        const auto &owner = state.cloths[slot];
        if (!owner || !owner->alive) continue;
        const std::uint64_t elements = owner->vertex_capacity +
            std::uint64_t{owner->index_count} / 3U;
        const auto status = add_realtime_registration(
            result[simulation::realtime_index(
                simulation::RealtimeResourceFamily::cloth)],
            slot, owner->generation, elements,
            common_phases |
                simulation::realtime_domain_execution_phase_mask |
                coupling_phase);
        if (status != PM_SUCCESS) return status;
    }
    for (std::uint32_t slot = 0U; slot < state.soft_bodies.size(); ++slot) {
        const auto &owner = state.soft_bodies[slot];
        if (!owner || !owner->alive) continue;
        const std::uint64_t elements = owner->node_count +
            std::uint64_t{owner->surface_vertex_count} +
            std::uint64_t{owner->surface_index_count} / 3U;
        const auto status = add_realtime_registration(
            result[simulation::realtime_index(
                simulation::RealtimeResourceFamily::soft_body)],
            slot, owner->generation, elements,
            common_phases |
                simulation::realtime_domain_execution_phase_mask |
                coupling_phase);
        if (status != PM_SUCCESS) return status;
    }
    for (std::uint32_t slot = 0U; slot < state.ropes.size(); ++slot) {
        const auto &owner = state.ropes[slot];
        if (!owner || !owner->alive) continue;
        const std::uint64_t elements = owner->data.count +
            std::uint64_t{owner->data.count - 1U};
        const auto status = add_realtime_registration(
            result[simulation::realtime_index(
                simulation::RealtimeResourceFamily::rope)],
            slot, owner->generation, elements,
            common_phases |
                simulation::realtime_domain_execution_phase_mask |
                coupling_phase);
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
    auto &coupling = limits.phase_maximum[simulation::realtime_index(
        simulation::RealtimeTelemetryPhase::coupling)];
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
    std::uint32_t staged_leaves{};
    std::uint32_t staged_queries{};
    if (native_realtime_spatial_counts(
            state, staged_leaves, staged_queries) != PM_SUCCESS)
        return limits;
    std::array<std::uint32_t,
               simulation::realtime_coupling_family_count> coupling_counts{};
    if (native_realtime_coupling_counts(state, coupling_counts) != PM_SUCCESS)
        return limits;
    const auto liquid_couplings = coupling_counts[simulation::realtime_index(
        simulation::RealtimeCouplingFamily::liquid)];
    const auto rigid_couplings = coupling_counts[simulation::realtime_index(
        simulation::RealtimeCouplingFamily::rigid)];
    const auto deformable_couplings = coupling_counts[
        simulation::realtime_index(
            simulation::RealtimeCouplingFamily::deformable_smoke)];
    commit.submissions = 1U;
    index.index_publications = 1U;
    if (staged_leaves == 0U) {
#if !defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
        dispatch(boundary, 1U, 1U);
        index.bytes_moved = sizeof(std::uint32_t);
        ++index.barriers;
        dispatch(index, 1U, 1U);
#endif
        commit.bytes_moved = sizeof(std::uint32_t);
        ++commit.barriers;
    } else {
        boundary.bytes_moved = sizeof(std::uint32_t);
        boundary.barriers = 1U;
#if !defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
        dispatch(boundary, 1U, 1U);
#endif
        if (rigid != 0U) {
            const std::uint64_t body_groups = (rigid + 127U) / 128U;
            boundary.bytes_moved +=
                std::uint64_t{rigid} * sizeof(RigidBodyState);
            ++boundary.barriers;
            dispatch(predict, 1U, body_groups);
            dispatch(index, 1U, body_groups);
        }
        for (const auto &owner : state.fluids) {
            if (!owner || !owner->alive) continue;
            const auto groups =
                (std::uint64_t{owner->options.capacity} + 127U) / 128U;
            boundary.bytes_moved +=
                2ULL * owner->options.capacity * sizeof(Vec3) +
                sizeof(std::uint32_t);
            boundary.barriers += 3U;
            dispatch(predict, 1U, groups);
            dispatch(index, 1U, groups);
            dispatch(candidates, 1U, groups);
            dispatch(domain,
                3U * simulation::fluid_realtime_density_sweeps + 1U,
                groups *
                    (3U * simulation::fluid_realtime_density_sweeps + 1U));
            dispatch(commit, 1U, groups);
        }
        for (const auto &owner : state.cloths) {
            if (!owner || !owner->alive) continue;
            const auto vertex_groups =
                (std::uint64_t{owner->vertex_capacity} + 127U) / 128U;
            const auto triangle_groups =
                (std::uint64_t{owner->index_count / 3U} + 127U) / 128U;
            boundary.bytes_moved +=
                2ULL * owner->vertex_capacity * sizeof(Vec3) +
                owner->vertex_capacity * sizeof(std::uint8_t) +
                sizeof(std::uint32_t) +
                owner->vertex_capacity * sizeof(Vec3);
            boundary.barriers += 5U;
            if (owner->rope_forces != nullptr) {
                boundary.bytes_moved +=
                    owner->vertex_capacity * sizeof(Vec3);
                ++boundary.barriers;
            }
            dispatch(predict, 1U, vertex_groups);
            dispatch(index, 1U, triangle_groups);
            dispatch(domain,
                3U * simulation::cloth_realtime_constraint_sweeps +
                    simulation::cloth_realtime_contact_passes,
                vertex_groups *
                        (2U * simulation::cloth_realtime_constraint_sweeps +
                         simulation::cloth_realtime_contact_passes) +
                    simulation::cloth_realtime_constraint_sweeps);
            dispatch(commit, 2U, vertex_groups + triangle_groups);
        }
        for (const auto &owner : state.soft_bodies) {
            if (!owner || !owner->alive) continue;
            const auto node_groups =
                (std::uint64_t{owner->node_count} + 127U) / 128U;
            const auto surface_groups =
                (std::uint64_t{owner->surface_vertex_count} + 127U) / 128U;
            const auto triangle_groups =
                (std::uint64_t{owner->surface_index_count / 3U} + 127U) /
                128U;
            boundary.bytes_moved +=
                2ULL * owner->node_count * sizeof(Vec3) +
                owner->node_count * sizeof(std::uint8_t) +
                sizeof(Quaternion) + sizeof(std::uint32_t) +
                4ULL * owner->node_count * sizeof(Vec3) +
                owner->node_count * sizeof(float) +
                2U * sizeof(std::uint32_t);
            boundary.barriers += 12U;
            dispatch(predict, 2U, node_groups + surface_groups);
            dispatch(index, 1U, triangle_groups);
            dispatch(domain,
                3U * simulation::soft_body_realtime_constraint_sweeps +
                    simulation::soft_body_realtime_contact_passes,
                node_groups *
                        (2U * simulation::soft_body_realtime_constraint_sweeps +
                         simulation::soft_body_realtime_contact_passes) +
                    simulation::soft_body_realtime_constraint_sweeps);
            dispatch(commit, 2U, node_groups + surface_groups);
        }
        for (const auto &owner : state.ropes) {
            if (!owner || !owner->alive) continue;
            const auto nodes = owner->data.count;
            const auto node_groups =
                (std::uint64_t{nodes} + 127U) / 128U;
            const auto segment_groups =
                (std::uint64_t{nodes - 1U} + 127U) / 128U;
            boundary.bytes_moved +=
                2ULL * nodes * sizeof(Vec3) +
                nodes * sizeof(std::uint8_t) +
                2U * sizeof(std::uint32_t);
            boundary.barriers += 5U;
            dispatch(predict, 1U, node_groups);
            dispatch(index, 1U, segment_groups);
            dispatch(domain,
                2U * simulation::rope_realtime_constraint_sweeps +
                    simulation::rope_realtime_contact_passes,
                simulation::rope_realtime_constraint_sweeps +
                    node_groups *
                        (simulation::rope_realtime_constraint_sweeps +
                         simulation::rope_realtime_contact_passes));
            dispatch(commit, 1U, node_groups);
        }
        const bool has_particle_smoke = std::any_of(
            state.smokes.begin(), state.smokes.end(),
            [](const auto &owner) {
                return owner && owner->alive &&
                    owner->options.grid_resolution == 0U;
            });
        if (has_particle_smoke) {
            boundary.bytes_moved +=
                std::uint64_t{state.options.realtime_spatial_leaf_capacity} *
                sizeof(std::uint32_t);
            ++boundary.barriers;
        }
        for (const auto &owner : state.smokes) {
            if (!owner || !owner->alive) continue;
            const auto particle_groups =
                (std::uint64_t{owner->options.capacity} + 127U) / 128U;
            const auto particle_bytes =
                std::uint64_t{owner->options.capacity};
            boundary.bytes_moved +=
                particle_bytes * (4U * sizeof(Vec3) + 4U * sizeof(float)) +
                2U * sizeof(std::uint32_t);
            boundary.barriers += 10U;
            dispatch(index, 1U, particle_groups);
            if (owner->options.grid_resolution == 0U) {
                const auto map_count =
                    std::uint64_t{owner->options.capacity} *
                    state.options.realtime_spatial_candidates_per_query;
                const auto compact_count =
                    std::uint64_t{owner->options.capacity} *
                    smoke_realtime_candidates_per_particle;
                const auto map_groups = (map_count + 127U) / 128U;
                const auto compact_groups = (compact_count + 127U) / 128U;
                boundary.bytes_moved +=
                    compact_count * sizeof(std::uint32_t);
                ++boundary.barriers;
                candidates.bytes_moved +=
                    (map_count + owner->options.capacity) *
                        sizeof(std::uint32_t) +
                    2U * sizeof(std::uint32_t);
                candidates.barriers += 4U;
                dispatch(candidates, 3U,
                    particle_groups + map_groups + compact_groups);
                dispatch(predict, 2U, 2U * particle_groups);
                dispatch(domain,
                    3U * simulation::smoke_realtime_particle_sweeps,
                    3U * simulation::smoke_realtime_particle_sweeps *
                        particle_groups);
                dispatch(commit, 1U, particle_groups);
            } else {
                const auto &grid = owner->grid;
                const auto cell_groups =
                    (std::uint64_t{grid.cell_count} + 127U) / 128U;
                std::uint64_t face_groups = 0U;
                std::uint64_t face_bytes = 0U;
                for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
                    face_groups +=
                        (std::uint64_t{grid.face_count[axis]} + 127U) / 128U;
                    face_bytes += std::uint64_t{grid.face_count[axis]} *
                        sizeof(float);
                }
                boundary.bytes_moved +=
                    2ULL * grid.cell_count * sizeof(float) + face_bytes;
                boundary.barriers += 5U;
                dispatch(predict, 14U, 5U * cell_groups +
                    3U * face_groups);
                dispatch(domain,
                    simulation::smoke_realtime_grid_pressure_cycles,
                    simulation::smoke_realtime_grid_pressure_cycles *
                        cell_groups);
                dispatch(commit, 4U, face_groups + particle_groups);
            }
        }
        index.index_builds = 1U;
#if !defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
        const auto spatial = simulation::spatial_build_dispatch_budget(
            state.realtime_spatial_plan.leaf_capacity, staged_leaves);
        dispatch(index, 1U + spatial.total_dispatches,
            state.realtime_spatial_plan.build.block_capacity +
                spatial.total_workgroups);
        if (state.realtime_spatial_plan.build.arrivals.size != 0U) {
            index.bytes_moved +=
                state.realtime_spatial_plan.build.arrivals.size;
            ++index.barriers;
        }
        dispatch(index, 1U, 1U);
        dispatch(candidates, 1U, (staged_queries + 63U) / 64U);
#endif
        if (rigid != 0U) {
            const auto body_groups = (rigid + 127U) / 128U;
            if (state.rigid_constraint_count != 0U)
                dispatch(candidates, 1U, 1U);
            const auto candidate_groups =
                (std::uint64_t{state.realtime_rigid_spatial_plan
                                   .candidate_slot_capacity} + 127U) / 128U;
            dispatch(candidates, 1U, candidate_groups);
            dispatch(candidates, 3U,
                2ULL * state.realtime_rigid_spatial_plan.compaction_blocks +
                    1U);
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
            dispatch(domain, 1U, contact_groups);
            dispatch(domain, 1U, contact_groups);
            dispatch(domain, 1U, contact_groups * leaf_blocks);
            if (reduces_leaf_manifolds)
                dispatch(domain, 1U, contact_groups);
            dispatch(domain, 1U, contact_groups);
            dispatch(domain, 1U, 1U);
            dispatch(domain, 1U, contact_groups);
            dispatch(domain, 1U,
                NativeScheduleBackend::realtime_geometry_prepare_workgroups(
                    avbd_grid));
            dispatch(domain,
                NativeScheduleBackend::realtime_geometry_solve_dispatches(
                    rigid),
                NativeScheduleBackend::realtime_geometry_solve_workgroups(
                    rigid, avbd_grid));
            if (!NativeScheduleBackend::geometry_constraints_inline_speed_clamp(
                    rigid, constraint_capacity, avbd_grid))
                dispatch(domain, 1U, body_groups);
            dispatch(domain, 1U, contact_groups);
            if (state.rigid_contact_revision != state.revision) {
                domain.bytes_moved += std::uint64_t{rigid} * rigid *
                    sizeof(*state.rigid_contact_cache_slots);
                ++domain.barriers;
            }
            dispatch(commit, 1U, body_groups);
        } else {
            commit.bytes_moved += sizeof(std::uint32_t);
            ++commit.barriers;
        }
        if (rigid_couplings != 0U || deformable_couplings != 0U) {
            constexpr std::uint64_t records = 64U;
            boundary.bytes_moved += records *
                (6U * sizeof(std::uint32_t) + 2U * sizeof(Vec3) +
                 3U * sizeof(float) + sizeof(std::uint8_t));
            boundary.bytes_moved += records *
                (4U * sizeof(Vec3) + 2U * sizeof(float));
            boundary.barriers += 18U;
            if (rigid != 0U) {
                boundary.bytes_moved += std::uint64_t{rigid} *
                    (sizeof(std::uint32_t) + sizeof(float));
                boundary.barriers += 2U;
            }
        }
        if (liquid_couplings != 0U) {
            constexpr std::uint64_t sweeps =
                simulation::liquid_coupling_realtime_sweeps;
            for (const auto &owner : state.fluid_pair_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled)
                    continue;
                const auto groups =
                    (std::uint64_t{owner->index.total_capacity} + 127U) /
                    128U;
                dispatch(coupling, 2U + 2U * sweeps,
                    2U * groups + 2U * sweeps * groups);
            }
            if (rigid != 0U) {
                const auto body_groups =
                    (std::uint64_t{rigid} + 127U) / 128U;
                const auto body_words =
                    (std::uint64_t{state.options.rigid_body_capacity} + 63U) /
                    64U;
                dispatch(coupling, 2U * sweeps,
                    2U * sweeps * body_groups);
                coupling.bytes_moved += sweeps *
                    (NativeScheduleBackend::fluid_body_buckets + 1ULL) *
                    body_words * sizeof(unsigned long long);
                coupling.barriers += 2U * sweeps;
                for (const auto &owner : state.fluids) {
                    if (!owner || !owner->alive) continue;
                    const auto groups =
                        (std::uint64_t{owner->options.capacity} + 127U) /
                        128U;
                    dispatch(coupling, 3U * sweeps,
                        sweeps * (2U * groups + rigid));
                    coupling.bytes_moved += sweeps * std::uint64_t{rigid} *
                        sizeof(std::uint32_t);
                    coupling.barriers += sweeps;
                }
            }
            for (const auto &item : state.fluid_cloth_couplings) {
                if (!item.alive || !item.options.enabled) continue;
                const auto &fluid = *state.fluids[item.options.fluid.index];
                const auto &cloth = *state.cloths[item.options.cloth.index];
                const auto fluid_groups =
                    (std::uint64_t{fluid.options.capacity} + 127U) / 128U;
                const auto cloth_groups =
                    (std::uint64_t{cloth.vertex_count} + 127U) / 128U;
                if (item.options.contact) {
                    dispatch(coupling, 2U * sweeps,
                        sweeps * (fluid_groups + cloth_groups));
                    coupling.bytes_moved += sweeps * cloth.vertex_count *
                        sizeof(Vec3);
                    coupling.barriers += sweeps;
                }
                if (item.options.containment)
                    dispatch(coupling, sweeps, sweeps * fluid_groups);
            }
            for (const auto &owner : state.fluid_rope_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled)
                    continue;
                const auto &fluid = *state.fluids[owner->options.fluid.index];
                const auto &rope = *state.ropes[owner->options.rope.index];
                const auto fluid_groups =
                    (std::uint64_t{fluid.options.capacity} + 127U) / 128U;
                const auto rope_groups =
                    (std::uint64_t{rope.data.count} + 127U) / 128U;
                dispatch(coupling, 2U * sweeps,
                    sweeps * (fluid_groups + rope_groups));
                coupling.bytes_moved += sweeps *
                    (sizeof(std::uint32_t) + sizeof(float));
                coupling.barriers += 2U * sweeps;
            }
            for (const auto &owner : state.fluid_soft_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled)
                    continue;
                const auto &fluid = *state.fluids[owner->options.fluid.index];
                const auto &body =
                    *state.soft_bodies[owner->options.soft_body.index];
                const auto fluid_groups =
                    (std::uint64_t{fluid.options.capacity} + 127U) / 128U;
                const auto node_groups =
                    (std::uint64_t{body.node_count} + 127U) / 128U;
                const auto surface_groups =
                    (std::uint64_t{body.surface_vertex_count} + 127U) / 128U;
                const auto tree_groups =
                    (std::uint64_t{owner->tree_count} + 127U) / 128U;
                dispatch(coupling, 5U * sweeps, sweeps *
                    (2U * fluid_groups + node_groups + surface_groups +
                     tree_groups));
                coupling.bytes_moved += sweeps *
                    (std::uint64_t{body.node_count} *
                         (sizeof(std::uint32_t) + 2U * sizeof(Vec3)) +
                     std::uint64_t{owner->tree_count} *
                         sizeof(std::uint32_t));
                coupling.barriers += 4U * sweeps;
            }
            for (const auto &owner : state.fluid_smoke_couplings) {
                if (!owner || !owner->alive) continue;
                const auto &fluid = *state.fluids[owner->options.fluid.index];
                const auto groups =
                    (std::uint64_t{fluid.options.capacity} + 127U) / 128U;
                dispatch(coupling, 2U * sweeps, 2U * sweeps * groups);
                coupling.bytes_moved += sweeps * sizeof(*owner->converted);
                coupling.barriers += sweeps;
            }
            coupling.numerical_iterations += sweeps;
        }
        if (rigid_couplings != 0U) {
            const auto body_groups = (rigid + 127U) / 128U;
            for (const auto &owner : state.cloths) {
                if (!owner || !owner->alive) continue;
                const auto vertex_groups =
                    (std::uint64_t{owner->vertex_count} + 127U) / 128U;
                const auto triangle_groups =
                    (std::uint64_t{owner->index_count / 3U} + 127U) / 128U;
                dispatch(coupling,
                    5U * simulation::rigid_coupling_realtime_sweeps,
                    simulation::rigid_coupling_realtime_sweeps *
                        (2U * vertex_groups + triangle_groups +
                         2ULL * rigid));
                coupling.bytes_moved +=
                    simulation::rigid_coupling_realtime_sweeps *
                    std::uint64_t{rigid} * sizeof(std::uint32_t);
                coupling.barriers +=
                    simulation::rigid_coupling_realtime_sweeps;
            }
            for (const auto &owner : state.soft_bodies) {
                if (!owner || !owner->alive) continue;
                const auto node_groups =
                    (std::uint64_t{owner->node_count} + 127U) / 128U;
                const auto surface_groups =
                    (std::uint64_t{owner->surface_vertex_count} + 127U) / 128U;
                dispatch(coupling,
                    4U * simulation::rigid_coupling_realtime_sweeps,
                    simulation::rigid_coupling_realtime_sweeps *
                        (2U * node_groups + rigid + surface_groups));
                coupling.bytes_moved +=
                    simulation::rigid_coupling_realtime_sweeps *
                    std::uint64_t{rigid} * sizeof(std::uint32_t);
                coupling.barriers +=
                    simulation::rigid_coupling_realtime_sweeps;
            }
            for (const auto &owner : state.ropes) {
                if (!owner || !owner->alive) continue;
                const auto node_groups =
                    (std::uint64_t{owner->data.count} + 127U) / 128U;
                dispatch(coupling,
                    2U * simulation::rigid_coupling_realtime_sweeps,
                    simulation::rigid_coupling_realtime_sweeps *
                        (node_groups + rigid));
                coupling.bytes_moved +=
                    simulation::rigid_coupling_realtime_sweeps *
                    std::uint64_t{rigid} * sizeof(std::uint32_t);
                coupling.barriers +=
                    simulation::rigid_coupling_realtime_sweeps;
                const auto attachments =
                    std::uint32_t(owner->data.options.first.enabled) +
                    std::uint32_t(owner->data.options.last.enabled);
                dispatch(coupling,
                    3U * attachments *
                        simulation::rigid_coupling_realtime_sweeps,
                    std::uint64_t{attachments} *
                        simulation::rigid_coupling_realtime_sweeps *
                        (2U + body_groups));
            }
            constexpr std::uint64_t sweeps =
                simulation::rigid_coupling_realtime_sweeps;
            for (const auto &owner : state.smoke_rigid_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled)
                    continue;
                const auto &smoke = *state.smokes[
                    owner->options.smoke.index];
                const auto particle_groups =
                    (std::uint64_t{smoke.options.capacity} + 127U) / 128U;
                if (owner->options.tracer_contact)
                    dispatch(coupling, sweeps, sweeps * particle_groups);
                if (owner->options.air_density > 0.0F &&
                    (smoke.options.grid_resolution != 0U ||
                     owner->options.tracer_contact))
                    dispatch(coupling, sweeps, sweeps);
            }
            coupling.numerical_iterations +=
                simulation::rigid_coupling_realtime_sweeps;
        }
        if (deformable_couplings != 0U) {
            for (const auto &owner : state.soft_cloth_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled)
                    continue;
                const auto &body = *state.soft_bodies[
                    owner->options.soft_body.index];
                const auto &cloth = *state.cloths[
                    owner->options.cloth.index];
                const auto body_groups =
                    (std::uint64_t{body.node_count} + 127U) / 128U;
                const auto cloth_groups =
                    (std::uint64_t{cloth.vertex_count} + 127U) / 128U;
                const auto triangle_groups =
                    (std::uint64_t{cloth.index_count / 3U} + 127U) / 128U;
                dispatch(coupling,
                    5U * simulation::deformable_smoke_coupling_sweeps,
                    simulation::deformable_smoke_coupling_sweeps *
                        (2U * body_groups + 2U * cloth_groups +
                         triangle_groups));
                coupling.bytes_moved +=
                    simulation::deformable_smoke_coupling_sweeps *
                    std::uint64_t{cloth.vertex_count} * sizeof(std::uint32_t);
                coupling.barriers +=
                    simulation::deformable_smoke_coupling_sweeps;
            }
            for (const auto &item : state.rope_cloth_couplings) {
                if (!item.alive || !item.options.enabled) continue;
                const auto endpoints =
                    std::uint32_t(item.options.first_vertex != UINT32_MAX) +
                    std::uint32_t(item.options.last_vertex != UINT32_MAX);
                dispatch(coupling,
                    3U * endpoints *
                        simulation::deformable_smoke_coupling_sweeps,
                    3ULL * endpoints *
                        simulation::deformable_smoke_coupling_sweeps);
            }
            for (const auto &owner : state.rope_soft_body_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled)
                    continue;
                const auto &body = *state.soft_bodies[
                    owner->options.soft_body.index];
                const auto records = 2U +
                    std::uint32_t(owner->options.attach_first) +
                    std::uint32_t(owner->options.attach_last);
                const auto node_groups =
                    (std::uint64_t{body.node_count} + 127U) / 128U;
                const auto surface_groups =
                    (std::uint64_t{body.surface_vertex_count} + 127U) / 128U;
                dispatch(coupling,
                    7U * records *
                        simulation::deformable_smoke_coupling_sweeps,
                    std::uint64_t{records} *
                        simulation::deformable_smoke_coupling_sweeps *
                        (5U + node_groups + surface_groups));
                coupling.bytes_moved += std::uint64_t{records} *
                    simulation::deformable_smoke_coupling_sweeps *
                    (sizeof(Vec3) +
                     std::uint64_t{body.node_count} * sizeof(Vec3));
                coupling.barriers += records *
                    simulation::deformable_smoke_coupling_sweeps * 2U;
            }
            constexpr std::uint64_t sweeps =
                simulation::deformable_smoke_coupling_sweeps;
            for (std::uint32_t smoke_slot = 0U;
                 smoke_slot < state.smokes.size(); ++smoke_slot) {
                const auto &smoke = state.smokes[smoke_slot];
                if (!smoke || !smoke->alive ||
                    smoke->options.grid_resolution != 0U)
                    continue;
                const auto references_smoke = [smoke_slot](
                        const auto &owner) noexcept {
                    return owner && owner->alive && owner->options.enabled &&
                        owner->options.smoke.index == smoke_slot;
                };
                if (std::any_of(state.smoke_cloth_couplings.begin(),
                                state.smoke_cloth_couplings.end(),
                                references_smoke) ||
                    std::any_of(state.smoke_soft_body_couplings.begin(),
                                state.smoke_soft_body_couplings.end(),
                                references_smoke) ||
                    std::any_of(state.smoke_rope_couplings.begin(),
                                state.smoke_rope_couplings.end(),
                                references_smoke)) {
                    const auto groups =
                        (std::uint64_t{smoke->options.capacity} + 127U) /
                        128U;
                    dispatch(coupling, 2U, 2U * groups);
                }
            }
            for (const auto &owner : state.smoke_cloth_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled)
                    continue;
                const auto &smoke = *state.smokes[
                    owner->options.smoke.index];
                const auto &cloth = *state.cloths[
                    owner->options.cloth.index];
                const auto smoke_groups =
                    (std::uint64_t{smoke.options.capacity} + 127U) / 128U;
                const auto vertex_groups =
                    (std::uint64_t{cloth.vertex_count} + 127U) / 128U;
                const auto triangle_groups =
                    (std::uint64_t{cloth.index_count / 3U} + 127U) / 128U;
                dispatch(coupling, 1U + sweeps, 1U + sweeps * smoke_groups);
                if (smoke.options.grid_resolution == 0U) {
                    dispatch(coupling, sweeps, sweeps * vertex_groups);
                } else {
                    dispatch(coupling, 2U * sweeps,
                        sweeps * (triangle_groups + vertex_groups));
                    coupling.bytes_moved += sweeps * cloth.vertex_count *
                        sizeof(Vec3);
                    coupling.barriers += sweeps;
                }
            }
            for (const auto &owner : state.smoke_soft_body_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled)
                    continue;
                const auto &smoke = *state.smokes[
                    owner->options.smoke.index];
                const auto &body = *state.soft_bodies[
                    owner->options.soft_body.index];
                const auto smoke_groups =
                    (std::uint64_t{smoke.options.capacity} + 127U) / 128U;
                const auto node_groups =
                    (std::uint64_t{body.node_count} + 127U) / 128U;
                const auto triangle_groups =
                    (std::uint64_t{body.surface_index_count / 3U} + 127U) /
                    128U;
                dispatch(coupling, 1U + sweeps,
                    1U + sweeps * smoke_groups);
                if (smoke.options.grid_resolution == 0U) {
                    dispatch(coupling, sweeps, sweeps * node_groups);
                } else {
                    dispatch(coupling, sweeps, sweeps * triangle_groups);
                    coupling.bytes_moved += sweeps * body.node_count *
                        sizeof(Vec3);
                    coupling.barriers += sweeps;
                }
            }
            for (const auto &owner : state.smoke_rope_couplings) {
                if (!owner || !owner->alive || !owner->options.enabled)
                    continue;
                const auto &smoke = *state.smokes[
                    owner->options.smoke.index];
                const auto &rope = *state.ropes[owner->options.rope.index];
                const auto smoke_groups =
                    (std::uint64_t{smoke.options.capacity} + 127U) / 128U;
                const auto node_groups =
                    (std::uint64_t{rope.data.count} + 127U) / 128U;
                dispatch(coupling, 1U + sweeps, 1U + sweeps * smoke_groups);
                dispatch(coupling, sweeps, sweeps * node_groups);
            }
            coupling.numerical_iterations +=
                simulation::deformable_smoke_coupling_sweeps;
        }
        domain.numerical_iterations =
            (rigid != 0U ? simulation::rigid_realtime_sweeps : 0U) +
            (state.fluid_count != 0U
                ? simulation::fluid_realtime_density_sweeps : 0U) +
            (std::any_of(state.cloths.begin(), state.cloths.end(),
                [](const auto &owner) { return owner && owner->alive; })
                ? simulation::cloth_realtime_constraint_sweeps : 0U) +
            (std::any_of(state.soft_bodies.begin(), state.soft_bodies.end(),
                [](const auto &owner) { return owner && owner->alive; })
                ? simulation::soft_body_realtime_constraint_sweeps : 0U) +
            (std::any_of(state.ropes.begin(), state.ropes.end(),
                [](const auto &owner) { return owner && owner->alive; })
                ? simulation::rope_realtime_constraint_sweeps : 0U) +
            (std::any_of(state.smokes.begin(), state.smokes.end(),
                [](const auto &owner) { return owner && owner->alive; })
                ? (std::any_of(state.smokes.begin(), state.smokes.end(),
                       [](const auto &owner) {
                           return owner && owner->alive &&
                               owner->options.grid_resolution != 0U;
                       })
                       ? simulation::smoke_realtime_grid_pressure_cycles
                       : simulation::smoke_realtime_particle_sweeps)
                : 0U);
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
    const bool fluid = state.fluid_count != 0U;
    const bool cloth = std::any_of(
        state.cloths.begin(), state.cloths.end(),
        [](const auto &owner) { return owner && owner->alive; });
    const bool soft = std::any_of(
        state.soft_bodies.begin(), state.soft_bodies.end(),
        [](const auto &owner) { return owner && owner->alive; });
    const bool rope = std::any_of(
        state.ropes.begin(), state.ropes.end(),
        [](const auto &owner) { return owner && owner->alive; });
    const bool smoke = std::any_of(
        state.smokes.begin(), state.smokes.end(),
        [](const auto &owner) { return owner && owner->alive; });
    const bool grid_smoke = std::any_of(
        state.smokes.begin(), state.smokes.end(), [](const auto &owner) {
            return owner && owner->alive &&
                owner->options.grid_resolution != 0U;
        });
    const bool domain = rigid || fluid || cloth || soft || rope || smoke;
    return {
        .dispatches = domain ? 1U : 0U,
        .workgroups = domain ? 1U : 0U,
        .bytes_moved = rigid
            ? std::uint64_t{state.rigid_body_count} * sizeof(RigidBodyState)
            : ((fluid || cloth || soft || rope || smoke)
                   ? sizeof(Vec3) : 0U),
        .index_builds = domain ? 1U : 0U,
        .index_refits = 0U,
        .index_publications = 1U,
        .numerical_iterations = rigid
            ? simulation::rigid_realtime_sweeps +
                (fluid ? simulation::fluid_realtime_density_sweeps : 0U) +
                (cloth ? simulation::cloth_realtime_constraint_sweeps : 0U) +
                (soft ? simulation::soft_body_realtime_constraint_sweeps : 0U)
                + (rope ? simulation::rope_realtime_constraint_sweeps : 0U)
                + (smoke ? (grid_smoke
                    ? simulation::smoke_realtime_grid_pressure_cycles
                    : simulation::smoke_realtime_particle_sweeps) : 0U)
            : (fluid ? simulation::fluid_realtime_density_sweeps : 0U) +
                (cloth ? simulation::cloth_realtime_constraint_sweeps : 0U) +
                (soft ? simulation::soft_body_realtime_constraint_sweeps : 0U) +
                (rope ? simulation::rope_realtime_constraint_sweeps : 0U) +
                (smoke ? (grid_smoke
                    ? simulation::smoke_realtime_grid_pressure_cycles
                    : simulation::smoke_realtime_particle_sweeps) : 0U),
        .barriers = domain ? 1U : 0U,
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
    const bool fluid = state.fluid_count != 0U;
    const bool cloth = std::any_of(
        state.cloths.begin(), state.cloths.end(),
        [](const auto &owner) { return owner && owner->alive; });
    const bool soft = std::any_of(
        state.soft_bodies.begin(), state.soft_bodies.end(),
        [](const auto &owner) { return owner && owner->alive; });
    const bool rope = std::any_of(
        state.ropes.begin(), state.ropes.end(),
        [](const auto &owner) { return owner && owner->alive; });
    const bool smoke = std::any_of(
        state.smokes.begin(), state.smokes.end(),
        [](const auto &owner) { return owner && owner->alive; });
    const bool grid_smoke = std::any_of(
        state.smokes.begin(), state.smokes.end(), [](const auto &owner) {
            return owner && owner->alive &&
                owner->options.grid_resolution != 0U;
        });
    const bool any_domain = rigid || fluid || cloth || soft || rope || smoke;
    auto &index = minimum[simulation::realtime_index(
        simulation::RealtimeTelemetryPhase::index)];
    index.index_publications = 1U;
    if (any_domain) index.index_builds = 1U;
    auto &domain = minimum[simulation::realtime_index(
        simulation::RealtimeTelemetryPhase::domain)];
    if (any_domain) {
        domain.dispatches = 1U;
        domain.workgroups = 1U;
        domain.numerical_iterations =
            (rigid ? simulation::rigid_realtime_sweeps : 0U) +
            (fluid ? simulation::fluid_realtime_density_sweeps : 0U) +
            (cloth ? simulation::cloth_realtime_constraint_sweeps : 0U) +
            (soft ? simulation::soft_body_realtime_constraint_sweeps : 0U) +
            (rope ? simulation::rope_realtime_constraint_sweeps : 0U) +
            (smoke ? (grid_smoke
                ? simulation::smoke_realtime_grid_pressure_cycles
                : simulation::smoke_realtime_particle_sweeps) : 0U);
        domain.barriers = 1U;
    }
    auto &commit = minimum[simulation::realtime_index(
        simulation::RealtimeTelemetryPhase::commit)];
    commit.submissions = 1U;
    if (any_domain) {
        commit.dispatches = 1U;
        commit.workgroups = 1U;
    }
    if (!rigid) {
        commit.bytes_moved = sizeof(std::uint32_t);
        ++commit.barriers;
    }
    if (any_domain) ++commit.barriers;
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
        std::array<std::uint32_t,
                   simulation::realtime_coupling_family_count>
            coupling_counts{};
        auto coupling_result = native_realtime_coupling_counts(
            state_, coupling_counts);
        if (coupling_result != PM_SUCCESS) return coupling_result;
        if (coupling_counts != scene.coupling_edges)
            return PM_ERROR_INVALID_STATE;
        coupling_result = stage_coupling_graph(scene);
        if (coupling_result != PM_SUCCESS) return coupling_result;
        auto count_result = native_realtime_spatial_counts(
            state_, staged_leaves_, staged_queries_);
        if (count_result != PM_SUCCESS) return count_result;
        if (!state_.options.realtime_spatial_enabled ||
            !state_.realtime_spatial.configured() ||
            staged_leaves_ >
                 state_.options.realtime_spatial_leaf_capacity ||
            staged_queries_ >
                 state_.options.realtime_spatial_query_capacity ||
            state_.options.realtime_spatial_candidates_per_query == 0U ||
            (scene.domains[simulation::realtime_index(
                 simulation::RealtimeDomain::smoke)] &&
             state_.options.realtime_spatial_candidates_per_query <
                 smoke_realtime_candidates_per_particle))
            return PM_ERROR_FEATURE_NOT_PRESENT;
        std::uint32_t fluid_offset = state_.rigid_body_count;
        for (auto &owner : state_.fluids) {
            if (!owner || !owner->alive) continue;
            const auto &fluid = *owner;
            const auto material = simulation::validate_fluid_realtime_input({
                0.0F, fluid.options.support_radius,
                fluid.options.particle_radius, fluid.options.rest_density,
                fluid.options.rest_particle_volume,
                fluid.options.maximum_speed,
                vector_length(options_.gravity) * scene.timestep *
                    scene.timestep,
                simulation::fluid_realtime_maximum_correction_fraction *
                    fluid.options.support_radius,
                fluid.options.repulsion, fluid.options.viscosity,
                fluid.options.normal_damping,
                fluid.options.velocity_damping});
            if (material != PM_SUCCESS) return material;
            owner->realtime_leaf_offset = fluid_offset;
            owner->realtime_query_offset = fluid_offset;
            fluid_offset += owner->options.capacity;
        }
        std::uint32_t cloth_offset = fluid_offset;
        for (auto &owner : state_.cloths) {
            if (!owner || !owner->alive) continue;
            auto &cloth = *owner;
            if (cloth.realtime_topology_revision == 0U ||
                !std::isfinite(cloth.realtime_characteristic_length) ||
                cloth.realtime_characteristic_length < 0.005F ||
                vector_length(options_.gravity) * scene.timestep *
                        scene.timestep >
                    0.25F * cloth.realtime_characteristic_length)
                return PM_ERROR_INVALID_ARGUMENT;
            cloth.realtime_leaf_offset = cloth_offset;
            cloth.realtime_query_offset = cloth_offset;
            cloth_offset += cloth.index_count / 3U;
        }
        std::uint32_t soft_offset = cloth_offset;
        for (auto &owner : state_.soft_bodies) {
            if (!owner || !owner->alive) continue;
            auto &body = *owner;
            if (body.realtime_topology_revision == 0U ||
                !std::isfinite(body.realtime_characteristic_length) ||
                body.realtime_characteristic_length < 0.005F ||
                vector_length(options_.gravity) * scene.timestep *
                        scene.timestep >
                    0.25F * body.realtime_characteristic_length)
                return PM_ERROR_INVALID_ARGUMENT;
            body.realtime_leaf_offset = soft_offset;
            body.realtime_query_offset = soft_offset;
            soft_offset += body.surface_index_count / 3U;
        }
        std::uint32_t rope_offset = soft_offset;
        for (auto &owner : state_.ropes) {
            if (!owner || !owner->alive) continue;
            auto &rope = *owner;
            const auto nodes = rope.data.count;
            const float inverse_node_mass =
                static_cast<float>(nodes) / rope.data.options.mass;
            simulation::RopeRealtimeInput material{};
            material.node_spacing = rope.data.options.node_spacing;
            material.radius = rope.data.options.radius;
            material.total_mass = rope.data.options.mass;
            material.node_count = nodes;
            material.linear_speed = rope.data.options.maximum_speed;
            material.acceleration_displacement =
                vector_length(options_.gravity) * scene.timestep *
                scene.timestep;
            material.solver_correction =
                simulation::rope_realtime_maximum_correction_fraction *
                rope.data.options.node_spacing;
            material.structural_compliance =
                rope.data.options.stretch_compliance;
            material.effective_inverse_mass = inverse_node_mass;
            material.velocity_damping =
                rope.data.options.velocity_damping;
            material.contact_friction = rope.data.options.friction;
            auto validation =
                simulation::validate_rope_realtime_input(material);
            if (validation != PM_SUCCESS) return validation;
            material.effective_inverse_mass = 2.0F * inverse_node_mass;
            validation = simulation::validate_rope_realtime_input(material);
            if (validation != PM_SUCCESS) return validation;
            rope.realtime_leaf_offset = rope_offset;
            rope.realtime_query_offset = rope_offset;
            rope_offset += nodes - 1U;
        }
        std::uint32_t smoke_offset = rope_offset;
        for (auto &owner : state_.smokes) {
            if (!owner || !owner->alive) continue;
            auto &smoke = *owner;
            const auto representation = smoke.options.grid_resolution == 0U
                ? simulation::SmokeRealtimeRepresentation::particles
                : simulation::SmokeRealtimeRepresentation::grid;
            if (representation != scene.smoke_representation)
                return PM_ERROR_FEATURE_NOT_PRESENT;
            const float characteristic = representation ==
                    simulation::SmokeRealtimeRepresentation::particles
                ? 3.0F * smoke.options.particle_radius
                : smoke.grid.spacing;
            const float viscosity = representation ==
                    simulation::SmokeRealtimeRepresentation::particles
                ? smoke.options.viscosity
                : smoke.options.grid_kinematic_viscosity;
            simulation::SmokeRealtimeInput material{};
            material.representation = representation;
            material.characteristic_length = characteristic;
            material.maximum_speed = smoke.options.maximum_speed;
            material.acceleration_displacement = smoke.options.buoyancy *
                scene.timestep * scene.timestep;
            material.solver_correction = representation ==
                    simulation::SmokeRealtimeRepresentation::particles
                ? simulation::smoke_realtime_maximum_correction_fraction *
                    characteristic
                : 0.0F;
            material.response = smoke.options.response;
            material.pressure_stiffness = smoke.options.pressure_stiffness;
            material.viscosity = viscosity;
            material.grid_resolution = smoke.options.grid_resolution;
            material.grid_vertical_resolution =
                smoke.options.grid_vertical_resolution;
            const auto validate_coordinate = [&](float coordinate) noexcept {
                material.world_coordinate = coordinate;
                return simulation::validate_smoke_realtime_input(material);
            };
            auto validation = validate_coordinate(
                smoke.options.emitter_center.x);
            if (validation == PM_SUCCESS)
                validation = validate_coordinate(
                    smoke.options.emitter_center.y);
            if (validation == PM_SUCCESS)
                validation = validate_coordinate(
                    smoke.options.emitter_center.z);
            if (representation ==
                    simulation::SmokeRealtimeRepresentation::grid) {
                const Vec3 maximum{
                    smoke.grid.minimum.x +
                        smoke.grid.spacing * float(smoke.grid.resolution),
                    smoke.grid.minimum.y +
                        smoke.grid.spacing * float(smoke.grid.height),
                    smoke.grid.minimum.z +
                        smoke.grid.spacing * float(smoke.grid.resolution)};
                if (validation == PM_SUCCESS)
                    validation = validate_coordinate(smoke.grid.minimum.x);
                if (validation == PM_SUCCESS)
                    validation = validate_coordinate(smoke.grid.minimum.y);
                if (validation == PM_SUCCESS)
                    validation = validate_coordinate(smoke.grid.minimum.z);
                if (validation == PM_SUCCESS)
                    validation = validate_coordinate(maximum.x);
                if (validation == PM_SUCCESS)
                    validation = validate_coordinate(maximum.y);
                if (validation == PM_SUCCESS)
                    validation = validate_coordinate(maximum.z);
                if (smoke.options.grid_pressure_iterations !=
                    simulation::smoke_realtime_grid_pressure_cycles)
                    validation = PM_ERROR_INVALID_ARGUMENT;
            }
            if (validation != PM_SUCCESS) return validation;
            smoke.realtime_leaf_offset = smoke_offset;
            smoke.realtime_query_offset = smoke_offset;
            smoke_offset += smoke.options.capacity;

            smoke.realtime_pending_count = smoke.count;
            smoke.realtime_pending_next_slot = smoke.next_slot;
            smoke.realtime_pending_emitted = smoke.emitted;
            smoke.realtime_pending_emit_count = 0U;
            smoke.realtime_pending_emission_fraction =
                smoke.emission_fraction;
            smoke.realtime_pending_time = float(std::fmod(
                double(smoke.time) + double(scene.timestep), 1000.0));
            if (representation ==
                simulation::SmokeRealtimeRepresentation::particles) {
                const double exact = double(smoke.emission_fraction) +
                    double(smoke.options.particles_per_second) *
                        scene.timestep;
                const auto requested = static_cast<std::uint32_t>(std::min(
                    std::floor(exact), double(smoke.options.capacity)));
                smoke.realtime_pending_emit_count = requested;
                smoke.realtime_pending_emission_fraction =
                    float(exact - std::floor(exact));
                smoke.realtime_pending_next_slot =
                    (smoke.next_slot + requested) % smoke.options.capacity;
                smoke.realtime_pending_count = std::min(
                    smoke.options.capacity, smoke.count + requested);
                smoke.realtime_pending_emitted = smoke.emitted + requested;
            }
        }
        if (smoke_offset != staged_leaves_)
            return PM_ERROR_INVALID_STATE;
        const auto candidates_per_query =
            state_.options.realtime_spatial_candidates_per_query;
        for (auto &owner : state_.smokes) {
            if (!owner || !owner->alive ||
                owner->options.grid_resolution != 0U)
                continue;
            auto &smoke = *owner;
            const auto capacity = smoke.options.capacity;
            const std::size_t map_count =
                std::size_t(capacity) * candidates_per_query;
            const std::size_t compact_count = std::size_t(capacity) *
                smoke_realtime_candidates_per_particle;
            if (map_count > std::numeric_limits<std::uint32_t>::max() ||
                compact_count > std::numeric_limits<std::uint32_t>::max() ||
                smoke.realtime_candidate_gather_slots.size() !=
                    compact_count)
                return PM_ERROR_CAPACITY_EXCEEDED;
            smoke.realtime_candidate_map_count =
                static_cast<std::uint32_t>(map_count);
            smoke.realtime_candidate_compact_count =
                static_cast<std::uint32_t>(compact_count);
            for (std::uint32_t particle = 0U; particle < capacity;
                 ++particle) {
                state_.realtime_smoke_payload_refs_host[
                    smoke.realtime_leaf_offset + particle] = particle;
                const auto source_base =
                    std::size_t(particle) * candidates_per_query;
                const auto compact_base = std::size_t(particle) *
                    smoke_realtime_candidates_per_particle;
                for (std::uint32_t candidate = 0U;
                     candidate < smoke_realtime_candidates_per_particle;
                     ++candidate) {
                    smoke.realtime_candidate_gather_slots[
                        compact_base + candidate] =
                            static_cast<std::uint32_t>(
                                source_base + candidate);
                }
            }
        }
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
            staged_leaves_);
#else
        result = state_.realtime_spatial.begin_device_tick(
            scene.tick, simulation::rigid_realtime_profile_version,
            scene.graph_revision, scene.resource_revision,
            staged_leaves_);
#endif
        if (result != PM_SUCCESS) return fail_begin(result);
        snapshot_ = state_.realtime_spatial.staging_snapshot();
        if (!snapshot_) return fail_begin(PM_ERROR_INVALID_STATE);
#if defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
        if (staged_leaves_ == 0U) {
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
             staged_leaves_,
             static_cast<std::uint32_t>(staged_leaves_ == 0U
                 ? simulation::SpatialUpdateKind::empty
                 : simulation::SpatialUpdateKind::build)});
        if (result != PM_SUCCESS) return fail_begin(result);
        dispatch(simulation::RealtimeTelemetryPhase::boundary, 1U);
        if (staged_leaves_ == 0U) {
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
        if (staged_leaves_ == 0U) {
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
        auto &boundary_work = work(
            simulation::RealtimeTelemetryPhase::boundary);
        if (rigid_coupling_count_ != 0U ||
            deformable_coupling_count_ != 0U) {
            auto coupling_result = stage_coupling_buffers(boundary_work);
            if (coupling_result != PM_SUCCESS) return coupling_result;
        }
        const bool has_particle_smoke = std::any_of(
            state_.smokes.begin(), state_.smokes.end(),
            [](const auto &owner) {
                return owner && owner->alive &&
                    owner->options.grid_resolution == 0U;
            });
        if (has_particle_smoke) {
            const auto bytes =
                std::size_t(state_.options.realtime_spatial_leaf_capacity) *
                sizeof(std::uint32_t);
            const auto result = native_.copy(
                state_.realtime_smoke_payload_refs,
                state_.realtime_smoke_payload_refs_host.data(), bytes,
                device::CopyDirection::host_to_device, native_.queue());
            if (result != PM_SUCCESS) return result;
            boundary_work.bytes_moved += bytes;
            ++boundary_work.barriers;
        }
        if (state_.rigid_body_count != 0U) {
            const std::size_t bytes = std::size_t(state_.rigid_body_count) *
                sizeof(RigidBodyState);
            const auto result = native_.copy(
                state_.render_previous_states,
                state_.states[state_.current_state], bytes,
                device::CopyDirection::device_to_device, native_.queue());
            if (result != PM_SUCCESS) return result;
            boundary_work.bytes_moved += bytes;
            ++boundary_work.barriers;
        }
        for (auto &owner : state_.fluids) {
            if (!owner || !owner->alive) continue;
            auto &fluid = *owner;
            const auto bytes = std::size_t(fluid.options.capacity) *
                sizeof(Vec3);
            auto result = native_.copy(
                fluid.next_positions, fluid.positions, bytes,
                device::CopyDirection::device_to_device, native_.queue());
            if (result == PM_SUCCESS)
                result = native_.copy(
                    fluid.next_velocities, fluid.velocities, bytes,
                    device::CopyDirection::device_to_device, native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(
                    fluid.realtime_time_advances, 0,
                    sizeof(*fluid.realtime_time_advances), native_.queue());
            if (result != PM_SUCCESS) return result;
            boundary_work.bytes_moved += 2U * bytes +
                sizeof(*fluid.realtime_time_advances);
            boundary_work.barriers += 3U;
        }
        for (auto &owner : state_.cloths) {
            if (!owner || !owner->alive) continue;
            auto &cloth = *owner;
            const auto bytes = std::size_t(cloth.vertex_capacity) *
                sizeof(Vec3);
            auto result = native_.copy(
                cloth.realtime_positions, cloth.positions, bytes,
                device::CopyDirection::device_to_device, native_.queue());
            if (result == PM_SUCCESS)
                result = native_.copy(
                    cloth.realtime_velocities, cloth.velocities, bytes,
                    device::CopyDirection::device_to_device,
                    native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(
                    cloth.realtime_time_advances, 0,
                    sizeof(*cloth.realtime_time_advances), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(
                    cloth.realtime_contact_active, 0,
                    cloth.vertex_capacity * sizeof(std::uint8_t),
                    native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(cloth.soft_body_forces, 0,
                    cloth.vertex_capacity * sizeof(Vec3), native_.queue());
            if (result == PM_SUCCESS && cloth.rope_forces != nullptr)
                result = native_.clear(cloth.rope_forces, 0,
                    cloth.vertex_capacity * sizeof(Vec3), native_.queue());
            if (result != PM_SUCCESS) return result;
            boundary_work.bytes_moved += 2U * bytes +
                sizeof(*cloth.realtime_time_advances) +
                cloth.vertex_capacity * sizeof(std::uint8_t);
            boundary_work.bytes_moved += cloth.vertex_capacity * sizeof(Vec3);
            ++boundary_work.barriers;
            if (cloth.rope_forces != nullptr) {
                boundary_work.bytes_moved +=
                    cloth.vertex_capacity * sizeof(Vec3);
                ++boundary_work.barriers;
            }
            boundary_work.barriers += 4U;
        }
        for (auto &owner : state_.soft_bodies) {
            if (!owner || !owner->alive) continue;
            auto &body = *owner;
            const auto bytes = std::size_t(body.node_count) * sizeof(Vec3);
            auto result = native_.copy(
                body.realtime_positions, body.positions, bytes,
                device::CopyDirection::device_to_device, native_.queue());
            if (result == PM_SUCCESS)
                result = native_.copy(
                    body.realtime_velocities, body.velocities, bytes,
                    device::CopyDirection::device_to_device,
                    native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(
                    body.realtime_time_advances, 0,
                    sizeof(*body.realtime_time_advances), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(
                    body.realtime_contact_active, 0,
                    body.node_count * sizeof(std::uint8_t), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(
                    body.realtime_shape_orientation, 0,
                    sizeof(*body.realtime_shape_orientation), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(body.contact_momentum_delta, 0,
                    body.node_count * sizeof(Vec3), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(body.contact_friction_delta, 0,
                    body.node_count * sizeof(Vec3), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(body.contact_normal_delta, 0,
                    body.node_count * sizeof(float), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(body.contact_count, 0,
                    sizeof(*body.contact_count), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(body.dynamic_contact_flag, 0,
                    sizeof(*body.dynamic_contact_flag), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(body.cloth_forces, 0,
                    body.node_count * sizeof(Vec3), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(body.rope_forces, 0,
                    body.node_count * sizeof(Vec3), native_.queue());
            if (result != PM_SUCCESS) return result;
            boundary_work.bytes_moved += 2U * bytes +
                sizeof(*body.realtime_time_advances) +
                body.node_count * sizeof(std::uint8_t) +
                sizeof(*body.realtime_shape_orientation);
            boundary_work.bytes_moved +=
                4ULL * body.node_count * sizeof(Vec3) +
                body.node_count * sizeof(float) +
                2U * sizeof(std::uint32_t);
            boundary_work.barriers += 12U;
        }
        for (auto &owner : state_.ropes) {
            if (!owner || !owner->alive) continue;
            auto &rope = *owner;
            const auto bytes =
                std::size_t(rope.data.count) * sizeof(Vec3);
            auto result = native_.copy(
                rope.realtime_positions, rope.data.positions, bytes,
                device::CopyDirection::device_to_device, native_.queue());
            if (result == PM_SUCCESS)
                result = native_.copy(
                    rope.realtime_velocities, rope.data.velocities, bytes,
                    device::CopyDirection::device_to_device,
                    native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(
                    rope.realtime_time_advances, 0,
                    sizeof(*rope.realtime_time_advances), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(
                    rope.realtime_contact_active, 0,
                    rope.data.count * sizeof(std::uint8_t), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.copy(rope.realtime_count, &rope.data.count,
                    sizeof(*rope.realtime_count),
                    device::CopyDirection::host_to_device, native_.queue());
            if (result != PM_SUCCESS) return result;
            boundary_work.bytes_moved += 2U * bytes +
                sizeof(*rope.realtime_time_advances) +
                rope.data.count * sizeof(std::uint8_t);
            boundary_work.bytes_moved += sizeof(*rope.realtime_count);
            boundary_work.barriers += 5U;
        }
        for (auto &owner : state_.smokes) {
            if (!owner || !owner->alive) continue;
            auto &smoke = *owner;
            const auto vector_bytes =
                std::size_t(smoke.options.capacity) * sizeof(Vec3);
            const auto scalar_bytes =
                std::size_t(smoke.options.capacity) * sizeof(float);
            auto result = native_.copy(
                smoke.realtime_positions, smoke.positions, vector_bytes,
                device::CopyDirection::device_to_device, native_.queue());
            if (result == PM_SUCCESS)
                result = native_.copy(smoke.realtime_previous_positions,
                    smoke.previous_positions, vector_bytes,
                    device::CopyDirection::device_to_device,
                    native_.queue());
            if (result == PM_SUCCESS)
                result = native_.copy(smoke.realtime_velocities,
                    smoke.velocities, vector_bytes,
                    device::CopyDirection::device_to_device,
                    native_.queue());
            if (result == PM_SUCCESS)
                result = native_.copy(smoke.realtime_vorticities,
                    smoke.vorticities, vector_bytes,
                    device::CopyDirection::device_to_device,
                    native_.queue());
            if (result == PM_SUCCESS)
                result = native_.copy(smoke.realtime_ages, smoke.ages,
                    scalar_bytes, device::CopyDirection::device_to_device,
                    native_.queue());
            if (result == PM_SUCCESS)
                result = native_.copy(smoke.realtime_thermal_lift,
                    smoke.thermal_lift, scalar_bytes,
                    device::CopyDirection::device_to_device,
                    native_.queue());
            if (result == PM_SUCCESS)
                result = native_.copy(smoke.realtime_number_densities,
                    smoke.number_densities, scalar_bytes,
                    device::CopyDirection::device_to_device,
                    native_.queue());
            if (result == PM_SUCCESS)
                result = native_.copy(smoke.realtime_pressures,
                    smoke.pressures, scalar_bytes,
                    device::CopyDirection::device_to_device,
                    native_.queue());
            if (result == PM_SUCCESS)
                result = native_.copy(smoke.realtime_count,
                    &smoke.realtime_pending_count,
                    sizeof(*smoke.realtime_count),
                    device::CopyDirection::host_to_device,
                    native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(smoke.realtime_time_advances, 0,
                    sizeof(*smoke.realtime_time_advances), native_.queue());
            if (result == PM_SUCCESS &&
                smoke.options.grid_resolution == 0U)
                result = native_.copy(smoke.realtime_candidate_indices,
                    smoke.realtime_candidate_gather_slots.data(),
                    std::size_t(smoke.realtime_candidate_compact_count) *
                        sizeof(std::uint32_t),
                    device::CopyDirection::host_to_device,
                    native_.queue());
            if (result != PM_SUCCESS) return result;
            boundary_work.bytes_moved += 4U * vector_bytes +
                4U * scalar_bytes + 2U * sizeof(std::uint32_t);
            boundary_work.barriers += 10U;

            if (smoke.options.grid_resolution == 0U) {
                boundary_work.bytes_moved +=
                    std::uint64_t{smoke.realtime_candidate_compact_count} *
                    sizeof(std::uint32_t);
                ++boundary_work.barriers;
            }

            if (smoke.options.grid_resolution == 0U) continue;
            auto &grid = smoke.grid;
            const auto cell_bytes =
                std::size_t(grid.cell_count) * sizeof(float);
            result = native_.copy(grid.realtime_density[0], grid.density,
                cell_bytes, device::CopyDirection::device_to_device,
                native_.queue());
            if (result == PM_SUCCESS)
                result = native_.copy(grid.realtime_temperature[0],
                    grid.temperature, cell_bytes,
                    device::CopyDirection::device_to_device,
                    native_.queue());
            std::uint64_t moved = 2U * cell_bytes;
            std::uint32_t barriers = 2U;
            for (std::uint32_t axis = 0U;
                 result == PM_SUCCESS && axis < 3U; ++axis) {
                const auto face_bytes =
                    std::size_t(grid.face_count[axis]) * sizeof(float);
                result = native_.copy(grid.face_velocity[axis][1],
                    grid.face_velocity[axis][0], face_bytes,
                    device::CopyDirection::device_to_device,
                    native_.queue());
                moved += face_bytes;
                ++barriers;
            }
            if (result != PM_SUCCESS) return result;
            boundary_work.bytes_moved += moved;
            boundary_work.barriers += barriers;
        }
        return PM_SUCCESS;
    }

    PmResult predict(simulation::RealtimeDomain domain,
                     const simulation::RealtimeScheduleScene &scene) noexcept {
        constexpr std::uint32_t block = 128U;
        if (domain == simulation::RealtimeDomain::fluid) {
            for (auto &owner : state_.fluids) {
                if (!owner || !owner->alive) continue;
                auto &fluid = *owner;
                const auto groups =
                    (fluid.options.capacity + block - 1U) / block;
                const auto constants = native_.fluid_constants(
                    fluid.options, options_.gravity, scene.timestep,
                    fluid.options.capacity);
                native_.predict_fluid_realtime(
                    groups, block, native_.queue(), fluid.next_positions,
                    fluid.next_velocities, fluid.previous, fluid.count,
                    fluid.options.capacity, constants, &options_.gravity.x,
                    scene.timestep,
                    simulation::fluid_realtime_profile_version,
                    fluid.realtime_accumulated_corrections,
                    fluid.realtime_time_advances, snapshot_.query_status);
                const auto result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatch(simulation::RealtimeTelemetryPhase::predict,
                         groups);
            }
            return PM_SUCCESS;
        }
        if (domain == simulation::RealtimeDomain::cloth) {
            for (auto &owner : state_.cloths) {
                if (!owner || !owner->alive) continue;
                auto &cloth = *owner;
                const auto groups =
                    (cloth.vertex_capacity + block - 1U) / block;
                native_.predict_cloth_realtime(
                    groups, block, native_.queue(),
                    cloth.realtime_positions, cloth.realtime_velocities,
                    cloth.previous, cloth.realtime_correction_used,
                    cloth.inverse_masses,
                    cloth.realtime_characteristic_lengths,
                    cloth.vertex_count, cloth.vertex_capacity,
                    &options_.gravity.x, cloth.velocity_damping,
                    cloth.realtime_maximum_speed, scene.timestep,
                    simulation::cloth_realtime_profile_version,
                    cloth.realtime_time_advances,
                    snapshot_.query_status);
                const auto result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatch(simulation::RealtimeTelemetryPhase::predict,
                         groups);
            }
            return PM_SUCCESS;
        }
        if (domain == simulation::RealtimeDomain::soft_body) {
            for (auto &owner : state_.soft_bodies) {
                if (!owner || !owner->alive) continue;
                auto &body = *owner;
                const auto groups =
                    (body.node_count + block - 1U) / block;
                const auto surface_groups =
                    (body.surface_vertex_count + block - 1U) / block;
                native_.predict_soft_body_realtime(
                    groups, block, native_.queue(),
                    body.realtime_positions, body.realtime_velocities,
                    body.previous, body.realtime_correction_used,
                    body.inverse_masses,
                    body.realtime_characteristic_lengths,
                    body.node_count, body.node_count, &options_.gravity.x,
                    body.velocity_damping, body.realtime_maximum_speed,
                    scene.timestep,
                    simulation::soft_body_realtime_profile_version,
                    body.realtime_time_advances, snapshot_.query_status);
                native_.update_soft_body_surface(
                    surface_groups, block, native_.queue(),
                    body.realtime_positions, body.rest_positions,
                    body.surface_rest_positions, body.surface_bindings,
                    body.realtime_surface_positions,
                    body.surface_vertex_count);
                const auto result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatches(simulation::RealtimeTelemetryPhase::predict, 2U,
                    groups + surface_groups);
            }
            return PM_SUCCESS;
        }
        if (domain == simulation::RealtimeDomain::rope) {
            for (auto &owner : state_.ropes) {
                if (!owner || !owner->alive) continue;
                auto &rope = *owner;
                const auto groups =
                    (rope.data.count + block - 1U) / block;
                native_.predict_rope_realtime(
                    groups, block, native_.queue(),
                    rope.realtime_positions, rope.realtime_velocities,
                    rope.realtime_previous_positions,
                    rope.realtime_correction_used,
                    rope.realtime_inverse_masses,
                    rope.realtime_anchor_targets,
                    rope.realtime_anchor_active, rope.data.count,
                    rope.data.count, rope.data.options.node_spacing,
                    rope.data.options.radius, &options_.gravity.x,
                    rope.data.options.velocity_damping,
                    rope.data.options.friction,
                    rope.data.options.maximum_speed, scene.timestep,
                    simulation::rope_realtime_profile_version,
                    rope.realtime_time_advances, snapshot_.query_status);
                const auto result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatch(simulation::RealtimeTelemetryPhase::predict,
                         groups);
            }
            return PM_SUCCESS;
        }
        if (domain == simulation::RealtimeDomain::smoke) {
            for (auto &owner : state_.smokes) {
                if (!owner || !owner->alive) continue;
                auto &smoke = *owner;
                const auto options = native_.smoke_domain_options(
                    smoke.options);
                const auto particle_groups =
                    (smoke.options.capacity + block - 1U) / block;
                if (smoke.options.grid_resolution == 0U) {
                    const auto emit_groups = std::max(1U,
                        (smoke.realtime_pending_emit_count + block - 1U) /
                            block);
                    native_.emit_smoke_realtime_particles(
                        emit_groups, block, native_.queue(),
                        smoke.realtime_positions,
                        smoke.realtime_previous_positions,
                        smoke.realtime_velocities, smoke.realtime_ages,
                        smoke.realtime_thermal_lift,
                        smoke.realtime_number_densities,
                        smoke.realtime_pressures, options, smoke.next_slot,
                        smoke.realtime_pending_emit_count, smoke.emitted,
                        scene.timestep,
                        simulation::smoke_realtime_profile_version,
                        snapshot_.query_status);
                    native_.predict_smoke_realtime_particles(
                        particle_groups, block, native_.queue(),
                        smoke.realtime_positions,
                        smoke.realtime_previous_positions,
                        smoke.realtime_velocities, smoke.realtime_ages,
                        smoke.realtime_thermal_lift,
                        smoke.realtime_pending_count, options,
                        &options_.gravity.x, scene.timestep,
                        simulation::smoke_realtime_profile_version,
                        smoke.realtime_accumulated_corrections,
                        smoke.realtime_time_advances,
                        snapshot_.query_status);
                    const auto result = native_.check_error();
                    if (result != PM_SUCCESS) return result;
                    dispatches(simulation::RealtimeTelemetryPhase::predict,
                        2U, std::uint64_t{emit_groups} + particle_groups);
                    continue;
                }

                auto &grid = smoke.grid;
                const auto cell_groups =
                    (grid.cell_count + block - 1U) / block;
                auto current = native_.smoke_dispatch_grid(grid);
                current.density = grid.realtime_density[0];
                current.pressure = grid.pressure[1];
                for (std::uint32_t axis = 0U; axis < 3U; ++axis)
                    current.face_velocity[axis] =
                        grid.face_velocity[axis][1];
                native_.begin_smoke_grid_realtime(
                    cell_groups, block, native_.queue(), current, options,
                    scene.timestep,
                    simulation::smoke_realtime_profile_version,
                    grid.pressure[1], grid.realtime_divergence,
                    smoke.realtime_time_advances, snapshot_.query_status);
                native_.apply_smoke_grid_realtime_source(
                    cell_groups, block, native_.queue(), current, options,
                    grid.realtime_density_rate,
                    grid.realtime_temperature_rate,
                    grid.realtime_density[0],
                    grid.realtime_temperature[0], scene.timestep,
                    simulation::smoke_realtime_profile_version,
                    snapshot_.query_status);
                native_.advect_smoke_grid_realtime_scalar(
                    cell_groups, block, native_.queue(), current, options,
                    grid.realtime_density[0], grid.realtime_density[1],
                    scene.timestep,
                    simulation::smoke_realtime_profile_version,
                    snapshot_.query_status);
                native_.advect_smoke_grid_realtime_scalar(
                    cell_groups, block, native_.queue(), current, options,
                    grid.realtime_temperature[0],
                    grid.realtime_temperature[1], scene.timestep,
                    simulation::smoke_realtime_profile_version,
                    snapshot_.query_status);
                std::uint64_t face_groups = 0U;
                for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
                    const auto groups =
                        (grid.face_count[axis] + block - 1U) / block;
                    face_groups += groups;
                    native_.advect_smoke_grid_realtime_face(
                        groups, block, native_.queue(), int(axis), current,
                        options, grid.face_velocity[axis][1],
                        grid.realtime_face_velocity[axis], scene.timestep,
                        simulation::smoke_realtime_profile_version,
                        snapshot_.query_status);
                }
                auto advected = current;
                advected.density = grid.realtime_density[1];
                for (std::uint32_t axis = 0U; axis < 3U; ++axis)
                    advected.face_velocity[axis] =
                        grid.realtime_face_velocity[axis];
                for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
                    const auto groups =
                        (grid.face_count[axis] + block - 1U) / block;
                    native_.apply_smoke_grid_realtime_face_forces(
                        groups, block, native_.queue(), int(axis), advected,
                        options, grid.realtime_face_velocity[axis],
                        grid.face_velocity[axis][1], grid.face_open[axis],
                        grid.face_wall_velocity[axis], grid.strain,
                        grid.subgrid_force, grid.realtime_temperature[1],
                        scene.timestep, &options_.gravity.x,
                        simulation::smoke_realtime_profile_version,
                        snapshot_.query_status);
                }
                auto forced = advected;
                for (std::uint32_t axis = 0U; axis < 3U; ++axis)
                    forced.face_velocity[axis] =
                        grid.face_velocity[axis][1];
                for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
                    const auto groups =
                        (grid.face_count[axis] + block - 1U) / block;
                    native_.apply_smoke_grid_realtime_boundaries(
                        groups, block, native_.queue(), int(axis), forced,
                        options, grid.face_velocity[axis][1],
                        grid.face_open[axis],
                        grid.face_wall_velocity[axis], scene.timestep,
                        simulation::smoke_realtime_profile_version,
                        snapshot_.query_status);
                }
                native_.calculate_smoke_grid_realtime_divergence(
                    cell_groups, block, native_.queue(), forced, options,
                    grid.realtime_divergence, scene.timestep,
                    simulation::smoke_realtime_profile_version,
                    snapshot_.query_status);
                const auto result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatches(simulation::RealtimeTelemetryPhase::predict,
                    14U, 5ULL * cell_groups + 3ULL * face_groups);
            }
            return PM_SUCCESS;
        }
        if (domain != simulation::RealtimeDomain::rigid)
            return PM_ERROR_FEATURE_NOT_PRESENT;
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
        auto *query_bounds = state_.realtime_query_bounds;
        auto *query_masks = state_.realtime_query_masks;
        auto *excluded = state_.realtime_excluded_sort_keys;
        auto result = PM_SUCCESS;
        if (state_.rigid_body_count != 0U) {
            result = native_.emit_rigid_spatial_proxies(
                groups, block, native_.queue(), state_.parameters,
                state_.states[state_.current_state],
                state_.states[output_state_], state_.meshes,
                state_.options.triangle_mesh_capacity, state_.ids,
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
        }
        auto *query_leaf_sources = fluid_spatial<std::uint32_t>(
            state_.realtime_fluid_spatial_plan.query_leaf_sources);
        for (std::uint32_t slot = 0U; slot < state_.fluids.size(); ++slot) {
            auto &owner = state_.fluids[slot];
            if (!owner || !owner->alive) continue;
            auto &fluid = *owner;
            const auto fluid_groups =
                (fluid.options.capacity + block - 1U) / block;
            const float fluid_correction[3]{
                simulation::fluid_realtime_maximum_correction_fraction *
                    fluid.options.support_radius,
                simulation::fluid_realtime_maximum_correction_fraction *
                    fluid.options.support_radius,
                simulation::fluid_realtime_maximum_correction_fraction *
                    fluid.options.support_radius};
            native_.emit_fluid_spatial_particles(
                fluid_groups, block, native_.queue(), fluid.positions,
                fluid.next_positions, fluid.ids, fluid.count,
                fluid.options.capacity, slot, fluid.generation,
                static_cast<std::uint32_t>(
                    scene.graph_revision %
                    (std::numeric_limits<std::uint32_t>::max() - 1U)) + 1U,
                fluid.realtime_leaf_offset,
                state_.realtime_spatial_plan.leaf_capacity,
                fluid.realtime_query_offset,
                state_.realtime_spatial_plan.query_capacity,
                fluid.realtime_leaf_offset,
                simulation::fluid_spatial_query_mask,
                simulation::fluid_spatial_query_mask, 0.0F,
                fluid.options.support_radius, path_deviation,
                fluid_correction, 1.0e-5F, snapshot_.metadata,
                snapshot_.bounds, snapshot_.build_flags, query_bounds,
                query_masks, excluded, query_leaf_sources,
                snapshot_.query_status);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            dispatch(simulation::RealtimeTelemetryPhase::index,
                     fluid_groups);
        }
        auto *deformable_query_leaf_sources = deformable_spatial<
            std::uint32_t>(state_.realtime_deformable_spatial_plan
                               .query_leaf_sources);
        for (std::uint32_t slot = 0U; slot < state_.cloths.size(); ++slot) {
            auto &owner = state_.cloths[slot];
            if (!owner || !owner->alive) continue;
            auto &cloth = *owner;
            const auto triangles = cloth.index_count / 3U;
            const auto cloth_groups = (triangles + block - 1U) / block;
            const float cloth_correction[3]{
                simulation::cloth_realtime_maximum_correction_fraction *
                    cloth.realtime_characteristic_length,
                simulation::cloth_realtime_maximum_correction_fraction *
                    cloth.realtime_characteristic_length,
                simulation::cloth_realtime_maximum_correction_fraction *
                    cloth.realtime_characteristic_length};
            native_.emit_deformable_spatial_primitives(
                cloth_groups, block, native_.queue(), cloth.previous,
                cloth.realtime_positions, cloth.vertex_count,
                cloth.indices, cloth.realtime_triangle_generations,
                triangles, 3U,
                simulation::deformable_spatial_cloth_family, slot,
                cloth.generation,
                static_cast<std::uint32_t>(cloth.realtime_topology_revision),
                simulation::deformable_spatial_triangle_primitive,
                cloth.realtime_leaf_offset,
                state_.realtime_spatial_plan.leaf_capacity,
                cloth.realtime_query_offset,
                state_.realtime_spatial_plan.query_capacity,
                cloth.realtime_leaf_offset,
                simulation::deformable_spatial_contact_mask,
                simulation::deformable_spatial_contact_mask,
                cloth.thickness, path_deviation, cloth_correction, 1.0e-5F,
                snapshot_.metadata, snapshot_.bounds,
                snapshot_.build_flags, query_bounds, query_masks, excluded,
                deformable_query_leaf_sources, snapshot_.query_status);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            dispatch(simulation::RealtimeTelemetryPhase::index,
                     cloth_groups);
        }
        for (std::uint32_t slot = 0U;
             slot < state_.soft_bodies.size(); ++slot) {
            auto &owner = state_.soft_bodies[slot];
            if (!owner || !owner->alive) continue;
            auto &body = *owner;
            const auto triangles = body.surface_index_count / 3U;
            const auto soft_groups = (triangles + block - 1U) / block;
            const float soft_correction[3]{
                simulation::soft_body_realtime_maximum_correction_fraction *
                    body.realtime_characteristic_length,
                simulation::soft_body_realtime_maximum_correction_fraction *
                    body.realtime_characteristic_length,
                simulation::soft_body_realtime_maximum_correction_fraction *
                    body.realtime_characteristic_length};
            native_.emit_deformable_spatial_primitives(
                soft_groups, block, native_.queue(), body.surface_positions,
                body.realtime_surface_positions, body.surface_vertex_count,
                body.surface_indices,
                body.realtime_surface_triangle_generations, triangles, 3U,
                simulation::deformable_spatial_soft_family, slot,
                body.generation,
                static_cast<std::uint32_t>(body.realtime_topology_revision),
                simulation::deformable_spatial_triangle_primitive,
                body.realtime_leaf_offset,
                state_.realtime_spatial_plan.leaf_capacity,
                body.realtime_query_offset,
                state_.realtime_spatial_plan.query_capacity,
                body.realtime_leaf_offset,
                simulation::deformable_spatial_contact_mask,
                simulation::deformable_spatial_contact_mask,
                body.node_radius, path_deviation, soft_correction, 1.0e-5F,
                snapshot_.metadata, snapshot_.bounds,
                snapshot_.build_flags, query_bounds, query_masks, excluded,
                deformable_query_leaf_sources, snapshot_.query_status);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            dispatch(simulation::RealtimeTelemetryPhase::index,
                     soft_groups);
        }
        for (std::uint32_t slot = 0U; slot < state_.ropes.size(); ++slot) {
            auto &owner = state_.ropes[slot];
            if (!owner || !owner->alive) continue;
            auto &rope = *owner;
            const auto segments = rope.data.count - 1U;
            const auto rope_groups = (segments + block - 1U) / block;
            const float rope_correction[3]{
                simulation::rope_realtime_maximum_correction_fraction *
                    rope.data.options.node_spacing,
                simulation::rope_realtime_maximum_correction_fraction *
                    rope.data.options.node_spacing,
                simulation::rope_realtime_maximum_correction_fraction *
                    rope.data.options.node_spacing};
            native_.emit_deformable_spatial_primitives(
                rope_groups, block, native_.queue(), rope.data.positions,
                rope.realtime_positions, rope.data.count,
                rope.realtime_segment_indices,
                rope.realtime_segment_generations, segments, 2U,
                simulation::deformable_spatial_rope_family, slot,
                rope.generation,
                static_cast<std::uint32_t>(rope.realtime_topology_revision),
                simulation::deformable_spatial_segment_primitive,
                rope.realtime_leaf_offset,
                state_.realtime_spatial_plan.leaf_capacity,
                rope.realtime_query_offset,
                state_.realtime_spatial_plan.query_capacity,
                rope.realtime_leaf_offset,
                simulation::deformable_spatial_contact_mask,
                simulation::deformable_spatial_contact_mask,
                rope.data.options.radius, path_deviation, rope_correction,
                1.0e-5F, snapshot_.metadata, snapshot_.bounds,
                snapshot_.build_flags, query_bounds, query_masks, excluded,
                deformable_query_leaf_sources, snapshot_.query_status);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            dispatch(simulation::RealtimeTelemetryPhase::index,
                     rope_groups);
        }
        for (std::uint32_t slot = 0U; slot < state_.smokes.size(); ++slot) {
            auto &owner = state_.smokes[slot];
            if (!owner || !owner->alive) continue;
            auto &smoke = *owner;
            const auto smoke_groups =
                (smoke.options.capacity + block - 1U) / block;
            const float support = smoke.options.grid_resolution == 0U
                ? 3.0F * smoke.options.particle_radius
                : smoke.grid.spacing;
            const float smoke_correction[3]{
                simulation::smoke_realtime_maximum_correction_fraction *
                    support,
                simulation::smoke_realtime_maximum_correction_fraction *
                    support,
                simulation::smoke_realtime_maximum_correction_fraction *
                    support};
            native_.emit_fluid_spatial_particles(
                smoke_groups, block, native_.queue(), smoke.positions,
                smoke.realtime_positions, smoke.realtime_stable_ids,
                smoke.realtime_count, smoke.options.capacity,
                slot | smoke_realtime_owner_tag, smoke.generation,
                static_cast<std::uint32_t>(
                    scene.graph_revision %
                    (std::numeric_limits<std::uint32_t>::max() - 1U)) + 1U,
                smoke.realtime_leaf_offset,
                state_.realtime_spatial_plan.leaf_capacity,
                smoke.realtime_query_offset,
                state_.realtime_spatial_plan.query_capacity,
                smoke.realtime_leaf_offset, smoke_realtime_spatial_mask,
                smoke_realtime_spatial_mask, 0.0F, support,
                path_deviation, smoke_correction, 1.0e-5F,
                snapshot_.metadata, snapshot_.bounds,
                snapshot_.build_flags, query_bounds, query_masks, excluded,
                query_leaf_sources, snapshot_.query_status);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            dispatch(simulation::RealtimeTelemetryPhase::index,
                     smoke_groups);
        }
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
            excluded, staged_leaves_);
        if (result != PM_SUCCESS) return result;
        result = state_.realtime_spatial.record_device_update(
            simulation::SpatialUpdateKind::build);
        if (result != PM_SUCCESS) return result;
        const auto budget = simulation::spatial_build_dispatch_budget(
            state_.realtime_spatial_plan.leaf_capacity,
            staged_leaves_);
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
        auto *query_bounds = state_.realtime_query_bounds;
        auto *query_masks = state_.realtime_query_masks;
        auto *excluded = state_.realtime_excluded_sort_keys;
#if defined(PARALLEL_MATER_SLANG_CPU_DRIVER)
        for (std::uint32_t index = 0U; index < staged_queries_;
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
            staged_queries_,
            state_.options.realtime_spatial_candidates_per_query);
        if (result == PM_SUCCESS)
            result = state_.realtime_spatial.record_device_queries(
                epoch_, staged_queries_);
        if (result != PM_SUCCESS) return result;
        if (staged_queries_ != 0U)
            dispatch(simulation::RealtimeTelemetryPhase::candidates,
                     (staged_queries_ + 63U) / 64U);
#endif
        constexpr std::uint32_t block = 128U;
        const auto candidates_per_query =
            state_.options.realtime_spatial_candidates_per_query;
        auto *query_leaf_sources = fluid_spatial<std::uint32_t>(
            state_.realtime_fluid_spatial_plan.query_leaf_sources);
        auto *filtered_counts = fluid_spatial<std::uint32_t>(
            state_.realtime_fluid_spatial_plan.filtered_counts);
        auto *filtered_sources = fluid_spatial<std::uint32_t>(
            state_.realtime_fluid_spatial_plan.filtered_sources);
        for (std::uint32_t slot = 0U; slot < state_.fluids.size(); ++slot) {
            auto &owner = state_.fluids[slot];
            if (!owner || !owner->alive) continue;
            auto &fluid = *owner;
            const auto groups =
                (fluid.options.capacity + block - 1U) / block;
            const auto query_offset = fluid.realtime_query_offset;
            native_.filter_fluid_spatial_candidates(
                groups, block, native_.queue(), snapshot_.metadata,
                snapshot_.admission,
                state_.realtime_spatial_plan.leaf_capacity,
                query_leaf_sources + query_offset, fluid.options.capacity,
                slot, fluid.generation, fluid.ids, fluid.count,
                fluid.options.capacity, slot, fluid.generation, fluid.ids,
                fluid.count, fluid.options.capacity,
                snapshot_.query_counts + query_offset,
                snapshot_.query_candidates +
                    std::size_t(query_offset) * candidates_per_query,
                candidates_per_query, filtered_counts + query_offset,
                filtered_sources +
                    std::size_t(query_offset) * candidates_per_query,
                snapshot_.query_status);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            dispatch(simulation::RealtimeTelemetryPhase::candidates,
                     groups);
        }
        for (std::uint32_t slot = 0U; slot < state_.smokes.size(); ++slot) {
            auto &owner = state_.smokes[slot];
            if (!owner || !owner->alive) continue;
            auto &smoke = *owner;
            if (smoke.options.grid_resolution != 0U) continue;
            const auto groups =
                (smoke.options.capacity + block - 1U) / block;
            const auto query_offset = smoke.realtime_query_offset;
            auto *raw_sources = filtered_sources +
                std::size_t(query_offset) * candidates_per_query;
            auto *dense_sources = deformable_spatial<std::uint32_t>(
                state_.realtime_deformable_spatial_plan.filtered_sources) +
                std::size_t(query_offset) * candidates_per_query;
            const auto raw_bytes =
                std::size_t(smoke.realtime_candidate_map_count) *
                sizeof(std::uint32_t);
            result = native_.clear(raw_sources, 0, raw_bytes,
                native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(smoke.realtime_candidate_counts, 0,
                    std::size_t(smoke.options.capacity) *
                        sizeof(std::uint32_t),
                    native_.queue());
            if (result != PM_SUCCESS) return result;
            native_.filter_fluid_spatial_candidates(
                groups, block, native_.queue(), snapshot_.metadata,
                snapshot_.admission,
                state_.realtime_spatial_plan.leaf_capacity,
                query_leaf_sources + query_offset, smoke.options.capacity,
                slot | smoke_realtime_owner_tag, smoke.generation,
                smoke.realtime_stable_ids, smoke.realtime_count,
                smoke.options.capacity, slot | smoke_realtime_owner_tag,
                smoke.generation, smoke.realtime_stable_ids,
                smoke.realtime_count, smoke.options.capacity,
                snapshot_.query_counts + query_offset,
                snapshot_.query_candidates +
                    std::size_t(query_offset) * candidates_per_query,
                candidates_per_query, smoke.realtime_candidate_counts,
                raw_sources,
                snapshot_.query_status);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            result = native_.copy(smoke.realtime_candidate_slot_count,
                &smoke.realtime_candidate_map_count,
                sizeof(*smoke.realtime_candidate_slot_count),
                device::CopyDirection::host_to_device, native_.queue());
            if (result != PM_SUCCESS) return result;
            const auto map_groups =
                (smoke.realtime_candidate_map_count + block - 1U) / block;
            result = native_.gather_rigid_spatial_pairs(
                map_groups, block, native_.queue(), raw_sources,
                smoke.realtime_candidate_slot_count,
                state_.realtime_smoke_payload_refs,
                state_.options.realtime_spatial_leaf_capacity,
                dense_sources, snapshot_.query_status);
            if (result != PM_SUCCESS) return result;
            result = native_.copy(smoke.realtime_candidate_slot_count,
                &smoke.realtime_candidate_compact_count,
                sizeof(*smoke.realtime_candidate_slot_count),
                device::CopyDirection::host_to_device, native_.queue());
            if (result != PM_SUCCESS) return result;
            const auto compact_groups =
                (smoke.realtime_candidate_compact_count + block - 1U) /
                block;
            result = native_.gather_rigid_spatial_pairs(
                compact_groups, block, native_.queue(),
                smoke.realtime_candidate_indices,
                smoke.realtime_candidate_slot_count, dense_sources,
                smoke.realtime_candidate_map_count,
                smoke.realtime_candidate_indices, snapshot_.query_status);
            if (result != PM_SUCCESS) return result;
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            dispatches(simulation::RealtimeTelemetryPhase::candidates, 3U,
                std::uint64_t{groups} + map_groups + compact_groups);
            auto &candidate_work = work(
                simulation::RealtimeTelemetryPhase::candidates);
            candidate_work.bytes_moved += raw_bytes +
                std::size_t(smoke.options.capacity) *
                    sizeof(std::uint32_t) +
                2U * sizeof(std::uint32_t);
            candidate_work.barriers += 4U;
        }
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
        if (domain == simulation::RealtimeDomain::fluid) {
            ++work(simulation::RealtimeTelemetryPhase::domain)
                  .numerical_iterations;
            const auto *counters = state_.realtime_spatial.counters(epoch_);
            if (counters == nullptr) return PM_ERROR_INVALID_STATE;
            constexpr std::uint32_t block = 128U;
            const auto candidates_per_query =
                state_.options.realtime_spatial_candidates_per_query;
            auto *query_leaf_sources = fluid_spatial<std::uint32_t>(
                state_.realtime_fluid_spatial_plan.query_leaf_sources);
            auto *filtered_counts = fluid_spatial<std::uint32_t>(
                state_.realtime_fluid_spatial_plan.filtered_counts);
            auto *filtered_sources = fluid_spatial<std::uint32_t>(
                state_.realtime_fluid_spatial_plan.filtered_sources);
            for (auto &owner : state_.fluids) {
                if (!owner || !owner->alive) continue;
                auto &fluid = *owner;
                const auto groups =
                    (fluid.options.capacity + block - 1U) / block;
                const auto query_offset = fluid.realtime_query_offset;
                auto *local_query_sources =
                    query_leaf_sources + query_offset;
                auto *local_filtered_counts =
                    filtered_counts + query_offset;
                auto *local_filtered_sources = filtered_sources +
                    std::size_t(query_offset) * candidates_per_query;
                const auto constants = native_.fluid_constants(
                    fluid.options, options_.gravity, scene.timestep,
                    fluid.options.capacity);
                if (iteration == 0U) {
                    const auto budget = simulation::fluid_realtime_work_budget(
                        fluid.options.capacity, fluid.options.capacity,
                        candidates_per_query);
                    auto result = fluid.realtime_contract.begin(
                        epoch_, *counters, budget);
                    if (result != PM_SUCCESS) return result;
                    result = fluid.realtime_contract.record_advance(
                        scene.timestep, scene.substeps);
                    if (result != PM_SUCCESS) return result;
                }
                native_.solve_fluid_realtime_density(
                    groups, block, native_.queue(), snapshot_.metadata,
                    snapshot_.admission,
                    state_.realtime_spatial_plan.leaf_capacity,
                    local_query_sources, fluid.options.capacity,
                    fluid.next_positions, fluid.count,
                    fluid.options.capacity, local_filtered_counts,
                    local_filtered_sources, candidates_per_query, constants,
                    scene.timestep,
                    simulation::fluid_realtime_profile_version,
                    fluid.foam_source, fluid.next_foam,
                    snapshot_.query_status);
                auto result = native_.check_error();
                if (result == PM_SUCCESS) {
                    native_.solve_fluid_realtime_position_delta(
                        groups, block, native_.queue(), snapshot_.metadata,
                        snapshot_.admission,
                        state_.realtime_spatial_plan.leaf_capacity,
                        local_query_sources, fluid.options.capacity,
                        fluid.next_positions, fluid.count,
                        fluid.options.capacity, local_filtered_counts,
                        local_filtered_sources, candidates_per_query,
                        constants, scene.timestep,
                        simulation::fluid_realtime_profile_version,
                        fluid.next_foam,
                        fluid.realtime_accumulated_corrections,
                        fluid.realtime_position_deltas,
                        snapshot_.query_status);
                    result = native_.check_error();
                }
                if (result == PM_SUCCESS) {
                    native_.apply_fluid_realtime_position_delta(
                        groups, block, native_.queue(), fluid.next_positions,
                        fluid.realtime_position_deltas,
                        fluid.realtime_accumulated_corrections, fluid.count,
                        fluid.options.capacity, scene.timestep,
                        simulation::fluid_realtime_profile_version, constants,
                        snapshot_.query_status);
                    result = native_.check_error();
                }
                if (result != PM_SUCCESS) return result;
                dispatches(simulation::RealtimeTelemetryPhase::domain, 3U,
                           3ULL * groups);
                result = fluid.realtime_contract.record_sweep(
                    epoch_, *counters);
                if (result != PM_SUCCESS) return result;
                if (iteration + 1U ==
                    simulation::fluid_realtime_density_sweeps) {
                    native_.solve_fluid_realtime_viscosity(
                        groups, block, native_.queue(), snapshot_.metadata,
                        snapshot_.admission,
                        state_.realtime_spatial_plan.leaf_capacity,
                        local_query_sources, fluid.options.capacity,
                        fluid.next_positions, fluid.next_velocities,
                        fluid.count, fluid.options.capacity,
                        local_filtered_counts, local_filtered_sources,
                        candidates_per_query, constants, scene.timestep,
                        simulation::fluid_realtime_profile_version,
                        fluid.foam_source, fluid.realtime_velocity_deltas,
                        snapshot_.query_status);
                    result = native_.check_error();
                    if (result != PM_SUCCESS) return result;
                    dispatch(simulation::RealtimeTelemetryPhase::domain,
                             groups);
                    result = fluid.realtime_contract.finish();
                    if (result != PM_SUCCESS) return result;
                }
            }
            return PM_SUCCESS;
        }
        if (domain == simulation::RealtimeDomain::cloth) {
            ++work(simulation::RealtimeTelemetryPhase::domain)
                  .numerical_iterations;
            const auto *counters = state_.realtime_spatial.counters(epoch_);
            if (counters == nullptr) return PM_ERROR_INVALID_STATE;
            constexpr std::uint32_t block = 128U;
            for (auto &owner : state_.cloths) {
                if (!owner || !owner->alive) continue;
                auto &cloth = *owner;
                const auto groups =
                    (cloth.vertex_capacity + block - 1U) / block;
                const auto triangles = cloth.index_count / 3U;
                if (iteration == 0U) {
                    const auto budget = simulation::cloth_realtime_work_budget(
                        cloth.vertex_count, cloth.vertex_capacity, triangles,
                        cloth.neighbor_count, cloth.vertex_count);
                    auto result = cloth.realtime_contract.begin(
                        epoch_, *counters, budget,
                        cloth.realtime_topology_revision);
                    if (result != PM_SUCCESS) return result;
                    result = cloth.realtime_contract.record_advance(
                        scene.timestep, scene.substeps);
                    if (result != PM_SUCCESS) return result;
                }
                native_.project_cloth_realtime_links(
                    groups, block, native_.queue(),
                    cloth.realtime_positions, cloth.scratch,
                    cloth.realtime_correction_used, cloth.inverse_masses,
                    cloth.realtime_characteristic_lengths, cloth.offsets,
                    cloth.neighbors, cloth.bond_active, cloth.vertex_count,
                    cloth.vertex_capacity, scene.timestep,
                    simulation::cloth_realtime_profile_version,
                    snapshot_.query_status);
                native_.apply_cloth_realtime_position_delta(
                    groups, block, native_.queue(),
                    cloth.realtime_positions, cloth.scratch,
                    cloth.realtime_correction_used, cloth.inverse_masses,
                    cloth.realtime_characteristic_lengths,
                    cloth.vertex_count, cloth.vertex_capacity,
                    scene.timestep,
                    simulation::cloth_realtime_profile_version,
                    snapshot_.query_status);
                native_.project_cloth_realtime_volume(
                    native_.queue(), cloth.realtime_positions,
                    cloth.inverse_masses,
                    cloth.realtime_characteristic_lengths,
                    cloth.realtime_correction_used, cloth.indices,
                    cloth.vertex_count, triangles,
                    cloth.volume_corner_offsets,
                    cloth.volume_corner_indices, cloth.volume_gradients,
                    cloth.target_volume, cloth.orientation,
                    cloth.volume_compliance, cloth.preserve_volume,
                    scene.timestep,
                    simulation::cloth_realtime_profile_version,
                    cloth.volume_lambda, iteration == 0U,
                    snapshot_.query_status);
                auto result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatches(simulation::RealtimeTelemetryPhase::domain, 3U,
                    2ULL * groups + 1U);
                const bool contact_pass = (iteration & 1U) != 0U;
                if (contact_pass) {
                    native_.project_cloth_realtime_contacts(
                        groups, block, native_.queue(),
                        cloth.realtime_positions, cloth.previous,
                        cloth.inverse_masses,
                        cloth.realtime_characteristic_lengths,
                        cloth.realtime_correction_used,
                        cloth.realtime_contact_normals,
                        cloth.realtime_contact_offsets,
                        cloth.realtime_contact_active,
                        cloth.vertex_count, cloth.vertex_capacity,
                        cloth.thickness, cloth.contact_friction,
                        scene.timestep,
                        simulation::cloth_realtime_profile_version,
                        snapshot_.query_status);
                    result = native_.check_error();
                    if (result != PM_SUCCESS) return result;
                    dispatch(simulation::RealtimeTelemetryPhase::domain,
                             groups);
                }
                result = cloth.realtime_contract.record_sweep(
                    epoch_, *counters, cloth.realtime_topology_revision,
                    contact_pass);
                if (result != PM_SUCCESS) return result;
                if (iteration + 1U ==
                    simulation::cloth_realtime_constraint_sweeps) {
                    result = cloth.realtime_contract.finish();
                    if (result != PM_SUCCESS) return result;
                }
            }
            return PM_SUCCESS;
        }
        if (domain == simulation::RealtimeDomain::soft_body) {
            ++work(simulation::RealtimeTelemetryPhase::domain)
                  .numerical_iterations;
            const auto *counters = state_.realtime_spatial.counters(epoch_);
            if (counters == nullptr) return PM_ERROR_INVALID_STATE;
            constexpr std::uint32_t block = 128U;
            for (auto &owner : state_.soft_bodies) {
                if (!owner || !owner->alive) continue;
                auto &body = *owner;
                const auto groups =
                    (body.node_count + block - 1U) / block;
                if (iteration == 0U) {
                    const auto budget =
                        simulation::soft_body_realtime_work_budget(
                            body.node_count, body.node_count,
                            body.neighbor_count, body.node_count);
                    auto result = body.realtime_contract.begin(
                        epoch_, *counters, budget,
                        body.realtime_topology_revision);
                    if (result != PM_SUCCESS) return result;
                    result = body.realtime_contract.record_advance(
                        scene.timestep, scene.substeps);
                    if (result != PM_SUCCESS) return result;
                }
                native_.project_soft_body_realtime_shape(
                    native_.queue(), body.realtime_positions, body.scratch,
                    body.rest_positions, body.inverse_masses,
                    body.realtime_characteristic_lengths,
                    body.realtime_correction_used, body.node_count,
                    body.node_count, body.movable_mass,
                    &body.shape_rest_center.x, &body.shape_inverse_rest,
                    body.realtime_shape_orientation,
                    body.shape_matching_stiffness, scene.timestep,
                    simulation::soft_body_realtime_profile_version,
                    snapshot_.query_status);
                native_.project_soft_body_realtime_links(
                    groups, block, native_.queue(),
                    body.realtime_positions, body.scratch,
                    body.realtime_correction_used, body.inverse_masses,
                    body.realtime_characteristic_lengths, body.offsets,
                    body.realtime_neighbors, body.node_count,
                    body.node_count, body.stretch_compliance,
                    scene.timestep,
                    simulation::soft_body_realtime_profile_version,
                    snapshot_.query_status);
                native_.apply_soft_body_realtime_position_delta(
                    groups, block, native_.queue(),
                    body.realtime_positions, body.scratch,
                    body.realtime_correction_used, body.inverse_masses,
                    body.realtime_characteristic_lengths,
                    body.node_count, body.node_count, scene.timestep,
                    simulation::soft_body_realtime_profile_version,
                    snapshot_.query_status);
                auto result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatches(simulation::RealtimeTelemetryPhase::domain, 3U,
                    2ULL * groups + 1U);
                const bool contact_pass = (iteration & 1U) != 0U;
                if (contact_pass) {
                    native_.project_soft_body_realtime_contacts(
                        groups, block, native_.queue(),
                        body.realtime_positions, body.previous,
                        body.inverse_masses,
                        body.realtime_characteristic_lengths,
                        body.realtime_correction_used,
                        body.realtime_contact_normals,
                        body.realtime_contact_offsets,
                        body.realtime_contact_active, body.node_count,
                        body.node_count, body.node_radius,
                        body.contact_friction, scene.timestep,
                        simulation::soft_body_realtime_profile_version,
                        snapshot_.query_status);
                    result = native_.check_error();
                    if (result != PM_SUCCESS) return result;
                    dispatch(simulation::RealtimeTelemetryPhase::domain,
                             groups);
                }
                result = body.realtime_contract.record_sweep(
                    epoch_, *counters, body.realtime_topology_revision,
                    contact_pass);
                if (result != PM_SUCCESS) return result;
                if (iteration + 1U ==
                    simulation::soft_body_realtime_constraint_sweeps) {
                    result = body.realtime_contract.finish();
                    if (result != PM_SUCCESS) return result;
                }
            }
            return PM_SUCCESS;
        }
        if (domain == simulation::RealtimeDomain::rope) {
            ++work(simulation::RealtimeTelemetryPhase::domain)
                  .numerical_iterations;
            const auto *counters = state_.realtime_spatial.counters(epoch_);
            if (counters == nullptr) return PM_ERROR_INVALID_STATE;
            constexpr std::uint32_t block = 128U;
            for (auto &owner : state_.ropes) {
                if (!owner || !owner->alive) continue;
                auto &rope = *owner;
                const auto groups =
                    (rope.data.count + block - 1U) / block;
                if (iteration == 0U) {
                    const auto budget = simulation::rope_realtime_work_budget(
                        rope.data.count, rope.data.count, rope.data.count);
                    auto result = rope.realtime_contract.begin(
                        epoch_, *counters, budget,
                        rope.realtime_topology_revision);
                    if (result != PM_SUCCESS) return result;
                    result = rope.realtime_contract.record_advance(
                        scene.timestep, scene.substeps);
                    if (result != PM_SUCCESS) return result;
                }
                native_.project_rope_realtime_constraints(
                    native_.queue(), rope.realtime_positions,
                    rope.realtime_position_deltas,
                    rope.realtime_correction_used,
                    rope.realtime_inverse_masses, rope.data.rest,
                    rope.realtime_rest_bend_lengths,
                    rope.realtime_anchor_active, rope.data.count,
                    rope.data.count, rope.data.options.node_spacing,
                    rope.data.options.stretch_compliance, scene.timestep,
                    simulation::rope_realtime_profile_version,
                    snapshot_.query_status);
                native_.apply_rope_realtime_position_delta(
                    groups, block, native_.queue(),
                    rope.realtime_positions,
                    rope.realtime_position_deltas,
                    rope.realtime_correction_used,
                    rope.realtime_inverse_masses,
                    rope.realtime_anchor_active, rope.data.count,
                    rope.data.count, rope.data.options.node_spacing,
                    scene.timestep,
                    simulation::rope_realtime_profile_version,
                    snapshot_.query_status);
                auto result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatches(simulation::RealtimeTelemetryPhase::domain, 2U,
                           1U + groups);
                const bool contact_pass = (iteration & 1U) != 0U;
                if (contact_pass) {
                    native_.project_rope_realtime_contacts(
                        groups, block, native_.queue(),
                        rope.realtime_positions,
                        rope.realtime_previous_positions,
                        rope.realtime_inverse_masses,
                        rope.realtime_correction_used,
                        rope.realtime_anchor_active,
                        rope.realtime_contact_normals,
                        rope.realtime_contact_offsets,
                        rope.realtime_contact_active, rope.data.count,
                        rope.data.count, rope.data.options.node_spacing,
                        rope.data.options.radius, rope.data.options.friction,
                        scene.timestep,
                        simulation::rope_realtime_profile_version,
                        snapshot_.query_status);
                    result = native_.check_error();
                    if (result != PM_SUCCESS) return result;
                    dispatch(simulation::RealtimeTelemetryPhase::domain,
                             groups);
                }
                result = rope.realtime_contract.record_sweep(
                    epoch_, *counters, rope.realtime_topology_revision,
                    contact_pass);
                if (result != PM_SUCCESS) return result;
                if (iteration + 1U ==
                    simulation::rope_realtime_constraint_sweeps) {
                    result = rope.realtime_contract.finish();
                    if (result != PM_SUCCESS) return result;
                }
            }
            return PM_SUCCESS;
        }
        if (domain == simulation::RealtimeDomain::smoke)
            return solve_smoke_domain(iteration, scene);
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
        simulation::RealtimeCouplingFamily family, std::uint32_t sweep,
        std::uint32_t edge_count,
        const simulation::RealtimeScheduleScene &scene) noexcept {
        if (family == simulation::RealtimeCouplingFamily::liquid)
            return solve_liquid_couplings(sweep, edge_count, scene);
        auto *counters = state_.realtime_spatial.counters(epoch_);
        if (counters == nullptr) return PM_ERROR_INVALID_STATE;
        auto &coupling_work = work(
            simulation::RealtimeTelemetryPhase::coupling);
        ++coupling_work.numerical_iterations;
        constexpr std::uint32_t block = 128U;

        if (family == simulation::RealtimeCouplingFamily::rigid) {
            if (edge_count != rigid_coupling_count_) return PM_ERROR_INVALID_STATE;
            if (sweep == 0U) {
                std::uint64_t endpoint_count{};
                for (const auto &owner : state_.cloths)
                    if (owner && owner->alive)
                        endpoint_count += owner->vertex_count;
                for (const auto &owner : state_.soft_bodies)
                    if (owner && owner->alive)
                        endpoint_count += owner->node_count;
                for (const auto &owner : state_.ropes)
                    if (owner && owner->alive)
                        endpoint_count += owner->data.count;
                for (const auto &owner : state_.smokes)
                    if (owner && owner->alive)
                        endpoint_count += owner->options.capacity;
                if (endpoint_count > std::numeric_limits<std::uint32_t>::max())
                    return PM_ERROR_CAPACITY_EXCEEDED;
                simulation::RigidCouplingRealtimeWorkBudget budget{};
                auto result = simulation::make_rigid_coupling_realtime_work_budget(
                    std::span<const simulation::RigidCouplingRegistration>(
                        rigid_couplings_.data(), rigid_coupling_count_),
                    static_cast<std::uint32_t>(endpoint_count),
                    state_.rigid_body_count, budget);
                if (result != PM_SUCCESS) return result;
                result = rigid_coupling_contract_.begin(epoch_, *counters,
                    budget,
                    std::span<const simulation::RigidCouplingRegistration>(
                        rigid_couplings_.data(), rigid_coupling_count_),
                    scene.graph_revision);
                if (result != PM_SUCCESS) return result;
                stage_smoke_rigid_measurement();
            }
            for (std::uint32_t slot = 0U; slot < state_.cloths.size(); ++slot) {
                auto &owner = state_.cloths[slot];
                if (!owner || !owner->alive) continue;
                auto &cloth = *owner;
                const auto groups =
                    (cloth.vertex_count + block - 1U) / block;
                auto result = native_.clear(state_.fluid_body_contact_flags, 0,
                    state_.rigid_body_count * sizeof(std::uint32_t),
                    native_.queue());
                if (result != PM_SUCCESS) return result;
                native_.collide_deformable_with_rigid_geometry(
                    groups, block, native_.queue(), cloth.realtime_positions,
                    cloth.realtime_velocities, cloth.previous,
                    cloth.inverse_masses, cloth.vertex_count, cloth.thickness,
                    scene.timestep, state_.parameters,
                    state_.render_previous_states,
                    state_.states[output_state_], state_.meshes,
                    state_.rigid_body_count, cloth.body_impulses,
                    cloth.rigid_contact_forces, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr, state_.fluid_body_contact_flags,
                    false, false, false, false,
                    cloth.realtime_maximum_speed);
                native_.reduce_fluid_body_impulses(
                    state_.rigid_body_count, block, native_.queue(),
                    cloth.body_impulses, cloth.count, state_.parameters,
                    state_.states[output_state_],
                    state_.fluid_body_contact_flags,
                    state_.rigid_body_count);
                const auto triangles = cloth.index_count / 3U;
                native_.update_cloth_surface(
                    (triangles + block - 1U) / block, block, native_.queue(),
                    cloth.realtime_positions, cloth.indices,
                    cloth.realtime_surface_positions, triangles);
                native_.constrain_rigid_bodies_with_cloth(
                    state_.rigid_body_count, block, native_.queue(),
                    cloth.realtime_positions, cloth.realtime_velocities,
                    cloth.indices, cloth.vertex_sources,
                    cloth.realtime_surface_positions, cloth.inverse_masses,
                    cloth.vertex_count, triangles, cloth.thickness,
                    scene.timestep, cloth.contact_friction, state_.parameters,
                    state_.render_previous_states,
                    state_.states[output_state_], state_.meshes, state_.ids,
                    state_.rigid_body_count, slot, cloth.generation,
                    nullptr, 0U, nullptr, 0U, cloth.body_corrections);
                native_.apply_cloth_body_corrections(
                    groups, block, native_.queue(), cloth.realtime_positions,
                    cloth.realtime_velocities, cloth.inverse_masses,
                    cloth.vertex_count, cloth.body_corrections,
                    state_.rigid_body_count);
                result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatches(simulation::RealtimeTelemetryPhase::coupling, 5U,
                    std::uint64_t{groups} * 2U +
                        (triangles + block - 1U) / block +
                        2ULL * state_.rigid_body_count);
                coupling_work.bytes_moved +=
                    state_.rigid_body_count * sizeof(std::uint32_t);
                ++coupling_work.barriers;
            }
            for (auto &owner : state_.soft_bodies) {
                if (!owner || !owner->alive) continue;
                auto &body = *owner;
                const auto groups = (body.node_count + block - 1U) / block;
                auto result = native_.clear(state_.fluid_body_contact_flags, 0,
                    state_.rigid_body_count * sizeof(std::uint32_t),
                    native_.queue());
                if (result != PM_SUCCESS) return result;
                native_.collide_deformable_with_rigid_geometry(
                    groups, block, native_.queue(), body.realtime_positions,
                    body.realtime_velocities, body.previous,
                    body.inverse_masses, body.node_count, body.node_radius,
                    scene.timestep, state_.parameters,
                    state_.render_previous_states,
                    state_.states[output_state_], state_.meshes,
                    state_.rigid_body_count, body.body_impulses,
                    body.rigid_contact_forces, body.contact_normals,
                    body.contact_arms, body.contact_momentum_delta,
                    body.contact_normal_delta, body.contact_count,
                    body.dynamic_contact_flag, state_.fluid_body_contact_flags,
                    true, false, false, false, body.realtime_maximum_speed,
                    body.body_position_corrections);
                native_.apply_soft_body_contact_friction(
                    groups, block, native_.queue(), body.realtime_positions,
                    body.realtime_velocities, body.inverse_masses,
                    body.rigid_contact_forces, body.contact_normals,
                    body.contact_arms, body.body_impulses, state_.parameters,
                    body.contact_momentum_delta, body.contact_friction_delta,
                    body.contact_normal_delta, body.contact_count,
                    body.node_count, body.movable_mass, &options_.gravity.x,
                    scene.timestep, body.contact_friction,
                    body.realtime_maximum_speed);
                native_.reduce_fluid_body_impulses(
                    state_.rigid_body_count, block, native_.queue(),
                    body.body_impulses, body.count, state_.parameters,
                    state_.states[output_state_],
                    state_.fluid_body_contact_flags,
                    state_.rigid_body_count, body.body_position_corrections);
                native_.update_soft_body_surface(
                    (body.surface_vertex_count + block - 1U) / block,
                    block, native_.queue(), body.realtime_positions,
                    body.rest_positions, body.surface_rest_positions,
                    body.surface_bindings, body.realtime_surface_positions,
                    body.surface_vertex_count);
                result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatches(simulation::RealtimeTelemetryPhase::coupling, 4U,
                    2ULL * groups + state_.rigid_body_count +
                        (body.surface_vertex_count + block - 1U) / block);
                coupling_work.bytes_moved +=
                    state_.rigid_body_count * sizeof(std::uint32_t);
                ++coupling_work.barriers;
            }
            for (auto &owner : state_.ropes) {
                if (!owner || !owner->alive) continue;
                auto &rope = *owner;
                const auto groups =
                    (rope.data.count + block - 1U) / block;
                auto result = native_.clear(state_.fluid_body_contact_flags, 0,
                    state_.rigid_body_count * sizeof(std::uint32_t),
                    native_.queue());
                if (result != PM_SUCCESS) return result;
                native_.collide_deformable_with_rigid_geometry(
                    groups, block, native_.queue(), rope.realtime_positions,
                    rope.realtime_velocities,
                    rope.realtime_previous_positions,
                    rope.realtime_inverse_masses, rope.data.count,
                    rope.data.options.radius, scene.timestep,
                    state_.parameters, state_.render_previous_states,
                    state_.states[output_state_], state_.meshes,
                    state_.rigid_body_count, rope.realtime_body_impulses,
                    rope.realtime_rigid_contact_forces, nullptr, nullptr,
                    nullptr, nullptr, nullptr, nullptr,
                    state_.fluid_body_contact_flags, true, false, false,
                    false, rope.data.options.maximum_speed,
                    rope.realtime_body_position_corrections);
                native_.reduce_fluid_body_impulses(
                    state_.rigid_body_count, block, native_.queue(),
                    rope.realtime_body_impulses, rope.realtime_count,
                    state_.parameters, state_.states[output_state_],
                    state_.fluid_body_contact_flags,
                    state_.rigid_body_count,
                    rope.realtime_body_position_corrections);
                result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatches(simulation::RealtimeTelemetryPhase::coupling, 2U,
                    groups + state_.rigid_body_count);
                coupling_work.bytes_moved +=
                    state_.rigid_body_count * sizeof(std::uint32_t);
                ++coupling_work.barriers;
            }
            std::uint32_t record{};
            for (auto &owner : state_.ropes) {
                if (!owner || !owner->alive) continue;
                auto &rope = *owner;
                const RopeAttachment attachments[2]{
                    rope.data.options.first, rope.data.options.last};
                for (std::uint32_t end = 0U; end < 2U; ++end) {
                    if (!attachments[end].enabled) continue;
                    auto result = solve_rigid_rope_attachment(
                        rope, end, record++, sweep, scene);
                    if (result != PM_SUCCESS) return result;
                }
            }
            if (record != rigid_attachment_records_)
                return PM_ERROR_INVALID_STATE;
            for (std::uint32_t index = 0U;
                 index < rigid_coupling_count_; ++index) {
                const auto result = rigid_coupling_contract_.record_edge(
                    epoch_, *counters, scene.graph_revision,
                    rigid_couplings_[index]);
                if (result != PM_SUCCESS) return result;
            }
            return sweep + 1U == simulation::rigid_coupling_realtime_sweeps
                ? rigid_coupling_contract_.finish() : PM_SUCCESS;
        }

        if (family != simulation::RealtimeCouplingFamily::deformable_smoke ||
            edge_count != deformable_coupling_count_)
            return PM_ERROR_INVALID_STATE;
        if (sweep == 0U) {
            std::uint64_t first_endpoints{};
            std::uint64_t second_endpoints{};
            for (const auto &owner : state_.cloths)
                if (owner && owner->alive) {
                    first_endpoints += owner->vertex_count;
                    second_endpoints += owner->vertex_count;
                }
            for (const auto &owner : state_.soft_bodies)
                if (owner && owner->alive) {
                    first_endpoints += owner->node_count;
                    second_endpoints += owner->node_count;
                }
            for (const auto &owner : state_.ropes)
                if (owner && owner->alive) {
                    first_endpoints += owner->data.count;
                    second_endpoints += owner->data.count;
                }
            for (const auto &owner : state_.smokes)
                if (owner && owner->alive)
                    second_endpoints += owner->options.capacity;
            if (first_endpoints > std::numeric_limits<std::uint32_t>::max() ||
                second_endpoints > std::numeric_limits<std::uint32_t>::max())
                return PM_ERROR_CAPACITY_EXCEEDED;
            simulation::DeformableSmokeCouplingWorkBudget budget{};
            auto result =
                simulation::make_deformable_smoke_coupling_work_budget(
                    std::span<const
                        simulation::DeformableSmokeCouplingRegistration>(
                            deformable_couplings_.data(),
                            deformable_coupling_count_),
                    static_cast<std::uint32_t>(first_endpoints),
                    static_cast<std::uint32_t>(second_endpoints), budget);
            if (result != PM_SUCCESS) return result;
            result = deformable_coupling_contract_.begin(epoch_, *counters,
                budget,
                std::span<const
                    simulation::DeformableSmokeCouplingRegistration>(
                        deformable_couplings_.data(),
                        deformable_coupling_count_), scene.graph_revision);
            if (result != PM_SUCCESS) return result;
            stage_smoke_deformable_measurement();
        }
        for (auto &owner : state_.soft_cloth_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            auto &coupling = *owner;
            auto &body = *state_.soft_bodies[
                coupling.options.soft_body.index];
            auto &cloth = *state_.cloths[coupling.options.cloth.index];
            auto result = native_.clear(coupling.cloth_contact_counts, 0,
                cloth.vertex_count * sizeof(std::uint32_t), native_.queue());
            if (result != PM_SUCCESS) return result;
            const auto body_groups =
                (body.node_count + block - 1U) / block;
            const auto cloth_groups =
                (cloth.vertex_count + block - 1U) / block;
            const auto cloth_triangles = cloth.index_count / 3U;
            native_.update_cloth_surface(
                (cloth_triangles + block - 1U) / block, block,
                native_.queue(), cloth.realtime_positions, cloth.indices,
                cloth.realtime_surface_positions, cloth_triangles);
            native_.update_soft_body_surface(
                (body.surface_vertex_count + block - 1U) / block, block,
                native_.queue(), body.realtime_positions,
                body.rest_positions, body.surface_rest_positions,
                body.surface_bindings, body.realtime_surface_positions,
                body.surface_vertex_count);
            const float distance = coupling.options.contact_distance > 0.0F
                ? coupling.options.contact_distance
                : body.node_radius + cloth.thickness;
            native_.detect_soft_body_cloth_contacts(
                body_groups, block, native_.queue(),
                body.realtime_positions, body.previous,
                body.realtime_velocities, body.inverse_masses,
                body.node_count, cloth.realtime_positions, cloth.previous,
                cloth.realtime_velocities, cloth.inverse_masses,
                cloth.indices, cloth.realtime_surface_positions,
                cloth_triangles, distance, coupling.options.friction,
                coupling.contacts, coupling.cloth_contact_counts);
            native_.apply_soft_body_cloth_contacts_to_soft_body(
                body_groups, block, native_.queue(),
                body.realtime_positions, body.realtime_velocities,
                body.cloth_forces, body.inverse_masses, body.node_count,
                coupling.contacts, coupling.cloth_contact_counts,
                scene.timestep, body.realtime_maximum_speed);
            native_.apply_soft_body_cloth_contacts_to_cloth(
                cloth_groups, block, native_.queue(),
                cloth.realtime_positions, cloth.realtime_velocities,
                cloth.soft_body_forces, cloth.inverse_masses,
                cloth.vertex_count, coupling.contacts, body.node_count,
                coupling.cloth_contact_counts, scene.timestep);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            dispatches(simulation::RealtimeTelemetryPhase::coupling, 5U,
                2ULL * body_groups + 2ULL * cloth_groups +
                    (cloth_triangles + block - 1U) / block);
            coupling_work.bytes_moved +=
                cloth.vertex_count * sizeof(std::uint32_t);
            ++coupling_work.barriers;
        }
        std::uint32_t record = 32U;
        for (auto &coupling : state_.rope_cloth_couplings) {
            if (!coupling.alive || !coupling.options.enabled) continue;
            auto &cloth = *state_.cloths[coupling.options.cloth.index];
            auto &rope = *state_.ropes[coupling.options.rope.index];
            for (std::uint32_t end = 0U; end < 2U; ++end) {
                const auto vertex = end == 0U
                    ? coupling.options.first_vertex
                    : coupling.options.last_vertex;
                if (vertex == UINT32_MAX) continue;
                auto result = solve_deformable_pair(
                    {cloth.realtime_positions + vertex,
                     cloth.previous + vertex,
                     cloth.realtime_velocities + vertex,
                     cloth.inverse_masses + vertex,
                     cloth.realtime_correction_used + vertex,
                     state_.realtime_coupling_first_generations + record, 1U},
                    {rope.realtime_positions +
                         (end == 0U ? 0U : rope.data.count - 1U),
                     rope.realtime_previous_positions +
                         (end == 0U ? 0U : rope.data.count - 1U),
                     rope.realtime_velocities +
                         (end == 0U ? 0U : rope.data.count - 1U),
                     rope.realtime_inverse_masses +
                         (end == 0U ? 0U : rope.data.count - 1U),
                     rope.realtime_correction_used +
                         (end == 0U ? 0U : rope.data.count - 1U),
                     state_.realtime_coupling_second_generations + record, 1U},
                    record, std::max(cloth.realtime_characteristic_length,
                        rope.data.options.node_spacing),
                    std::max(cloth.realtime_maximum_speed,
                        rope.data.options.maximum_speed), sweep, scene);
                if (result != PM_SUCCESS) return result;
                ++record;
            }
        }
        constexpr float zero_offset[3]{};
        for (auto &owner : state_.rope_soft_body_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            auto &coupling = *owner;
            auto &body = *state_.soft_bodies[
                coupling.options.soft_body.index];
            auto &rope = *state_.ropes[coupling.options.rope.index];
            for (std::uint32_t end = 0U; end < 2U; ++end) {
                native_.sample_rope_soft_body_anchor(native_.queue(),
                    rope.data.soft_anchor_positions,
                    rope.data.soft_anchor_velocities,
                    rope.data.soft_anchor_impulses,
                    rope.data.soft_anchor_inverse_masses, end,
                    body.realtime_surface_positions,
                    body.realtime_velocities, body.surface_indices,
                    body.surface_bindings, coupling.anchor_triangle[end],
                    &coupling.anchor_weights[end].x, zero_offset);
                dispatch(simulation::RealtimeTelemetryPhase::coupling, 1U);
                auto result = solve_soft_rope_pair(
                    coupling, body, rope, end, record++, sweep, scene);
                if (result != PM_SUCCESS) return result;
            }
        }
        for (auto &owner : state_.rope_soft_body_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            auto &coupling = *owner;
            auto &body = *state_.soft_bodies[
                coupling.options.soft_body.index];
            auto &rope = *state_.ropes[coupling.options.rope.index];
            for (std::uint32_t end = 0U; end < 2U; ++end) {
                if (!(end == 0U ? coupling.options.attach_first
                                : coupling.options.attach_last))
                    continue;
                native_.sample_rope_soft_body_anchor(native_.queue(),
                    rope.data.soft_anchor_positions,
                    rope.data.soft_anchor_velocities,
                    rope.data.soft_anchor_impulses,
                    rope.data.soft_anchor_inverse_masses, end,
                    body.realtime_surface_positions,
                    body.realtime_velocities, body.surface_indices,
                    body.surface_bindings, coupling.anchor_triangle[end],
                    &coupling.anchor_weights[end].x,
                    &coupling.anchor_offset[end].x);
                dispatch(simulation::RealtimeTelemetryPhase::coupling, 1U);
                auto result = solve_soft_rope_pair(
                    coupling, body, rope, end, record++, sweep, scene);
                if (result != PM_SUCCESS) return result;
            }
        }
        if (record != 32U + deformable_attachment_records_)
            return PM_ERROR_INVALID_STATE;
        for (std::uint32_t index = 0U;
             index < deformable_coupling_count_; ++index) {
            const auto result = deformable_coupling_contract_.record_edge(
                epoch_, *counters, scene.graph_revision,
                deformable_couplings_[index]);
            if (result != PM_SUCCESS) return result;
        }
        return sweep + 1U ==
                simulation::deformable_smoke_coupling_sweeps
            ? deformable_coupling_contract_.finish() : PM_SUCCESS;
    }

    PmResult commit_tick(
        const simulation::RealtimeScheduleScene &scene) noexcept {
        auto result = state_.realtime_resources.prepare_commit();
        if (result != PM_SUCCESS) return result;
        constexpr std::uint32_t block = 128U;
        for (auto &owner : state_.fluids) {
            if (!owner || !owner->alive) continue;
            auto &fluid = *owner;
            const auto groups =
                (fluid.options.capacity + block - 1U) / block;
            const auto constants = native_.fluid_constants(
                fluid.options, options_.gravity, scene.timestep,
                fluid.options.capacity);
            native_.commit_fluid_realtime(
                groups, block, native_.queue(), fluid.next_positions,
                fluid.previous, fluid.next_velocities,
                fluid.realtime_velocity_deltas, fluid.count,
                fluid.options.capacity, constants, scene.timestep,
                simulation::fluid_realtime_profile_version,
                snapshot_.query_status);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
        }
        for (auto &owner : state_.cloths) {
            if (!owner || !owner->alive) continue;
            auto &cloth = *owner;
            const auto groups =
                (cloth.vertex_capacity + block - 1U) / block;
            native_.commit_cloth_realtime(
                groups, block, native_.queue(), cloth.realtime_positions,
                cloth.previous, cloth.realtime_velocities,
                cloth.inverse_masses,
                cloth.realtime_characteristic_lengths,
                cloth.realtime_contact_normals,
                cloth.realtime_contact_offsets,
                cloth.realtime_contact_active, cloth.vertex_count,
                cloth.vertex_capacity, cloth.thickness,
                cloth.contact_friction, cloth.realtime_maximum_speed,
                scene.timestep, simulation::cloth_realtime_profile_version,
                snapshot_.query_status);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            const auto triangles = cloth.index_count / 3U;
            native_.update_cloth_surface(
                (triangles + block - 1U) / block, block, native_.queue(),
                cloth.realtime_positions, cloth.indices,
                cloth.realtime_surface_positions, triangles);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
        }
        for (auto &owner : state_.soft_bodies) {
            if (!owner || !owner->alive) continue;
            auto &body = *owner;
            const auto groups =
                (body.node_count + block - 1U) / block;
            native_.commit_soft_body_realtime(
                groups, block, native_.queue(), body.realtime_positions,
                body.previous, body.realtime_velocities,
                body.inverse_masses,
                body.realtime_characteristic_lengths,
                body.realtime_contact_normals,
                body.realtime_contact_offsets,
                body.realtime_contact_active, body.node_count,
                body.node_count, body.node_radius, body.contact_friction,
                body.realtime_maximum_speed, scene.timestep,
                simulation::soft_body_realtime_profile_version,
                snapshot_.query_status);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            native_.update_soft_body_surface(
                (body.surface_vertex_count + block - 1U) / block,
                block, native_.queue(), body.realtime_positions,
                body.rest_positions, body.surface_rest_positions,
                body.surface_bindings, body.realtime_surface_positions,
                body.surface_vertex_count);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
        }
        for (auto &owner : state_.ropes) {
            if (!owner || !owner->alive) continue;
            auto &rope = *owner;
            const auto groups =
                (rope.data.count + block - 1U) / block;
            native_.commit_rope_realtime(
                groups, block, native_.queue(),
                rope.realtime_positions,
                rope.realtime_previous_positions,
                rope.realtime_velocities, rope.realtime_inverse_masses,
                rope.realtime_anchor_active,
                rope.realtime_contact_normals,
                rope.realtime_contact_offsets,
                rope.realtime_contact_active, rope.data.count,
                rope.data.count, rope.data.options.node_spacing,
                rope.data.options.radius, rope.data.options.friction,
                rope.data.options.maximum_speed, scene.timestep,
                simulation::rope_realtime_profile_version,
                snapshot_.query_status);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
        }
        for (auto &owner : state_.smokes) {
            if (!owner || !owner->alive) continue;
            auto &smoke = *owner;
            const auto groups =
                (smoke.options.capacity + block - 1U) / block;
            const auto smoke_options = native_.smoke_domain_options(
                smoke.options);
            if (smoke.options.grid_resolution == 0U) {
                native_.commit_smoke_realtime_particles(
                    groups, block, native_.queue(),
                    smoke.realtime_positions,
                    smoke.realtime_previous_positions,
                    smoke.realtime_velocities, smoke.realtime_ages,
                    smoke.realtime_thermal_lift,
                    smoke.realtime_pending_count, smoke_options,
                    scene.timestep,
                    simulation::smoke_realtime_profile_version,
                    snapshot_.query_status);
            } else {
                auto &grid = smoke.grid;
                auto field = native_.smoke_dispatch_grid(grid);
                field.density = grid.realtime_density[1];
                field.pressure = grid.pressure[1];
                for (std::uint32_t axis = 0U; axis < 3U; ++axis)
                    field.face_velocity[axis] =
                        grid.face_velocity[axis][1];
                for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
                    const auto face_groups =
                        (grid.face_count[axis] + block - 1U) / block;
                    native_.project_smoke_grid_realtime_face(
                        face_groups, block, native_.queue(), int(axis),
                        field, smoke_options, grid.pressure[1],
                        grid.face_velocity[axis][1], grid.face_open[axis],
                        grid.face_wall_velocity[axis], scene.timestep,
                        simulation::smoke_realtime_profile_version,
                        snapshot_.query_status);
                }
                native_.trace_smoke_grid_realtime_particles(
                    groups, block, native_.queue(),
                    smoke.realtime_positions,
                    smoke.realtime_previous_positions,
                    smoke.realtime_velocities, smoke.realtime_ages,
                    smoke.realtime_thermal_lift,
                    smoke.realtime_number_densities,
                    smoke.realtime_pressures,
                    smoke.realtime_vorticities,
                    smoke.realtime_pending_count, smoke_options, field,
                    scene.timestep,
                    simulation::smoke_realtime_profile_version,
                    snapshot_.query_status);
            }
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
        }
        result = apply_smoke_couplings(scene);
        if (result != PM_SUCCESS) return result;
        if (state_.rigid_body_count != 0U) {
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
        for (auto &owner : state_.fluids) {
            if (!owner || !owner->alive) continue;
            std::swap(owner->positions, owner->next_positions);
            std::swap(owner->velocities, owner->next_velocities);
        }
        for (auto &owner : state_.cloths) {
            if (!owner || !owner->alive) continue;
            std::swap(owner->positions, owner->realtime_positions);
            std::swap(owner->velocities, owner->realtime_velocities);
            if (owner->surface_positions != nullptr)
                std::swap(owner->surface_positions,
                          owner->realtime_surface_positions);
        }
        for (auto &owner : state_.soft_bodies) {
            if (!owner || !owner->alive) continue;
            std::swap(owner->positions, owner->realtime_positions);
            std::swap(owner->velocities, owner->realtime_velocities);
            std::swap(owner->surface_positions,
                      owner->realtime_surface_positions);
        }
        for (auto &owner : state_.ropes) {
            if (!owner || !owner->alive) continue;
            std::swap(owner->data.positions, owner->realtime_positions);
            std::swap(owner->data.velocities, owner->realtime_velocities);
        }
        for (auto &owner : state_.smokes) {
            if (!owner || !owner->alive) continue;
            auto &smoke = *owner;
            std::swap(smoke.positions, smoke.realtime_positions);
            std::swap(smoke.previous_positions,
                      smoke.realtime_previous_positions);
            std::swap(smoke.velocities, smoke.realtime_velocities);
            std::swap(smoke.vorticities, smoke.realtime_vorticities);
            std::swap(smoke.ages, smoke.realtime_ages);
            std::swap(smoke.thermal_lift, smoke.realtime_thermal_lift);
            std::swap(smoke.number_densities,
                      smoke.realtime_number_densities);
            std::swap(smoke.pressures, smoke.realtime_pressures);
            if (smoke.options.grid_resolution != 0U) {
                auto &grid = smoke.grid;
                std::swap(grid.density, grid.realtime_density[1]);
                std::swap(grid.temperature, grid.realtime_temperature[1]);
                std::swap(grid.pressure[0], grid.pressure[1]);
                std::swap(grid.divergence, grid.realtime_divergence);
                for (std::uint32_t axis = 0U; axis < 3U; ++axis)
                    std::swap(grid.face_velocity[axis][0],
                              grid.face_velocity[axis][1]);
            }
            smoke.count = smoke.realtime_pending_count;
            smoke.next_slot = smoke.realtime_pending_next_slot;
            smoke.emitted = smoke.realtime_pending_emitted;
            smoke.emission_fraction =
                smoke.realtime_pending_emission_fraction;
            smoke.time = smoke.realtime_pending_time;
            smoke.index_dirty = true;
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
        for (auto &owner : state_.cloths)
            if (owner && owner->alive) owner->realtime_contract = {};
        for (auto &owner : state_.soft_bodies)
            if (owner && owner->alive) owner->realtime_contract = {};
        for (auto &owner : state_.ropes)
            if (owner && owner->alive) owner->realtime_contract = {};
        for (auto &owner : state_.smokes)
            if (owner && owner->alive) owner->realtime_contract = {};
        for (auto &owner : state_.fluids)
            if (owner && owner->alive) owner->realtime_contract = {};
        liquid_coupling_contract_ = {};
        rigid_coupling_contract_ = {};
        deformable_coupling_contract_ = {};
        return PM_SUCCESS;
    }

private:
    PmResult solve_liquid_couplings(
        std::uint32_t sweep, std::uint32_t edge_count,
        const simulation::RealtimeScheduleScene &scene) noexcept {
        if (edge_count != liquid_coupling_count_)
            return PM_ERROR_INVALID_STATE;
        auto *counters = state_.realtime_spatial.counters(epoch_);
        if (counters == nullptr) return PM_ERROR_INVALID_STATE;
        auto &coupling_work = work(
            simulation::RealtimeTelemetryPhase::coupling);
        ++coupling_work.numerical_iterations;
        constexpr std::uint32_t block = 128U;
        if (sweep == 0U) {
            std::uint64_t liquid_particles{};
            std::uint64_t partner_endpoints = state_.rigid_body_count;
            for (const auto &owner : state_.fluids)
                if (owner && owner->alive)
                    liquid_particles += owner->options.capacity;
            for (const auto &owner : state_.cloths)
                if (owner && owner->alive)
                    partner_endpoints += owner->vertex_count;
            for (const auto &owner : state_.soft_bodies)
                if (owner && owner->alive)
                    partner_endpoints += owner->node_count;
            for (const auto &owner : state_.ropes)
                if (owner && owner->alive)
                    partner_endpoints += owner->data.count;
            for (const auto &owner : state_.smokes)
                if (owner && owner->alive)
                    partner_endpoints += owner->options.capacity;
            if (liquid_particles > std::numeric_limits<std::uint32_t>::max() ||
                partner_endpoints > std::numeric_limits<std::uint32_t>::max())
                return PM_ERROR_CAPACITY_EXCEEDED;
            simulation::LiquidCouplingRealtimeWorkBudget budget{};
            auto result = simulation::make_liquid_coupling_realtime_work_budget(
                std::span<const simulation::LiquidCouplingRegistration>(
                    liquid_couplings_.data(), liquid_coupling_count_),
                static_cast<std::uint32_t>(liquid_particles),
                static_cast<std::uint32_t>(partner_endpoints), budget);
            if (result != PM_SUCCESS) return result;
            result = liquid_coupling_contract_.begin(epoch_, *counters,
                budget,
                std::span<const simulation::LiquidCouplingRegistration>(
                    liquid_couplings_.data(), liquid_coupling_count_),
                scene.graph_revision);
            if (result != PM_SUCCESS) return result;
        }

        for (auto &owner : state_.fluid_pair_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            auto &pair = *owner;
            auto &first = *state_.fluids[pair.options.first.index];
            auto &second = *state_.fluids[pair.options.second.index];
            const auto pair_groups =
                (pair.index.total_capacity + block - 1U) / block;
            if (sweep == 0U) {
                native_.fluid_pair_cell_keys(pair_groups, block,
                    native_.queue(), first.next_positions, first.count,
                    first.options.capacity, second.next_positions,
                    second.count, second.options.capacity,
                    1.0F / first.options.support_radius,
                    pair.index.keys[0], pair.index.identities[0]);
                auto result = native_.radix_sort_pairs_u64_u32(
                    native_.queue(), pair.index.keys[0], pair.index.keys[1],
                    pair.index.identities[0], pair.index.identities[1],
                    pair.index.total_capacity, pair.index.sort_workspace,
                    pair.index.sort_workspace_size);
                if (result != PM_SUCCESS) return result;
                dispatches(simulation::RealtimeTelemetryPhase::coupling,
                    2U, 2U * pair_groups);
            }
            auto constants = native_.fluid_constants(first.options,
                options_.gravity, scene.timestep, first.options.capacity);
            const float response_step = scene.timestep /
                static_cast<float>(simulation::liquid_coupling_realtime_sweeps);
            constants.repulsion *= response_step;
            constants.viscosity *= response_step;
            constants.normal_damping *= response_step;
            native_.fluid_pair_pressure_contact(pair_groups, block,
                native_.queue(), first.next_positions,
                first.next_velocities, first.count, first.options.capacity,
                first.realtime_velocity_deltas, second.next_positions,
                second.next_velocities, second.count,
                second.options.capacity, second.realtime_velocity_deltas,
                pair.index.keys[1], pair.index.identities[1],
                pair.index.total_capacity, constants);
            native_.fluid_pair_viscosity(pair_groups, block, native_.queue(),
                first.next_positions, first.next_velocities, first.count,
                first.options.capacity, first.realtime_velocity_deltas,
                second.next_positions, second.next_velocities, second.count,
                second.options.capacity, second.realtime_velocity_deltas,
                pair.index.keys[1], pair.index.identities[1],
                pair.index.total_capacity, constants);
            auto result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            dispatches(simulation::RealtimeTelemetryPhase::coupling,
                2U, 2U * pair_groups);
        }

        if (state_.rigid_body_count != 0U) {
            const auto body_groups =
                (state_.rigid_body_count + block - 1U) / block;
            native_.calculate_fluid_body_bounds(body_groups, block,
                native_.queue(), state_.parameters,
                state_.render_previous_states, state_.states[output_state_],
                state_.meshes, state_.rigid_body_count,
                state_.fluid_body_bounds);
            auto result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            dispatch(simulation::RealtimeTelemetryPhase::coupling,
                body_groups);
            const auto body_words =
                (state_.options.rigid_body_capacity + 63U) / 64U;
            result = native_.clear(state_.fluid_body_masks, 0,
                NativeScheduleBackend::fluid_body_buckets * body_words *
                    sizeof(unsigned long long), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(state_.fluid_global_body_masks, 0,
                    body_words * sizeof(unsigned long long), native_.queue());
            if (result != PM_SUCCESS) return result;
            coupling_work.bytes_moved +=
                (NativeScheduleBackend::fluid_body_buckets + 1ULL) *
                body_words * sizeof(unsigned long long);
            coupling_work.barriers += 2U;
            native_.index_fluid_body_cells(body_groups, block, native_.queue(),
                state_.parameters, state_.fluid_body_bounds,
                state_.rigid_body_count, body_words, 0.0F,
                state_.fluid_body_masks, state_.fluid_global_body_masks);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            dispatch(simulation::RealtimeTelemetryPhase::coupling,
                body_groups);
            for (std::uint32_t slot = 0U;
                 slot < state_.fluids.size(); ++slot) {
                auto &owner = state_.fluids[slot];
                if (!owner || !owner->alive) continue;
                auto &fluid = *owner;
                const auto groups =
                    (fluid.options.capacity + block - 1U) / block;
                result = native_.clear(state_.fluid_body_contact_flags, 0,
                    state_.rigid_body_count * sizeof(std::uint32_t),
                    native_.queue());
                if (result != PM_SUCCESS) return result;
                const float diameter = 2.0F * fluid.options.particle_radius;
                const float particle_mass = fluid.options.rest_density *
                    (fluid.options.rest_particle_volume > 0.0F
                        ? fluid.options.rest_particle_volume
                        : diameter * diameter * diameter);
                native_.detect_fluid_moving_contacts(groups, block,
                    native_.queue(), fluid.next_positions, fluid.previous,
                    fluid.count, fluid.options.particle_radius,
                    state_.parameters, state_.render_previous_states,
                    state_.states[output_state_], state_.meshes,
                    state_.fluid_body_bounds, state_.rigid_body_count,
                    state_.fluid_body_masks, state_.fluid_global_body_masks,
                    body_words, sweep == 0U, fluid.moving_contacts,
                    state_.fluid_body_contact_flags);
                native_.resolve_fluid_moving_contacts(groups, block,
                    native_.queue(), fluid.next_positions,
                    fluid.next_velocities, fluid.next_foam, fluid.count,
                    fluid.options.particle_radius, particle_mass,
                    scene.timestep /
                        simulation::liquid_coupling_realtime_sweeps,
                    fluid.options.maximum_speed,
                    state_.parameters, state_.states[output_state_],
                    state_.meshes, fluid.moving_contacts,
                    state_.fluid_body_contact_flags, fluid.body_impulses,
                    false, fluid.contact_samples, fluid.contact_flags, slot,
                    fluid.generation, state_.ids, false, state_.paint_fields,
                    state_.options.paint_field_capacity, state_.paint_rules,
                    state_.options.paint_rule_capacity);
                native_.reduce_fluid_body_impulses(
                    state_.rigid_body_count, block, native_.queue(),
                    fluid.body_impulses, fluid.count, state_.parameters,
                    state_.states[output_state_],
                    state_.fluid_body_contact_flags,
                    state_.rigid_body_count);
                result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatches(simulation::RealtimeTelemetryPhase::coupling,
                    3U, 2ULL * groups + state_.rigid_body_count);
                coupling_work.bytes_moved +=
                    state_.rigid_body_count * sizeof(std::uint32_t);
                ++coupling_work.barriers;
            }
        }

        for (auto &coupling : state_.fluid_cloth_couplings) {
            if (!coupling.alive || !coupling.options.enabled) continue;
            auto &fluid = *state_.fluids[coupling.options.fluid.index];
            auto &cloth = *state_.cloths[coupling.options.cloth.index];
            const auto fluid_groups =
                (fluid.options.capacity + block - 1U) / block;
            const auto cloth_groups =
                (cloth.vertex_count + block - 1U) / block;
            auto result = PM_SUCCESS;
            if (coupling.options.contact) {
                result = native_.clear(cloth.fluid_forces, 0,
                    cloth.vertex_count * sizeof(Vec3), native_.queue());
                if (result != PM_SUCCESS) return result;
                const float diameter = 2.0F * fluid.options.particle_radius;
                const float particle_mass = fluid.options.rest_density *
                    (fluid.options.rest_particle_volume > 0.0F
                        ? fluid.options.rest_particle_volume
                        : diameter * diameter * diameter);
                const float contact = coupling.options.contact_distance > 0.0F
                    ? coupling.options.contact_distance
                    : fluid.options.particle_radius + cloth.thickness;
                const float radius = coupling.options.interaction_radius > 0.0F
                    ? coupling.options.interaction_radius
                    : fluid.options.support_radius;
                const float response_step = scene.timestep /
                    static_cast<float>(simulation::liquid_coupling_realtime_sweeps);
                native_.apply_fluid_cloth_forces(fluid_groups, block,
                    native_.queue(), fluid.next_positions,
                    fluid.next_velocities, fluid.count,
                    fluid.realtime_velocity_deltas, fluid.foam_source,
                    particle_mass, cloth.realtime_positions,
                    cloth.realtime_velocities, cloth.indices,
                    cloth.index_count / 3U, cloth.orientation,
                    simulation::FluidClothDispatchOptions{contact, radius,
                        coupling.options.stiffness * response_step,
                        coupling.options.damping * response_step,
                        coupling.options.tangential_drag * response_step,
                        coupling.options.maximum_force * response_step},
                    cloth.fluid_forces);
                native_.apply_fluid_forces_to_cloth(cloth_groups, block,
                    native_.queue(), cloth.realtime_positions,
                    cloth.realtime_velocities, cloth.inverse_masses,
                    cloth.fluid_forces, cloth.vertex_count, 1.0F);
                result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatches(simulation::RealtimeTelemetryPhase::coupling,
                    2U, fluid_groups + cloth_groups);
                coupling_work.bytes_moved +=
                    cloth.vertex_count * sizeof(Vec3);
                ++coupling_work.barriers;
            }
            if (coupling.options.containment) {
                const float contact = coupling.options.contact_distance > 0.0F
                    ? coupling.options.contact_distance
                    : fluid.options.particle_radius + cloth.thickness;
                native_.project_fluid_inside_cloth(fluid_groups, block,
                    native_.queue(), fluid.next_positions,
                    fluid.next_velocities, fluid.count, contact,
                    cloth.realtime_positions, cloth.realtime_velocities,
                    cloth.indices, cloth.index_count / 3U,
                    cloth.orientation);
                result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatch(simulation::RealtimeTelemetryPhase::coupling,
                    fluid_groups);
            }
        }

        for (auto &owner : state_.fluid_rope_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            auto &coupling = *owner;
            auto &fluid = *state_.fluids[coupling.options.fluid.index];
            auto &rope = *state_.ropes[coupling.options.rope.index];
            const auto fluid_groups =
                (fluid.options.capacity + block - 1U) / block;
            const auto rope_groups =
                (rope.data.count + block - 1U) / block;
            auto result = native_.clear(coupling.contact_count, 0,
                sizeof(*coupling.contact_count), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(coupling.maximum_penetration, 0,
                    sizeof(*coupling.maximum_penetration), native_.queue());
            if (result != PM_SUCCESS) return result;
            const float diameter = 2.0F * fluid.options.particle_radius;
            const float particle_mass = fluid.options.rest_density *
                (fluid.options.rest_particle_volume > 0.0F
                    ? fluid.options.rest_particle_volume
                    : diameter * diameter * diameter);
            const float distance = coupling.options.contact_distance > 0.0F
                ? coupling.options.contact_distance
                : fluid.options.particle_radius + rope.data.options.radius;
            native_.detect_fluid_rope_contacts(fluid_groups, block,
                native_.queue(), fluid.next_positions,
                fluid.next_velocities, fluid.count, particle_mass,
                rope.realtime_positions, rope.realtime_velocities,
                rope.data.count, distance, coupling.options.friction,
                coupling.contribution_nodes, coupling.contribution_impulses,
                coupling.contact_count, coupling.maximum_penetration, true);
            native_.apply_fluid_rope_impulses(rope_groups, block,
                native_.queue(), rope.realtime_velocities,
                rope.data.fluid_contact_forces, rope.data.count,
                rope.data.options.mass, rope.data.options.maximum_speed,
                rope.data.options.first.enabled,
                rope.data.options.last.enabled, fluid.count,
                coupling.contribution_nodes, coupling.contribution_impulses,
                coupling.options.maximum_rope_acceleration,
                scene.timestep /
                    simulation::liquid_coupling_realtime_sweeps,
                1.0F / scene.timestep);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            dispatches(simulation::RealtimeTelemetryPhase::coupling,
                2U, fluid_groups + rope_groups);
            coupling_work.bytes_moved += sizeof(std::uint32_t) + sizeof(float);
            coupling_work.barriers += 2U;
        }

        for (auto &owner : state_.fluid_soft_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            auto &coupling = *owner;
            auto &fluid = *state_.fluids[coupling.options.fluid.index];
            auto &body = *state_.soft_bodies[
                coupling.options.soft_body.index];
            const auto fluid_groups =
                (fluid.options.capacity + block - 1U) / block;
            const auto node_groups =
                (body.node_count + block - 1U) / block;
            const auto surface_groups =
                (body.surface_vertex_count + block - 1U) / block;
            auto result = native_.clear(coupling.counts, 0,
                body.node_count * sizeof(std::uint32_t), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(coupling.position_deltas, 0,
                    body.node_count * sizeof(Vec3), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(coupling.impulses, 0,
                    body.node_count * sizeof(Vec3), native_.queue());
            if (result == PM_SUCCESS)
                result = native_.clear(coupling.ready, 0,
                    coupling.tree_count * sizeof(std::uint32_t),
                    native_.queue());
            if (result != PM_SUCCESS) return result;
            native_.refit_fluid_soft_surface(
                (coupling.tree_count + block - 1U) / block, block,
                native_.queue(), body.realtime_surface_positions,
                coupling.previous_surface, body.surface_indices,
                coupling.triangle_order, coupling.tree, coupling.parents,
                coupling.ready, coupling.tree_count, coupling.bounds);
            const float distance = coupling.options.contact_distance > 0.0F
                ? coupling.options.contact_distance
                : fluid.options.particle_radius;
            native_.detect_fluid_soft_contacts(fluid_groups, block,
                native_.queue(), fluid.next_positions, fluid.previous,
                fluid.next_velocities, fluid.count,
                body.realtime_surface_positions, coupling.previous_surface,
                body.surface_indices, body.surface_index_count,
                body.surface_bindings, body.realtime_velocities,
                coupling.orientation, distance, coupling.bounds,
                sweep == 0U, coupling.tree, coupling.triangle_order,
                coupling.contacts, coupling.counts, coupling.contact_count,
                coupling.maximum_penetration);
            const float diameter = 2.0F * fluid.options.particle_radius;
            const float particle_mass = fluid.options.rest_density *
                (fluid.options.rest_particle_volume > 0.0F
                    ? fluid.options.rest_particle_volume
                    : diameter * diameter * diameter);
            native_.solve_fluid_soft_contacts(fluid_groups, block,
                native_.queue(), fluid.next_positions,
                fluid.next_velocities, fluid.realtime_velocity_deltas,
                fluid.next_foam, fluid.count, coupling.contacts,
                coupling.counts, body.inverse_masses,
                body.realtime_velocities, body.realtime_maximum_speed,
                1.0F / particle_mass, coupling.options.friction,
                0.2F * body.node_radius,
                1.0F,
                coupling.position_deltas, coupling.impulses);
            native_.apply_fluid_soft_contacts(node_groups, block,
                native_.queue(), body.realtime_positions,
                body.realtime_velocities, body.inverse_masses,
                coupling.position_deltas, coupling.impulses,
                body.fluid_forces, body.node_count, 1.0F / scene.timestep);
            native_.update_soft_body_surface(surface_groups, block,
                native_.queue(), body.realtime_positions,
                body.rest_positions, body.surface_rest_positions,
                body.surface_bindings, body.realtime_surface_positions,
                body.surface_vertex_count);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            dispatches(simulation::RealtimeTelemetryPhase::coupling,
                5U, 2ULL * fluid_groups + node_groups + surface_groups +
                    (coupling.tree_count + block - 1U) / block);
            coupling_work.bytes_moved +=
                body.node_count *
                    (sizeof(std::uint32_t) + 2U * sizeof(Vec3)) +
                coupling.tree_count * sizeof(std::uint32_t);
            coupling_work.barriers += 4U;
        }

        for (auto &owner : state_.fluid_smoke_couplings) {
            if (!owner || !owner->alive) continue;
            auto &coupling = *owner;
            auto &fluid = *state_.fluids[coupling.options.fluid.index];
            auto &smoke = *state_.smokes[coupling.options.smoke.index];
            const auto groups =
                (fluid.options.capacity + block - 1U) / block;
            simulation::FluidSmokeDispatchOptions dispatch{};
            dispatch.smoke = native_.smoke_dispatch_options(smoke.options);
            dispatch.heater_center[0] = coupling.options.heater.center.x;
            dispatch.heater_center[1] = coupling.options.heater.center.y;
            dispatch.heater_center[2] = coupling.options.heater.center.z;
            dispatch.heater_orientation[0] =
                coupling.options.heater.orientation.x;
            dispatch.heater_orientation[1] =
                coupling.options.heater.orientation.y;
            dispatch.heater_orientation[2] =
                coupling.options.heater.orientation.z;
            dispatch.heater_orientation[3] =
                coupling.options.heater.orientation.w;
            dispatch.heater_half_extents[0] =
                coupling.options.heater.half_extents.x;
            dispatch.heater_half_extents[1] =
                coupling.options.heater.half_extents.y;
            dispatch.heater_temperature = coupling.options.heater_temperature;
            dispatch.boiling_temperature =
                std::numeric_limits<float>::max();
            dispatch.heat_transfer_rate = coupling.options.heat_transfer_rate;
            dispatch.wind_drag = coupling.options.wind_drag;
            dispatch.steam_rise_speed = coupling.options.steam_rise_speed;
            const float dt = scene.timestep /
                simulation::liquid_coupling_realtime_sweeps;
            if (smoke.options.grid_resolution == 0U) {
                native_.drag_fluid_with_smoke_particles(groups, block,
                    native_.queue(), fluid.next_positions,
                    fluid.realtime_velocity_deltas, fluid.count, dispatch,
                    smoke.realtime_positions, smoke.realtime_velocities,
                    smoke.keys[1], smoke.indices[1], dt);
            } else {
                auto grid = native_.smoke_dispatch_grid(smoke.grid);
                for (std::uint32_t axis = 0U; axis < 3U; ++axis)
                    grid.face_velocity[axis] =
                        smoke.grid.face_velocity[axis][1];
                native_.drag_fluid_with_smoke_grid(groups, block,
                    native_.queue(), fluid.next_positions,
                    fluid.realtime_velocity_deltas, fluid.count, dispatch,
                    grid, dt);
            }
            auto result = native_.clear(coupling.converted, 0,
                sizeof(*coupling.converted), native_.queue());
            if (result != PM_SUCCESS) return result;
            const float gravity[3]{
                options_.gravity.x, options_.gravity.y, options_.gravity.z};
            native_.exchange_fluid_with_smoke(groups, block, native_.queue(),
                fluid.next_positions, fluid.realtime_velocity_deltas,
                fluid.temperatures, fluid.count,
                fluid.options.particle_radius, dispatch,
                smoke.realtime_positions, smoke.realtime_velocities,
                smoke.realtime_ages, smoke.realtime_thermal_lift,
                smoke.realtime_number_densities, smoke.realtime_pressures,
                smoke.realtime_vorticities, smoke.realtime_pending_next_slot,
                coupling.converted, fluid.keep, dt, gravity);
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            dispatches(simulation::RealtimeTelemetryPhase::coupling,
                2U, 2U * groups);
            coupling_work.bytes_moved += sizeof(*coupling.converted);
            ++coupling_work.barriers;
        }

        for (std::uint32_t index = 0U;
             index < liquid_coupling_count_; ++index) {
            const auto result = liquid_coupling_contract_.record_edge(
                epoch_, *counters, scene.graph_revision,
                liquid_couplings_[index]);
            if (result != PM_SUCCESS) return result;
        }
        return sweep + 1U == simulation::liquid_coupling_realtime_sweeps
            ? liquid_coupling_contract_.finish() : PM_SUCCESS;
    }

    simulation::RigidCouplingConstraintBuffers rigid_constraints(
        std::uint32_t record) const noexcept {
        return {state_.realtime_coupling_first_endpoints + record,
            state_.realtime_coupling_second_endpoints + record,
            state_.realtime_coupling_first_generations + record,
            state_.realtime_coupling_second_generations + record,
            state_.realtime_coupling_pairs + record,
            state_.realtime_coupling_modes + record,
            state_.realtime_coupling_normals + record,
            state_.realtime_coupling_body_local_points + record,
            state_.realtime_coupling_contact_distances + record,
            state_.realtime_coupling_frictions + record,
            state_.realtime_coupling_responses + record,
            state_.realtime_coupling_active + record, 1U};
    }

    simulation::RigidCouplingScratchBuffers rigid_scratch(
        std::uint32_t record) const noexcept {
        return {state_.realtime_coupling_first_position_deltas + record,
            state_.realtime_coupling_second_position_deltas + record,
            state_.realtime_coupling_first_impulses + record,
            state_.realtime_coupling_second_impulses + record};
    }

    PmResult solve_rigid_rope_attachment(RopeStorage &rope,
        std::uint32_t end, std::uint32_t record, std::uint32_t sweep,
        const simulation::RealtimeScheduleScene &scene) noexcept {
        const auto node = end == 0U ? 0U : rope.data.count - 1U;
        const simulation::RigidCouplingEndpointBuffers endpoints{
            rope.realtime_positions + node,
            rope.realtime_previous_positions + node,
            rope.realtime_velocities + node,
            rope.realtime_inverse_masses + node,
            rope.realtime_correction_used + node,
            state_.realtime_coupling_first_generations + record, 1U};
        const simulation::RigidCouplingBodyBuffers bodies{
            state_.parameters, state_.render_previous_states,
            state_.states[output_state_],
            state_.realtime_rigid_correction_used,
            state_.realtime_rigid_generations, state_.rigid_body_count};
        const auto constraints = rigid_constraints(record);
        const auto scratch = rigid_scratch(record);
        native_.project_rigid_couplings(1U, 128U, native_.queue(), endpoints,
            bodies, constraints, scratch, rope.data.options.node_spacing,
            rope.data.options.maximum_speed, scene.timestep,
            simulation::rigid_coupling_realtime_profile_version, sweep,
            snapshot_.query_status);
        native_.apply_rigid_coupling_endpoints(1U, 128U, native_.queue(),
            endpoints, constraints, scratch, rope.data.options.node_spacing,
            rope.data.options.maximum_speed, scene.timestep,
            simulation::rigid_coupling_realtime_profile_version, sweep,
            snapshot_.query_status);
        native_.apply_rigid_coupling_bodies(
            (state_.rigid_body_count + 127U) / 128U, 128U,
            native_.queue(), bodies, constraints, scratch,
            rope.data.options.node_spacing, scene.timestep,
            simulation::rigid_coupling_realtime_profile_version, sweep,
            snapshot_.query_status);
        const auto result = native_.check_error();
        if (result == PM_SUCCESS)
            dispatches(simulation::RealtimeTelemetryPhase::coupling, 3U,
                2U + (state_.rigid_body_count + 127U) / 128U);
        return result;
    }

    simulation::DeformableSmokeCouplingConstraintBuffers
    deformable_constraints(std::uint32_t record) const noexcept {
        return {state_.realtime_coupling_first_endpoints + record,
            state_.realtime_coupling_second_endpoints + record,
            state_.realtime_coupling_first_generations + record,
            state_.realtime_coupling_second_generations + record,
            state_.realtime_coupling_pairs + record,
            state_.realtime_coupling_modes + record,
            state_.realtime_coupling_normals + record,
            state_.realtime_coupling_contact_distances + record,
            state_.realtime_coupling_frictions + record,
            state_.realtime_coupling_responses + record,
            state_.realtime_coupling_active + record, 1U};
    }

    simulation::DeformableSmokeCouplingScratchBuffers deformable_scratch(
        std::uint32_t record) const noexcept {
        return {state_.realtime_coupling_first_position_deltas + record,
            state_.realtime_coupling_second_position_deltas + record,
            state_.realtime_coupling_first_impulses + record,
            state_.realtime_coupling_second_impulses + record};
    }

    PmResult solve_deformable_pair(
        const simulation::DeformableSmokeCouplingEndpointBuffers &first,
        const simulation::DeformableSmokeCouplingEndpointBuffers &second,
        std::uint32_t record, float characteristic_length,
        float maximum_speed, std::uint32_t sweep,
        const simulation::RealtimeScheduleScene &scene) noexcept {
        const auto constraints = deformable_constraints(record);
        const auto scratch = deformable_scratch(record);
        native_.project_deformable_smoke_couplings(
            1U, 128U, native_.queue(), first, second, constraints, scratch,
            characteristic_length, maximum_speed, scene.timestep,
            simulation::deformable_smoke_coupling_profile_version, sweep,
            snapshot_.query_status);
        native_.apply_deformable_smoke_coupling_first(
            1U, 128U, native_.queue(), first, constraints, scratch,
            characteristic_length, maximum_speed, scene.timestep,
            simulation::deformable_smoke_coupling_profile_version, sweep,
            snapshot_.query_status);
        native_.apply_deformable_smoke_coupling_second(
            1U, 128U, native_.queue(), second, constraints, scratch,
            characteristic_length, maximum_speed, scene.timestep,
            simulation::deformable_smoke_coupling_profile_version, sweep,
            snapshot_.query_status);
        const auto result = native_.check_error();
        if (result == PM_SUCCESS)
            dispatches(simulation::RealtimeTelemetryPhase::coupling, 3U, 3U);
        return result;
    }

    PmResult solve_soft_rope_pair(RopeSoftBodyCouplingStorage &coupling,
        SoftBodyStorage &body, RopeStorage &rope, std::uint32_t end,
        std::uint32_t record, std::uint32_t sweep,
        const simulation::RealtimeScheduleScene &scene) noexcept {
        const auto node = end == 0U ? 0U : rope.data.count - 1U;
        const simulation::DeformableSmokeCouplingEndpointBuffers first{
            rope.data.soft_anchor_positions + end,
            rope.data.soft_anchor_positions + end,
            rope.data.soft_anchor_velocities + end,
            rope.data.soft_anchor_inverse_masses + end,
            state_.realtime_coupling_first_correction_used + record,
            state_.realtime_coupling_first_generations + record, 1U};
        const simulation::DeformableSmokeCouplingEndpointBuffers second{
            rope.realtime_positions + node,
            rope.realtime_previous_positions + node,
            rope.realtime_velocities + node,
            rope.realtime_inverse_masses + node,
            rope.realtime_correction_used + node,
            state_.realtime_coupling_second_generations + record, 1U};
        auto result = solve_deformable_pair(first, second, record,
            std::max(body.realtime_characteristic_length,
                rope.data.options.node_spacing),
            std::max(body.realtime_maximum_speed,
                rope.data.options.maximum_speed), sweep, scene);
        if (result != PM_SUCCESS) return result;
        result = native_.copy(rope.data.soft_anchor_impulses + end,
            state_.realtime_coupling_first_impulses + record, sizeof(Vec3),
            device::CopyDirection::device_to_device, native_.queue());
        if (result != PM_SUCCESS) return result;
        result = native_.clear(coupling.node_impulses, 0,
            body.node_count * sizeof(Vec3), native_.queue());
        if (result != PM_SUCCESS) return result;
        native_.scatter_rope_soft_body_anchor(1U, 256U, native_.queue(),
            rope.data.soft_anchor_positions, rope.data.soft_anchor_impulses,
            end, body.realtime_positions, body.node_count,
            coupling.options.anchor_support_radius_scale * body.node_radius,
            coupling.options.anchor_contact_support_radius_scale *
                body.node_radius,
            coupling.contact_count, body.surface_indices,
            body.surface_bindings, coupling.anchor_triangle[end],
            &coupling.anchor_weights[end].x, coupling.node_impulses);
        native_.apply_rope_soft_body_impulses(
            (body.node_count + 127U) / 128U, 128U, native_.queue(),
            body.realtime_positions, body.realtime_velocities,
            body.rope_forces, body.inverse_masses, coupling.node_impulses,
            body.node_count, coupling.options.maximum_soft_body_acceleration,
            body.realtime_maximum_speed, scene.timestep,
            1.0F / scene.timestep);
        native_.update_soft_body_surface(
            (body.surface_vertex_count + 127U) / 128U, 128U,
            native_.queue(), body.realtime_positions, body.rest_positions,
            body.surface_rest_positions, body.surface_bindings,
            body.realtime_surface_positions, body.surface_vertex_count);
        result = native_.check_error();
        if (result == PM_SUCCESS) {
            dispatches(simulation::RealtimeTelemetryPhase::coupling, 3U,
                1U + (body.node_count + 127U) / 128U +
                    (body.surface_vertex_count + 127U) / 128U);
            auto &coupling_work = work(
                simulation::RealtimeTelemetryPhase::coupling);
            coupling_work.bytes_moved +=
                sizeof(Vec3) + body.node_count * sizeof(Vec3);
            coupling_work.barriers += 2U;
        }
        return result;
    }

    void stage_smoke_rigid_measurement() noexcept {
        constexpr std::uint64_t sweeps =
            simulation::rigid_coupling_realtime_sweeps;
        for (const auto &owner : state_.smoke_rigid_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            const auto &smoke = *state_.smokes[owner->options.smoke.index];
            const auto particle_groups =
                (std::uint64_t{smoke.options.capacity} + 127U) / 128U;
            if (owner->options.tracer_contact)
                dispatches(simulation::RealtimeTelemetryPhase::coupling,
                    sweeps, sweeps * particle_groups);
            if (owner->options.air_density > 0.0F &&
                (smoke.options.grid_resolution != 0U ||
                 owner->options.tracer_contact))
                dispatches(simulation::RealtimeTelemetryPhase::coupling,
                    sweeps, sweeps);
        }
    }

    void stage_smoke_deformable_measurement() noexcept {
        auto &coupling = work(simulation::RealtimeTelemetryPhase::coupling);
        constexpr std::uint64_t sweeps =
            simulation::deformable_smoke_coupling_sweeps;
        for (std::uint32_t smoke_slot = 0U;
             smoke_slot < state_.smokes.size(); ++smoke_slot) {
            const auto &smoke = state_.smokes[smoke_slot];
            if (!smoke || !smoke->alive ||
                smoke->options.grid_resolution != 0U)
                continue;
            const auto references_smoke = [smoke_slot](
                    const auto &owner) noexcept {
                return owner && owner->alive && owner->options.enabled &&
                    owner->options.smoke.index == smoke_slot;
            };
            if (std::any_of(state_.smoke_cloth_couplings.begin(),
                            state_.smoke_cloth_couplings.end(),
                            references_smoke) ||
                std::any_of(state_.smoke_soft_body_couplings.begin(),
                            state_.smoke_soft_body_couplings.end(),
                            references_smoke) ||
                std::any_of(state_.smoke_rope_couplings.begin(),
                            state_.smoke_rope_couplings.end(),
                            references_smoke)) {
                const auto groups =
                    (std::uint64_t{smoke->options.capacity} + 127U) / 128U;
                dispatches(simulation::RealtimeTelemetryPhase::coupling,
                    2U, 2U * groups);
            }
        }
        for (const auto &owner : state_.smoke_cloth_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            const auto &smoke = *state_.smokes[owner->options.smoke.index];
            const auto &cloth = *state_.cloths[owner->options.cloth.index];
            const auto smoke_groups =
                (std::uint64_t{smoke.options.capacity} + 127U) / 128U;
            const auto vertex_groups =
                (std::uint64_t{cloth.vertex_count} + 127U) / 128U;
            const auto triangle_groups =
                (std::uint64_t{cloth.index_count / 3U} + 127U) / 128U;
            dispatches(simulation::RealtimeTelemetryPhase::coupling,
                1U + sweeps, 1U + sweeps * smoke_groups);
            if (smoke.options.grid_resolution == 0U) {
                dispatches(simulation::RealtimeTelemetryPhase::coupling,
                    sweeps, sweeps * vertex_groups);
            } else {
                dispatches(simulation::RealtimeTelemetryPhase::coupling,
                    2U * sweeps,
                    sweeps * (triangle_groups + vertex_groups));
                coupling.bytes_moved += sweeps * cloth.vertex_count *
                    sizeof(Vec3);
                coupling.barriers += sweeps;
            }
        }
        for (const auto &owner : state_.smoke_soft_body_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            const auto &smoke = *state_.smokes[owner->options.smoke.index];
            const auto &body =
                *state_.soft_bodies[owner->options.soft_body.index];
            const auto smoke_groups =
                (std::uint64_t{smoke.options.capacity} + 127U) / 128U;
            const auto node_groups =
                (std::uint64_t{body.node_count} + 127U) / 128U;
            const auto triangle_groups =
                (std::uint64_t{body.surface_index_count / 3U} + 127U) /
                128U;
            dispatches(simulation::RealtimeTelemetryPhase::coupling,
                1U + sweeps, 1U + sweeps * smoke_groups);
            if (smoke.options.grid_resolution == 0U) {
                dispatches(simulation::RealtimeTelemetryPhase::coupling,
                    sweeps, sweeps * node_groups);
            } else {
                dispatches(simulation::RealtimeTelemetryPhase::coupling,
                    sweeps, sweeps * triangle_groups);
                coupling.bytes_moved += sweeps * body.node_count *
                    sizeof(Vec3);
                coupling.barriers += sweeps;
            }
        }
        for (const auto &owner : state_.smoke_rope_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            const auto &smoke = *state_.smokes[owner->options.smoke.index];
            const auto &rope = *state_.ropes[owner->options.rope.index];
            const auto smoke_groups =
                (std::uint64_t{smoke.options.capacity} + 127U) / 128U;
            const auto node_groups =
                (std::uint64_t{rope.data.count} + 127U) / 128U;
            dispatches(simulation::RealtimeTelemetryPhase::coupling,
                1U + sweeps, 1U + sweeps * smoke_groups);
            dispatches(simulation::RealtimeTelemetryPhase::coupling,
                sweeps, sweeps * node_groups);
        }
    }

    [[nodiscard]] bool smoke_has_deformable_coupling(
        std::uint32_t smoke_slot) const noexcept {
        const auto references_smoke = [smoke_slot](
                const auto &owner) noexcept {
            return owner && owner->alive && owner->options.enabled &&
                owner->options.smoke.index == smoke_slot;
        };
        return std::any_of(state_.smoke_cloth_couplings.begin(),
                           state_.smoke_cloth_couplings.end(),
                           references_smoke) ||
            std::any_of(state_.smoke_soft_body_couplings.begin(),
                        state_.smoke_soft_body_couplings.end(),
                        references_smoke) ||
            std::any_of(state_.smoke_rope_couplings.begin(),
                        state_.smoke_rope_couplings.end(),
                        references_smoke);
    }

    simulation::SmokeGridDispatchField staged_smoke_grid(
        SmokeStorage &smoke) noexcept {
        auto field = native_.smoke_dispatch_grid(smoke.grid);
        if (smoke.options.grid_resolution == 0U) return field;
        field.density = smoke.grid.realtime_density[1];
        field.pressure = smoke.grid.pressure[1];
        for (std::uint32_t axis = 0U; axis < 3U; ++axis)
            field.face_velocity[axis] =
                smoke.grid.face_velocity[axis][1];
        return field;
    }

    PmResult index_smoke_for_deformable_drag() noexcept {
        constexpr std::uint32_t block = 128U;
        for (std::uint32_t slot = 0U; slot < state_.smokes.size(); ++slot) {
            auto &owner = state_.smokes[slot];
            if (!owner || !owner->alive ||
                owner->options.grid_resolution != 0U ||
                !smoke_has_deformable_coupling(slot))
                continue;
            auto &smoke = *owner;
            const auto groups =
                (smoke.options.capacity + block - 1U) / block;
            native_.index_smoke_particles(
                groups, block, native_.queue(), smoke.realtime_positions,
                smoke.realtime_ages, smoke.realtime_pending_count,
                native_.smoke_domain_options(smoke.options), smoke.keys[0],
                smoke.indices[0]);
            auto result = native_.check_error();
            if (result != PM_SUCCESS) return result;
            result = native_.radix_sort_pairs_u64_u32(
                native_.queue(), smoke.keys[0], smoke.keys[1],
                smoke.indices[0], smoke.indices[1], smoke.options.capacity,
                smoke.sort_workspace, smoke.sort_workspace_size);
            if (result != PM_SUCCESS) return result;
            result = native_.check_error();
            if (result != PM_SUCCESS) return result;
        }
        return PM_SUCCESS;
    }

    PmResult apply_smoke_rigid_couplings(
        const simulation::RealtimeScheduleScene &scene) noexcept {
        constexpr std::uint32_t block = 128U;
        const float dt = scene.timestep /
            static_cast<float>(simulation::rigid_coupling_realtime_sweeps);
        constexpr simulation::RigidCouplingFamily families[]{
            simulation::RigidCouplingFamily::smoke_particle,
            simulation::RigidCouplingFamily::smoke_grid};
        constexpr simulation::RigidCouplingMode modes[]{
            simulation::RigidCouplingMode::contact,
            simulation::RigidCouplingMode::drag};
        for (std::uint32_t sweep = 0U;
             sweep < simulation::rigid_coupling_realtime_sweeps; ++sweep) {
            for (const auto family : families) {
                for (const auto mode : modes) {
                    for (auto &owner : state_.smoke_rigid_couplings) {
                        if (!owner || !owner->alive ||
                            !owner->options.enabled)
                            continue;
                        auto &coupling = *owner;
                        auto &smoke = *state_.smokes[
                            coupling.options.smoke.index];
                        const bool grid =
                            smoke.options.grid_resolution != 0U;
                        const auto actual_family = grid
                            ? simulation::RigidCouplingFamily::smoke_grid
                            : simulation::RigidCouplingFamily::smoke_particle;
                        if (family != actual_family) continue;
                        const auto &body_slot =
                            state_.slots[coupling.options.body.index];
                        const auto body = body_slot.dense_index;
                        const auto field = staged_smoke_grid(smoke);
                        if (mode == simulation::RigidCouplingMode::contact) {
                            if (!coupling.options.tracer_contact) continue;
                            const auto groups =
                                (smoke.options.capacity + block - 1U) /
                                block;
                            const float clearance =
                                coupling.options.contact_distance > 0.0F
                                ? coupling.options.contact_distance
                                : smoke.options.particle_radius;
                            const float diameter =
                                2.0F * smoke.options.particle_radius;
                            const float particle_mass = grid ? 0.0F :
                                coupling.options.air_density * diameter *
                                    diameter * diameter;
                            native_.collide_smoke_with_rigid_body(
                                groups, block, native_.queue(),
                                smoke.realtime_positions,
                                smoke.realtime_previous_positions,
                                smoke.realtime_velocities,
                                smoke.realtime_ages,
                                smoke.realtime_pressures,
                                smoke.realtime_pending_count,
                                smoke.options.lifetime,
                                state_.meshes + coupling.mesh.index,
                                state_.states[output_state_],
                                state_.render_previous_states, body,
                                clearance, particle_mass,
                                coupling.options.drag_coefficient,
                                smoke.options.pressure_stiffness, dt,
                                smoke.options.maximum_speed, field,
                                smoke.rigid_impulses);
                        } else if (grid &&
                                   coupling.options.air_density > 0.0F) {
                            native_.apply_smoke_grid_force_to_rigid_body(
                                1U, block, native_.queue(),
                                state_.meshes + coupling.mesh.index, body,
                                state_.parameters,
                                state_.states[output_state_], field,
                                coupling.options.air_density,
                                coupling.options.drag_coefficient,
                                smoke.options.rest_number_density,
                                smoke.options.grid_kinematic_viscosity,
                                smoke.options.grid_les_coefficient, dt);
                        } else if (!grid &&
                                   coupling.options.air_density > 0.0F &&
                                   coupling.options.tracer_contact) {
                            native_.apply_smoke_rigid_body_impulses(
                                1U, block, native_.queue(),
                                smoke.rigid_impulses,
                                smoke.realtime_pending_count, body,
                                state_.parameters,
                                state_.states[output_state_]);
                        }
                    }
                }
            }
        }
        return native_.check_error();
    }

    PmResult prepare_smoke_deformable_contacts() noexcept {
        for (auto &owner : state_.smoke_cloth_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            auto &coupling = *owner;
            auto &cloth = *state_.cloths[coupling.options.cloth.index];
            native_.calculate_smoke_deformable_bounds(
                1U, 1U, native_.queue(), cloth.realtime_positions,
                cloth.vertex_count, coupling.minimum, coupling.maximum);
        }
        for (auto &owner : state_.smoke_soft_body_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            auto &coupling = *owner;
            auto &body = *state_.soft_bodies[
                coupling.options.soft_body.index];
            native_.calculate_smoke_deformable_bounds(
                1U, 1U, native_.queue(), body.realtime_surface_positions,
                body.surface_vertex_count, coupling.minimum,
                coupling.maximum);
        }
        for (auto &owner : state_.smoke_rope_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            auto &coupling = *owner;
            auto &rope = *state_.ropes[coupling.options.rope.index];
            native_.calculate_smoke_deformable_bounds(
                1U, 1U, native_.queue(), rope.realtime_positions,
                rope.data.count, coupling.minimum, coupling.maximum);
        }
        return native_.check_error();
    }

    PmResult apply_smoke_cloth_couplings(
        simulation::DeformableSmokeCouplingPair pair,
        simulation::DeformableSmokeCouplingMode mode,
        float dt) noexcept {
        constexpr std::uint32_t block = 128U;
        for (auto &owner : state_.smoke_cloth_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            auto &coupling = *owner;
            auto &smoke = *state_.smokes[coupling.options.smoke.index];
            const bool grid = smoke.options.grid_resolution != 0U;
            const auto actual_pair = grid
                ? simulation::DeformableSmokeCouplingPair::cloth_smoke_grid
                : simulation::DeformableSmokeCouplingPair::
                      cloth_smoke_particle;
            if (pair != actual_pair) continue;
            auto &cloth = *state_.cloths[coupling.options.cloth.index];
            const auto field = staged_smoke_grid(smoke);
            if (mode ==
                simulation::DeformableSmokeCouplingMode::contact) {
                const auto groups =
                    (smoke.options.capacity + block - 1U) / block;
                const float clearance =
                    coupling.options.contact_distance > 0.0F
                    ? coupling.options.contact_distance
                    : smoke.options.particle_radius + cloth.thickness;
                native_.collide_smoke_with_cloth(
                    groups, block, native_.queue(),
                    smoke.realtime_positions,
                    smoke.realtime_previous_positions,
                    smoke.realtime_velocities, smoke.realtime_ages,
                    smoke.realtime_pressures, smoke.realtime_pending_count,
                    native_.smoke_dispatch_options(smoke.options),
                    cloth.realtime_positions, cloth.realtime_velocities,
                    cloth.indices, cloth.index_count, coupling.minimum,
                    coupling.maximum, clearance, field);
            } else if (!grid) {
                const auto groups =
                    (cloth.vertex_count + block - 1U) / block;
                native_.apply_smoke_wind_to_cloth(
                    groups, block, native_.queue(),
                    cloth.realtime_positions, cloth.realtime_velocities,
                    cloth.inverse_masses, cloth.vertex_count,
                    native_.smoke_dispatch_options(smoke.options),
                    smoke.realtime_positions, smoke.realtime_velocities,
                    smoke.keys[1], smoke.indices[1], field,
                    coupling.options.wind_drag,
                    coupling.options.maximum_wind_acceleration, dt);
            } else {
                auto result = native_.clear(
                    cloth.smoke_forces, 0,
                    cloth.vertex_count * sizeof(Vec3), native_.queue());
                if (result != PM_SUCCESS) return result;
                const auto triangles = cloth.index_count / 3U;
                native_.accumulate_smoke_grid_force_on_cloth(
                    (triangles + block - 1U) / block, block,
                    native_.queue(), cloth.realtime_positions,
                    cloth.realtime_velocities, cloth.inverse_masses,
                    cloth.indices, cloth.index_count, field,
                    coupling.options.wind_drag,
                    smoke.options.rest_number_density,
                    smoke.options.grid_kinematic_viscosity,
                    smoke.options.grid_les_coefficient,
                    cloth.smoke_forces);
                native_.apply_smoke_grid_force_to_cloth(
                    (cloth.vertex_count + block - 1U) / block, block,
                    native_.queue(), cloth.realtime_velocities,
                    cloth.inverse_masses, cloth.smoke_forces,
                    cloth.vertex_count,
                    coupling.options.maximum_wind_acceleration, dt);
            }
        }
        return native_.check_error();
    }

    PmResult apply_smoke_soft_body_couplings(
        simulation::DeformableSmokeCouplingPair pair,
        simulation::DeformableSmokeCouplingMode mode,
        float dt) noexcept {
        constexpr std::uint32_t block = 128U;
        for (auto &owner : state_.smoke_soft_body_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            auto &coupling = *owner;
            auto &smoke = *state_.smokes[coupling.options.smoke.index];
            const bool grid = smoke.options.grid_resolution != 0U;
            const auto actual_pair = grid
                ? simulation::DeformableSmokeCouplingPair::
                      soft_body_smoke_grid
                : simulation::DeformableSmokeCouplingPair::
                      soft_body_smoke_particle;
            if (pair != actual_pair) continue;
            auto &body = *state_.soft_bodies[
                coupling.options.soft_body.index];
            const auto field = staged_smoke_grid(smoke);
            if (mode ==
                simulation::DeformableSmokeCouplingMode::contact) {
                const auto groups =
                    (smoke.options.capacity + block - 1U) / block;
                const float clearance =
                    coupling.options.contact_distance > 0.0F
                    ? coupling.options.contact_distance
                    : smoke.options.particle_radius + body.node_radius;
                native_.collide_smoke_with_soft_body(
                    groups, block, native_.queue(),
                    smoke.realtime_positions,
                    smoke.realtime_previous_positions,
                    smoke.realtime_velocities, smoke.realtime_ages,
                    smoke.realtime_pending_count,
                    native_.smoke_dispatch_options(smoke.options),
                    body.realtime_surface_positions,
                    body.surface_bindings, body.surface_indices,
                    body.surface_index_count, body.realtime_velocities,
                    coupling.minimum, coupling.maximum, clearance, field);
            } else if (!grid) {
                const auto groups =
                    (body.node_count + block - 1U) / block;
                native_.apply_smoke_wind_to_soft_body(
                    groups, block, native_.queue(),
                    body.realtime_positions, body.realtime_velocities,
                    body.inverse_masses, body.node_count,
                    native_.smoke_dispatch_options(smoke.options),
                    smoke.realtime_positions, smoke.realtime_velocities,
                    smoke.keys[1], smoke.indices[1], field,
                    coupling.options.wind_drag,
                    coupling.options.maximum_wind_acceleration, dt,
                    body.realtime_maximum_speed);
            } else {
                auto result = native_.copy(
                    body.velocity_scratch, body.realtime_velocities,
                    body.node_count * sizeof(Vec3),
                    device::CopyDirection::device_to_device,
                    native_.queue());
                if (result != PM_SUCCESS) return result;
                native_.apply_smoke_grid_force_to_soft_body(
                    (body.surface_index_count / 3U + block - 1U) / block,
                    block, native_.queue(),
                    body.realtime_surface_positions, body.surface_indices,
                    body.surface_index_count, body.surface_bindings,
                    body.velocity_scratch, body.realtime_velocities,
                    body.inverse_masses, field,
                    coupling.options.wind_drag,
                    smoke.options.rest_number_density,
                    smoke.options.grid_kinematic_viscosity,
                    smoke.options.grid_les_coefficient,
                    coupling.options.maximum_wind_acceleration, dt);
            }
        }
        return native_.check_error();
    }

    PmResult apply_smoke_rope_couplings(
        simulation::DeformableSmokeCouplingPair pair,
        simulation::DeformableSmokeCouplingMode mode,
        float dt) noexcept {
        constexpr std::uint32_t block = 128U;
        for (auto &owner : state_.smoke_rope_couplings) {
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            auto &coupling = *owner;
            auto &smoke = *state_.smokes[coupling.options.smoke.index];
            const bool grid = smoke.options.grid_resolution != 0U;
            const auto actual_pair = grid
                ? simulation::DeformableSmokeCouplingPair::rope_smoke_grid
                : simulation::DeformableSmokeCouplingPair::
                      rope_smoke_particle;
            if (pair != actual_pair) continue;
            auto &rope = *state_.ropes[coupling.options.rope.index];
            const auto field = staged_smoke_grid(smoke);
            if (mode ==
                simulation::DeformableSmokeCouplingMode::contact) {
                const auto groups =
                    (smoke.options.capacity + block - 1U) / block;
                const float clearance =
                    coupling.options.contact_distance > 0.0F
                    ? coupling.options.contact_distance
                    : smoke.options.particle_radius +
                          rope.data.options.radius;
                native_.collide_smoke_with_rope(
                    groups, block, native_.queue(),
                    smoke.realtime_positions,
                    smoke.realtime_previous_positions,
                    smoke.realtime_velocities, smoke.realtime_ages,
                    smoke.realtime_pending_count,
                    native_.smoke_dispatch_options(smoke.options),
                    rope.realtime_positions, rope.realtime_velocities,
                    rope.data.count, coupling.minimum, coupling.maximum,
                    clearance);
            } else {
                const auto groups =
                    (rope.data.count + block - 1U) / block;
                native_.apply_smoke_wind_to_rope(
                    groups, block, native_.queue(),
                    rope.realtime_positions, rope.realtime_velocities,
                    rope.data.count, rope.data.options.maximum_speed,
                    rope.data.options.first.enabled,
                    rope.data.options.last.enabled, rope.data.soft_first,
                    rope.data.soft_last,
                    native_.smoke_dispatch_options(smoke.options),
                    smoke.realtime_positions, smoke.realtime_velocities,
                    smoke.keys[1], smoke.indices[1], field,
                    coupling.options.wind_drag,
                    coupling.options.maximum_wind_acceleration, dt);
            }
        }
        return native_.check_error();
    }

    PmResult apply_smoke_deformable_couplings(
        const simulation::RealtimeScheduleScene &scene) noexcept {
        auto result = prepare_smoke_deformable_contacts();
        if (result != PM_SUCCESS) return result;
        result = index_smoke_for_deformable_drag();
        if (result != PM_SUCCESS) return result;
        const float dt = scene.timestep / static_cast<float>(
            simulation::deformable_smoke_coupling_sweeps);
        constexpr simulation::DeformableSmokeCouplingMode modes[]{
            simulation::DeformableSmokeCouplingMode::contact,
            simulation::DeformableSmokeCouplingMode::drag};
        constexpr simulation::DeformableSmokeCouplingPair cloth_pairs[]{
            simulation::DeformableSmokeCouplingPair::cloth_smoke_particle,
            simulation::DeformableSmokeCouplingPair::cloth_smoke_grid};
        constexpr simulation::DeformableSmokeCouplingPair soft_pairs[]{
            simulation::DeformableSmokeCouplingPair::
                soft_body_smoke_particle,
            simulation::DeformableSmokeCouplingPair::soft_body_smoke_grid};
        constexpr simulation::DeformableSmokeCouplingPair rope_pairs[]{
            simulation::DeformableSmokeCouplingPair::rope_smoke_particle,
            simulation::DeformableSmokeCouplingPair::rope_smoke_grid};
        for (std::uint32_t sweep = 0U;
             sweep < simulation::deformable_smoke_coupling_sweeps; ++sweep) {
            for (const auto pair : cloth_pairs)
                for (const auto mode : modes) {
                    result = apply_smoke_cloth_couplings(pair, mode, dt);
                    if (result != PM_SUCCESS) return result;
                }
            for (const auto pair : soft_pairs)
                for (const auto mode : modes) {
                    result = apply_smoke_soft_body_couplings(pair, mode, dt);
                    if (result != PM_SUCCESS) return result;
                }
            for (const auto pair : rope_pairs)
                for (const auto mode : modes) {
                    result = apply_smoke_rope_couplings(pair, mode, dt);
                    if (result != PM_SUCCESS) return result;
                }
        }
        return PM_SUCCESS;
    }

    PmResult apply_smoke_couplings(
        const simulation::RealtimeScheduleScene &scene) noexcept {
        if (std::any_of(state_.smoke_rigid_couplings.begin(),
                        state_.smoke_rigid_couplings.end(),
                        [](const auto &owner) {
                            return owner && owner->alive &&
                                owner->options.enabled;
                        })) {
            const auto result = apply_smoke_rigid_couplings(scene);
            if (result != PM_SUCCESS) return result;
        }
        if (deformable_coupling_count_ != 0U) {
            const auto result = apply_smoke_deformable_couplings(scene);
            if (result != PM_SUCCESS) return result;
        }
        return PM_SUCCESS;
    }

    PmResult stage_coupling_buffers(
        simulation::RealtimeWorkCounters &boundary_work) noexcept {
        for (const auto &slot : state_.slots)
            if (slot.alive && slot.dense_index < state_.rigid_body_count)
                state_.realtime_rigid_generations_host[slot.dense_index] =
                    slot.generation;
        const auto copy = [&](void *destination, const void *source,
                              std::size_t bytes) noexcept {
            const auto result = native_.copy(destination, source, bytes,
                device::CopyDirection::host_to_device, native_.queue());
            if (result == PM_SUCCESS) {
                boundary_work.bytes_moved += bytes;
                ++boundary_work.barriers;
            }
            return result;
        };
        if (state_.rigid_body_count != 0U) {
            const auto bytes = std::size_t(state_.rigid_body_count) *
                sizeof(std::uint32_t);
            auto result = copy(state_.realtime_rigid_generations,
                state_.realtime_rigid_generations_host.data(), bytes);
            if (result != PM_SUCCESS) return result;
            result = native_.clear(state_.realtime_rigid_correction_used, 0,
                std::size_t(state_.rigid_body_count) * sizeof(float),
                native_.queue());
            if (result != PM_SUCCESS) return result;
            boundary_work.bytes_moved +=
                std::size_t(state_.rigid_body_count) * sizeof(float);
            ++boundary_work.barriers;
        }
#define PM_COPY_COUPLING(member) do { \
    const auto result = copy(state_.member, state_.member##_host.data(), \
        sizeof(state_.member##_host)); \
    if (result != PM_SUCCESS) return result; \
} while (false)
        PM_COPY_COUPLING(realtime_coupling_first_endpoints);
        PM_COPY_COUPLING(realtime_coupling_second_endpoints);
        PM_COPY_COUPLING(realtime_coupling_first_generations);
        PM_COPY_COUPLING(realtime_coupling_second_generations);
        PM_COPY_COUPLING(realtime_coupling_pairs);
        PM_COPY_COUPLING(realtime_coupling_modes);
        PM_COPY_COUPLING(realtime_coupling_normals);
        PM_COPY_COUPLING(realtime_coupling_body_local_points);
        PM_COPY_COUPLING(realtime_coupling_contact_distances);
        PM_COPY_COUPLING(realtime_coupling_frictions);
        PM_COPY_COUPLING(realtime_coupling_responses);
        PM_COPY_COUPLING(realtime_coupling_active);
#undef PM_COPY_COUPLING
        constexpr std::size_t records = 64U;
        const std::size_t vector_bytes = records * sizeof(Vec3);
        const std::size_t scalar_bytes = records * sizeof(float);
        void *vectors[]{state_.realtime_coupling_first_position_deltas,
            state_.realtime_coupling_second_position_deltas,
            state_.realtime_coupling_first_impulses,
            state_.realtime_coupling_second_impulses};
        for (auto *buffer : vectors) {
            const auto result = native_.clear(
                buffer, 0, vector_bytes, native_.queue());
            if (result != PM_SUCCESS) return result;
            boundary_work.bytes_moved += vector_bytes;
            ++boundary_work.barriers;
        }
        void *scalars[]{state_.realtime_coupling_first_correction_used,
            state_.realtime_coupling_second_correction_used};
        for (auto *buffer : scalars) {
            const auto result = native_.clear(
                buffer, 0, scalar_bytes, native_.queue());
            if (result != PM_SUCCESS) return result;
            boundary_work.bytes_moved += scalar_bytes;
            ++boundary_work.barriers;
        }
        return PM_SUCCESS;
    }

    PmResult stage_coupling_graph(
        const simulation::RealtimeScheduleScene &scene) noexcept {
        liquid_coupling_count_ = 0U;
        rigid_coupling_count_ = 0U;
        deformable_coupling_count_ = 0U;
        rigid_attachment_records_ = 0U;
        deformable_attachment_records_ = 0U;
        state_.realtime_coupling_first_endpoints_host.fill(0U);
        state_.realtime_coupling_second_endpoints_host.fill(0U);
        state_.realtime_coupling_first_generations_host.fill(0U);
        state_.realtime_coupling_second_generations_host.fill(0U);
        state_.realtime_coupling_pairs_host.fill(0U);
        state_.realtime_coupling_modes_host.fill(0U);
        state_.realtime_coupling_normals_host.fill({});
        state_.realtime_coupling_body_local_points_host.fill({});
        state_.realtime_coupling_contact_distances_host.fill(0.0F);
        state_.realtime_coupling_frictions_host.fill(0.0F);
        state_.realtime_coupling_responses_host.fill(0.0F);
        state_.realtime_coupling_active_host.fill(0U);

        const auto add_rigid = [&](std::uint64_t edge_id,
                                   simulation::RigidCouplingFamily family,
                                   simulation::RigidCouplingMode mode,
                                   std::uint32_t endpoint_generation,
                                   std::uint32_t rigid_generation,
                                   std::uint32_t constraints = 1U) noexcept {
            if (rigid_coupling_count_ >= rigid_couplings_.size())
                return false;
            rigid_couplings_[rigid_coupling_count_++] = {edge_id, family,
                mode, endpoint_generation, rigid_generation, constraints};
            return true;
        };
        const auto graph_generation = static_cast<std::uint32_t>(
            scene.graph_revision %
                (std::numeric_limits<std::uint32_t>::max() - 1U)) + 1U;

        const auto add_liquid = [&](std::uint64_t edge_id,
                simulation::LiquidCouplingFamily family,
                simulation::LiquidCouplingMode mode,
                std::uint32_t liquid_generation,
                std::uint32_t partner_generation) noexcept {
            if (liquid_coupling_count_ >= liquid_couplings_.size())
                return false;
            liquid_couplings_[liquid_coupling_count_++] = {edge_id, family,
                mode, liquid_generation, partner_generation, 1U};
            return true;
        };
        constexpr simulation::LiquidCouplingMode liquid_modes[]{
            simulation::LiquidCouplingMode::contact,
            simulation::LiquidCouplingMode::pressure,
            simulation::LiquidCouplingMode::viscosity};
        for (const auto mode : liquid_modes) {
            for (std::uint32_t slot = 0U;
                 slot < state_.fluid_pair_couplings.size(); ++slot) {
                const auto &owner = state_.fluid_pair_couplings[slot];
                if (!owner || !owner->alive || !owner->options.enabled)
                    continue;
                if (owner->options.material_profile !=
                    FluidPairMaterialProfile::equal)
                    return PM_ERROR_FEATURE_NOT_PRESENT;
                const auto &options = owner->options;
                if (options.first.index >= state_.fluids.size() ||
                    options.second.index >= state_.fluids.size() ||
                    !state_.fluids[options.first.index] ||
                    !state_.fluids[options.second.index] ||
                    !state_.fluids[options.first.index]->alive ||
                    !state_.fluids[options.second.index]->alive ||
                    state_.fluids[options.first.index]->generation !=
                        options.first.generation ||
                    state_.fluids[options.second.index]->generation !=
                        options.second.generation)
                    return PM_ERROR_INVALID_STATE;
                if (!add_liquid(0x510000000ULL +
                        std::uint64_t{static_cast<std::uint32_t>(mode)} *
                            0x100000ULL + slot,
                        simulation::LiquidCouplingFamily::liquid, mode,
                        options.first.generation,
                        options.second.generation))
                    return PM_ERROR_CAPACITY_EXCEEDED;
            }
        }
        if (state_.rigid_body_count != 0U) {
            constexpr simulation::LiquidCouplingMode rigid_modes[]{
                simulation::LiquidCouplingMode::contact,
                simulation::LiquidCouplingMode::containment};
            for (const auto mode : rigid_modes) {
                for (std::uint32_t slot = 0U;
                     slot < state_.fluids.size(); ++slot) {
                    const auto &owner = state_.fluids[slot];
                    if (!owner || !owner->alive) continue;
                    if (!add_liquid(0x520000000ULL +
                            std::uint64_t{
                                static_cast<std::uint32_t>(mode)} *
                                0x100000ULL + slot,
                            simulation::LiquidCouplingFamily::rigid, mode,
                            owner->generation, graph_generation))
                        return PM_ERROR_CAPACITY_EXCEEDED;
                }
            }
        }
        constexpr simulation::LiquidCouplingMode cloth_modes[]{
            simulation::LiquidCouplingMode::contact,
            simulation::LiquidCouplingMode::containment};
        for (const auto mode : cloth_modes) {
            for (std::uint32_t slot = 0U;
                 slot < state_.fluid_cloth_couplings.size(); ++slot) {
                const auto &coupling = state_.fluid_cloth_couplings[slot];
                if (!coupling.alive || !coupling.options.enabled ||
                    (mode == simulation::LiquidCouplingMode::contact
                        ? !coupling.options.contact
                        : !coupling.options.containment))
                    continue;
                const auto &options = coupling.options;
                if (options.fluid.index >= state_.fluids.size() ||
                    options.cloth.index >= state_.cloths.size() ||
                    !state_.fluids[options.fluid.index] ||
                    !state_.cloths[options.cloth.index] ||
                    !state_.fluids[options.fluid.index]->alive ||
                    !state_.cloths[options.cloth.index]->alive ||
                    state_.fluids[options.fluid.index]->generation !=
                        options.fluid.generation ||
                    state_.cloths[options.cloth.index]->generation !=
                        options.cloth.generation)
                    return PM_ERROR_INVALID_STATE;
                if (!add_liquid(0x530000000ULL +
                        std::uint64_t{static_cast<std::uint32_t>(mode)} *
                            0x100000ULL + slot,
                        simulation::LiquidCouplingFamily::cloth, mode,
                        options.fluid.generation,
                        options.cloth.generation))
                    return PM_ERROR_CAPACITY_EXCEEDED;
            }
        }
        for (std::uint32_t slot = 0U;
             slot < state_.fluid_soft_couplings.size(); ++slot) {
            const auto &owner = state_.fluid_soft_couplings[slot];
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            const auto &options = owner->options;
            if (options.fluid.index >= state_.fluids.size() ||
                options.soft_body.index >= state_.soft_bodies.size() ||
                !state_.fluids[options.fluid.index] ||
                !state_.soft_bodies[options.soft_body.index] ||
                !state_.fluids[options.fluid.index]->alive ||
                !state_.soft_bodies[options.soft_body.index]->alive ||
                state_.fluids[options.fluid.index]->generation !=
                    options.fluid.generation ||
                state_.soft_bodies[options.soft_body.index]->generation !=
                    options.soft_body.generation)
                return PM_ERROR_INVALID_STATE;
            if (!add_liquid(0x540000000ULL + slot,
                    simulation::LiquidCouplingFamily::soft_body,
                    simulation::LiquidCouplingMode::contact,
                    options.fluid.generation, options.soft_body.generation))
                return PM_ERROR_CAPACITY_EXCEEDED;
        }
        for (std::uint32_t slot = 0U;
             slot < state_.fluid_rope_couplings.size(); ++slot) {
            const auto &owner = state_.fluid_rope_couplings[slot];
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            const auto &options = owner->options;
            if (options.fluid.index >= state_.fluids.size() ||
                options.rope.index >= state_.ropes.size() ||
                !state_.fluids[options.fluid.index] ||
                !state_.ropes[options.rope.index] ||
                !state_.fluids[options.fluid.index]->alive ||
                !state_.ropes[options.rope.index]->alive ||
                state_.fluids[options.fluid.index]->generation !=
                    options.fluid.generation ||
                state_.ropes[options.rope.index]->generation !=
                    options.rope.generation)
                return PM_ERROR_INVALID_STATE;
            if (!add_liquid(0x550000000ULL + slot,
                    simulation::LiquidCouplingFamily::rope,
                    simulation::LiquidCouplingMode::contact,
                    options.fluid.generation, options.rope.generation))
                return PM_ERROR_CAPACITY_EXCEEDED;
        }
        constexpr simulation::LiquidCouplingMode smoke_modes[]{
            simulation::LiquidCouplingMode::drag,
            simulation::LiquidCouplingMode::thermal};
        constexpr simulation::LiquidCouplingFamily smoke_families[]{
            simulation::LiquidCouplingFamily::smoke_particle,
            simulation::LiquidCouplingFamily::smoke_grid};
        for (const auto family : smoke_families) {
            for (const auto mode : smoke_modes) {
                for (std::uint32_t slot = 0U;
                     slot < state_.fluid_smoke_couplings.size(); ++slot) {
                    const auto &owner = state_.fluid_smoke_couplings[slot];
                    if (!owner || !owner->alive) continue;
                    const auto &options = owner->options;
                    if (options.fluid.index >= state_.fluids.size() ||
                        options.smoke.index >= state_.smokes.size() ||
                        !state_.fluids[options.fluid.index] ||
                        !state_.smokes[options.smoke.index] ||
                        !state_.fluids[options.fluid.index]->alive ||
                        !state_.smokes[options.smoke.index]->alive ||
                        state_.fluids[options.fluid.index]->generation !=
                            options.fluid.generation ||
                        state_.smokes[options.smoke.index]->generation !=
                            options.smoke.generation)
                        return PM_ERROR_INVALID_STATE;
                    const auto actual_family =
                        state_.smokes[options.smoke.index]
                        ->options.grid_resolution == 0U
                            ? simulation::LiquidCouplingFamily::smoke_particle
                            : simulation::LiquidCouplingFamily::smoke_grid;
                    if (actual_family != family) continue;
                    if (!add_liquid(0x560000000ULL +
                            std::uint64_t{static_cast<std::uint32_t>(mode)} *
                                0x100000ULL + slot,
                            family, mode, options.fluid.generation,
                            options.smoke.generation))
                        return PM_ERROR_CAPACITY_EXCEEDED;
                }
            }
        }
        if (state_.rigid_body_count != 0U) {
            for (std::uint32_t slot = 0U; slot < state_.cloths.size(); ++slot) {
                const auto &owner = state_.cloths[slot];
                if (!owner || !owner->alive) continue;
                if (!add_rigid(0x100000000ULL + slot,
                        simulation::RigidCouplingFamily::cloth,
                        simulation::RigidCouplingMode::contact,
                        owner->generation, graph_generation))
                    return PM_ERROR_CAPACITY_EXCEEDED;
            }
            for (std::uint32_t slot = 0U;
                 slot < state_.soft_bodies.size(); ++slot) {
                const auto &owner = state_.soft_bodies[slot];
                if (!owner || !owner->alive) continue;
                if (!add_rigid(0x200000000ULL + slot,
                        simulation::RigidCouplingFamily::soft_body,
                        simulation::RigidCouplingMode::contact,
                        owner->generation, graph_generation))
                    return PM_ERROR_CAPACITY_EXCEEDED;
            }
            for (std::uint32_t slot = 0U; slot < state_.ropes.size(); ++slot) {
                const auto &owner = state_.ropes[slot];
                if (!owner || !owner->alive) continue;
                if (!add_rigid(0x300000000ULL + slot,
                        simulation::RigidCouplingFamily::rope,
                        simulation::RigidCouplingMode::contact,
                        owner->generation, graph_generation))
                    return PM_ERROR_CAPACITY_EXCEEDED;
            }
            for (std::uint32_t slot = 0U; slot < state_.ropes.size(); ++slot) {
                const auto &owner = state_.ropes[slot];
                if (!owner || !owner->alive) continue;
                const RopeAttachment attachments[2]{
                    owner->data.options.first, owner->data.options.last};
                for (std::uint32_t end = 0U; end < 2U; ++end) {
                    const auto &attachment = attachments[end];
                    if (!attachment.enabled ||
                        attachment.body.index >= state_.slots.size())
                        continue;
                    const auto &body_slot = state_.slots[attachment.body.index];
                    if (!body_slot.alive ||
                        body_slot.generation != attachment.body.generation ||
                        body_slot.dense_index >= state_.rigid_body_count)
                        return PM_ERROR_INVALID_STATE;
                    if (!add_rigid(0x380000000ULL +
                            std::uint64_t{slot} * 2U + end,
                            simulation::RigidCouplingFamily::rope,
                            simulation::RigidCouplingMode::attachment,
                            owner->generation, attachment.body.generation))
                        return PM_ERROR_CAPACITY_EXCEEDED;
                    const auto record = rigid_attachment_records_++;
                    state_.realtime_coupling_second_endpoints_host[record] =
                        body_slot.dense_index;
                    state_.realtime_coupling_first_generations_host[record] =
                        owner->generation;
                    state_.realtime_coupling_second_generations_host[record] =
                        attachment.body.generation;
                    state_.realtime_coupling_pairs_host[record] =
                        static_cast<std::uint32_t>(
                            simulation::RigidCouplingFamily::rope);
                    state_.realtime_coupling_modes_host[record] =
                        static_cast<std::uint32_t>(
                            simulation::RigidCouplingMode::attachment);
                    state_.realtime_coupling_body_local_points_host[record] =
                        attachment.local_anchor;
                    state_.realtime_coupling_responses_host[record] = 0.995F;
                    state_.realtime_coupling_active_host[record] = 1U;
                }
            }
            constexpr simulation::RigidCouplingFamily smoke_families[]{
                simulation::RigidCouplingFamily::smoke_particle,
                simulation::RigidCouplingFamily::smoke_grid};
            constexpr simulation::RigidCouplingMode smoke_modes[]{
                simulation::RigidCouplingMode::contact,
                simulation::RigidCouplingMode::drag};
            for (const auto family : smoke_families) {
                for (const auto mode : smoke_modes) {
                    for (std::uint32_t slot = 0U;
                         slot < state_.smoke_rigid_couplings.size(); ++slot) {
                        const auto &owner =
                            state_.smoke_rigid_couplings[slot];
                        if (!owner || !owner->alive ||
                            !owner->options.enabled)
                            continue;
                        const auto &options = owner->options;
                        if (options.smoke.index >= state_.smokes.size() ||
                            !state_.smokes[options.smoke.index] ||
                            !state_.smokes[options.smoke.index]->alive ||
                            state_.smokes[options.smoke.index]->generation !=
                                options.smoke.generation ||
                            options.body.index >= state_.slots.size())
                            return PM_ERROR_INVALID_STATE;
                        const auto &body_slot =
                            state_.slots[options.body.index];
                        if (!body_slot.alive ||
                            body_slot.generation != options.body.generation ||
                            body_slot.dense_index >= state_.rigid_body_count ||
                            owner->mesh.index >=
                                state_.options.triangle_mesh_capacity ||
                            !state_.meshes[owner->mesh.index].alive ||
                            state_.meshes[owner->mesh.index].generation !=
                                owner->mesh.generation)
                            return PM_ERROR_INVALID_STATE;
                        const auto actual_family =
                            state_.smokes[options.smoke.index]
                                    ->options.grid_resolution == 0U
                                ? simulation::RigidCouplingFamily::
                                      smoke_particle
                                : simulation::RigidCouplingFamily::smoke_grid;
                        if (actual_family != family) continue;
                        if (!add_rigid(0x400000000ULL +
                                std::uint64_t{
                                    static_cast<std::uint32_t>(family)} *
                                    0x1000000ULL +
                                std::uint64_t{
                                    static_cast<std::uint32_t>(mode)} *
                                    0x100000ULL + slot,
                                family, mode, options.smoke.generation,
                                options.body.generation))
                            return PM_ERROR_CAPACITY_EXCEEDED;
                    }
                }
            }
        }

        const auto add_deformable = [&](std::uint64_t edge_id,
                simulation::DeformableSmokeCouplingPair pair,
                simulation::DeformableSmokeCouplingMode mode,
                std::uint32_t first_generation,
                std::uint32_t second_generation,
                std::uint32_t constraints = 1U) noexcept {
            if (deformable_coupling_count_ >= deformable_couplings_.size())
                return false;
            deformable_couplings_[deformable_coupling_count_++] = {edge_id,
                pair, mode, first_generation, second_generation, constraints};
            return true;
        };
        for (std::uint32_t slot = 0U;
             slot < state_.soft_cloth_couplings.size(); ++slot) {
            const auto &owner = state_.soft_cloth_couplings[slot];
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            const auto &options = owner->options;
            if (options.cloth.index >= state_.cloths.size() ||
                options.soft_body.index >= state_.soft_bodies.size() ||
                !state_.cloths[options.cloth.index] ||
                !state_.soft_bodies[options.soft_body.index])
                return PM_ERROR_INVALID_STATE;
            if (!add_deformable(0x410000000ULL + slot,
                    simulation::DeformableSmokeCouplingPair::cloth_soft_body,
                    simulation::DeformableSmokeCouplingMode::contact,
                    options.cloth.generation,
                    options.soft_body.generation))
                return PM_ERROR_CAPACITY_EXCEEDED;
        }
        for (std::uint32_t slot = 0U;
             slot < state_.rope_cloth_couplings.size(); ++slot) {
            const auto &coupling = state_.rope_cloth_couplings[slot];
            if (!coupling.alive || !coupling.options.enabled) continue;
            const auto constraints =
                std::uint32_t(coupling.options.first_vertex != UINT32_MAX) +
                std::uint32_t(coupling.options.last_vertex != UINT32_MAX);
            if (!add_deformable(0x420000000ULL + slot,
                    simulation::DeformableSmokeCouplingPair::cloth_rope,
                    simulation::DeformableSmokeCouplingMode::attachment,
                    coupling.options.cloth.generation,
                    coupling.options.rope.generation, constraints))
                return PM_ERROR_CAPACITY_EXCEEDED;
            for (std::uint32_t end = 0U; end < 2U; ++end) {
                const auto vertex = end == 0U
                    ? coupling.options.first_vertex
                    : coupling.options.last_vertex;
                if (vertex == UINT32_MAX) continue;
                const auto record = 32U + deformable_attachment_records_++;
                if (record >= state_.realtime_coupling_active_host.size())
                    return PM_ERROR_CAPACITY_EXCEEDED;
                state_.realtime_coupling_first_generations_host[record] =
                    coupling.options.cloth.generation;
                state_.realtime_coupling_second_generations_host[record] =
                    coupling.options.rope.generation;
                state_.realtime_coupling_pairs_host[record] =
                    static_cast<std::uint32_t>(
                        simulation::DeformableSmokeCouplingPair::cloth_rope);
                state_.realtime_coupling_modes_host[record] =
                    static_cast<std::uint32_t>(
                        simulation::DeformableSmokeCouplingMode::attachment);
                state_.realtime_coupling_responses_host[record] = 0.995F;
                state_.realtime_coupling_active_host[record] = 1U;
            }
        }
        constexpr simulation::DeformableSmokeCouplingPair
            cloth_smoke_pairs[]{
                simulation::DeformableSmokeCouplingPair::
                    cloth_smoke_particle,
                simulation::DeformableSmokeCouplingPair::cloth_smoke_grid};
        constexpr simulation::DeformableSmokeCouplingMode
            deformable_smoke_modes[]{
            simulation::DeformableSmokeCouplingMode::contact,
            simulation::DeformableSmokeCouplingMode::drag};
        for (const auto pair : cloth_smoke_pairs) {
            for (const auto mode : deformable_smoke_modes) {
                for (std::uint32_t slot = 0U;
                     slot < state_.smoke_cloth_couplings.size(); ++slot) {
                    const auto &owner = state_.smoke_cloth_couplings[slot];
                    if (!owner || !owner->alive || !owner->options.enabled)
                        continue;
                    const auto &options = owner->options;
                    if (options.smoke.index >= state_.smokes.size() ||
                        options.cloth.index >= state_.cloths.size() ||
                        !state_.smokes[options.smoke.index] ||
                        !state_.cloths[options.cloth.index] ||
                        !state_.smokes[options.smoke.index]->alive ||
                        !state_.cloths[options.cloth.index]->alive ||
                        state_.smokes[options.smoke.index]->generation !=
                            options.smoke.generation ||
                        state_.cloths[options.cloth.index]->generation !=
                            options.cloth.generation)
                        return PM_ERROR_INVALID_STATE;
                    const auto actual_pair =
                        state_.smokes[options.smoke.index]
                                ->options.grid_resolution == 0U
                            ? simulation::DeformableSmokeCouplingPair::
                                  cloth_smoke_particle
                            : simulation::DeformableSmokeCouplingPair::
                                  cloth_smoke_grid;
                    if (actual_pair != pair) continue;
                    if (!add_deformable(0x430000000ULL +
                            std::uint64_t{
                                static_cast<std::uint32_t>(pair)} *
                                0x1000000ULL +
                            std::uint64_t{
                                static_cast<std::uint32_t>(mode)} *
                                0x100000ULL + slot,
                            pair, mode, options.cloth.generation,
                            options.smoke.generation))
                        return PM_ERROR_CAPACITY_EXCEEDED;
                }
            }
        }
        for (std::uint32_t slot = 0U;
             slot < state_.rope_soft_body_couplings.size(); ++slot) {
            const auto &owner = state_.rope_soft_body_couplings[slot];
            if (!owner || !owner->alive || !owner->options.enabled) continue;
            if (!add_deformable(0x450000000ULL + slot,
                    simulation::DeformableSmokeCouplingPair::soft_body_rope,
                    simulation::DeformableSmokeCouplingMode::contact,
                    owner->options.soft_body.generation,
                    owner->options.rope.generation, 2U))
                return PM_ERROR_CAPACITY_EXCEEDED;
            const auto &rope = *state_.ropes[owner->options.rope.index];
            for (std::uint32_t end = 0U; end < 2U; ++end) {
                const auto record = 32U + deformable_attachment_records_++;
                if (record >= state_.realtime_coupling_active_host.size())
                    return PM_ERROR_CAPACITY_EXCEEDED;
                state_.realtime_coupling_first_generations_host[record] =
                    owner->options.soft_body.generation;
                state_.realtime_coupling_second_generations_host[record] =
                    owner->options.rope.generation;
                state_.realtime_coupling_pairs_host[record] =
                    static_cast<std::uint32_t>(
                        simulation::DeformableSmokeCouplingPair::soft_body_rope);
                state_.realtime_coupling_modes_host[record] =
                    static_cast<std::uint32_t>(
                        simulation::DeformableSmokeCouplingMode::contact);
                state_.realtime_coupling_normals_host[record] = normalized_or(
                    multiply(owner->anchor_offset[end], -1.0F),
                    Vec3{0.0F, 1.0F, 0.0F});
                state_.realtime_coupling_contact_distances_host[record] =
                    owner->options.contact_distance > 0.0F
                        ? owner->options.contact_distance
                        : rope.data.options.radius;
                state_.realtime_coupling_frictions_host[record] =
                    owner->options.friction;
                state_.realtime_coupling_active_host[record] = 1U;
            }
        }
        for (std::uint32_t slot = 0U;
             slot < state_.rope_soft_body_couplings.size(); ++slot) {
            const auto &owner = state_.rope_soft_body_couplings[slot];
            if (!owner || !owner->alive || !owner->options.enabled ||
                (!owner->options.attach_first &&
                 !owner->options.attach_last))
                continue;
            const auto constraints =
                std::uint32_t(owner->options.attach_first) +
                std::uint32_t(owner->options.attach_last);
            if (!add_deformable(0x458000000ULL + slot,
                    simulation::DeformableSmokeCouplingPair::soft_body_rope,
                    simulation::DeformableSmokeCouplingMode::attachment,
                    owner->options.soft_body.generation,
                    owner->options.rope.generation, constraints))
                return PM_ERROR_CAPACITY_EXCEEDED;
            for (std::uint32_t end = 0U; end < 2U; ++end) {
                if (!(end == 0U ? owner->options.attach_first
                                : owner->options.attach_last))
                    continue;
                const auto record = 32U + deformable_attachment_records_++;
                if (record >= state_.realtime_coupling_active_host.size())
                    return PM_ERROR_CAPACITY_EXCEEDED;
                state_.realtime_coupling_first_generations_host[record] =
                    owner->options.soft_body.generation;
                state_.realtime_coupling_second_generations_host[record] =
                    owner->options.rope.generation;
                state_.realtime_coupling_pairs_host[record] =
                    static_cast<std::uint32_t>(
                        simulation::DeformableSmokeCouplingPair::soft_body_rope);
                state_.realtime_coupling_modes_host[record] =
                    static_cast<std::uint32_t>(
                        simulation::DeformableSmokeCouplingMode::attachment);
                state_.realtime_coupling_responses_host[record] = 0.995F;
                state_.realtime_coupling_active_host[record] = 1U;
            }
        }
        constexpr simulation::DeformableSmokeCouplingPair
            soft_smoke_pairs[]{
                simulation::DeformableSmokeCouplingPair::
                    soft_body_smoke_particle,
                simulation::DeformableSmokeCouplingPair::
                    soft_body_smoke_grid};
        for (const auto pair : soft_smoke_pairs) {
            for (const auto mode : deformable_smoke_modes) {
                for (std::uint32_t slot = 0U;
                     slot < state_.smoke_soft_body_couplings.size(); ++slot) {
                    const auto &owner =
                        state_.smoke_soft_body_couplings[slot];
                    if (!owner || !owner->alive || !owner->options.enabled)
                        continue;
                    const auto &options = owner->options;
                    if (options.smoke.index >= state_.smokes.size() ||
                        options.soft_body.index >= state_.soft_bodies.size() ||
                        !state_.smokes[options.smoke.index] ||
                        !state_.soft_bodies[options.soft_body.index] ||
                        !state_.smokes[options.smoke.index]->alive ||
                        !state_.soft_bodies[options.soft_body.index]->alive ||
                        state_.smokes[options.smoke.index]->generation !=
                            options.smoke.generation ||
                        state_.soft_bodies[options.soft_body.index]
                                ->generation != options.soft_body.generation)
                        return PM_ERROR_INVALID_STATE;
                    const auto actual_pair =
                        state_.smokes[options.smoke.index]
                                ->options.grid_resolution == 0U
                            ? simulation::DeformableSmokeCouplingPair::
                                  soft_body_smoke_particle
                            : simulation::DeformableSmokeCouplingPair::
                                  soft_body_smoke_grid;
                    if (actual_pair != pair) continue;
                    if (!add_deformable(0x460000000ULL +
                            std::uint64_t{
                                static_cast<std::uint32_t>(pair)} *
                                0x1000000ULL +
                            std::uint64_t{
                                static_cast<std::uint32_t>(mode)} *
                                0x100000ULL + slot,
                            pair, mode, options.soft_body.generation,
                            options.smoke.generation))
                        return PM_ERROR_CAPACITY_EXCEEDED;
                }
            }
        }
        constexpr simulation::DeformableSmokeCouplingPair
            rope_smoke_pairs[]{
                simulation::DeformableSmokeCouplingPair::
                    rope_smoke_particle,
                simulation::DeformableSmokeCouplingPair::rope_smoke_grid};
        for (const auto pair : rope_smoke_pairs) {
            for (const auto mode : deformable_smoke_modes) {
                for (std::uint32_t slot = 0U;
                     slot < state_.smoke_rope_couplings.size(); ++slot) {
                    const auto &owner = state_.smoke_rope_couplings[slot];
                    if (!owner || !owner->alive || !owner->options.enabled)
                        continue;
                    const auto &options = owner->options;
                    if (options.smoke.index >= state_.smokes.size() ||
                        options.rope.index >= state_.ropes.size() ||
                        !state_.smokes[options.smoke.index] ||
                        !state_.ropes[options.rope.index] ||
                        !state_.smokes[options.smoke.index]->alive ||
                        !state_.ropes[options.rope.index]->alive ||
                        state_.smokes[options.smoke.index]->generation !=
                            options.smoke.generation ||
                        state_.ropes[options.rope.index]->generation !=
                            options.rope.generation)
                        return PM_ERROR_INVALID_STATE;
                    const auto actual_pair =
                        state_.smokes[options.smoke.index]
                                ->options.grid_resolution == 0U
                            ? simulation::DeformableSmokeCouplingPair::
                                  rope_smoke_particle
                            : simulation::DeformableSmokeCouplingPair::
                                  rope_smoke_grid;
                    if (actual_pair != pair) continue;
                    if (!add_deformable(0x480000000ULL +
                            std::uint64_t{
                                static_cast<std::uint32_t>(pair)} *
                                0x1000000ULL +
                            std::uint64_t{
                                static_cast<std::uint32_t>(mode)} *
                                0x100000ULL + slot,
                            pair, mode, options.rope.generation,
                            options.smoke.generation))
                        return PM_ERROR_CAPACITY_EXCEEDED;
                }
            }
        }
        return liquid_coupling_count_ == scene.coupling_edges[
                   simulation::realtime_index(
                       simulation::RealtimeCouplingFamily::liquid)] &&
                rigid_coupling_count_ == scene.coupling_edges[
                   simulation::realtime_index(
                       simulation::RealtimeCouplingFamily::rigid)] &&
                deformable_coupling_count_ == scene.coupling_edges[
                   simulation::realtime_index(
                       simulation::RealtimeCouplingFamily::deformable_smoke)]
            ? PM_SUCCESS : PM_ERROR_INVALID_STATE;
    }

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

    template <typename T>
    T *fluid_spatial(simulation::SpatialBufferRange range) const noexcept {
        return realtime_range<T>(state_.realtime_fluid_spatial_scratch,
                                 range);
    }

    template <typename T>
    T *deformable_spatial(
        simulation::SpatialBufferRange range) const noexcept {
        return realtime_range<T>(
            state_.realtime_deformable_spatial_scratch, range);
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
        for (const auto &owner : state_.fluids) {
            if (!owner || !owner->alive) continue;
            const auto groups =
                (owner->options.capacity + 127U) / 128U;
            dispatch(simulation::RealtimeTelemetryPhase::commit, groups);
        }
        for (const auto &owner : state_.cloths) {
            if (!owner || !owner->alive) continue;
            const auto vertex_groups =
                (owner->vertex_capacity + 127U) / 128U;
            const auto triangle_groups =
                (owner->index_count / 3U + 127U) / 128U;
            dispatches(simulation::RealtimeTelemetryPhase::commit, 2U,
                       vertex_groups + triangle_groups);
        }
        for (const auto &owner : state_.soft_bodies) {
            if (!owner || !owner->alive) continue;
            const auto node_groups =
                (owner->node_count + 127U) / 128U;
            const auto surface_groups =
                (owner->surface_vertex_count + 127U) / 128U;
            dispatches(simulation::RealtimeTelemetryPhase::commit, 2U,
                       node_groups + surface_groups);
        }
        for (const auto &owner : state_.ropes) {
            if (!owner || !owner->alive) continue;
            const auto groups =
                (owner->data.count + 127U) / 128U;
            dispatch(simulation::RealtimeTelemetryPhase::commit, groups);
        }
        for (const auto &owner : state_.smokes) {
            if (!owner || !owner->alive) continue;
            const auto groups =
                (owner->options.capacity + 127U) / 128U;
            if (owner->options.grid_resolution == 0U) {
                dispatch(simulation::RealtimeTelemetryPhase::commit,
                         groups);
                continue;
            }
            std::uint64_t face_groups = 0U;
            for (std::uint32_t axis = 0U; axis < 3U; ++axis)
                face_groups +=
                    (std::uint64_t{owner->grid.face_count[axis]} + 127U) /
                    128U;
            dispatches(simulation::RealtimeTelemetryPhase::commit, 4U,
                       face_groups + groups);
        }
    }

    PmResult solve_smoke_domain(
        std::uint32_t iteration,
        const simulation::RealtimeScheduleScene &scene) noexcept {
        ++work(simulation::RealtimeTelemetryPhase::domain)
              .numerical_iterations;
        const auto *counters = state_.realtime_spatial.counters(epoch_);
        if (counters == nullptr) return PM_ERROR_INVALID_STATE;
        constexpr std::uint32_t block = 128U;
        for (auto &owner : state_.smokes) {
            if (!owner || !owner->alive) continue;
            auto &smoke = *owner;
            const auto options = native_.smoke_domain_options(smoke.options);
            if (iteration == 0U) {
                const auto budget = smoke.options.grid_resolution == 0U
                    ? simulation::smoke_realtime_particle_work_budget(
                          smoke.realtime_pending_count,
                          smoke.options.capacity,
                          smoke_realtime_candidates_per_particle)
                    : simulation::smoke_realtime_grid_work_budget(
                          smoke.realtime_pending_count,
                          smoke.options.capacity, smoke.grid.resolution,
                          smoke.grid.height);
                auto result = smoke.realtime_contract.begin(
                    epoch_, *counters, budget);
                if (result != PM_SUCCESS) return result;
                result = smoke.realtime_contract.record_advance(
                    scene.timestep, scene.substeps);
                if (result != PM_SUCCESS) return result;
            }
            if (smoke.options.grid_resolution == 0U) {
                const auto groups =
                    (smoke.options.capacity + block - 1U) / block;
                native_.solve_smoke_realtime_density_pressure(
                    groups, block, native_.queue(),
                    smoke.realtime_positions, smoke.realtime_ages,
                    smoke.realtime_pending_count, options,
                    smoke.realtime_candidate_counts,
                    smoke.realtime_candidate_indices,
                    smoke_realtime_candidates_per_particle,
                    scene.timestep,
                    simulation::smoke_realtime_profile_version,
                    smoke.realtime_number_densities,
                    smoke.realtime_pressures, snapshot_.query_status);
                native_.solve_smoke_realtime_position_delta(
                    groups, block, native_.queue(),
                    smoke.realtime_positions, smoke.realtime_velocities,
                    smoke.realtime_ages, smoke.realtime_pending_count,
                    options, smoke.realtime_candidate_counts,
                    smoke.realtime_candidate_indices,
                    smoke_realtime_candidates_per_particle,
                    smoke.realtime_number_densities,
                    smoke.realtime_pressures,
                    smoke.realtime_accumulated_corrections,
                    smoke.realtime_position_deltas, scene.timestep,
                    simulation::smoke_realtime_profile_version,
                    snapshot_.query_status);
                native_.apply_smoke_realtime_position_delta(
                    groups, block, native_.queue(),
                    smoke.realtime_positions,
                    smoke.realtime_position_deltas,
                    smoke.realtime_accumulated_corrections,
                    smoke.realtime_ages, smoke.realtime_pending_count,
                    options, scene.timestep,
                    simulation::smoke_realtime_profile_version,
                    snapshot_.query_status);
                auto result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatches(simulation::RealtimeTelemetryPhase::domain, 3U,
                           3ULL * groups);
                result = smoke.realtime_contract.record_spatial_pass(
                    epoch_, *counters);
                if (result != PM_SUCCESS) return result;
            } else {
                auto &grid = smoke.grid;
                const auto groups =
                    (grid.cell_count + block - 1U) / block;
                auto field = native_.smoke_dispatch_grid(grid);
                field.density = grid.realtime_density[1];
                field.pressure = grid.pressure[1];
                for (std::uint32_t axis = 0U; axis < 3U; ++axis)
                    field.face_velocity[axis] =
                        grid.face_velocity[axis][1];
                const float *pressure = (iteration & 1U) == 0U
                    ? grid.pressure[1] : grid.residual;
                float *next = (iteration & 1U) == 0U
                    ? grid.residual : grid.pressure[1];
                native_.solve_smoke_grid_realtime_pressure_cycle(
                    groups, block, native_.queue(), field, options,
                    grid.realtime_divergence, pressure, next,
                    grid.face_open[0], grid.face_open[1],
                    grid.face_open[2], scene.timestep,
                    simulation::smoke_realtime_profile_version,
                    snapshot_.query_status);
                auto result = native_.check_error();
                if (result != PM_SUCCESS) return result;
                dispatch(simulation::RealtimeTelemetryPhase::domain,
                         groups);
                if (iteration == 0U) {
                    result = smoke.realtime_contract.record_spatial_pass(
                        epoch_, *counters);
                    if (result != PM_SUCCESS) return result;
                }
            }
            auto result = smoke.realtime_contract.record_iteration();
            if (result != PM_SUCCESS) return result;
            const auto expected = smoke.options.grid_resolution == 0U
                ? simulation::smoke_realtime_particle_sweeps
                : simulation::smoke_realtime_grid_pressure_cycles;
            if (iteration + 1U == expected) {
                result = smoke.realtime_contract.finish();
                if (result != PM_SUCCESS) return result;
            }
        }
        return PM_SUCCESS;
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
    simulation::LiquidCouplingRealtimeTickContract
        liquid_coupling_contract_{};
    simulation::RigidCouplingRealtimeTickContract rigid_coupling_contract_{};
    simulation::DeformableSmokeCouplingTickContract
        deformable_coupling_contract_{};
    std::array<simulation::RigidCouplingRegistration,
               simulation::rigid_coupling_realtime_maximum_edges>
        rigid_couplings_{};
    std::array<simulation::LiquidCouplingRegistration,
               simulation::liquid_coupling_realtime_maximum_edges>
        liquid_couplings_{};
    std::array<simulation::DeformableSmokeCouplingRegistration,
               simulation::deformable_smoke_coupling_maximum_edges>
        deformable_couplings_{};
    std::uint32_t liquid_coupling_count_{};
    std::uint32_t rigid_coupling_count_{};
    std::uint32_t deformable_coupling_count_{};
    std::uint32_t rigid_attachment_records_{};
    std::uint32_t deformable_attachment_records_{};
    std::uint32_t output_state_{};
    std::uint64_t pending_contact_epoch_{};
    std::uint32_t staged_leaves_{};
    std::uint32_t staged_queries_{};
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

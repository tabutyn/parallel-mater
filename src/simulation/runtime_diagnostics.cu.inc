// SPDX-License-Identifier: MIT
// Included by runtime.cu; kept in one CUDA translation unit.

ContactDeviceView SimulationRuntime::contacts() const noexcept {
    if (!impl_ || (impl_->frame && !impl_->frame->acknowledged)) return {};
    std::uint32_t count = 0U;
    std::uint32_t overflow = 0U;
    if (!read_device_u32(nullptr, impl_->fluid_contact_count, count,
                         "fluid contact count readback failed") ||
        !read_device_u32(nullptr, impl_->fluid_contact_overflow, overflow,
                         "fluid contact overflow readback failed")) {
        return {};
    }
    count = std::min(count, impl_->options.contact_capacity);
    return {{impl_->fluid_contact_events, count}, count,
            overflow != 0U, impl_->frame_index};
}

RigidContactDeviceView SimulationRuntime::rigid_contacts() const noexcept {
    if (!impl_ || (impl_->frame && !impl_->frame->acknowledged)) {
        return {};
    }
    std::uint32_t count = 0U;
    if (!read_device_u32(nullptr, impl_->rigid_contact_count, count,
                         "rigid contact count readback failed")) {
        return {};
    }
    count = std::min(count, impl_->rigid_contact_capacity);
    return {{impl_->rigid_contact_events, count}, count, impl_->frame_index};
}

Status SimulationRuntime::physics_debug_frame(
    PhysicsDebugFrameView &output) const noexcept {
    output = {};
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    }
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (impl_->options.physics_debug.frame_capacity == 0U) {
        return failure(StatusCode::not_supported,
                       "physics debug capture was not enabled at world creation");
    }
    if (impl_->frame && !impl_->frame->acknowledged) {
        status = wait_for_completion(impl_->frame);
        if (!status) return status;
    }
    if (impl_->debug_frame_count == 0U) {
        return failure(StatusCode::not_supported,
                       "physics debug capture has no completed frame");
    }
    const std::size_t index =
        (impl_->debug_next_frame + impl_->debug_frames.size() - 1U) %
        impl_->debug_frames.size();
    const PhysicsDebugFrame &frame = impl_->debug_frames[index];
    output.frame_index = frame.frame_index;
    output.timestep = frame.timestep;
    output.gravity = frame.gravity;
    output.maximum_fluid_neighbor_count =
        frame.maximum_fluid_neighbor_count;
    output.rigid_bodies = {
        frame.rigid_bodies.data(), frame.rigid_bodies.size()};
    output.fluid_particles = {
        frame.fluid_particles.data(), frame.fluid_particles.size()};
    output.cloth_vertices = {
        frame.cloth_vertices.data(), frame.cloth_vertices.size()};
    output.soft_body_nodes = {
        frame.soft_body_nodes.data(), frame.soft_body_nodes.size()};
    output.rope_nodes = {frame.rope_nodes.data(), frame.rope_nodes.size()};
    output.rigid_contacts = {
        frame.rigid_contacts.data(), frame.rigid_contacts.size()};
    output.fluid_contacts = {
        frame.fluid_contacts.data(), frame.fluid_contacts.size()};
    return success();
}

Status SimulationRuntime::copy_physics_debug_capture(
    PhysicsDebugCapture &output) const noexcept {
    output.frames.clear();
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    }
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (impl_->options.physics_debug.frame_capacity == 0U) {
        return failure(StatusCode::not_supported,
                       "physics debug capture was not enabled at world creation");
    }
    if (impl_->frame && !impl_->frame->acknowledged) {
        status = wait_for_completion(impl_->frame);
        if (!status) return status;
    }
    try {
        output.frames.reserve(impl_->debug_frame_count);
        const std::size_t first =
            (impl_->debug_next_frame + impl_->debug_frames.size() -
             impl_->debug_frame_count) % impl_->debug_frames.size();
        for (std::size_t offset = 0U;
             offset < impl_->debug_frame_count; ++offset) {
            output.frames.push_back(impl_->debug_frames[
                (first + offset) % impl_->debug_frames.size()]);
        }
    } catch (...) {
        output.frames.clear();
        return failure(StatusCode::out_of_memory,
                       "physics debug capture copy failed");
    }
    return success();
}

Status SimulationRuntime::collect_step_timings(
    SimulationStepTimings &output) const noexcept {
    output = {};
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    }
    Status status = impl_->require_current_device();
    if (!status) {
        return status;
    }
    if (impl_->frame && !impl_->frame->acknowledged) {
        status = wait_for_completion(impl_->frame);
        if (!status) {
            return status;
        }
    }
    if (!impl_->timing_available || impl_->timing_boundary_count < 2U) {
        output.frame_index = impl_->frame_index;
        return success();
    }

    output.frame_index = impl_->timing_frame_index;
    output.available = true;
    cudaError_t error = cudaEventElapsedTime(
        &output.total_gpu_milliseconds, impl_->timing_events[0],
        impl_->timing_events[impl_->timing_boundary_count - 1U]);
    if (error != cudaSuccess) {
        return cuda_failure(error, "failed to collect total kernel timing");
    }
    for (std::size_t index = 0; index < impl_->timing_stages.size(); ++index) {
        float milliseconds = 0.0F;
        error = cudaEventElapsedTime(&milliseconds, impl_->timing_events[index],
                                     impl_->timing_events[index + 1U]);
        if (error != cudaSuccess) {
            return cuda_failure(error, "failed to collect kernel stage timing");
        }
        KernelTiming *timing = nullptr;
        bool contact_generation_stage = false;
        switch (impl_->timing_stages[index]) {
        case TimingStage::rigid_integration:
            timing = &output.rigid_integration;
            break;
        case TimingStage::rigid_world_bounds:
            timing = &output.rigid_world_bounds;
            contact_generation_stage = true;
            break;
        case TimingStage::rigid_pair_filter:
            timing = &output.rigid_pair_filter;
            contact_generation_stage = true;
            break;
        case TimingStage::rigid_pair_compaction:
            timing = &output.rigid_pair_compaction;
            contact_generation_stage = true;
            break;
        case TimingStage::rigid_leaf_pair_generation:
            timing = &output.rigid_leaf_pair_generation;
            contact_generation_stage = true;
            break;
        case TimingStage::rigid_contact_evaluation:
            timing = &output.rigid_contact_evaluation;
            contact_generation_stage = true;
            break;
        case TimingStage::rigid_contact_solve:
            timing = &output.rigid_contact_solve;
            break;
        case TimingStage::rigid_input_clear:
            timing = &output.rigid_input_clear;
            break;
        case TimingStage::rope_solve:
            timing = &output.rope_solve;
            break;
        case TimingStage::smoke_advection:
            timing = &output.smoke_advection;
            break;
        case TimingStage::smoke_grid:
            timing = &output.smoke_grid;
            break;
        case TimingStage::smoke_emission:
            timing = &output.smoke_emission;
            break;
        case TimingStage::cloth_prediction:
            timing = &output.cloth_prediction;
            break;
        case TimingStage::cloth_constraints:
            timing = &output.cloth_constraints;
            break;
        case TimingStage::cloth_contacts:
            timing = &output.cloth_contacts;
            break;
        case TimingStage::soft_body_prediction:
            timing = &output.soft_body_prediction;
            break;
        case TimingStage::soft_body_constraints:
            timing = &output.soft_body_constraints;
            break;
        case TimingStage::soft_body_contacts:
        case TimingStage::soft_body_contact_cleanup:
            timing = &output.soft_body_contacts;
            break;
        case TimingStage::soft_body_cloth_contacts:
            timing = &output.soft_body_cloth_contacts;
            break;
        case TimingStage::fluid_cloth_contacts:
            timing = &output.fluid_cloth_contacts;
            break;
        case TimingStage::fluid_soft_body_contacts:
            timing = &output.fluid_soft_body_contacts;
            break;
        case TimingStage::fluid_rope_contacts:
            timing = &output.fluid_rope_contacts;
            break;
        case TimingStage::rope_soft_body_contacts:
            timing = &output.rope_soft_body_contacts;
            break;
        case TimingStage::fluid_spawn:
            timing = &output.fluid_spawn;
            break;
        case TimingStage::fluid_neighbor_sort:
            timing = &output.fluid_neighbor_sort;
            break;
        case TimingStage::fluid_neighbor_forces:
            timing = &output.fluid_neighbor_forces;
            break;
        case TimingStage::fluid_integration:
            timing = &output.fluid_integration;
            break;
        case TimingStage::fluid_static_contacts:
            timing = &output.fluid_static_contacts;
            break;
        case TimingStage::fluid_body_index:
            timing = &output.fluid_body_index;
            break;
        case TimingStage::fluid_moving_contacts:
            timing = &output.fluid_moving_contacts;
            break;
        case TimingStage::fluid_contact_events:
            timing = &output.fluid_contact_events;
            break;
        case TimingStage::fluid_outflow_compaction:
            timing = &output.fluid_outflow_compaction;
            break;
        case TimingStage::fluid_smoke_exchange:
            timing = &output.fluid_smoke_exchange;
            break;
        }
        timing->total_milliseconds += milliseconds;
        const std::uint32_t launches =
            impl_->timing_launch_counts[index] != 0U ? impl_->timing_launch_counts[index] :
            impl_->timing_stages[index] == TimingStage::soft_body_cloth_contacts
                ? impl_->soft_cloth_kernels_per_substep
                : impl_->timing_stages[index] == TimingStage::soft_body_contact_cleanup
                ? 2U + (impl_->rigid_body_count != 0U
                    ? k_soft_contact_cleanup_passes * 5U / 2U : 0U)
                : impl_->timing_stages[index] == TimingStage::rigid_contact_solve
                ? impl_->rigid_solve_kernels_per_substep
                : impl_->timing_stages[index] ==
                          TimingStage::rigid_contact_evaluation
                    ? 3U + (impl_->rigid_leaf_manifolds != nullptr ? 1U : 0U) : 1U;
        timing->launch_count += launches;
        if (contact_generation_stage) {
            output.rigid_contact_generation.total_milliseconds += milliseconds;
            output.rigid_contact_generation.launch_count += launches;
        }
    }
    return success();
}

Status SimulationRuntime::collect_statistics(SimulationStatistics &output,
                                 device::ExecutionQueue queue) const noexcept {
    (void)queue;
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    }
    Status device_status = impl_->require_current_device();
    if (!device_status) {
        return device_status;
    }
    if (impl_->frame && !impl_->frame->acknowledged) {
        Status status = wait_for_completion(impl_->frame);
        if (!status) {
            return status;
        }
    }
    const std::size_t capacity = impl_->options.rigid_body_capacity;
    output = {};
    output.frame_index = impl_->frame_index;
    output.fluid_count = impl_->fluid_count;
    output.boiled_particle_count = impl_->boiled_particle_count;
    for (const auto &smoke : impl_->smokes) {
        if (!smoke || !smoke->alive) continue;
        ++output.smoke_system_count;
        output.smoke_particle_count += smoke->count;
        output.emitted_smoke_particle_count += smoke->emitted;
        output.allocated_bytes += static_cast<std::size_t>(smoke->options.capacity) *
            (2U * sizeof(Vec3) + 2U * sizeof(float));
    }
    for (const auto &cloth : impl_->cloths) {
        if (!cloth || !cloth->alive) continue;
        ++output.cloth_count;
        output.cloth_vertex_count += cloth->vertex_count;
        output.allocated_bytes +=
            static_cast<std::size_t>(cloth->vertex_capacity) *
                (6U * sizeof(Vec3) + sizeof(float) + sizeof(FluidBodyImpulse)) +
            cloth->index_count * sizeof(std::uint32_t) +
            (static_cast<std::size_t>(cloth->vertex_capacity) + 1U) *
                sizeof(std::uint32_t) +
            cloth->neighbor_capacity * sizeof(DeformableNeighbor) +
            cloth->bond_count * (sizeof(ClothBond) +
                2U * sizeof(std::uint8_t)) +
            (cloth->surface_positions != nullptr
                ? cloth->index_count * (sizeof(Vec3) +
                    3U * sizeof(std::uint32_t)) +
                    cloth->vertex_capacity * (sizeof(std::uint32_t) + sizeof(std::uint8_t)) : 0U) +
            impl_->options.rigid_body_capacity *
                sizeof(ClothBodyCorrection) +
            (cloth->volume_gradients != nullptr
                ? static_cast<std::size_t>(cloth->vertex_count) * sizeof(Vec3) +
                    (static_cast<std::size_t>(cloth->vertex_count) + 1U +
                        cloth->index_count) * sizeof(std::uint32_t) +
                    sizeof(float) : 0U) +
            (cloth->fluid_forces != nullptr
                ? static_cast<std::size_t>(cloth->vertex_capacity) * sizeof(Vec3)
                : 0U) +
            sizeof(std::uint32_t);
    }
    for (const auto &body : impl_->soft_bodies) {
        if (!body || !body->alive) continue;
        ++output.soft_body_count;
        output.soft_body_node_count += body->node_count;
        output.allocated_bytes +=
            static_cast<std::size_t>(body->node_count) *
                (14U * sizeof(Vec3) + 2U * sizeof(float) +
                 sizeof(FluidBodyImpulse)) +
            static_cast<std::size_t>(body->bond_count) *
                (sizeof(SoftBodyBond) + sizeof(std::uint8_t)) +
            (static_cast<std::size_t>(body->node_count) + 1U) *
                sizeof(std::uint32_t) +
            static_cast<std::size_t>(body->neighbor_count) *
                sizeof(DeformableNeighbor) +
            (body->warp_neighbors != nullptr
                ? static_cast<std::size_t>(body->neighbor_count) *
                    sizeof(SoftBodyNeighbor) +
                    static_cast<std::size_t>(body->node_count) * sizeof(float)
                : 0U) +
            static_cast<std::size_t>(body->surface_vertex_count) *
                (2U * sizeof(Vec3) + sizeof(SoftBodySurfaceBinding)) +
            static_cast<std::size_t>(body->surface_index_count) *
                sizeof(std::uint32_t) + 3U * sizeof(std::uint32_t) +
                sizeof(Vec3) + sizeof(Quaternion);
    }
    for (const auto &coupling : impl_->soft_cloth_couplings) {
        if (!coupling || !coupling->alive) continue;
        output.allocated_bytes +=
            impl_->soft_bodies[coupling->options.soft_body.index]->node_count *
                sizeof(SoftClothContact) +
            impl_->cloths[coupling->options.cloth.index]->vertex_capacity *
                sizeof(std::uint32_t);
    }
    output.emitted_particle_count = impl_->emitted_particle_count;
    for(const auto &rope:impl_->ropes) {
        if(!rope || !rope->alive)continue;
        ++output.rope_count;output.rope_node_count+=rope->data.count;
        output.allocated_bytes+=rope->data.count*(10*sizeof(Vec3)+2*sizeof(float))+
            2*impl_->options.rigid_body_capacity*sizeof(Vec3);
    }
    for (const auto &coupling : impl_->fluid_soft_couplings) {
        if (!coupling || !coupling->alive) continue;
        const auto &body = *impl_->soft_bodies[coupling->options.soft_body.index];
        const auto &fluid = *impl_->fluids[coupling->options.fluid.index];
        output.allocated_bytes += fluid.options.capacity * sizeof(FluidSoftContact) +
            body.node_count * (sizeof(std::uint32_t) + 2U * sizeof(Vec3)) +
            body.surface_vertex_count * sizeof(Vec3) + 2U * sizeof(Vec3) +
            sizeof(std::uint32_t) + sizeof(float);
        output.fluid_soft_body_contact_count += *coupling->contact_count;
        output.maximum_fluid_soft_body_penetration = std::max(
            output.maximum_fluid_soft_body_penetration, *coupling->maximum_penetration);
    }
    for (const auto &coupling : impl_->fluid_rope_couplings) {
        if (!coupling || !coupling->alive) continue;
        output.allocated_bytes += impl_->ropes[coupling->options.rope.index]->data.count *
            sizeof(Vec3) + sizeof(std::uint32_t) + sizeof(float);
        output.fluid_rope_contact_count += *coupling->contact_count;
        output.maximum_fluid_rope_penetration = std::max(
            output.maximum_fluid_rope_penetration, *coupling->maximum_penetration);
    }
    for (const auto &coupling : impl_->rope_soft_body_couplings) {
        if (!coupling || !coupling->alive) continue;
        output.allocated_bytes += impl_->soft_bodies[coupling->options.soft_body.index]->node_count *
            sizeof(Vec3) + impl_->soft_bodies[coupling->options.soft_body.index]->surface_vertex_count*
            sizeof(Vec3) + 2U*sizeof(Vec3) +
            coupling->tree_count*(sizeof(BvhNode)+2U*sizeof(std::uint32_t))+
            impl_->soft_bodies[coupling->options.soft_body.index]->surface_index_count/3U*
                sizeof(std::uint32_t)+sizeof(std::uint32_t)+sizeof(float);
        output.rope_soft_body_contact_count += *coupling->contact_count;
        output.maximum_rope_soft_body_penetration = std::max(
            output.maximum_rope_soft_body_penetration,*coupling->maximum_penetration);
    }
    output.destroyed_particle_count = impl_->destroyed_particle_count;
    output.spawn_capacity_miss_count = impl_->spawn_capacity_miss_count;
    for (const auto &source : impl_->particle_sources)
        if (source.alive)
            output.allocated_bytes += source.data->count * (sizeof(Vec3) + sizeof(std::uint8_t)) + sizeof(std::uint32_t);
    output.contact_count = *impl_->fluid_contact_count;
    output.contact_overflow_count = *impl_->fluid_contact_overflow;
    output.maximum_fluid_neighbor_count =
        *impl_->fluid_maximum_neighbor_count;
    for (const auto &fluid : impl_->fluids) {
        if (!fluid || !fluid->alive) continue;
        const std::uint32_t live = *fluid->count;
        output.particle_count += live;
        output.destroyed_particle_count +=
            fluid->initial_count + fluid->emitted_count - live;
        const std::size_t capacity = fluid->options.capacity;
        output.allocated_bytes += capacity *
            (6U * sizeof(Vec3) + 5U * sizeof(std::uint32_t) +
             5U * sizeof(float) + 3U * sizeof(std::uint8_t) +
             2U * sizeof(std::uint64_t) + sizeof(FluidBodyImpulse) +
             2U * sizeof(FluidContactSample)) +
             fluid->sort_workspace_size + fluid->select_workspace_size +
             3U * sizeof(std::uint32_t);
    }
    output.rigid_body_count = impl_->rigid_body_count;
    output.rigid_constraint_count = impl_->rigid_constraint_count;
    output.triangle_mesh_count = impl_->triangle_mesh_count;
    if (impl_->frame_index != 0U && impl_->rigid_body_count != 0U) {
        const auto &schedule = *impl_->rigid_contact_schedule;
        output.rigid_contact_island_count = schedule.island_count;
        output.rigid_contact_early_exit_count = schedule.early_exit_count;
        output.rigid_contact_pose_passes = schedule.counts[6];
        output.rigid_contact_impact_passes = schedule.counts[7];
        output.rigid_contact_maximum_passes = schedule.maximum_passes;
        output.rigid_contact_color_count = schedule.remaining;
        output.rigid_contact_overflow_pairs = schedule.counts[28];
        output.rigid_contact_grid_blocks = schedule.blocks;
        output.rigid_contact_candidate_pairs = schedule.candidates;
        output.rigid_contact_live_pairs = schedule.contacts;
    }
    output.allocated_bytes +=
        capacity * (sizeof(BodyParameters) + sizeof(BodyAccumulator) +
                    sizeof(simulation::GeometryConstraintBody) +
                    sizeof(KinematicTarget) + sizeof(RigidBodyId) +
                    4U * sizeof(RigidBodyState) +
                    (impl_->fixed_contact_projection != nullptr
                         ? sizeof(FixedContactProjection)
                         : 0U) +
                    (impl_->debug_applied_forces != nullptr
                         ? 2U * sizeof(Vec3) + sizeof(PhysicsDebugRigidSample) : 0U) +
                    sizeof(std::uint8_t)) +
        capacity * (capacity - 1U) / 2U * sizeof(ContactManifold) +
        impl_->rigid_leaf_pair_slot_capacity * sizeof(CachedContactPair) +
        capacity * capacity * sizeof(std::uint32_t) +
        sizeof(ContactSchedule) +
        capacity * (capacity - 1U) / 2U * sizeof(std::uint32_t) +
        capacity * (2U * sizeof(WorldAabb) + sizeof(std::uint32_t)) +
        capacity * capacity *
            (sizeof(std::uint8_t) + sizeof(std::uint32_t)) +
        sizeof(std::uint32_t) + impl_->rigid_broad_phase_workspace_size +
        impl_->rigid_leaf_pair_capacity * sizeof(LeafPair) +
        (impl_->rigid_leaf_manifolds != nullptr
             ? impl_->rigid_leaf_pair_capacity * sizeof(ContactManifold) : 0U) +
        capacity * capacity * sizeof(std::uint32_t) +
        impl_->rigid_contact_capacity * sizeof(RigidContactEvent) +
        impl_->options.contact_capacity * sizeof(ContactEvent) +
        k_fluid_body_buckets *
            ((capacity + 63U) / 64U) * sizeof(unsigned long long) +
        ((capacity + 63U) / 64U) * sizeof(unsigned long long) +
        4U * sizeof(std::uint32_t) +
        impl_->options.triangle_mesh_capacity * sizeof(TriangleMeshResource) +
        impl_->options.paint_field_capacity * sizeof(PaintFieldResource) +
        impl_->options.paint_rule_capacity * sizeof(PaintRuleResource) +
        impl_->options.rigid_constraint_capacity *
            sizeof(RigidConstraintResource);
    output.allocated_bytes +=
        impl_->debug_frames.capacity() * sizeof(PhysicsDebugFrame);
    for (const PhysicsDebugFrame &frame : impl_->debug_frames) {
        output.allocated_bytes +=
            frame.rigid_bodies.capacity() * sizeof(PhysicsDebugRigidSample) +
            frame.fluid_particles.capacity() * sizeof(PhysicsDebugFluidSample) +
            frame.cloth_vertices.capacity() * sizeof(PhysicsDebugClothSample) +
            frame.soft_body_nodes.capacity() *
                sizeof(PhysicsDebugSoftBodySample) +
            frame.rope_nodes.capacity() * sizeof(PhysicsDebugRopeSample) +
            frame.rigid_contacts.capacity() * sizeof(RigidContactEvent) +
            frame.fluid_contacts.capacity() * sizeof(ContactEvent);
    }
    for (std::uint32_t index = 0;
         index < impl_->options.paint_field_capacity; ++index) {
        const PaintFieldResource &field = impl_->paint_fields[index];
        if (!field.alive) continue;
        output.allocated_bytes +=
            field.options.vertex_uvs.size * sizeof(Vec2) +
            static_cast<std::size_t>(field.options.width) *
                field.options.height * sizeof(std::uint32_t);
    }
    for (std::uint32_t index = 0;
         index < impl_->options.triangle_mesh_capacity; ++index) {
        if (impl_->meshes[index].alive) {
            output.allocated_bytes +=
                impl_->meshes[index].vertex_count * sizeof(Vec3) +
                impl_->meshes[index].index_count * sizeof(std::uint32_t) +
                impl_->meshes[index].bvh_node_count * sizeof(BvhNode) +
                impl_->meshes[index].bvh_leaf_count * sizeof(std::uint32_t) +
                (impl_->meshes[index].solid_planes != nullptr
                     ? (impl_->meshes[index].index_count / 3U +
                        impl_->meshes[index].solid_unique_plane_count) * sizeof(CollisionPlane) : 0U) +
                (impl_->meshes[index].shell_normals != nullptr
                     ? impl_->meshes[index].index_count / 3U * sizeof(Vec3) : 0U);
        }
    }
    return success();
}

int SimulationRuntime::device_ordinal() const noexcept {
    return impl_ ? impl_->device_ordinal : -1;
}

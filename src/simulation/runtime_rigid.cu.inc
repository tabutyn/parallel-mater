// SPDX-License-Identifier: MIT
// Included by runtime.cu; kept in one CUDA translation unit.

Status SimulationRuntime::add_rigid_body(RigidBodyOptions options,
                             RigidBodyId &output) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    status = validate_body_options(options);
    if (!status) {
        return status;
    }
    status = impl_->validate_handle(options.mesh);
    if (!status) {
        return status;
    }
    if (impl_->rigid_body_count >= impl_->options.rigid_body_capacity) {
        return failure(StatusCode::capacity_exceeded,
                       "rigid body capacity is exhausted");
    }

    std::uint32_t slot_index = k_invalid_dense;
    for (std::uint32_t index = 0; index < impl_->slots.size(); ++index) {
        if (!impl_->slots[index].alive) {
            slot_index = index;
            break;
        }
    }
    if (slot_index == k_invalid_dense) {
        return failure(StatusCode::internal_error,
                       "no free rigid body handle slot was found");
    }

    RigidBodyState normalized_state = options.initial_state;
    normalized_state.orientation =
        normalized_quaternion(normalized_state.orientation);
    const TriangleMeshResource &mesh = impl_->meshes[options.mesh.index];

    Impl::Slot &slot = impl_->slots[slot_index];
    const std::uint32_t dense = impl_->rigid_body_count;
    slot.alive = true;
    slot.dense_index = dense;
    if (slot.generation == 0U) {
        slot.generation = 1U;
    }
    const RigidBodyId id{slot_index, slot.generation};
    impl_->parameters[dense] =
        make_parameters(options, mesh);
    impl_->accumulators[dense] = {};
    impl_->targets[dense] = {};
    impl_->ids[dense] = id;
    impl_->avbd_bodies[dense] = {};
    impl_->states[0][dense] = normalized_state;
    impl_->states[1][dense] = normalized_state;
    impl_->render_previous_states[dense] = normalized_state;
    ++impl_->rigid_body_count;
    ++impl_->revision;
    output = id;
    return success();
}
Status SimulationRuntime::remove_rigid_body(RigidBodyId body) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    for (std::uint32_t index = 0U;
         index < impl_->options.rigid_constraint_capacity; ++index) {
        const RigidConstraintResource &constraint =
            impl_->rigid_constraints[index];
        if (constraint.alive &&
            (constraint.options.body_a == body ||
             constraint.options.body_b == body))
            return failure(StatusCode::invalid_argument,
                           "rigid body is still referenced by a constraint");
    }
    for (const auto &coupling : impl_->smoke_rigid_couplings)
        if (coupling && coupling->alive && coupling->options.body == body)
            return failure(StatusCode::invalid_argument,
                           "rigid body is still referenced by smoke coupling");
    for (std::uint32_t index = 0;
         index < impl_->options.paint_field_capacity; ++index)
        if (impl_->paint_fields[index].alive &&
            impl_->paint_fields[index].options.body == body)
            return failure(StatusCode::invalid_argument,
                           "rigid body is still referenced by a paint field");
    for (std::uint32_t index = 0;
         index < impl_->options.paint_rule_capacity; ++index)
        if (impl_->paint_rules[index].alive &&
            impl_->paint_rules[index].options.rigid_source == body)
            return failure(StatusCode::invalid_argument,
                           "rigid body is still referenced by a paint rule");
    for(const auto &rope:impl_->ropes)
        if(rope && rope->alive &&
           ((rope->data.options.first.enabled && rope->data.options.first.body==body) ||
            (rope->data.options.last.enabled && rope->data.options.last.body==body)))
            return failure(StatusCode::invalid_argument,"rigid body is still referenced by a rope attachment");
    const std::uint32_t last = impl_->rigid_body_count - 1U;
    if (dense != last) {
        impl_->parameters[dense] = impl_->parameters[last];
        impl_->accumulators[dense] = impl_->accumulators[last];
        impl_->targets[dense] = impl_->targets[last];
        impl_->ids[dense] = impl_->ids[last];
        impl_->states[0][dense] = impl_->states[0][last];
        impl_->states[1][dense] = impl_->states[1][last];
        impl_->avbd_bodies[dense] = impl_->avbd_bodies[last];
        impl_->render_previous_states[dense] =
            impl_->render_previous_states[last];
        impl_->slots[impl_->ids[dense].index].dense_index = dense;
    }
    --impl_->rigid_body_count;
    Impl::Slot &slot = impl_->slots[body.index];
    slot.alive = false;
    slot.dense_index = k_invalid_dense;
    ++slot.generation;
    if (slot.generation == 0U) {
        slot.generation = 1U;
    }
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::set_rigid_body_state(RigidBodyId body,
                                   RigidBodyState state) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (!finite(state)) {
        return failure(StatusCode::invalid_argument,
                       "rigid body state must contain finite values");
    }
    const float quaternion_size = state.orientation.x * state.orientation.x +
                                  state.orientation.y * state.orientation.y +
                                  state.orientation.z * state.orientation.z +
                                  state.orientation.w * state.orientation.w;
    if (quaternion_size <= k_epsilon * k_epsilon) {
        return failure(StatusCode::invalid_argument,
                       "rigid body orientation must be nonzero");
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    state.orientation = normalized_quaternion(state.orientation);
    impl_->states[0][dense] = state;
    impl_->states[1][dense] = state;
    impl_->render_previous_states[dense] = state;
    impl_->avbd_bodies[dense] = {};
    impl_->accumulators[dense] = {};
    impl_->targets[dense] = {};
    for (unsigned index = 0; index < impl_->options.rigid_constraint_capacity; ++index) {
        auto &joint = impl_->rigid_constraints[index];
        if (joint.alive && (joint.options.body_a == body || joint.options.body_b == body))
            std::fill_n(joint.avbd_rows, 18U, avbd::Row{});
    }
    for (auto &smoke : impl_->smokes)
        if (smoke && smoke->alive)
            smoke->grid.static_metadata_valid = false;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::set_kinematic_target(RigidBodyId body,
                                   RigidBodyState target) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (!finite(target)) {
        return failure(StatusCode::invalid_argument,
                       "kinematic target must contain finite values");
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    if (impl_->parameters[dense].motion != MotionType::kinematic) {
        return failure(StatusCode::invalid_argument,
                       "kinematic targets require a kinematic body");
    }
    const float quaternion_size = target.orientation.x * target.orientation.x +
                                  target.orientation.y * target.orientation.y +
                                  target.orientation.z * target.orientation.z +
                                  target.orientation.w * target.orientation.w;
    if (quaternion_size <= k_epsilon * k_epsilon) {
        return failure(StatusCode::invalid_argument,
                       "kinematic orientation must be nonzero");
    }
    target.orientation = normalized_quaternion(target.orientation);
    impl_->targets[dense] = {target, true};
    return success();
}

Status SimulationRuntime::apply_force(RigidBodyId body, Vec3 force,
                          Vec3 world_point) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (!finite(force) || !finite(world_point)) {
        return failure(StatusCode::invalid_argument,
                       "force and application point must be finite");
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    if (impl_->parameters[dense].motion != MotionType::dynamic) {
        return failure(StatusCode::invalid_argument,
                       "forces may only be applied to dynamic bodies");
    }
    BodyAccumulator &accumulator = impl_->accumulators[dense];
    accumulator.force = add(accumulator.force, force);
    const Vec3 arm = subtract(
        world_point, impl_->states[impl_->current_state][dense].position);
    accumulator.torque = add(accumulator.torque, cross(arm, force));
    return success();
}

Status SimulationRuntime::apply_rigid_input(
    RigidBodyId body, Vec3 force, Vec3 torque, Vec3 linear_impulse,
    Vec3 angular_impulse) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) return status;
    if (!finite(force) || !finite(torque) || !finite(linear_impulse) ||
        !finite(angular_impulse)) {
        return failure(StatusCode::invalid_argument,
                       "rigid inputs must be finite");
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) return status;
    if (impl_->parameters[dense].motion != MotionType::dynamic) {
        return failure(StatusCode::invalid_argument,
                       "rigid inputs require a dynamic body");
    }
    BodyAccumulator &accumulator = impl_->accumulators[dense];
    accumulator.force = add(accumulator.force, force);
    accumulator.torque = add(accumulator.torque, torque);
    accumulator.impulse = add(accumulator.impulse, linear_impulse);
    accumulator.angular_impulse = add(accumulator.angular_impulse,
                                      angular_impulse);
    return success();
}

Status SimulationRuntime::apply_central_acceleration(
    HostSpan<RigidBodyId> bodies, Vec3 acceleration) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument,
                       "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (!finite(acceleration) ||
        (bodies.size != 0U && bodies.data == nullptr) ||
        bodies.size > impl_->rigid_body_count)
        return failure(StatusCode::invalid_argument,
                       "central acceleration batch is invalid");
    for (std::uint64_t index = 0U; index < bodies.size; ++index) {
        std::uint32_t dense = 0U;
        status = impl_->validate_handle(bodies.data[index], dense);
        if (!status) return status;
        if (impl_->parameters[dense].motion != MotionType::dynamic)
            return failure(StatusCode::invalid_argument,
                           "central acceleration requires dynamic bodies");
        for (std::uint64_t prior = 0U; prior < index; ++prior)
            if (bodies.data[prior] == bodies.data[index])
                return failure(StatusCode::invalid_argument,
                               "central acceleration body is duplicated");
    }
    for (std::uint64_t index = 0U; index < bodies.size; ++index) {
        std::uint32_t dense = 0U;
        status = impl_->validate_handle(bodies.data[index], dense);
        if (!status) return status;
        const float mass = 1.0F / impl_->parameters[dense].inverse_mass;
        impl_->accumulators[dense].force = add(
            impl_->accumulators[dense].force,
            multiply(acceleration, mass));
    }
    return success();
}

Status SimulationRuntime::apply_impulse(RigidBodyId body, Vec3 impulse,
                            Vec3 world_point) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (!finite(impulse) || !finite(world_point)) {
        return failure(StatusCode::invalid_argument,
                       "impulse and application point must be finite");
    }
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    if (impl_->parameters[dense].motion != MotionType::dynamic) {
        return failure(StatusCode::invalid_argument,
                       "impulses may only be applied to dynamic bodies");
    }
    BodyAccumulator &accumulator = impl_->accumulators[dense];
    accumulator.impulse = add(accumulator.impulse, impulse);
    const Vec3 arm = subtract(
        world_point, impl_->states[impl_->current_state][dense].position);
    accumulator.angular_impulse =
        add(accumulator.angular_impulse, cross(arm, impulse));
    return success();
}

Status SimulationRuntime::rigid_body_view(RigidBodyDeviceView &output) const noexcept {
    output = {};
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    output.ids = {impl_->ids, impl_->rigid_body_count};
    output.states = {impl_->states[impl_->current_state],
                     impl_->rigid_body_count};
    output.previous_states = {
        impl_->render_previous_states, impl_->rigid_body_count};
    if (impl_->debug_applied_forces != nullptr) {
        output.applied_forces = {
            impl_->debug_applied_forces, impl_->rigid_body_count};
        output.applied_torques = {
            impl_->debug_applied_torques, impl_->rigid_body_count};
    }
    output.revision = impl_->revision;
    return success();
}

Status SimulationRuntime::read_rigid_body_state(RigidBodyId body, RigidBodyState &output,
                                    device::ExecutionQueue queue) const noexcept {
    const cudaStream_t stream = native_stream(queue);
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
    std::uint32_t dense = 0U;
    status = impl_->validate_handle(body, dense);
    if (!status) {
        return status;
    }
    RigidBodyState temporary{};
    cudaError_t error = cudaMemcpyAsync(
        &temporary, impl_->states[impl_->current_state] + dense,
        sizeof(RigidBodyState), cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) {
        return cuda_failure(error, "rigid body state readback failed");
    }
    error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        return cuda_failure(error, "rigid body state readback synchronization failed");
    }
    output = temporary;
    return success();
}

Status SimulationRuntime::query_hit_box(HitBox box, HitBoxResult &output,
                            device::ExecutionQueue queue) const noexcept {
    const cudaStream_t stream = native_stream(queue);
    if (!impl_)
        return failure(StatusCode::invalid_argument,
                       "simulation is not initialized");
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (impl_->frame && !impl_->frame->acknowledged) {
        status = wait_for_completion(impl_->frame);
        if (!status) return status;
    }
    const float orientation_length_squared =
        box.orientation.x * box.orientation.x +
        box.orientation.y * box.orientation.y +
        box.orientation.z * box.orientation.z +
        box.orientation.w * box.orientation.w;
    if (!finite(box.center) || !finite(box.orientation) ||
        !finite(box.half_extents) || box.half_extents.x <= 0.0F ||
        box.half_extents.y <= 0.0F || box.half_extents.z <= 0.0F ||
        orientation_length_squared <= k_epsilon * k_epsilon) {
        return failure(StatusCode::invalid_argument,
                       "hit box transform and half extents are invalid");
    }
    box.orientation = normalized_quaternion(box.orientation);

    if (impl_->rigid_body_count != 0U) {
        const PmResult query_result = simulation::query_rigid_hit_box(
            queue,
            box, impl_->parameters, impl_->states[impl_->current_state],
            impl_->meshes, impl_->rigid_body_count,
            impl_->hit_box_rigid_flags);
        if (query_result != PM_SUCCESS)
            return failure(StatusCode::backend_failure,
                           "rigid hit-box query launch failed");
    }
    for (const auto &owner : impl_->fluids) {
        if (!owner || !owner->alive || *owner->count == 0U) continue;
        const PmResult query_result = simulation::query_particle_hit_box(
            queue,
            box, owner->positions, *owner->count, owner->hit_box_flags);
        if (query_result != PM_SUCCESS)
            return failure(StatusCode::backend_failure,
                           "particle hit-box query launch failed");
    }
    cudaError_t error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess)
        return cuda_failure(error, "hit-box query synchronization failed");

    HitBoxResult temporary;
    try {
        temporary.rigid_bodies.reserve(impl_->rigid_body_count);
        std::size_t particle_count = 0U;
        for (const auto &owner : impl_->fluids)
            if (owner && owner->alive) particle_count += *owner->count;
        temporary.particles.reserve(particle_count);
        for (std::uint32_t body = 0U; body < impl_->rigid_body_count; ++body)
            if (impl_->hit_box_rigid_flags[body] != 0U)
                temporary.rigid_bodies.push_back(impl_->ids[body]);
        for (std::uint32_t slot = 0U; slot < impl_->fluids.size(); ++slot) {
            const auto &owner = impl_->fluids[slot];
            if (!owner || !owner->alive) continue;
            const FluidId fluid{slot, owner->generation};
            for (std::uint32_t particle = 0U; particle < *owner->count;
                 ++particle)
                if (owner->hit_box_flags[particle] != 0U)
                    temporary.particles.push_back(
                        {fluid, owner->ids[particle]});
        }
        std::sort(temporary.rigid_bodies.begin(),
                  temporary.rigid_bodies.end(),
                  [](RigidBodyId left, RigidBodyId right) {
                      return left.index < right.index ||
                          (left.index == right.index &&
                           left.generation < right.generation);
                  });
        std::sort(temporary.particles.begin(), temporary.particles.end(),
                  [](HitBoxParticle left, HitBoxParticle right) {
                      if (left.fluid.index != right.fluid.index)
                          return left.fluid.index < right.fluid.index;
                      if (left.fluid.generation != right.fluid.generation)
                          return left.fluid.generation < right.fluid.generation;
                      return left.stable_particle_id <
                             right.stable_particle_id;
                  });
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "hit-box query result allocation failed");
    }
    output = std::move(temporary);
    return success();
}

Status SimulationRuntime::add_rigid_constraint(
    RigidConstraintOptions options, RigidConstraintId &output) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    status = validate_constraint_options(options);
    if (!status) return status;
    std::uint32_t dense_a = 0U;
    std::uint32_t dense_b = 0U;
    if (!(status = impl_->validate_handle(options.body_a, dense_a)) ||
        !(status = impl_->validate_handle(options.body_b, dense_b))) return status;
    if (impl_->parameters[dense_a].motion != MotionType::dynamic &&
        impl_->parameters[dense_b].motion != MotionType::dynamic)
        return failure(StatusCode::invalid_argument,
                       "rigid constraint needs at least one dynamic body");
    if (impl_->rigid_constraint_count >= impl_->options.rigid_constraint_capacity)
        return failure(StatusCode::capacity_exceeded,
                       "rigid constraint capacity is exhausted");
    std::uint32_t slot = k_invalid_dense;
    for (std::uint32_t index = 0U;
         index < impl_->options.rigid_constraint_capacity; ++index)
        if (!impl_->rigid_constraints[index].alive) {
            slot = index;
            break;
        }
    if (slot == k_invalid_dense)
        return failure(StatusCode::internal_error,
                       "no free rigid constraint slot was found");
    options.local_orientation_a = normalized_quaternion(
        options.local_orientation_a);
    options.local_orientation_b = normalized_quaternion(
        options.local_orientation_b);
    RigidConstraintResource &resource = impl_->rigid_constraints[slot];
    if (resource.generation == 0U) resource.generation = 1U;
    resource.options = options;
    resource.state = {};
    resource.state.enabled = options.enabled;
    std::fill_n(resource.avbd_rows, 18U, avbd::Row{});
    resource.alive = true;
    ++impl_->rigid_constraint_count;
    ++impl_->revision;
    output = {slot, resource.generation};
    return success();
}

Status SimulationRuntime::update_rigid_constraint(
    RigidConstraintId id, RigidConstraintOptions options) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    status = validate_constraint_options(options);
    if (!status) return status;
    RigidConstraintResource *resource = nullptr;
    status = impl_->validate_handle(id, resource);
    if (!status) return status;
    std::uint32_t dense_a = 0U;
    std::uint32_t dense_b = 0U;
    if (!(status = impl_->validate_handle(options.body_a, dense_a)) ||
        !(status = impl_->validate_handle(options.body_b, dense_b))) return status;
    if (impl_->parameters[dense_a].motion != MotionType::dynamic &&
        impl_->parameters[dense_b].motion != MotionType::dynamic)
        return failure(StatusCode::invalid_argument,
                       "rigid constraint needs at least one dynamic body");
    options.local_orientation_a = normalized_quaternion(
        options.local_orientation_a);
    options.local_orientation_b = normalized_quaternion(
        options.local_orientation_b);
    const bool contact_topology_changed = !(resource->options.body_a == options.body_a)
        || !(resource->options.body_b == options.body_b)
        || resource->options.enabled != options.enabled
        || resource->options.disable_collisions != options.disable_collisions
        || resource->options.type != options.type || resource->state.broken;
    const bool contact_cache_current = impl_->rigid_contact_revision == impl_->revision;
    resource->options = options;
    resource->state = {};
    resource->state.enabled = options.enabled;
    std::fill_n(resource->avbd_rows, 18U, avbd::Row{});
    ++impl_->revision;
    if (!contact_topology_changed && contact_cache_current)
        impl_->rigid_contact_revision = impl_->revision;
    return success();
}

Status SimulationRuntime::remove_rigid_constraint(RigidConstraintId id) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    RigidConstraintResource *resource = nullptr;
    status = impl_->validate_handle(id, resource);
    if (!status) return status;
    resource->alive = false;
    resource->state = {};
    ++resource->generation;
    if (resource->generation == 0U) resource->generation = 1U;
    --impl_->rigid_constraint_count;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::read_rigid_constraint_state(
    RigidConstraintId id, RigidConstraintState &output,
    device::ExecutionQueue queue) const noexcept {
    const cudaStream_t stream = native_stream(queue);
    if (!impl_)
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (impl_->frame && !impl_->frame->acknowledged) {
        status = wait_for_completion(impl_->frame);
        if (!status) return status;
    }
    if (id.index >= impl_->options.rigid_constraint_capacity ||
        impl_->rigid_constraints == nullptr) {
        return failure(StatusCode::invalid_handle,
                       "rigid constraint handle index is invalid");
    }
    RigidConstraintResource *resource = &impl_->rigid_constraints[id.index];
    RigidConstraintState temporary{};
    std::uint32_t generation = 0U;
    bool alive = false;
    cudaError_t error = cudaMemcpyAsync(
        &temporary, &resource->state, sizeof(temporary),
        cudaMemcpyDeviceToHost, stream);
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(&generation, &resource->generation,
                                sizeof(generation), cudaMemcpyDeviceToHost,
                                stream);
    }
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(&alive, &resource->alive, sizeof(alive),
                                cudaMemcpyDeviceToHost, stream);
    }
    const cudaError_t synchronization_error = cudaStreamSynchronize(stream);
    if (error == cudaSuccess) error = synchronization_error;
    if (error != cudaSuccess)
        return cuda_failure(error, "rigid constraint state readback failed");
    if (!alive || generation != id.generation) {
        return failure(StatusCode::invalid_handle,
                       "rigid constraint handle is stale");
    }
    output = temporary;
    return success();
}

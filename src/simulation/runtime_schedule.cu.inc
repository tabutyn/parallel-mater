// SPDX-License-Identifier: MIT
// Included by runtime.cpp; shared backend-neutral runtime implementation.

Status SimulationRuntime::step_async(StepOptions options, FrameToken &completion,
                         device::ExecutionQueue queue) noexcept {
    const device::ExecutionQueue stream = queue;
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (completion.pending()) {
        return failure(StatusCode::busy,
                       "completion token already represents a pending frame");
    }
    const bool realtime = options.profile ==
        PhysicsStepProfile::realtime_single_step_v1;
    const bool realtime_step = simulation::rigid_realtime_step_request_valid(
        options.timestep, options.substeps);
    if ((options.profile != PhysicsStepProfile::legacy_default && !realtime) ||
        (realtime && !realtime_step) ||
        !finite(options.timestep) || options.timestep <= 0.0F ||
        options.substeps == 0U || options.substeps > 1'024U ||
        !finite(options.gravity) || options.rigid_contact_pass_limit > 64U ||
        options.rigid_contact_iterations == 0U ||
        options.rigid_contact_iterations > 64U) {
        return failure(StatusCode::invalid_argument,
                       "step timestep, substeps, gravity, or contact pass limit is invalid");
    }
    // Legacy ropes may request smaller shared steps so rigid attachments stay
    // on the same clock. Realtime v1 ignores that legacy knob: its contract is
    // always one 1/60 advance and eight fixed constraint sweeps.
    if (!realtime)
        for (const auto &rope : impl_->ropes) if (rope && rope->alive) {
            const float required = std::ceil(options.timestep /
                rope->data.options.maximum_substep_timestep);
            if (!finite(required) || required > 1'024.0F)
                return failure(StatusCode::invalid_argument,
                    "rope timestep limit requires more than 1024 substeps");
            options.substeps = std::max(options.substeps,
                static_cast<std::uint32_t>(required));
        }
    if (!realtime)
        for (auto &cloth : impl_->cloths) if (cloth && cloth->alive) {
            status = rebuild_cloth_topology(*cloth, queue);
            if (!status) return status;
        }
    const bool debug_enabled =
        impl_->options.physics_debug.frame_capacity != 0U;
    if (!realtime && impl_->rigid_body_count == 0U) {
        const auto cleared = clear_bytes(impl_->rigid_contact_count, 0,
                                         sizeof(std::uint32_t), queue);
        if (cleared != PM_SUCCESS) return backend_failure(cleared, "contact count clear failed");
    }
    if (!realtime)
        for (auto *counter : {impl_->fluid_neighbor_overflow, impl_->fluid_maximum_neighbor_count,
                              impl_->fluid_contact_count, impl_->fluid_contact_overflow}) {
            const auto cleared = clear_bytes(counter, 0, sizeof(std::uint32_t), queue);
            if (cleared != PM_SUCCESS) return backend_failure(cleared, "fluid count clear failed");
        }

    std::unique_ptr<FrameToken::Impl> token_impl;
    if (completion.impl_) {
        token_impl = std::move(completion.impl_);
    } else {
        try {
            token_impl = std::make_unique<FrameToken::Impl>();
        } catch (...) {
            return failure(StatusCode::out_of_memory,
                           "failed to allocate completion token state");
        }
    }

    std::shared_ptr<CompletionState> frame;
    try {
        frame = std::make_shared<CompletionState>();
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to allocate frame completion state");
    }
    frame->fluid_neighbor_overflow = impl_->fluid_neighbor_overflow;
    frame->execution = impl_->execution;
    if (debug_enabled || realtime) {
        try {
            frame->on_complete = [implementation = impl_.get(), options,
                                  queue, debug_enabled, realtime]() {
                if (realtime) {
                    auto status =
                        implementation->finalize_realtime_lifecycle();
                    if (!status) return status;
                }
                return debug_enabled
                    ? implementation->record_debug_frame(options, queue)
                    : success();
            };
        } catch (...) {
            return failure(StatusCode::out_of_memory,
                           "physics debug completion allocation failed");
        }
    }
    PmResult error = impl_->execution.services->create_event(false, frame->event);
    if (error != PM_SUCCESS) {
        return backend_failure(error, "failed to create frame completion event");
    }
    NativeScheduleBackend backend(queue);
    status = realtime
        ? schedule_native_realtime(*impl_, backend, options)
        : simulation::schedule_frame(*impl_, backend, options);
    if (!status) {
        if (!realtime) (void)wait_queue(stream);
        return status;
    }
    error = impl_->execution.services->record_event(frame->event);
    if (error != PM_SUCCESS) {
        wait_queue(stream);
        return backend_failure(error, "failed to record frame completion event");
    }

    impl_->frame = frame;
    token_impl->completion = std::move(frame);
    completion.impl_ = std::move(token_impl);
    return success();
}
Status SimulationRuntime::step(
    StepOptions options, device::ExecutionQueue queue) noexcept {
    FrameToken completion;
    Status status = step_async(options, completion, queue);
    if (!status) {
        return status;
    }
    return completion.wait();
}

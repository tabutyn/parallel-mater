// SPDX-License-Identifier: MIT
// Included by runtime.cpp; shared backend-neutral runtime implementation.

Status SimulationRuntime::step_async(StepOptions options, FrameToken &completion,
                         device::ExecutionQueue queue) noexcept {
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
    const bool realtime_step = simulation::rigid_realtime_step_request_valid(
        options.timestep, options.substeps);
    if (!realtime_step ||
        !finite(options.gravity) || options.rigid_contact_pass_limit > 64U ||
        options.rigid_contact_iterations == 0U ||
        options.rigid_contact_iterations > 64U) {
        return failure(StatusCode::invalid_argument,
                       "step timestep, substeps, gravity, or contact pass limit is invalid");
    }
    const bool debug_enabled =
        impl_->options.physics_debug.frame_capacity != 0U;

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
    {
        try {
            frame->on_complete = [implementation = impl_.get(), options,
                                  queue, debug_enabled]() {
                auto status = implementation->finalize_realtime_lifecycle();
                if (!status) return status;
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
    status = schedule_native_realtime(*impl_, backend, options);
    if (!status) {
        return status;
    }
    error = impl_->execution.services->record_event(frame->event);
    if (error != PM_SUCCESS) {
        (void)wait_queue(queue);
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

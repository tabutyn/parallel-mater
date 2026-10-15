// SPDX-License-Identifier: MIT
// Included by runtime.cu; kept in one CUDA translation unit.

Status SimulationRuntime::step_async(StepOptions options, FrameToken &completion,
                         device::ExecutionQueue queue) noexcept {
    const cudaStream_t stream = native_stream(queue);
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
    if (!finite(options.timestep) || options.timestep <= 0.0F ||
        options.substeps == 0U || options.substeps > 1'024U ||
        !finite(options.gravity) || options.rigid_contact_pass_limit > 64U ||
        options.rigid_contact_iterations == 0U ||
        options.rigid_contact_iterations > 64U) {
        return failure(StatusCode::invalid_argument,
                       "step timestep, substeps, gravity, or contact pass limit is invalid");
    }
    // Advance rigid attachments and ropes on the same clock. Splitting only
    // the rope after a coarse rigid step leaves endpoints discontinuous.
    for (const auto &rope : impl_->ropes) if (rope && rope->alive) {
        const float required = std::ceil(options.timestep /
            rope->data.options.maximum_substep_timestep);
        if (!finite(required) || required > 1'024.0F)
            return failure(StatusCode::invalid_argument,
                           "rope timestep limit requires more than 1024 substeps");
        options.substeps = std::max(options.substeps,
                                   static_cast<std::uint32_t>(required));
    }
    for (auto &cloth : impl_->cloths) if (cloth && cloth->alive) {
        status = rebuild_cloth_topology(*cloth);
        if (!status) return status;
    }
    const bool debug_enabled =
        impl_->options.physics_debug.frame_capacity != 0U;
    if (impl_->rigid_body_count == 0U) {
        *impl_->rigid_contact_count = 0U;
    }
    *impl_->fluid_neighbor_overflow = 0U;
    *impl_->fluid_maximum_neighbor_count = 0U;
    *impl_->fluid_contact_count = 0U;
    *impl_->fluid_contact_overflow = 0U;

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
    if (debug_enabled) {
        try {
            frame->on_complete = [implementation = impl_.get(), options]() {
                return implementation->record_debug_frame(options);
            };
        } catch (...) {
            return failure(StatusCode::out_of_memory,
                           "physics debug completion allocation failed");
        }
    }
    cudaError_t error =
        cudaEventCreateWithFlags(&frame->event, cudaEventDisableTiming);
    if (error != cudaSuccess) {
        return cuda_failure(error, "failed to create frame completion event");
    }
    NativeScheduleBackend backend(queue);
    status = simulation::schedule_frame(*impl_, backend, options);
    if (!status) {
        (void)cudaStreamSynchronize(stream);
        return status;
    }
    error = cudaEventRecord(frame->event, stream);
    if (error != cudaSuccess) {
        cudaStreamSynchronize(stream);
        return cuda_failure(error, "failed to record frame completion event");
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

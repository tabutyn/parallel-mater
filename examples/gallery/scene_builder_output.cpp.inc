// Public-command stepping, readback, and recorded scene updates.
[[nodiscard]] PmBufferRange range(PmBuffer buffer,
                                  auto region) noexcept {
    return {buffer, region.offset, region.size};
}

template <typename T>
[[nodiscard]] bool read_domain(const std::byte *base,
                               auto region,
                               std::vector<T> &records,
                               std::uint64_t &revision,
                               std::string &error) {
    if (region.size < sizeof(PmDomainOutputHeader)) {
        records.clear();
        return true;
    }
    const auto *header = reinterpret_cast<const PmDomainOutputHeader *>(
        base + region.offset);
    const PmDeviceSize needed = sizeof(*header) +
        static_cast<PmDeviceSize>(header->activeCount) * sizeof(T);
    if (header->activeCount > header->capacity || needed > region.size) {
        error = "simulation returned an invalid output header";
        return false;
    }
    try {
        const auto *begin = reinterpret_cast<const T *>(
            base + region.offset + sizeof(*header));
        records.assign(begin, begin + header->activeCount);
    } catch (const std::bad_alloc &) {
        error = "read simulation output: out of host memory";
        return false;
    }
    revision = std::max(revision, header->revision);
    return true;
}

} // namespace

bool CommandScene::step(float timestep, std::uint32_t substeps, Vec3 gravity,
                        CommandSceneFrame &frame, std::string &error,
                        bool collect_contacts, bool readback) {
    error.clear();
    if (impl_ == nullptr || !(timestep > 0.0F) || substeps == 0U) {
        error = "step simulation: invalid argument";
        return false;
    }
    impl_->output_ready = false;
    if (readback) frame.previous_rigid_states = frame.rigid_states;
    if (!impl_->begin(error)) return false;
    if (!checked(pmCmdResetQueryPool(impl_->commands, impl_->statistics_query,
                                     0U, 1U),
                 "reset statistics query", error) ||
        !checked(pmCmdBeginStatisticsQuery(impl_->commands,
                                           impl_->statistics_query, 0U),
                 "begin statistics query", error))
        return false;
    PmStepInfo step_info{};
    step_info.timestep = timestep;
    step_info.substeps = substeps;
    copy(step_info.gravity, gravity);
    step_info.flags = collect_contacts
        ? static_cast<PmFlags>(PM_STEP_COLLECT_CONTACT_EVENTS_BIT) : 0U;
    if (!checked(pmCmdStepSimulation(impl_->commands, impl_->simulation,
                                     &step_info),
                 "record simulation step", error) ||
        !checked(pmCmdEndStatisticsQuery(impl_->commands,
                                         impl_->statistics_query, 0U),
                 "end statistics query", error))
        return false;
    const auto copy_range = [&](auto function, Impl::Region region,
                                std::string_view operation) {
        const PmBufferRange destination = range(impl_->output.handle, region);
        return checked(function(&destination), operation, error);
    };
    if (impl_->rigid_count != 0U &&
        !copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyRigidStates(impl_->commands, impl_->simulation,
                    0U, impl_->rigid_count, destination);
            }, impl_->rigid_output, "copy rigid output"))
        return false;
    if (!impl_->fluid_capacities.empty() &&
        !copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyFluidParticles(impl_->commands,
                    impl_->simulation, 0U, destination);
            }, impl_->fluid_output, "copy fluid output"))
        return false;
    for (std::uint32_t index = 0U; index < impl_->cloth_outputs.size(); ++index)
    {
        if (!copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyClothState(impl_->commands, impl_->simulation,
                    index, destination);
            }, impl_->cloth_outputs[index], "copy cloth output"))
            return false;
        if (!copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyClothSurface(impl_->commands,
                    impl_->simulation, index, destination);
            }, impl_->cloth_surface_outputs[index], "copy cloth surface"))
            return false;
    }
    for (std::uint32_t index = 0U; index < impl_->rope_outputs.size(); ++index)
        if (!copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyRopeState(impl_->commands, impl_->simulation,
                    index, destination);
            }, impl_->rope_outputs[index], "copy rope output"))
            return false;
    for (std::uint32_t index = 0U;
         index < impl_->soft_body_outputs.size(); ++index)
    {
        if (!copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopySoftBodyState(impl_->commands,
                    impl_->simulation, index, destination);
            }, impl_->soft_body_outputs[index], "copy soft body output"))
            return false;
        if (!copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopySoftBodySurface(impl_->commands,
                    impl_->simulation, index, destination);
            }, impl_->soft_body_surface_outputs[index],
            "copy soft body surface"))
            return false;
    }
    if (!impl_->smoke_capacities.empty() &&
        !copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopySmokeState(impl_->commands, impl_->simulation,
                    0U, destination);
            }, impl_->smoke_output, "copy smoke output"))
        return false;
    for (std::uint32_t index = 0U; index < impl_->paint_outputs.size(); ++index)
        if (!copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyPaintField(impl_->commands, impl_->simulation,
                    index, destination);
            }, impl_->paint_outputs[index], "copy paint output"))
            return false;
    if (!impl_->constraints.empty() &&
        !copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyConstraintStates(impl_->commands,
                    impl_->simulation, 0U,
                    static_cast<std::uint32_t>(impl_->constraints.size()),
                    destination);
            }, impl_->constraint_output, "copy constraint output"))
        return false;
    if (collect_contacts &&
        !copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyContactEvents(impl_->commands,
                                              impl_->simulation, destination);
            }, impl_->contact_output, "copy contact events"))
        return false;
    if (impl_->debug_output.size != 0U &&
        !copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyDebugData(impl_->commands, impl_->simulation,
                                          destination);
            }, impl_->debug_output, "copy debug output"))
        return false;
    if (!impl_->submit(error)) return false;
    impl_->output_ready = true;

    // GPU rendering consumes this public output buffer directly. Readback is
    // retained only for explicit validation/capture and CPU scene controllers.
    if (!readback) return true;
    return readback_frame(frame, error, collect_contacts);
}

bool CommandScene::readback_frame(CommandSceneFrame &frame, std::string &error, bool collect_contacts) const {
    error.clear();
    if (!impl_ || !impl_->output_ready) { error = "No completed frame to read"; return false; }

    const auto *base = static_cast<const std::byte *>(impl_->output.mapped);
    try {
        const auto *rigid = reinterpret_cast<const PmRigidState *>(
            base + impl_->rigid_output.offset);
        frame.rigid_states.assign(rigid, rigid + impl_->rigid_count);
        if (frame.previous_rigid_states.size() != frame.rigid_states.size())
            frame.previous_rigid_states = frame.rigid_states;
        frame.cloth_vertices.resize(impl_->cloth_outputs.size());
        frame.cloth_surface_vertices.resize(
            impl_->cloth_surface_outputs.size());
        frame.rope_nodes.resize(impl_->rope_outputs.size());
        frame.soft_body_nodes.resize(impl_->soft_body_outputs.size());
        frame.soft_body_surface_vertices.resize(
            impl_->soft_body_surface_outputs.size());
        frame.paint_fields.resize(impl_->paint_outputs.size());
    } catch (const std::bad_alloc &) {
        error = "read rigid simulation output: out of host memory";
        return false;
    }
    frame.revision = 0U;
    if (!impl_->fluid_capacities.empty() &&
        !read_domain(base, impl_->fluid_output, frame.fluid_particles,
                     frame.revision, error))
        return false;
    for (std::size_t index = 0U; index < impl_->cloth_outputs.size(); ++index)
    {
        if (!read_domain(base, impl_->cloth_outputs[index],
                         frame.cloth_vertices[index], frame.revision, error))
            return false;
        if (!read_domain(base, impl_->cloth_surface_outputs[index],
                         frame.cloth_surface_vertices[index],
                         frame.revision, error))
            return false;
    }
    for (std::size_t index = 0U; index < impl_->rope_outputs.size(); ++index)
        if (!read_domain(base, impl_->rope_outputs[index],
                         frame.rope_nodes[index], frame.revision, error))
            return false;
    for (std::size_t index = 0U;
         index < impl_->soft_body_outputs.size(); ++index)
    {
        if (!read_domain(base, impl_->soft_body_outputs[index],
                         frame.soft_body_nodes[index], frame.revision, error))
            return false;
        if (!read_domain(base, impl_->soft_body_surface_outputs[index],
                         frame.soft_body_surface_vertices[index],
                         frame.revision, error))
            return false;
    }
    if (!impl_->smoke_capacities.empty() &&
        !read_domain(base, impl_->smoke_output, frame.smoke_particles,
                     frame.revision, error))
        return false;
    for (std::size_t index = 0U; index < impl_->paint_outputs.size(); ++index)
        if (!read_domain(base, impl_->paint_outputs[index],
                         frame.paint_fields[index], frame.revision, error))
            return false;
    try {
        const auto *constraints = reinterpret_cast<const PmConstraintState *>(
            base + impl_->constraint_output.offset);
        frame.constraint_states.assign(
            constraints, constraints + impl_->constraints.size());
    } catch (const std::bad_alloc &) {
        error = "read constraint output: out of host memory";
        return false;
    }
    if (collect_contacts) {
        if (!read_domain(base, impl_->contact_output, frame.contacts,
                         frame.revision, error))
            return false;
    } else {
        frame.contacts.clear();
    }
    if (impl_->debug_output.size != 0U &&
        !read_domain(base, impl_->debug_output, frame.debug_records,
                     frame.revision, error))
        return false;
    const PmResult query_result = pmGetQueryPoolResults(
        impl_->device, impl_->statistics_query, 0U, 1U,
        sizeof(frame.statistics), &frame.statistics,
        sizeof(frame.statistics), PM_QUERY_RESULT_WAIT_BIT);
    return checked(query_result, "read statistics query", error);
}

bool CommandScene::gpu_frame(CommandSceneGpuFrame &frame, std::string &error) const {
    error.clear();
    frame = {};
    if (!impl_ || !impl_->output_ready) { error = "GPU frame: no completed output"; return false; }
    if (!checked(pmGetNativeBuffer(impl_->device, impl_->output.handle, &frame.buffer),
        "export public render buffer", error)) return false;
    const auto region = [](auto value) { return CommandSceneGpuFrame::Region{value.offset, value.size}; };
    frame.rigid = region(impl_->rigid_output);
    frame.fluid = region(impl_->fluid_output);
    frame.smoke = region(impl_->smoke_output);
    for (auto value : impl_->cloth_surface_outputs) frame.cloth_surfaces.push_back(region(value));
    for (auto value : impl_->soft_body_surface_outputs) frame.soft_surfaces.push_back(region(value));
    for (auto value : impl_->rope_outputs) frame.ropes.push_back(region(value));
    for (auto value : impl_->paint_outputs) frame.paint.push_back(region(value));
    return true;
}

bool CommandScene::set_rigid_state(std::uint32_t body,
                                   const RigidBodyState &state,
                                   std::string &error) {
    if (impl_ == nullptr || body >= impl_->rigid_count) {
        error = "set rigid state: invalid argument";
        return false;
    }
    const PmRigidState converted = pm(state);
    return impl_->begin(error) &&
        checked(pmCmdSetRigidStates(impl_->commands, impl_->simulation,
                                    body, 1U, &converted),
                "record rigid state", error) && impl_->submit(error);
}

bool CommandScene::set_kinematic_target(std::uint32_t body,
                                        const RigidBodyState &state,
                                        std::string &error) {
    if (impl_ == nullptr || body >= impl_->rigid_count) {
        error = "set kinematic target: invalid argument";
        return false;
    }
    const PmRigidState converted = pm(state);
    return impl_->begin(error) &&
        checked(pmCmdSetKinematicTargets(impl_->commands, impl_->simulation,
                                         body, 1U, &converted),
                "record kinematic target", error) && impl_->submit(error);
}

bool CommandScene::apply_rigid_inputs(const std::vector<PmRigidInput> &inputs,
                                      std::string &error) {
    if (impl_ == nullptr || inputs.empty()) return inputs.empty();
    return impl_->begin(error) &&
        checked(pmCmdApplyRigidInputs(impl_->commands, impl_->simulation,
                                      static_cast<std::uint32_t>(inputs.size()),
                                      inputs.data()),
                "record rigid inputs", error) && impl_->submit(error);
}

bool CommandScene::write_constraint(
    std::uint32_t constraint, const RigidConstraintDefinition &definition,
    std::string &error) {
    if (impl_ == nullptr || constraint >= impl_->constraints.size()) {
        error = "write rigid constraint: invalid argument";
        return false;
    }
    const PmRigidConstraintInfo converted = pm(definition);
    if (!impl_->begin(error) ||
        !checked(pmCmdWriteRigidConstraints(impl_->commands, impl_->simulation,
                                            constraint, 1U, &converted),
                 "record rigid constraint", error) ||
        !impl_->submit(error))
        return false;
    impl_->constraints[constraint] = converted;
    return true;
}

bool CommandScene::set_constraint_active(std::uint32_t constraint,
                                         bool active, std::string &error) {
    if (impl_ == nullptr || constraint >= impl_->constraints.size()) {
        error = "set rigid constraint active: invalid argument";
        return false;
    }
    PmRigidConstraintInfo converted = impl_->constraints[constraint];
    converted.active = active ? PM_TRUE : PM_FALSE;
    if (!impl_->begin(error) ||
        !checked(pmCmdWriteRigidConstraints(impl_->commands, impl_->simulation,
                                            constraint, 1U, &converted),
                 "record rigid constraint state", error) ||
        !impl_->submit(error))
        return false;
    impl_->constraints[constraint] = converted;
    return true;
}

bool CommandScene::reset(std::string &error) {
    if (impl_ == nullptr) {
        error = "reset simulation: invalid state";
        return false;
    }
    CommandScene replacement;
    CommandSceneInstance bindings;
    if (!create(impl_->original_scene, impl_->debug_frame_capacity,
                replacement, bindings, error))
        return false;
    impl_ = std::move(replacement.impl_);
    return true;
}

PmDevice CommandScene::device() const noexcept {
    return impl_ != nullptr ? impl_->device : nullptr;
}

PmPhysicalDevice CommandScene::physical_device() const noexcept {
    return impl_ != nullptr ? impl_->physical : nullptr;
}

PmSimulation CommandScene::simulation() const noexcept {
    return impl_ != nullptr ? impl_->simulation : nullptr;
}

PmQueue CommandScene::queue() const noexcept {
    return impl_ != nullptr ? impl_->queue : nullptr;
}

} // namespace parallel_mater::gallery

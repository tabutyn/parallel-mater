// Public-command stepping, readback, and recorded scene updates.
[[nodiscard]] PmBufferRange range(PmBuffer buffer,
                                  auto region) noexcept {
    return {buffer, region.offset, region.size};
}

struct OutputMapping {
    PmDevice device{};
    PmDeviceMemory memory{};
    void *data{};
    ~OutputMapping() { if (data) pmUnmapMemory(device, memory); }
    bool open(PmDevice device_value, PmDeviceMemory memory_value,
              PmDeviceSize offset, PmDeviceSize size, std::string &error) {
        device = device_value; memory = memory_value;
        return checked(pmMapMemory(device, memory, offset, size, &data),
                       "map live simulation output", error);
    }
};

template <typename T>
[[nodiscard]] bool read_records(PmDevice device, PmDeviceMemory memory,
    PmDeviceSize offset, std::size_t count, std::vector<T> &records,
    std::string &error) {
    records.clear();
    if (count == 0U) return true;
    OutputMapping mapping;
    if (!mapping.open(device, memory, offset, count * sizeof(T), error)) return false;
    try {
        const auto *begin = static_cast<const T *>(mapping.data);
        records.assign(begin, begin + count);
    } catch (const std::bad_alloc &) {
        error = "read simulation output: out of host memory";
        return false;
    }
    return true;
}

template <typename T>
[[nodiscard]] bool read_domain(PmDevice device, PmDeviceMemory memory,
    auto region, std::vector<T> &records, std::uint64_t &revision, std::string &error) {
    if (region.size < sizeof(PmDomainOutputHeader)) { records.clear(); return true; }
    PmDomainOutputHeader header{};
    {
        OutputMapping mapping;
        if (!mapping.open(device, memory, region.offset, sizeof(header), error)) return false;
        std::memcpy(&header, mapping.data, sizeof(header));
    }
    const PmDeviceSize needed = sizeof(header) + static_cast<PmDeviceSize>(header.activeCount) * sizeof(T);
    if (header.activeCount > header.capacity || needed > region.size) {
        error = "simulation returned an invalid output header";
        return false;
    }
    revision = std::max(revision, header.revision);
    return read_records(device, memory, region.offset + sizeof(header), header.activeCount, records, error);
}

} // namespace

bool CommandScene::step(float timestep, std::uint32_t substeps, Vec3 gravity,
                        CommandSceneFrame &frame, std::string &error,
                        bool collect_contacts, bool read_deformables, bool read_host) {
    error.clear();
    if (impl_ == nullptr || !(timestep > 0.0F) || substeps == 0U) {
        error = "step simulation: invalid argument";
        return false;
    }
    impl_->output_ready = false;
    if (read_host) frame.previous_rigid_states = frame.rigid_states;
    if (!impl_->begin(error)) return false;
    if (impl_->rigid_count != 0U) {
        const PmBufferCopy snapshot{impl_->rigid_output.offset,
            impl_->previous_rigid_output.offset, impl_->rigid_output.size};
        if (!checked(pmCmdCopyBuffer(impl_->commands, impl_->output.handle,
                impl_->output.handle, 1U, &snapshot), "snapshot GPU rigid output", error)) return false;
    }
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

    // Statistics are part of every completed step, including native GPU
    // rendering where the scene output deliberately stays on the device.
    // Consume the query before the next command buffer tries to reset it.
    const PmResult query_result = pmGetQueryPoolResults(
        impl_->device, impl_->statistics_query, 0U, 1U,
        sizeof(frame.statistics), &frame.statistics,
        sizeof(frame.statistics), PM_QUERY_RESULT_WAIT_BIT);
    if (!checked(query_result, "read statistics query", error)) return false;

    // Native rendering borrows completed public exports. Host reads are explicit.
    impl_->contacts_available = collect_contacts;
    frame.full_host_output = false;
    if (!read_host) return true;
    return read_output(frame, error, read_deformables);
}

bool CommandScene::read_output(CommandSceneFrame &frame, std::string &error,
                               bool read_deformables) {
    error.clear();
    if (!impl_ || !impl_->output_ready) {
        error = "read simulation output: no completed frame";
        return false;
    }
    const auto read = [&](auto region, auto &records) {
        return read_domain(impl_->device, impl_->output.memory, region, records, frame.revision, error);
    };
    if (!read_records(impl_->device, impl_->output.memory, impl_->rigid_output.offset,
            impl_->rigid_count, frame.rigid_states, error) ||
        !read_records(impl_->device, impl_->output.memory, impl_->constraint_output.offset,
            impl_->constraints.size(), frame.constraint_states, error)) return false;
    if (frame.previous_rigid_states.size() != frame.rigid_states.size())
        frame.previous_rigid_states = frame.rigid_states;
    frame.full_host_output = false;
    if (read_deformables) {
        try {
            frame.cloth_vertices.resize(impl_->cloth_outputs.size());
            frame.cloth_surface_vertices.resize(impl_->cloth_surface_outputs.size());
            frame.rope_nodes.resize(impl_->rope_outputs.size());
            frame.soft_body_nodes.resize(impl_->soft_body_outputs.size());
            frame.soft_body_surface_vertices.resize(impl_->soft_body_surface_outputs.size());
            frame.paint_fields.resize(impl_->paint_outputs.size());
        } catch (const std::bad_alloc &) {
            error = "read simulation output: out of host memory"; return false;
        }
        if (!impl_->fluid_capacities.empty() && !read(impl_->fluid_output, frame.fluid_particles)) return false;
        for (std::size_t index = 0; index < impl_->cloth_outputs.size(); ++index)
            if (!read(impl_->cloth_outputs[index], frame.cloth_vertices[index]) ||
                !read(impl_->cloth_surface_outputs[index], frame.cloth_surface_vertices[index])) return false;
        for (std::size_t index = 0; index < impl_->rope_outputs.size(); ++index)
            if (!read(impl_->rope_outputs[index], frame.rope_nodes[index])) return false;
        for (std::size_t index = 0; index < impl_->soft_body_outputs.size(); ++index)
            if (!read(impl_->soft_body_outputs[index], frame.soft_body_nodes[index]) ||
                !read(impl_->soft_body_surface_outputs[index], frame.soft_body_surface_vertices[index])) return false;
        if (!impl_->smoke_capacities.empty() && !read(impl_->smoke_output, frame.smoke_particles)) return false;
        for (std::size_t index = 0; index < impl_->paint_outputs.size(); ++index)
            if (!read(impl_->paint_outputs[index], frame.paint_fields[index])) return false;
        if (impl_->debug_output.size && !read(impl_->debug_output, frame.debug_records)) return false;
        frame.full_host_output = true;
    } else {
        frame.fluid_particles.clear(); frame.cloth_vertices.clear();
        frame.cloth_surface_vertices.clear(); frame.rope_nodes.clear();
        frame.soft_body_nodes.clear(); frame.soft_body_surface_vertices.clear();
        frame.smoke_particles.clear(); frame.paint_fields.clear(); frame.debug_records.clear();
    }
    if (impl_->contacts_available) {
        if (!read(impl_->contact_output, frame.contacts)) return false;
    } else frame.contacts.clear();
    frame.revision = std::max(frame.revision, frame.statistics.stepCount);
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
                replacement, bindings, error, impl_->selected_backend))
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

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
                        bool collect_contacts, bool read_deformables,
                        bool read_host, PmResult expected_result) {
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
    for (std::uint32_t first = 0U; first < impl_->rigid_count;) {
        while (first < impl_->rigid_count &&
               impl_->rigid_bodies[first].active == PM_FALSE)
            ++first;
        if (first == impl_->rigid_count) break;
        std::uint32_t count = 1U;
        while (first + count < impl_->rigid_count &&
               impl_->rigid_bodies[first + count].active == PM_TRUE)
            ++count;
        const PmBufferRange destination{
            impl_->output.handle,
            impl_->rigid_output.offset +
                static_cast<PmDeviceSize>(first) * sizeof(PmRigidState),
            static_cast<PmDeviceSize>(count) * sizeof(PmRigidState)};
        if (!checked(pmCmdCopyRigidStates(impl_->commands, impl_->simulation,
                                         first, count, &destination),
                     "copy rigid output", error))
            return false;
        first += count;
    }
    if (!impl_->fluid_capacities.empty() &&
        !copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyFluidParticles(impl_->commands,
                    impl_->simulation, 0U, destination);
            }, impl_->fluid_output, "copy fluid output"))
        return false;
    for (std::uint32_t index = 0U; index < impl_->cloth_outputs.size(); ++index)
    {
        if (impl_->cloth_active[index] == 0U) continue;
        if (!copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyClothState(impl_->commands, impl_->simulation,
                    index, destination);
            }, impl_->cloth_outputs[index], "copy cloth output"))
            return false;
        if (impl_->cloth_capacities[index].triangleIndexCapacity != 0U &&
            !copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyClothSurface(impl_->commands,
                    impl_->simulation, index, destination);
            }, impl_->cloth_surface_outputs[index], "copy cloth surface"))
            return false;
    }
    for (std::uint32_t index = 0U; index < impl_->rope_outputs.size(); ++index)
        if (impl_->rope_active[index] != 0U &&
            !copy_range([&](const PmBufferRange *destination) {
                return pmCmdCopyRopeState(impl_->commands, impl_->simulation,
                    index, destination);
            }, impl_->rope_outputs[index], "copy rope output"))
            return false;
    for (std::uint32_t index = 0U;
         index < impl_->soft_body_outputs.size(); ++index)
    {
        if (impl_->soft_body_active[index] == 0U) continue;
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
    if (!impl_->smoke_capacities.empty() && impl_->smoke_active &&
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
    if (!impl_->submit(error, expected_result)) return false;
    impl_->output_ready = true;

    // Statistics are part of every completed step, including native GPU
    // rendering where the scene output deliberately stays on the device.
    // Consume the query before the next command buffer tries to reset it.
    const PmResult query_result = pmGetQueryPoolResults(
        impl_->device, impl_->statistics_query, 0U, 1U,
        sizeof(frame.realtime_statistics), &frame.realtime_statistics,
        sizeof(frame.realtime_statistics), PM_QUERY_RESULT_WAIT_BIT);
    if (!checked(query_result, "read statistics query", error)) return false;
    frame.statistics = frame.realtime_statistics.step;

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
    std::vector<PmRigidState> rigid_slots;
    if (!read_records(impl_->device, impl_->output.memory, impl_->rigid_output.offset,
            impl_->rigid_count, rigid_slots, error) ||
        !read_records(impl_->device, impl_->output.memory, impl_->constraint_output.offset,
            impl_->constraints.size(), frame.constraint_states, error)) return false;
    frame.rigid_states.clear();
    try {
        frame.rigid_states.reserve(rigid_slots.size());
        for (std::size_t index = 0U; index < rigid_slots.size(); ++index)
            if (impl_->rigid_bodies[index].active == PM_TRUE)
                frame.rigid_states.push_back(rigid_slots[index]);
    } catch (const std::bad_alloc &) {
        error = "read simulation output: out of host memory";
        return false;
    }
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
        for (std::size_t index = 0; index < impl_->cloth_outputs.size(); ++index) {
            if (impl_->cloth_active[index] == 0U) {
                frame.cloth_vertices[index].clear();
                frame.cloth_surface_vertices[index].clear();
                continue;
            }
            if (!read(impl_->cloth_outputs[index], frame.cloth_vertices[index]))
                return false;
            if (impl_->cloth_capacities[index].triangleIndexCapacity == 0U)
                frame.cloth_surface_vertices[index].clear();
            else if (!read(impl_->cloth_surface_outputs[index],
                           frame.cloth_surface_vertices[index]))
                return false;
        }
        for (std::size_t index = 0; index < impl_->rope_outputs.size(); ++index)
            if (impl_->rope_active[index] == 0U)
                frame.rope_nodes[index].clear();
            else if (!read(impl_->rope_outputs[index], frame.rope_nodes[index]))
                return false;
        for (std::size_t index = 0; index < impl_->soft_body_outputs.size(); ++index) {
            if (impl_->soft_body_active[index] == 0U) {
                frame.soft_body_nodes[index].clear();
                frame.soft_body_surface_vertices[index].clear();
                continue;
            }
            if (!read(impl_->soft_body_outputs[index], frame.soft_body_nodes[index]) ||
                !read(impl_->soft_body_surface_outputs[index],
                      frame.soft_body_surface_vertices[index])) return false;
        }
        if (!impl_->smoke_capacities.empty()) {
            if (!impl_->smoke_active) frame.smoke_particles.clear();
            else if (!read(impl_->smoke_output, frame.smoke_particles)) return false;
        }
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

bool CommandScene::set_fluid_particles(
    const std::vector<FluidParticle> &particles, std::string &error) {
    if (impl_ == nullptr || impl_->fluid_capacities.empty() ||
        particles.size() > impl_->fluid_capacities[0].particleCapacity ||
        particles.size() * sizeof(PmParticleState) > impl_->particle_input.size) {
        error = "set fluid particles: invalid argument";
        return false;
    }
    std::vector<PmParticleState> converted(particles.size());
    for (std::size_t index = 0U; index < particles.size(); ++index) {
        copy(converted[index].position, particles[index].position);
        copy(converted[index].velocity, particles[index].velocity);
        converted[index].temperatureCelsius = particles[index].temperature;
    }
    if (!converted.empty())
        std::memcpy(impl_->particle_input.mapped, converted.data(),
                    converted.size() * sizeof(PmParticleState));
    const PmMappedMemoryRange mapped{
        impl_->particle_input.memory, 0U, impl_->particle_input.size};
    if (!checked(pmFlushMappedMemoryRanges(impl_->device, 1U, &mapped),
                 "flush fluid particles", error))
        return false;
    const PmBufferView view{{impl_->particle_input.handle, 0U,
                             impl_->particle_input.size},
                            PM_FORMAT_PARTICLE_STATE,
                            static_cast<std::uint32_t>(converted.size()),
                            sizeof(PmParticleState)};
    return impl_->begin(error) &&
        checked(pmCmdSetFluidParticles(impl_->commands, impl_->simulation, 0U,
                                       &view),
                "record fluid particles", error) && impl_->submit(error);
}

bool CommandScene::set_particle_emitters_active(bool active,
                                                std::string &error) {
    if (impl_ == nullptr || impl_->original_scene.particle_sources.empty()) {
        error = "set particle emitters active: invalid argument";
        return false;
    }
    std::vector<std::vector<PmFloat3>> vertices(
        impl_->original_scene.particle_sources.size());
    std::vector<PmParticleEmitterInfo> emitters(vertices.size());
    for (std::size_t index = 0U; index < emitters.size(); ++index) {
        const auto &source = impl_->original_scene.particle_sources[index];
        vertices[index].reserve(source.vertices.size());
        for (Vec3 vertex : source.vertices)
            vertices[index].push_back(pm(vertex));
        emitters[index] = {active ? PM_TRUE : PM_FALSE, 0U,
            static_cast<std::uint32_t>(vertices[index].size()),
            vertices[index].data(),
            static_cast<std::uint32_t>(source.indices.size()),
            source.indices.data(), source.spacing,
            pm(source.options.initial_velocity),
            source.options.initial_temperature};
    }
    return impl_->begin(error) &&
        checked(pmCmdWriteParticleEmitters(impl_->commands, impl_->simulation,
                0U, static_cast<std::uint32_t>(emitters.size()),
                emitters.data()),
            "record particle emitters", error) && impl_->submit(error);
}

bool CommandScene::set_rigid_active(std::uint32_t body, bool active,
                                    std::string &error) {
    if (impl_ == nullptr || body >= impl_->rigid_bodies.size()) {
        error = "set rigid active: invalid argument";
        return false;
    }
    PmRigidBodyInfo converted = impl_->rigid_bodies[body];
    converted.active = active ? PM_TRUE : PM_FALSE;
    if (!impl_->begin(error) ||
        !checked(pmCmdWriteRigidBodies(impl_->commands, impl_->simulation, body,
                                       1U, &converted),
                 "record rigid active state", error) ||
        !impl_->submit(error))
        return false;
    impl_->rigid_bodies[body] = converted;
    return true;
}

bool CommandScene::set_cloth_active(std::uint32_t cloth, bool active,
                                    std::string &error) {
    if (impl_ == nullptr || cloth >= impl_->original_scene.cloths.size()) {
        error = "set cloth active: invalid argument";
        return false;
    }
    const auto &definition = impl_->original_scene.cloths[cloth];
    const auto &mesh = impl_->original_scene.meshes[definition.mesh_index];
    std::vector<PmFloat3> vertices;
    vertices.reserve(mesh.vertices.size());
    for (const auto &vertex : mesh.vertices) vertices.push_back(pm(vertex.position));
    PmClothInfo converted{};
    converted.active = active ? PM_TRUE : PM_FALSE;
    converted.materialIndex = cloth;
    if (active) {
        converted.vertexCount = static_cast<std::uint32_t>(vertices.size());
        converted.pVertices = vertices.data();
        converted.triangleIndexCount =
            static_cast<std::uint32_t>(mesh.indices.size());
        converted.pTriangleIndices = mesh.indices.data();
        converted.pInverseMasses = definition.inverse_masses.data();
    }
    const bool submitted = impl_->begin(error) &&
        checked(pmCmdWriteCloths(impl_->commands, impl_->simulation, cloth, 1U,
                                 &converted),
                "record cloth active state", error) && impl_->submit(error);
    if (submitted) impl_->cloth_active[cloth] = active ? 1U : 0U;
    return submitted;
}

bool CommandScene::set_soft_body_active(std::uint32_t soft_body, bool active,
                                        std::string &error) {
    if (impl_ == nullptr ||
        soft_body >= impl_->original_scene.soft_bodies.size()) {
        error = "set soft body active: invalid argument";
        return false;
    }
    const auto &definition = impl_->original_scene.soft_bodies[soft_body];
    const auto &mesh = impl_->original_scene.meshes[definition.mesh_index];
    std::vector<PmFloat3> nodes, surface_vertices;
    std::vector<PmSoftBodyBond> bonds;
    std::vector<PmSoftBodySurfaceBinding> bindings;
    for (const Vec3 node : definition.nodes) nodes.push_back(pm(node));
    for (const auto &bond : definition.bonds)
        bonds.push_back({bond.first, bond.second, bond.rest_length});
    for (const auto &vertex : mesh.vertices)
        surface_vertices.push_back(pm(vertex.position));
    for (const auto &source : definition.surface_bindings) {
        PmSoftBodySurfaceBinding target{};
        std::copy_n(source.nodes, 4U, target.nodes);
        std::copy_n(source.weights, 4U, target.weights);
        bindings.push_back(target);
    }
    PmSoftBodyInfo converted{};
    converted.active = active ? PM_TRUE : PM_FALSE;
    converted.materialIndex = soft_body;
    if (active) {
        converted.nodeCount = static_cast<std::uint32_t>(nodes.size());
        converted.pNodes = nodes.data();
        converted.bondCount = static_cast<std::uint32_t>(bonds.size());
        converted.pBonds = bonds.data();
        converted.pInverseMasses = definition.inverse_masses.data();
        converted.surfaceVertexCount =
            static_cast<std::uint32_t>(surface_vertices.size());
        converted.pSurfaceVertices = surface_vertices.data();
        converted.surfaceTriangleIndexCount =
            static_cast<std::uint32_t>(mesh.indices.size());
        converted.pSurfaceTriangleIndices = mesh.indices.data();
        converted.pSurfaceBindings = bindings.data();
    }
    const bool submitted = impl_->begin(error) &&
        checked(pmCmdWriteSoftBodies(impl_->commands, impl_->simulation,
                                     soft_body, 1U, &converted),
                "record soft body active state", error) && impl_->submit(error);
    if (submitted) impl_->soft_body_active[soft_body] = active ? 1U : 0U;
    return submitted;
}

bool CommandScene::set_rope_active(std::uint32_t rope, bool active,
                                   std::string &error) {
    if (impl_ == nullptr || rope >= impl_->original_scene.ropes.size()) {
        error = "set rope active: invalid argument";
        return false;
    }
    if (active) {
        const bool submitted = write_rope(
            rope, impl_->original_scene.ropes[rope], error);
        if (submitted) impl_->rope_active[rope] = 1U;
        return submitted;
    }
    PmRopeInfo converted{};
    converted.materialIndex = rope;
    const bool submitted = impl_->begin(error) &&
        checked(pmCmdWriteRopes(impl_->commands, impl_->simulation, rope, 1U,
                                &converted),
                "record rope inactive state", error) && impl_->submit(error);
    if (submitted) impl_->rope_active[rope] = 0U;
    return submitted;
}

bool CommandScene::set_smoke_active(bool active, std::string &error) {
    if (impl_ == nullptr || !impl_->original_scene.has_smoke) {
        error = "set smoke active: invalid argument";
        return false;
    }
    const auto &source = impl_->original_scene.smoke_options;
    PmSmokeInfo converted{active ? PM_TRUE : PM_FALSE, 0U, source.capacity,
        pm(source.emitter_center),
        {source.emitter_half_extents.x, source.emitter_half_extents.y},
        pm(source.initial_velocity), pm(source.wind),
        source.particles_per_second, source.lifetime,
        pm(source.grid_minimum), source.grid_edge_length};
    const bool submitted = impl_->begin(error) &&
        checked(pmCmdWriteSmoke(impl_->commands, impl_->simulation, 0U, 1U,
                                &converted),
                "record smoke active state", error) && impl_->submit(error);
    if (submitted) impl_->smoke_active = active;
    return submitted;
}

bool CommandScene::set_smoke_emitter_active(bool active, std::string &error) {
    if (impl_ == nullptr || !impl_->original_scene.has_smoke || active) {
        error = "set smoke emitter active: invalid argument";
        return false;
    }
    const auto &source = impl_->original_scene.smoke_options;
    PmSmokeInfo converted{PM_TRUE, 0U, source.capacity,
        pm(source.emitter_center),
        {source.emitter_half_extents.x, source.emitter_half_extents.y},
        pm(source.initial_velocity), pm(source.wind), 0.0F, source.lifetime,
        pm(source.grid_minimum), source.grid_edge_length};
    return impl_->begin(error) &&
        checked(pmCmdWriteSmoke(impl_->commands, impl_->simulation, 0U, 1U,
                                &converted),
                "record smoke emitter active state", error) &&
        impl_->submit(error);
}

bool CommandScene::write_rope(std::uint32_t rope,
                              const RopeDefinition &definition,
                              std::string &error) {
    if (impl_ == nullptr || rope >= impl_->rope_capacities.size() ||
        definition.centerline.size() < 2U) {
        error = "write rope: invalid argument";
        return false;
    }
    std::vector<PmFloat3> points;
    points.reserve(definition.centerline.size());
    for (const Vec3 point : definition.centerline) points.push_back(pm(point));
    PmRopeInfo converted{};
    converted.active = PM_TRUE;
    converted.materialIndex = rope;
    converted.pointCount = static_cast<std::uint32_t>(points.size());
    converted.pCenterline = points.data();
    if (definition.first_body >= 0)
        converted.first = {static_cast<std::uint32_t>(definition.first_body),
            pm(definition.options.first.local_anchor), PM_TRUE};
    if (definition.last_body >= 0)
        converted.last = {static_cast<std::uint32_t>(definition.last_body),
            pm(definition.options.last.local_anchor), PM_TRUE};
    return impl_->begin(error) &&
        checked(pmCmdWriteRopes(impl_->commands, impl_->simulation, rope, 1U,
                                &converted),
                "record rope replacement", error) && impl_->submit(error);
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
    if (!impl_->submit_reset(error))
        return false;
    impl_->rigid_bodies = impl_->initial_rigid_bodies;
    impl_->constraints = impl_->initial_constraints;
    impl_->cloth_active.assign(impl_->cloth_active.size(), 1U);
    impl_->rope_active.assign(impl_->rope_active.size(), 1U);
    impl_->soft_body_active.assign(impl_->soft_body_active.size(), 1U);
    impl_->smoke_active = !impl_->smoke_capacities.empty();
    impl_->output_ready = false;
    impl_->contacts_available = false;
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

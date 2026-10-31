// Public-command resource creation and initial upload.
bool CommandScene::create(const SceneDefinition &scene,
                          std::uint32_t debug_frame_capacity,
                          CommandScene &output,
                          CommandSceneInstance &instance,
                          std::string &error,
                          std::optional<PmNativeBackend> backend,
                          PmResult *creation_category,
                          bool enable_validation) {
    if (creation_category) *creation_category = PM_ERROR_INVALID_STATE;
    error.clear();
    output = {};
    instance = {};
    auto next = std::make_unique<Impl>();

    PmInstanceCreateInfo instance_info{};
    instance_info.sType = PM_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.apiVersion = PM_API_VERSION_0_3;
    instance_info.pApplicationName = "parallel-mater-gallery";
    instance_info.enableValidation = enable_validation ? PM_TRUE : PM_FALSE;
    if (!checked(pmCreateInstance(&instance_info, nullptr, &next->instance),
                 "create Parallel Mater instance", error))
        return false;
    std::uint32_t physical_count = 0U;
    if (!checked(pmEnumeratePhysicalDevices(next->instance, &physical_count,
                                            nullptr),
                 "enumerate physical devices", error) ||
        physical_count == 0U) {
        if (error.empty()) error = "no Parallel Mater physical device";
        return false;
    }
    std::vector<PmPhysicalDevice> physical_devices(physical_count);
    if (!checked(pmEnumeratePhysicalDevices(next->instance, &physical_count,
                                            physical_devices.data()),
                 "enumerate physical devices", error))
        return false;
    PmPhysicalDeviceFeatures features{};
    const bool requires_fluid = scene.fluid_options.capacity != 0U;
    for (PmPhysicalDevice candidate : physical_devices) {
        if (backend) {
            PmNativeInteropProperties interop{};
            if (pmGetPhysicalDeviceInteropProperties(
                    candidate, *backend, &interop) != PM_SUCCESS)
                continue;
        }
        PmPhysicalDeviceFeatures candidate_features{};
        pmGetPhysicalDeviceFeatures(candidate, &candidate_features);
        const bool compatible = candidate_features.rigidBodies == PM_TRUE &&
            (scene.rigid_constraints.empty() ||
             candidate_features.rigidConstraints == PM_TRUE) &&
            (!requires_fluid ||
             candidate_features.particleLiquids == PM_TRUE) &&
            (scene.cloths.empty() || candidate_features.cloth == PM_TRUE) &&
            (scene.ropes.empty() || candidate_features.rope == PM_TRUE) &&
            (scene.soft_bodies.empty() ||
             candidate_features.softBodies == PM_TRUE) &&
            (!scene.has_smoke || candidate_features.smoke == PM_TRUE);
        if (!compatible) continue;
        next->physical = candidate;
        features = candidate_features;
        break;
    }
    if (next->physical == nullptr) {
        if (creation_category) *creation_category = PM_ERROR_FEATURE_NOT_PRESENT;
        error = "selected device does not expose every scene capability";
        return false;
    }
    float priority = 1.0F;
    const PmDeviceQueueCreateInfo queue_info{0U, 1U, &priority};
    PmDeviceCreateInfo device_info{};
    device_info.sType = PM_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueInfos = &queue_info;
    device_info.enabledFeatures = features;
    if (!checked(pmCreateDevice(next->physical, &device_info, &next->device),
                 "create Parallel Mater device", error))
        return false;
    pmGetDeviceQueue(next->device, 0U, 0U, &next->queue);
    if (next->queue == nullptr) {
        error = "get Parallel Mater physics queue: invalid state";
        return false;
    }
    PmCommandPoolCreateInfo pool_info{};
    pool_info.sType = PM_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = PM_COMMAND_POOL_RESET_BUFFER_BIT;
    if (!checked(pmCreateCommandPool(next->device, &pool_info,
                                     &next->command_pool),
                 "create command pool", error) ||
        !checked(pmAllocateCommandBuffers(next->device, next->command_pool, 1U,
                                          &next->commands),
                 "allocate command buffer", error) ||
        !checked(pmAllocateCommandBuffers(next->device, next->command_pool, 1U,
                                          &next->reset_commands),
                 "allocate reset command buffer", error) ||
        !checked(pmCreateFence(next->device, 0U, &next->fence),
                 "create queue fence", error))
        return false;

    std::vector<std::uint32_t> body_geometry(scene.rigid_bodies.size());
    try {
        next->geometry_cache.reserve(scene.rigid_bodies.size() + 16U);
        for (std::size_t body_index = 0U;
             body_index < scene.rigid_bodies.size(); ++body_index) {
            const auto &body = scene.rigid_bodies[body_index];
            const bool collision = !body.collision_mesh_indices.empty();
            const auto &mesh_indices = collision ? body.collision_mesh_indices
                                                 : body.mesh_indices;
            const std::string key = mesh_key(collision, mesh_indices);
            const auto cached = next->geometry_cache.find(key);
            if (cached != next->geometry_cache.end()) {
                body_geometry[body_index] = cached->second;
                continue;
            }
            std::uint32_t geometry_index = 0U;
            if (!next->create_geometry(
                    collision ? scene.collision_meshes : scene.meshes,
                    mesh_indices, geometry_index, error))
                return false;
            next->geometry_cache.emplace(key, geometry_index);
            body_geometry[body_index] = geometry_index;
        }
    } catch (const std::bad_alloc &) {
        error = "build scene geometry cache: out of host memory";
        return false;
    }

    struct PaintDefinition {
        std::uint32_t body{PM_INVALID_INDEX};
        std::uint32_t cloth{PM_INVALID_INDEX};
        std::uint32_t mesh{};
        std::uint32_t geometry{PM_INVALID_INDEX};
        std::uint32_t resolution{};
        std::uint32_t rigid_source{PM_INVALID_INDEX};
        float brush_radius{};
        bool liquid_source{};
    };
    std::vector<PaintDefinition> paints;
    try {
        for (std::uint32_t body_index = 0U;
             body_index < scene.rigid_bodies.size(); ++body_index) {
            const auto &body = scene.rigid_bodies[body_index];
            if (!body.paintable) continue;
            for (std::uint32_t mesh_index : body.mesh_indices) {
                const std::string key = "paint:" + std::to_string(mesh_index);
                auto found = next->geometry_cache.find(key);
                if (found == next->geometry_cache.end()) {
                    std::uint32_t geometry_index = 0U;
                    if (!next->create_geometry(scene.meshes, {mesh_index},
                                               geometry_index, error))
                        return false;
                    found = next->geometry_cache.emplace(key, geometry_index).first;
                }
                paints.push_back({body_index, PM_INVALID_INDEX, mesh_index,
                    found->second, body.paint_resolution, PM_INVALID_INDEX,
                    0.15F, true});
            }
        }
        for (std::uint32_t cloth_index = 0U;
             cloth_index < scene.cloths.size(); ++cloth_index) {
            const auto &cloth = scene.cloths[cloth_index];
            if (!cloth.paintable) continue;
            const auto source = std::find_if(
                scene.rigid_bodies.begin(), scene.rigid_bodies.end(),
                [&](const RigidBodyDefinition &body) {
                    return body.source_name == cloth.paint_source;
                });
            if (source == scene.rigid_bodies.end()) {
                error = "cloth paint source is missing";
                return false;
            }
            paints.push_back({PM_INVALID_INDEX, cloth_index, cloth.mesh_index,
                PM_INVALID_INDEX, cloth.paint_resolution,
                static_cast<std::uint32_t>(source - scene.rigid_bodies.begin()),
                cloth.paint_brush_radius, false});
        }
    } catch (const std::bad_alloc &) {
        error = "build paint bindings: out of host memory";
        return false;
    }

    if (requires_fluid) {
        next->fluid_capacities.push_back({scene.fluid_options.capacity,
                                          scene.fluid_options.maximum_neighbors});
    }
    for (const auto &cloth : scene.cloths) {
        if (cloth.mesh_index >= scene.meshes.size()) {
            error = "cloth references an invalid mesh";
            return false;
        }
        const auto &mesh = scene.meshes[cloth.mesh_index];
        next->cloth_capacities.push_back({
            static_cast<std::uint32_t>(
                mesh.vertices.size() + ((cloth.break_strain > 0 ||
                    cloth.impact_break_impulse > 0) ? mesh.indices.size() : 0U)),
            static_cast<std::uint32_t>(mesh.indices.size()),
            static_cast<std::uint32_t>(
                std::max<std::size_t>(mesh.indices.size(), 1U) * 2U)});
    }
    std::vector<std::vector<Vec3>> sampled_ropes(scene.ropes.size());
    for (std::size_t index = 0U; index < scene.ropes.size(); ++index) {
        const auto &rope = scene.ropes[index];
        const Status sampled = sample_rope_centerline(
            {rope.centerline.data(), rope.centerline.size()},
            rope.options.node_spacing, sampled_ropes[index]);
        if (!sampled) {
            error = sampled.message != nullptr ? sampled.message
                                               : "sample rope centerline failed";
            return false;
        }
        next->rope_capacities.push_back(
            {static_cast<std::uint32_t>(sampled_ropes[index].size())});
    }
    for (const auto &body : scene.soft_bodies) {
        if (body.mesh_index >= scene.meshes.size()) {
            error = "soft body references an invalid mesh";
            return false;
        }
        const auto &mesh = scene.meshes[body.mesh_index];
        next->soft_body_capacities.push_back({
            static_cast<std::uint32_t>(body.nodes.size()),
            static_cast<std::uint32_t>(body.bonds.size()),
            static_cast<std::uint32_t>(mesh.vertices.size()),
            static_cast<std::uint32_t>(mesh.indices.size())});
    }
    if (scene.has_smoke) {
        const std::uint64_t cells = grid_cells(scene.smoke_options);
        if (cells > std::numeric_limits<std::uint32_t>::max()) {
            error = "smoke grid exceeds the public capacity range";
            return false;
        }
        next->smoke_capacities.push_back(
            {scene.smoke_options.capacity, static_cast<std::uint32_t>(cells)});
    }
    for (const PaintDefinition &paint : paints) {
        const std::size_t vertices = scene.meshes[paint.mesh].vertices.size();
        next->paint_capacities.push_back({
            static_cast<std::uint32_t>(vertices), paint.resolution,
            paint.resolution});
    }

    std::vector<PmCouplingState> couplings;
    const PmInteractionMaterialProfile smoke_profile =
        scene.has_smoke && scene.smoke_options.grid_resolution != 0U &&
            scene.smoke_options.grid_vertical_resolution != 0U
        ? PM_INTERACTION_MATERIAL_SMOKE_GRID
        : PM_INTERACTION_MATERIAL_SMOKE_PARTICLE;
    const auto add_mechanical = [&](PmCouplingType type, std::uint32_t first,
                                    std::uint32_t second,
                                    PmMechanicalCouplingState parameters = {}) {
        PmCouplingState state{};
        state.type = type;
        state.first = first;
        state.second = second;
        state.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
        switch (type) {
        case PM_COUPLING_LIQUID_RIGID:
        case PM_COUPLING_LIQUID_CLOTH:
            state.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
                PM_INTERACTION_MODE_CONTAINMENT_BIT;
            break;
        case PM_COUPLING_ROPE_SOFT_BODY:
            state.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
                PM_INTERACTION_MODE_ATTACHMENT_BIT;
            break;
        default:
            state.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT;
            break;
        }
        state.parameters.mechanical = parameters;
        if (type == PM_COUPLING_LIQUID_RIGID) state.parameters.mechanical.twoWay = PM_TRUE;
        couplings.push_back(state);
    };
    // Explicit scene/corpus defaults, independent of private solver options.
    try {
        if (scene.create_default_couplings) {
        if (requires_fluid)
            add_mechanical(PM_COUPLING_LIQUID_RIGID, 0U, 0U);
        for (std::uint32_t cloth = 0U; cloth < scene.cloths.size(); ++cloth) {
            if (requires_fluid && scene.cloths[cloth].contains_fluid) {
                add_mechanical(PM_COUPLING_LIQUID_CLOTH, 0U, cloth,
                    {0.0F, 2'000.0F,
                     12.0F, 0.0F, 1.44F,
                     960.0F, 0.0F,
                     PM_TRUE});
            }
            for (std::uint32_t soft = 0U;
                 soft < scene.soft_bodies.size(); ++soft) {
                add_mechanical(PM_COUPLING_SOFT_BODY_CLOTH, soft, cloth,
                    {0.0F, 0.0F, 0.0F,
                     scene.cloths[cloth].contact_friction, 0.0F, 0.0F, 0.0F,
                     PM_TRUE});
            }
        }
        for (std::uint32_t soft = 0U; soft < scene.soft_bodies.size(); ++soft) {
            if (requires_fluid) {
                add_mechanical(PM_COUPLING_LIQUID_SOFT_BODY, 0U, soft,
                    {0.0F, 0.0F, 0.0F,
                     0.05F, 0.0F, 0.0F, 0.0F, PM_TRUE});
            }
        }
        for (std::uint32_t rope = 0U; rope < scene.ropes.size(); ++rope) {
            if (requires_fluid) {
                add_mechanical(PM_COUPLING_LIQUID_ROPE, 0U, rope,
                    {0.0F, 0.0F, 0.0F,
                     0.05F, 0.0F, 0.0F,
                     30.0F, PM_TRUE});
            }
            for (std::uint32_t soft = 0U;
                 soft < scene.soft_bodies.size(); ++soft) {
                const auto &definition = scene.ropes[rope];
                PmCouplingState state{};
                state.type = PM_COUPLING_ROPE_SOFT_BODY;
                state.first = rope;
                state.second = soft;
                state.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
                    PM_INTERACTION_MODE_ATTACHMENT_BIT;
                state.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
                state.parameters.mechanical.contactDistance =
                    0.0F;
                state.parameters.mechanical.friction = 0.4F;
                state.parameters.attachment = {
                    PM_INVALID_INDEX, PM_INVALID_INDEX,
                    0.0F, 4.0F,
                    1.5F,
                    100.0F,
                    definition.first_soft_body == static_cast<int>(soft)
                        ? PM_TRUE : PM_FALSE,
                    definition.last_soft_body == static_cast<int>(soft)
                        ? PM_TRUE : PM_FALSE};
                couplings.push_back(state);
            }
            for (std::uint32_t cloth = 0U;
                 cloth < scene.cloths.size(); ++cloth) {
                const auto &definition = scene.ropes[rope];
                const bool first =
                    definition.first_cloth == static_cast<int>(cloth);
                const bool last =
                    definition.last_cloth == static_cast<int>(cloth);
                if (!first && !last) continue;
                PmCouplingState state{};
                state.type = PM_COUPLING_ROPE_CLOTH;
                state.first = rope;
                state.second = cloth;
                state.modeFlags = PM_INTERACTION_MODE_ATTACHMENT_BIT;
                state.materialProfile = PM_INTERACTION_MATERIAL_ENDPOINT_NATIVE;
                state.parameters.attachment = {
                    first ? definition.first_cloth_vertex : PM_INVALID_INDEX,
                    last ? definition.last_cloth_vertex : PM_INVALID_INDEX,
                    0.05F, 0.0F, 0.0F,
                    20'000.0F, PM_FALSE, PM_FALSE};
                couplings.push_back(state);
            }
        }
        for (const auto &surface : scene.thermal_surfaces) {
            PmCouplingState state{};
            state.type = PM_COUPLING_LIQUID_SMOKE;
            state.first = 0U;
            state.second = 0U;
            state.modeFlags = PM_INTERACTION_MODE_DRAG_BIT |
                PM_INTERACTION_MODE_THERMAL_BIT | PM_INTERACTION_MODE_PHASE_BIT;
            state.materialProfile = smoke_profile;
            copy(state.parameters.thermal.heaterCenter, surface.plane.center);
            copy(state.parameters.thermal.heaterOrientation,
                 surface.plane.orientation);
            state.parameters.thermal.heaterHalfExtents[0] =
                surface.plane.half_extents.x;
            state.parameters.thermal.heaterHalfExtents[1] =
                surface.plane.half_extents.y;
            state.parameters.thermal.heaterTemperature = surface.temperature;
            state.parameters.thermal.boilingTemperature = 100.0F;
            state.parameters.thermal.heatTransferRate =
                surface.heat_transfer_rate;
            state.parameters.thermal.windDrag = surface.smoke_drag;
            state.parameters.thermal.steamRiseSpeed = surface.steam_rise_speed;
            couplings.push_back(state);
        }
        if (scene.has_smoke) {
            for (std::uint32_t cloth = 0U; cloth < scene.cloths.size(); ++cloth) {
                PmCouplingState state{};
                state.type = PM_COUPLING_SMOKE_CLOTH;
                state.first = 0U;
                state.second = cloth;
                state.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
                    PM_INTERACTION_MODE_DRAG_BIT;
                state.materialProfile = smoke_profile;
                state.parameters.smoke = {2.0F,
                    20.0F, 0.0F, 0.0F,
                    0.0F, PM_FALSE};
                couplings.push_back(state);
            }
            for (std::uint32_t rope = 0U; rope < scene.ropes.size(); ++rope) {
                PmCouplingState state{};
                state.type = PM_COUPLING_SMOKE_ROPE;
                state.first = 0U;
                state.second = rope;
                state.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
                    PM_INTERACTION_MODE_DRAG_BIT;
                state.materialProfile = smoke_profile;
                state.parameters.smoke = {2.0F,
                    20.0F, 0.0F, 0.0F,
                    0.0F, PM_FALSE};
                couplings.push_back(state);
            }
            for (std::uint32_t soft = 0U;
                 soft < scene.soft_bodies.size(); ++soft) {
                PmCouplingState state{};
                state.type = PM_COUPLING_SMOKE_SOFT_BODY;
                state.first = 0U;
                state.second = soft;
                state.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
                    PM_INTERACTION_MODE_DRAG_BIT;
                state.materialProfile = smoke_profile;
                state.parameters.smoke = {0.5F,
                    2.0F, 0.0F, 0.0F,
                    0.0F, PM_FALSE};
                couplings.push_back(state);
            }
            for (std::uint32_t body = 0U;
                 body < scene.rigid_bodies.size(); ++body) {
                const auto &definition = scene.rigid_bodies[body];
                const bool named_obstacle = !scene.smoke_obstacle_name.empty() &&
                    (definition.name == scene.smoke_obstacle_name ||
                     definition.source_name == scene.smoke_obstacle_name);
                if (!definition.smoke_collider && !named_obstacle &&
                    definition.options.motion != MotionType::dynamic)
                    continue;
                PmCouplingState state{};
                state.type = PM_COUPLING_SMOKE_RIGID;
                state.first = 0U;
                state.second = body;
                state.modeFlags = PM_INTERACTION_MODE_CONTACT_BIT |
                    PM_INTERACTION_MODE_DRAG_BIT;
                state.materialProfile = smoke_profile;
                state.parameters.smoke = {0.0F, 0.0F, 1.5F,
                    4.0F, 0.0F,
                    PM_TRUE};
                couplings.push_back(state);
            }
        }
        }
    } catch (const std::bad_alloc &) {
        error = "build coupling pipeline: out of host memory";
        return false;
    }

    next->rigid_count = static_cast<std::uint32_t>(scene.rigid_bodies.size());
    next->contact_capacity = 65'536U;
    next->debug_capacity = debug_frame_capacity == 0U ? 0U :
        std::max<std::uint32_t>(1024U, next->rigid_count * 16U);
    PmSimulationCreateInfo simulation_info{};
    simulation_info.sType = PM_STRUCTURE_TYPE_SIMULATION_CREATE_INFO;
    simulation_info.rigidBodyCapacity = next->rigid_count;
    simulation_info.constraintCapacity = std::max<std::uint32_t>(
        static_cast<std::uint32_t>(scene.rigid_constraints.size()),
        next->rigid_count * 2U);
    simulation_info.contactCapacity = next->contact_capacity;
    simulation_info.fluidCount =
        static_cast<std::uint32_t>(next->fluid_capacities.size());
    simulation_info.pFluidCapacities = next->fluid_capacities.data();
    simulation_info.clothCount =
        static_cast<std::uint32_t>(next->cloth_capacities.size());
    simulation_info.pClothCapacities = next->cloth_capacities.data();
    simulation_info.ropeCount =
        static_cast<std::uint32_t>(next->rope_capacities.size());
    simulation_info.pRopeCapacities = next->rope_capacities.data();
    simulation_info.softBodyCount =
        static_cast<std::uint32_t>(next->soft_body_capacities.size());
    simulation_info.pSoftBodyCapacities = next->soft_body_capacities.data();
    simulation_info.smokeCount =
        static_cast<std::uint32_t>(next->smoke_capacities.size());
    simulation_info.pSmokeCapacities = next->smoke_capacities.data();
    simulation_info.couplingCapacity =
        static_cast<std::uint32_t>(couplings.size());
    simulation_info.emitterCapacity =
        static_cast<std::uint32_t>(scene.particle_sources.size());
    simulation_info.destroyerCapacity =
        static_cast<std::uint32_t>(scene.destroy_planes.size());
    simulation_info.paintFieldCount =
        static_cast<std::uint32_t>(next->paint_capacities.size());
    simulation_info.pPaintFieldCapacities = next->paint_capacities.data();
    simulation_info.paintRuleCapacity =
        static_cast<std::uint32_t>(paints.size());
    simulation_info.debugFrameCapacity = debug_frame_capacity;
    if (!checked(pmCreateSimulation(next->device, &simulation_info,
                                    &next->simulation),
                 "create simulation", error))
        return false;
    PmSimulationMemoryRequirements simulation_memory{};
    pmGetSimulationMemoryRequirements(next->device, next->simulation,
                                      &simulation_memory);
    PmMemoryAllocateInfo allocation{};
    allocation.sType = PM_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.memoryTypeIndex =
        (simulation_memory.persistent.memoryTypeBits & 0x2U) != 0U ? 1U : 0U;
    allocation.allocationSize = simulation_memory.persistent.size;
    if (!checked(pmAllocateMemory(next->device, &allocation,
                                  &next->persistent_memory),
                 "allocate persistent simulation memory", error))
        return false;
    allocation.memoryTypeIndex =
        (simulation_memory.scratch.memoryTypeBits & 0x2U) != 0U ? 1U : 0U;
    allocation.allocationSize = simulation_memory.scratch.size;
    if (!checked(pmAllocateMemory(next->device, &allocation,
                                  &next->scratch_memory),
                 "allocate scratch simulation memory", error))
        return false;
    const PmSimulationMemoryBinding binding{
        next->persistent_memory, 0U, next->scratch_memory, 0U};
    if (!checked(pmBindSimulationMemory(next->device, next->simulation, &binding),
                 "bind simulation memory", error))
        return false;

    PmRigidSolverState rigid_solver{};
    rigid_solver.rigidIterations = 10U;
    rigid_solver.enableSleeping = PM_FALSE;
    rigid_solver.sleepLinearThreshold = 0.02F;
    rigid_solver.sleepAngularThreshold = 0.02F;
    rigid_solver.sleepTimeSeconds = 0.5F;

    std::vector<PmLiquidMaterialState> fluid_materials;
    std::vector<PmClothMaterialState> cloth_materials;
    std::vector<PmRopeMaterialState> rope_materials;
    std::vector<PmSoftBodyMaterialState> soft_body_materials;
    std::vector<PmSmokeMaterialState> smoke_materials;
    try {
        if (requires_fluid) {
            const auto &source = scene.fluid_options;
            fluid_materials.push_back({
                source.rest_density, source.particle_radius,
                source.support_radius, source.rest_particle_volume,
                source.repulsion, source.viscosity, source.normal_damping,
                source.velocity_damping, source.maximum_speed,
                source.maximum_pair_acceleration, source.solver_iterations});
        }
        for (const auto &source : scene.cloths)
            cloth_materials.push_back({
                source.vertex_mass, source.thickness,
                source.stretch_compliance, source.bending_compliance,
                source.velocity_damping,
                source.contact_friction, source.solver_iterations,
                source.break_strain, source.fracture_persistence_substeps,
                source.impact_break_impulse,
                source.preserve_volume ? PM_TRUE : PM_FALSE,
                source.target_volume, source.volume_compliance});
        for (const auto &source : scene.ropes) {
            const auto &options = source.options;
            rope_materials.push_back({
                options.node_spacing, options.radius, options.mass,
                options.stretch_compliance, options.velocity_damping,
                options.maximum_substep_timestep, options.friction,
                options.maximum_speed, options.solver_iterations,
                options.self_collision ? PM_TRUE : PM_FALSE});
        }
        for (const auto &source : scene.soft_bodies)
            soft_body_materials.push_back({
                source.node_mass, source.node_radius,
                source.stretch_compliance, source.velocity_damping,
                source.spring_damping, source.contact_friction,
                source.shape_matching_stiffness,
                source.maximum_projection_fraction,
                source.constraint_velocity_response, source.maximum_speed,
                source.solver_iterations});
        if (scene.has_smoke) {
            const auto &source = scene.smoke_options;
            smoke_materials.push_back({
                source.particle_radius, source.buoyancy, source.response,
                source.rest_number_density, source.pressure_stiffness,
                source.viscosity, source.vorticity_confinement,
                source.maximum_speed, source.grid_resolution,
                source.grid_vertical_resolution,
                source.grid_pressure_iterations,
                source.grid_kinematic_viscosity,
                source.grid_les_coefficient,
                source.grid_pressure_tolerance});
        }
    } catch (const std::bad_alloc &) {
        error = "assemble physics pipeline: out of host memory";
        return false;
    }
    PmPhysicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = PM_STRUCTURE_TYPE_PHYSICS_PIPELINE_CREATE_INFO;
    pipeline_info.pRigidState = &rigid_solver;
    pipeline_info.fluidMaterialCount =
        static_cast<std::uint32_t>(fluid_materials.size());
    pipeline_info.pFluidMaterials = fluid_materials.data();
    pipeline_info.couplingCount = static_cast<std::uint32_t>(couplings.size());
    pipeline_info.pCouplings = couplings.data();
    pipeline_info.clothMaterialCount =
        static_cast<std::uint32_t>(cloth_materials.size());
    pipeline_info.pClothMaterials = cloth_materials.data();
    pipeline_info.ropeMaterialCount =
        static_cast<std::uint32_t>(rope_materials.size());
    pipeline_info.pRopeMaterials = rope_materials.data();
    pipeline_info.softBodyMaterialCount =
        static_cast<std::uint32_t>(soft_body_materials.size());
    pipeline_info.pSoftBodyMaterials = soft_body_materials.data();
    pipeline_info.smokeMaterialCount =
        static_cast<std::uint32_t>(smoke_materials.size());
    pipeline_info.pSmokeMaterials = smoke_materials.data();
    if (!checked(pmCreatePhysicsPipeline(next->device, &pipeline_info,
                                         &next->pipeline),
                 "create physics pipeline", error))
        return false;

    PmQueryPoolCreateInfo query_info{};
    query_info.sType = PM_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query_info.queryType = PM_QUERY_REALTIME_STEP_STATISTICS;
    query_info.queryCount = 1U;
    if (!checked(pmCreateQueryPool(next->device, &query_info,
                                   &next->statistics_query),
                 "create statistics query", error))
        return false;

    std::vector<PmRigidBodyInfo> rigid_bodies;
    std::vector<PmClothInfo> cloths;
    std::vector<std::vector<PmFloat3>> cloth_vertices;
    std::vector<PmRopeInfo> ropes;
    std::vector<std::vector<PmFloat3>> rope_centerlines;
    std::vector<PmSoftBodyInfo> soft_bodies;
    std::vector<std::vector<PmFloat3>> soft_nodes;
    std::vector<std::vector<PmFloat3>> soft_surface_vertices;
    std::vector<std::vector<PmSoftBodyBond>> soft_bonds;
    std::vector<std::vector<PmSoftBodySurfaceBinding>> soft_bindings;
    std::vector<PmParticleState> initial_particles;
    std::vector<PmParticleEmitterInfo> emitters;
    std::vector<std::vector<PmFloat3>> emitter_vertices;
    std::vector<PmParticleDestroyerInfo> destroyers;
    std::vector<PmPaintFieldInfo> paint_fields;
    std::vector<std::vector<PmFloat2>> paint_uvs;
    std::vector<PmPaintRuleInfo> paint_rules;
    try {
        rigid_bodies.resize(scene.rigid_bodies.size());
        for (std::size_t index = 0U; index < scene.rigid_bodies.size(); ++index) {
            const auto &source = scene.rigid_bodies[index].options;
            auto &target = rigid_bodies[index];
            target.active = PM_TRUE;
            target.motionType = pm(source.motion);
            target.geometry = next->geometries[body_geometry[index]].handle;
            target.state = pm(source.initial_state);
            target.mass = source.mass;
            copy(target.inertiaDiagonal, source.inertia_diagonal);
            target.friction = source.friction;
            target.restitution = source.restitution;
            target.linearDamping = source.linear_damping;
            target.angularDamping = source.angular_damping;
            target.maximumLinearSpeed = source.maximum_linear_speed;
            target.maximumAngularSpeed = source.maximum_angular_speed;
            target.collisionMargin = source.collision_margin;
            target.userData = source.user_data;
        }
        next->constraints.resize(simulation_info.constraintCapacity);
        next->authored_constraint_count = static_cast<std::uint32_t>(
            scene.rigid_constraints.size());
        for (std::size_t index = 0U;
             index < scene.rigid_constraints.size(); ++index)
            next->constraints[index] = pm(scene.rigid_constraints[index]);

        cloth_vertices.resize(scene.cloths.size());
        cloths.resize(scene.cloths.size());
        for (std::size_t index = 0U; index < scene.cloths.size(); ++index) {
            const auto &definition = scene.cloths[index];
            const auto &mesh = scene.meshes[definition.mesh_index];
            auto &vertices = cloth_vertices[index];
            vertices.reserve(mesh.vertices.size());
            for (const auto &vertex : mesh.vertices)
                vertices.push_back(pm(vertex.position));
            cloths[index] = {PM_TRUE, static_cast<std::uint32_t>(index),
                static_cast<std::uint32_t>(vertices.size()), vertices.data(),
                static_cast<std::uint32_t>(mesh.indices.size()),
                mesh.indices.data(), definition.inverse_masses.data()};
        }
        rope_centerlines.resize(scene.ropes.size());
        ropes.resize(scene.ropes.size());
        for (std::size_t index = 0U; index < scene.ropes.size(); ++index) {
            const auto &definition = scene.ropes[index];
            auto &points = rope_centerlines[index];
            points.reserve(definition.centerline.size());
            for (Vec3 point : definition.centerline) points.push_back(pm(point));
            auto &target = ropes[index];
            target.active = PM_TRUE;
            target.materialIndex = static_cast<std::uint32_t>(index);
            target.pointCount = static_cast<std::uint32_t>(points.size());
            target.pCenterline = points.data();
            if (definition.first_body >= 0) {
                target.first = {
                    static_cast<std::uint32_t>(definition.first_body),
                    pm(definition.options.first.local_anchor), PM_TRUE};
            }
            if (definition.last_body >= 0) {
                target.last = {
                    static_cast<std::uint32_t>(definition.last_body),
                    pm(definition.options.last.local_anchor), PM_TRUE};
            }
        }
        soft_nodes.resize(scene.soft_bodies.size());
        soft_surface_vertices.resize(scene.soft_bodies.size());
        soft_bonds.resize(scene.soft_bodies.size());
        soft_bindings.resize(scene.soft_bodies.size());
        soft_bodies.resize(scene.soft_bodies.size());
        for (std::size_t index = 0U; index < scene.soft_bodies.size(); ++index) {
            const auto &definition = scene.soft_bodies[index];
            const auto &mesh = scene.meshes[definition.mesh_index];
            for (Vec3 node : definition.nodes)
                soft_nodes[index].push_back(pm(node));
            for (const auto &bond : definition.bonds)
                soft_bonds[index].push_back(
                    {bond.first, bond.second, bond.rest_length});
            for (const auto &vertex : mesh.vertices)
                soft_surface_vertices[index].push_back(pm(vertex.position));
            for (const auto &binding_source : definition.surface_bindings) {
                PmSoftBodySurfaceBinding binding_target{};
                std::copy_n(binding_source.nodes, 4U, binding_target.nodes);
                std::copy_n(binding_source.weights, 4U, binding_target.weights);
                soft_bindings[index].push_back(binding_target);
            }
            soft_bodies[index] = {PM_TRUE, static_cast<std::uint32_t>(index),
                static_cast<std::uint32_t>(soft_nodes[index].size()),
                soft_nodes[index].data(),
                static_cast<std::uint32_t>(soft_bonds[index].size()),
                soft_bonds[index].data(), definition.inverse_masses.data(),
                static_cast<std::uint32_t>(
                    soft_surface_vertices[index].size()),
                soft_surface_vertices[index].data(),
                static_cast<std::uint32_t>(mesh.indices.size()),
                mesh.indices.data(), soft_bindings[index].data()};
        }

        if (requires_fluid) {
            const std::size_t requested = std::min<std::size_t>(
                scene.fluid_options.capacity, scene.initial_particles.size());
            initial_particles.reserve(requested);
            for (std::size_t index = 0U; index < requested; ++index) {
                const std::size_t source_index =
                    index * scene.initial_particles.size() / requested;
                const auto &source = scene.initial_particles[source_index];
                PmParticleState target{};
                copy(target.position, source.position);
                copy(target.velocity, source.velocity);
                target.temperatureCelsius = source.temperature;
                initial_particles.push_back(target);
            }
        }
        emitter_vertices.resize(scene.particle_sources.size());
        emitters.resize(scene.particle_sources.size());
        for (std::size_t index = 0U; index < scene.particle_sources.size(); ++index) {
            const auto &source = scene.particle_sources[index];
            for (Vec3 vertex : source.vertices)
                emitter_vertices[index].push_back(pm(vertex));
            emitters[index] = {source.options.enabled ? PM_TRUE : PM_FALSE, 0U,
                static_cast<std::uint32_t>(emitter_vertices[index].size()),
                emitter_vertices[index].data(),
                static_cast<std::uint32_t>(source.indices.size()),
                source.indices.data(), source.spacing,
                pm(source.options.initial_velocity),
                source.options.initial_temperature};
        }
        destroyers.resize(scene.destroy_planes.size());
        for (std::size_t index = 0U; index < scene.destroy_planes.size(); ++index) {
            const auto &source = scene.destroy_planes[index];
            auto &target = destroyers[index];
            target.active = source.enabled ? PM_TRUE : PM_FALSE;
            target.fluidIndex = 0U;
            target.center = pm(source.plane.center);
            copy(target.orientation, source.plane.orientation);
            target.halfExtents = {source.plane.half_extents.x,
                                  source.plane.half_extents.y};
            switch (source.crossing) {
            case CrossingDirection::along_normal:
                target.crossing = PM_CROSSING_ALONG_NORMAL;
                break;
            case CrossingDirection::against_normal:
                target.crossing = PM_CROSSING_AGAINST_NORMAL;
                break;
            case CrossingDirection::either:
                target.crossing = PM_CROSSING_EITHER;
                break;
            }
        }
        paint_uvs.resize(paints.size());
        paint_fields.resize(paints.size());
        paint_rules.resize(paints.size());
        for (std::size_t index = 0U; index < paints.size(); ++index) {
            const auto &paint = paints[index];
            const auto &mesh = scene.meshes[paint.mesh];
            for (const auto &vertex : mesh.vertices)
                paint_uvs[index].push_back({vertex.uv.x, vertex.uv.y});
            auto &field = paint_fields[index];
            field.active = PM_TRUE;
            field.targetType = paint.body != PM_INVALID_INDEX
                ? PM_PAINT_TARGET_RIGID : PM_PAINT_TARGET_CLOTH;
            field.targetIndex = paint.body != PM_INVALID_INDEX
                ? paint.body : paint.cloth;
            field.geometry = paint.geometry == PM_INVALID_INDEX
                ? nullptr : next->geometries[paint.geometry].handle;
            field.vertexCount =
                static_cast<std::uint32_t>(paint_uvs[index].size());
            field.pVertexUvs = paint_uvs[index].data();
            field.width = paint.resolution;
            field.height = paint.resolution;
            auto &rule = paint_rules[index];
            rule.active = PM_TRUE;
            rule.sourceType = paint.liquid_source
                ? PM_PAINT_SOURCE_LIQUID : PM_PAINT_SOURCE_RIGID;
            rule.sourceIndex = paint.liquid_source ? 0U : paint.rigid_source;
            rule.targetFieldIndex = static_cast<std::uint32_t>(index);
            rule.reach = 0.025F;
            rule.brushRadius = paint.brush_radius;
        }
        next->original_scene = scene;
        next->rigid_bodies = rigid_bodies;
        next->initial_rigid_bodies = rigid_bodies;
        next->initial_constraints = next->constraints;
        next->cloth_active.assign(scene.cloths.size(), 1U);
        next->rope_active.assign(scene.ropes.size(), 1U);
        next->soft_body_active.assign(scene.soft_bodies.size(), 1U);
        next->smoke_active = scene.has_smoke;
    } catch (const std::bad_alloc &) {
        error = "assemble scene resources: out of host memory";
        return false;
    }

    if (requires_fluid) {
        const PmDeviceSize bytes = std::max<PmDeviceSize>(
            sizeof(PmParticleState),
            static_cast<PmDeviceSize>(scene.fluid_options.capacity) *
                sizeof(PmParticleState));
        if (!next->create_buffer(bytes, next->particle_input, error)) return false;
        if (!initial_particles.empty())
            std::memcpy(next->particle_input.mapped, initial_particles.data(),
                        initial_particles.size() * sizeof(PmParticleState));
    }

    const auto record_initial_state = [&](PmCommandBuffer commands) {
        for (const auto &geometry : next->geometries)
            if (!checked(pmCmdBuildGeometry(commands, geometry.handle),
                         "record geometry build", error))
                return false;
        if (!rigid_bodies.empty() &&
            !checked(pmCmdWriteRigidBodies(
                         commands, next->simulation, 0U,
                         static_cast<std::uint32_t>(rigid_bodies.size()),
                         rigid_bodies.data()),
                     "record rigid bodies", error))
            return false;
        if (!next->constraints.empty() &&
            !checked(pmCmdWriteRigidConstraints(
                         commands, next->simulation, 0U,
                         static_cast<std::uint32_t>(next->constraints.size()),
                         next->constraints.data()),
                     "record rigid constraints", error))
            return false;
        if (requires_fluid) {
            const PmBufferView view{{next->particle_input.handle, 0U,
                                     next->particle_input.size},
                                    PM_FORMAT_PARTICLE_STATE,
                                    static_cast<std::uint32_t>(
                                        initial_particles.size()),
                                    sizeof(PmParticleState)};
            if (!checked(pmCmdSetFluidParticles(
                             commands, next->simulation, 0U, &view),
                         "record fluid particles", error))
                return false;
        }
        if (!cloths.empty() &&
            !checked(pmCmdWriteCloths(
                         commands, next->simulation, 0U,
                         static_cast<std::uint32_t>(cloths.size()),
                         cloths.data()),
                     "record cloths", error))
            return false;
        if (!ropes.empty() &&
            !checked(pmCmdWriteRopes(
                         commands, next->simulation, 0U,
                         static_cast<std::uint32_t>(ropes.size()), ropes.data()),
                     "record ropes", error))
            return false;
        if (!soft_bodies.empty() &&
            !checked(pmCmdWriteSoftBodies(
                         commands, next->simulation, 0U,
                         static_cast<std::uint32_t>(soft_bodies.size()),
                         soft_bodies.data()),
                     "record soft bodies", error))
            return false;
        if (scene.has_smoke) {
            const auto &source = scene.smoke_options;
            const PmSmokeInfo smoke{
                PM_TRUE, 0U, source.capacity, pm(source.emitter_center),
                {source.emitter_half_extents.x,
                 source.emitter_half_extents.y},
                pm(source.initial_velocity), pm(source.wind),
                source.particles_per_second, source.lifetime,
                pm(source.grid_minimum), source.grid_edge_length};
            if (!checked(pmCmdWriteSmoke(
                             commands, next->simulation, 0U, 1U, &smoke),
                         "record smoke", error))
                return false;
        }
        if (!emitters.empty() &&
            !checked(pmCmdWriteParticleEmitters(
                         commands, next->simulation, 0U,
                         static_cast<std::uint32_t>(emitters.size()),
                         emitters.data()),
                     "record particle emitters", error))
            return false;
        if (!destroyers.empty() &&
            !checked(pmCmdWriteParticleDestroyers(
                         commands, next->simulation, 0U,
                         static_cast<std::uint32_t>(destroyers.size()),
                         destroyers.data()),
                     "record particle destroyers", error))
            return false;
        if (!paint_fields.empty() &&
            (!checked(pmCmdWritePaintFields(
                          commands, next->simulation, 0U,
                          static_cast<std::uint32_t>(paint_fields.size()),
                          paint_fields.data()),
                      "record paint fields", error) ||
             !checked(pmCmdWritePaintRules(
                          commands, next->simulation, 0U,
                          static_cast<std::uint32_t>(paint_rules.size()),
                          paint_rules.data()),
                      "record paint rules", error)))
            return false;
        return true;
    };

    if (!next->begin(error) ||
        !record_initial_state(next->commands) ||
        !next->submit(error))
        return false;

    PmCommandBufferBeginInfo reset_begin{};
    reset_begin.sType = PM_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (!checked(pmBeginCommandBuffer(next->reset_commands, &reset_begin),
                 "begin reusable reset commands", error) ||
        !checked(pmCmdResetSimulation(
                     next->reset_commands, next->simulation),
                 "record simulation reset", error) ||
        !checked(pmCmdBindPhysicsPipeline(
                     next->reset_commands, next->pipeline),
                 "bind reset physics pipeline", error) ||
        !record_initial_state(next->reset_commands) ||
        !checked(pmEndCommandBuffer(next->reset_commands),
                 "end reusable reset commands", error))
        return false;

    PmDeviceSize output_size = 0U;
    const auto append_region = [&](PmDeviceSize bytes, Impl::Region &region) {
        output_size = (output_size + 15U) & ~PmDeviceSize{15U};
        region = {output_size, bytes};
        output_size += bytes;
    };
    append_region(next->rigid_count * sizeof(PmRigidState), next->rigid_output);
    append_region(next->rigid_count * sizeof(PmRigidState), next->previous_rigid_output);
    if (requires_fluid)
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(next->fluid_capacities[0].particleCapacity) *
                sizeof(PmFluidParticleOutput), next->fluid_output);
    next->cloth_outputs.resize(next->cloth_capacities.size());
    next->cloth_surface_outputs.resize(next->cloth_capacities.size());
    for (std::size_t index = 0U; index < next->cloth_outputs.size(); ++index)
    {
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(next->cloth_capacities[index].vertexCapacity) *
                sizeof(PmClothVertexState), next->cloth_outputs[index]);
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(
                next->cloth_capacities[index].triangleIndexCapacity) *
                sizeof(PmFloat3), next->cloth_surface_outputs[index]);
    }
    next->rope_outputs.resize(next->rope_capacities.size());
    for (std::size_t index = 0U; index < next->rope_outputs.size(); ++index)
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(next->rope_capacities[index].nodeCapacity) *
                sizeof(PmRopeNodeState), next->rope_outputs[index]);
    next->soft_body_outputs.resize(next->soft_body_capacities.size());
    next->soft_body_surface_outputs.resize(next->soft_body_capacities.size());
    for (std::size_t index = 0U; index < next->soft_body_outputs.size(); ++index)
    {
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(next->soft_body_capacities[index].nodeCapacity) *
                sizeof(PmSoftBodyNodeState), next->soft_body_outputs[index]);
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(
                next->soft_body_capacities[index].surfaceVertexCapacity) *
                sizeof(PmFloat3), next->soft_body_surface_outputs[index]);
    }
    if (scene.has_smoke)
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(next->smoke_capacities[0].particleCapacity) *
                sizeof(PmSmokeParticleState), next->smoke_output);
    next->paint_outputs.resize(next->paint_capacities.size());
    for (std::size_t index = 0U; index < next->paint_outputs.size(); ++index)
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(next->paint_capacities[index].width) *
                next->paint_capacities[index].height * sizeof(std::uint32_t),
            next->paint_outputs[index]);
    if (!next->constraints.empty())
        append_region(next->constraints.size() * sizeof(PmConstraintState),
                      next->constraint_output);
    append_region(sizeof(PmDomainOutputHeader) +
        static_cast<PmDeviceSize>(next->contact_capacity) * sizeof(PmContactEvent),
        next->contact_output);
    next->debug_capacity = next->rigid_count;
    if (requires_fluid)
        next->debug_capacity += next->fluid_capacities[0].particleCapacity;
    for (const auto &capacity : next->cloth_capacities)
        next->debug_capacity += capacity.vertexCapacity;
    for (const auto &capacity : next->rope_capacities)
        next->debug_capacity += capacity.nodeCapacity;
    for (const auto &capacity : next->soft_body_capacities)
        next->debug_capacity += capacity.nodeCapacity;
    if (debug_frame_capacity != 0U)
        append_region(sizeof(PmDomainOutputHeader) +
            static_cast<PmDeviceSize>(next->debug_capacity) *
                sizeof(PmDebugRecord), next->debug_output);
    if (!next->create_buffer(output_size, next->output, error)) return false;
    // Output remains GPU-owned between submissions. CPU consumers map only
    // the requested live records; the native renderer borrows this buffer.
    pmUnmapMemory(next->device, next->output.memory);
    next->output.mapped = nullptr;

    instance.rigid_bodies.resize(scene.rigid_bodies.size());
    std::iota(instance.rigid_bodies.begin(), instance.rigid_bodies.end(), 0U);
    instance.rigid_constraints.resize(scene.rigid_constraints.size());
    std::iota(instance.rigid_constraints.begin(),
              instance.rigid_constraints.end(), 0U);
    instance.cloths.resize(scene.cloths.size());
    std::iota(instance.cloths.begin(), instance.cloths.end(), 0U);
    instance.ropes.resize(scene.ropes.size());
    std::iota(instance.ropes.begin(), instance.ropes.end(), 0U);
    instance.soft_bodies.resize(scene.soft_bodies.size());
    std::iota(instance.soft_bodies.begin(), instance.soft_bodies.end(), 0U);
    instance.fluid = requires_fluid ? 0U : PM_INVALID_INDEX;
    instance.smoke = scene.has_smoke ? 0U : PM_INVALID_INDEX;
    instance.render_buffers.device = next->device;
    instance.render_buffers.queue = next->queue;
    instance.render_buffers.output = next->output.handle;
    instance.render_buffers.resource_generation =
        next_output_resource_generation.fetch_add(1U, std::memory_order_relaxed);
    const auto region = [](auto value) { return GpuSceneBuffers::Region{value.offset, value.size}; };
    instance.render_buffers.rigid = region(next->rigid_output);
    instance.render_buffers.previous_rigid = region(next->previous_rigid_output);
    instance.render_buffers.fluid = region(next->fluid_output);
    instance.render_buffers.smoke = region(next->smoke_output);
    instance.render_buffers.fluid_capacity = requires_fluid
        ? next->fluid_capacities[0].particleCapacity : 0U;
    for (const auto value : next->cloth_surface_outputs)
        instance.render_buffers.cloth_surfaces.push_back(region(value));
    for (const auto value : next->soft_body_surface_outputs)
        instance.render_buffers.soft_surfaces.push_back(region(value));
    for (const auto value : next->rope_outputs)
        instance.render_buffers.ropes.push_back(region(value));
    for (const auto value : next->paint_outputs)
        instance.render_buffers.paint.push_back(region(value));
    for (std::uint32_t index = 0U; index < paints.size(); ++index)
        instance.paint_bindings.push_back(
            {paints[index].body, paints[index].mesh, index});

    output.impl_ = std::move(next);
    if (creation_category) *creation_category = PM_SUCCESS;
    return true;
}

namespace {

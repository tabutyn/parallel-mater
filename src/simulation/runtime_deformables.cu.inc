// SPDX-License-Identifier: MIT
// Included by runtime.cpp; shared backend-neutral runtime implementation.

Status SimulationRuntime::add_triangle_mesh(
    DeviceSpan<const Vec3> vertices,
    DeviceSpan<const std::uint32_t> triangle_indices, TriangleMeshId &output,
    device::ExecutionQueue queue) noexcept {
    const device::ExecutionQueue stream = queue;
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    if (vertices.data == nullptr || triangle_indices.data == nullptr ||
        vertices.size < 3U || triangle_indices.size < 3U ||
        triangle_indices.size % 3U != 0U ||
        vertices.size > std::numeric_limits<std::uint32_t>::max() ||
        triangle_indices.size > std::numeric_limits<std::uint32_t>::max()) {
        return failure(StatusCode::invalid_argument,
                       "triangle mesh requires device vertices and triangle indices");
    }
    if (impl_->triangle_mesh_count >= impl_->options.triangle_mesh_capacity) {
        return failure(StatusCode::capacity_exceeded,
                       "triangle mesh capacity is exhausted");
    }
    std::uint32_t slot = k_invalid_dense;
    for (std::uint32_t index = 0;
         index < impl_->options.triangle_mesh_capacity; ++index) {
        if (!impl_->meshes[index].alive) {
            slot = index;
            break;
        }
    }
    if (slot == k_invalid_dense) {
        return failure(StatusCode::internal_error,
                       "no free triangle mesh slot was found");
    }

    struct GeometryStorageOwner {
        simulation::GeometryStoragePlan plan{};
        ~GeometryStorageOwner() {
            release_managed(plan.shell_normals);
            release_managed(plan.solid_planes);
            release_managed(plan.bvh_leaves);
            release_managed(plan.bvh_nodes);
            release_managed(plan.indices);
            release_managed(plan.vertices);
        }
        void disarm() noexcept {
            plan.vertices = nullptr;
            plan.indices = nullptr;
            plan.bvh_nodes = nullptr;
            plan.bvh_leaves = nullptr;
            plan.solid_planes = nullptr;
            plan.shell_normals = nullptr;
        }
    } owned;
    StorageAllocator allocator;
    status = simulation::visit_geometry_storage(
        owned.plan, static_cast<std::size_t>(vertices.size),
        static_cast<std::size_t>(triangle_indices.size), allocator);
    if (!status) return status;
    Vec3 *const owned_vertices = owned.plan.vertices;
    std::uint32_t *const owned_indices = owned.plan.indices;
    BvhNode *const owned_bvh_nodes = owned.plan.bvh_nodes;
    std::uint32_t *const owned_bvh_leaves = owned.plan.bvh_leaves;
    CollisionPlane *const owned_solid_planes = owned.plan.solid_planes;
    Vec3 *const owned_shell_normals = owned.plan.shell_normals;
    PmResult error = copy_bytes(
        owned_vertices, vertices.data, sizeof(Vec3) * vertices.size,
        device::CopyDirection::automatic, stream);
    if (error == PM_SUCCESS) {
        error = copy_bytes(owned_indices, triangle_indices.data,
                                sizeof(std::uint32_t) * triangle_indices.size,
                                device::CopyDirection::automatic, stream);
    }
    if (error == PM_SUCCESS) {
        error = wait_queue(stream);
    }
    if (error == PM_SUCCESS)
        error = stream.services->read_host(owned_vertices, sizeof(Vec3) * vertices.size);
    if (error == PM_SUCCESS)
        error = stream.services->read_host(owned_indices, sizeof(std::uint32_t) * triangle_indices.size);
    if (error != PM_SUCCESS) {
        return backend_failure(error, "triangle mesh upload failed");
    }
    // Native adapters may map uncached device-local memory. BVH and convex
    // preparation revisit vertices many times; take one contiguous host copy
    // instead of repeatedly reading those mappings over the device bus.
    std::vector<Vec3> host_vertices;
    std::vector<std::uint32_t> host_indices;
    try {
        host_vertices.resize(static_cast<std::size_t>(vertices.size));
        host_indices.resize(static_cast<std::size_t>(triangle_indices.size));
        std::memcpy(host_vertices.data(), owned_vertices, sizeof(Vec3) * vertices.size);
        std::memcpy(host_indices.data(), owned_indices,
                    sizeof(std::uint32_t) * triangle_indices.size);
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to stage triangle mesh host preparation");
    }
    for (std::uint64_t index = 0; index < vertices.size; ++index) {
        if (!finite(host_vertices[index])) {
            return failure(StatusCode::invalid_argument,
                           "triangle mesh contains a non-finite vertex");
        }
    }

    Vec3 minimum = host_vertices[0];
    Vec3 maximum = host_vertices[0];
    for (std::uint64_t index = 1; index < vertices.size; ++index) {
        minimum = component_min(minimum, host_vertices[index]);
        maximum = component_max(maximum, host_vertices[index]);
    }
    for (std::uint64_t index = 0; index < triangle_indices.size; index += 3U) {
        const std::uint32_t first = host_indices[index];
        const std::uint32_t second = host_indices[index + 1U];
        const std::uint32_t third = host_indices[index + 2U];
        if (first >= vertices.size || second >= vertices.size ||
            third >= vertices.size) {
            return failure(StatusCode::invalid_argument,
                           "triangle mesh index is outside the vertex buffer");
        }
        const Vec3 area = cross(subtract(host_vertices[second],
                                         host_vertices[first]),
                                subtract(host_vertices[third],
                                         host_vertices[first]));
        if (length_squared(area) <= k_epsilon * k_epsilon) {
            return failure(StatusCode::invalid_argument,
                           "triangle mesh contains a degenerate triangle");
        }
    }

    std::vector<std::uint32_t> triangle_order;
    std::vector<BvhNode> bvh_nodes;
    std::vector<std::uint32_t> reordered_indices;
    try {
        const std::uint32_t triangle_count =
            static_cast<std::uint32_t>(triangle_indices.size / 3U);
        triangle_order.resize(triangle_count);
        std::iota(triangle_order.begin(), triangle_order.end(), 0U);
        bvh_nodes.reserve(triangle_count * 2U);
        const auto coordinate = [](Vec3 value, int axis) {
            return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
        };
        const auto centroid = [&](std::uint32_t triangle) {
            const std::uint32_t offset = triangle * 3U;
            return multiply(
                add(add(host_vertices[host_indices[offset]],
                        host_vertices[host_indices[offset + 1U]]),
                    host_vertices[host_indices[offset + 2U]]),
                1.0F / 3.0F);
        };
        std::function<std::uint32_t(std::uint32_t, std::uint32_t)> build =
            [&](std::uint32_t begin, std::uint32_t end) {
                BvhNode node{};
                node.minimum = {FLT_MAX, FLT_MAX, FLT_MAX};
                node.maximum = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
                for (std::uint32_t item = begin; item < end; ++item) {
                    const std::uint32_t offset = triangle_order[item] * 3U;
                    for (std::uint32_t corner = 0; corner < 3U; ++corner) {
                        const Vec3 vertex =
                            host_vertices[host_indices[offset + corner]];
                        node.minimum = component_min(node.minimum, vertex);
                        node.maximum = component_max(node.maximum, vertex);
                    }
                }
                const std::uint32_t node_index =
                    static_cast<std::uint32_t>(bvh_nodes.size());
                bvh_nodes.push_back(node);
                if (end - begin <= 4U) {
                    bvh_nodes[node_index].first_triangle = begin;
                    bvh_nodes[node_index].triangle_count = end - begin;
                    return node_index;
                }
                const Vec3 extent = subtract(node.maximum, node.minimum);
                const int axis = extent.x >= extent.y && extent.x >= extent.z
                                     ? 0
                                     : (extent.y >= extent.z ? 1 : 2);
                std::stable_sort(
                    triangle_order.begin() + begin, triangle_order.begin() + end,
                    [&](std::uint32_t first, std::uint32_t second) {
                        const float first_value = coordinate(centroid(first), axis);
                        const float second_value = coordinate(centroid(second), axis);
                        return first_value < second_value ||
                               (first_value == second_value && first < second);
                    });
                const std::uint32_t middle = begin + (end - begin) / 2U;
                const std::uint32_t left = build(begin, middle);
                const std::uint32_t right = build(middle, end);
                bvh_nodes[node_index].left = left;
                bvh_nodes[node_index].right = right;
                return node_index;
            };
        (void)build(0U, triangle_count);
        reordered_indices.resize(triangle_indices.size);
        for (std::uint32_t triangle = 0; triangle < triangle_count; ++triangle) {
            const std::uint32_t source = triangle_order[triangle] * 3U;
            const std::uint32_t destination = triangle * 3U;
            reordered_indices[destination] = host_indices[source];
            reordered_indices[destination + 1U] = host_indices[source + 1U];
            reordered_indices[destination + 2U] = host_indices[source + 2U];
        }
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to build triangle mesh acceleration data");
    }

    std::vector<std::uint32_t> bvh_leaves;
    std::vector<CollisionPlane> solid_planes;
    std::vector<CollisionPlane> unique_solid_planes;
    std::vector<Vec3> shell_normals;
    try {
        solid_planes = closed_convex_planes(host_vertices.data(),
            static_cast<std::uint32_t>(vertices.size), reordered_indices);
        unique_solid_planes = detail::unique_convex_planes(solid_planes);
        shell_normals = closed_shell_normals(host_vertices.data(),
            static_cast<std::uint32_t>(vertices.size), reordered_indices);
        for (std::uint32_t index = 0U; index < bvh_nodes.size(); ++index) {
            if (bvh_nodes[index].triangle_count != 0U) {
                bvh_leaves.push_back(index);
            }
        }
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to index triangle mesh BVH leaves");
    }
    if (bvh_nodes.size() > owned.plan.bvh_node_capacity ||
        bvh_leaves.size() > owned.plan.bvh_leaf_capacity ||
        solid_planes.size() + unique_solid_planes.size() >
            owned.plan.solid_plane_capacity ||
        shell_normals.size() > owned.plan.shell_normal_capacity)
        return failure(StatusCode::internal_error,
                       "triangle geometry plan capacity was exceeded");
    std::copy(reordered_indices.begin(), reordered_indices.end(), owned_indices);
    std::copy(bvh_nodes.begin(), bvh_nodes.end(), owned_bvh_nodes);
    std::copy(bvh_leaves.begin(), bvh_leaves.end(), owned_bvh_leaves);
    if (!solid_planes.empty())
        std::copy(solid_planes.begin(), solid_planes.end(), owned_solid_planes);
    if (!unique_solid_planes.empty())
        std::copy(unique_solid_planes.begin(), unique_solid_planes.end(),
                  owned_solid_planes + solid_planes.size());
    else if (solid_planes.empty())
        release_managed(owned.plan.solid_planes);

    if (!shell_normals.empty())
        std::copy(shell_normals.begin(), shell_normals.end(), owned_shell_normals);
    else
        release_managed(owned.plan.shell_normals);

    TriangleMeshResource &mesh = impl_->meshes[slot];
    mesh.vertices = owned_vertices;
    mesh.indices = owned_indices;
    mesh.vertex_count = static_cast<std::uint32_t>(vertices.size);
    mesh.index_count = static_cast<std::uint32_t>(triangle_indices.size);
    mesh.minimum = minimum;
    mesh.maximum = maximum;
    mesh.bounding_center = multiply(add(minimum, maximum), 0.5F);
    float squared_radius = 0.0F;
    for (std::uint64_t index = 0U; index < vertices.size; ++index) {
        squared_radius = fmaxf(
            squared_radius,
            length_squared(subtract(owned_vertices[index],
                                    mesh.bounding_center)));
    }
    mesh.bounding_radius = sqrtf(squared_radius);
    const Vec3 half_extents = multiply(subtract(maximum, minimum), 0.5F);
    mesh.unit_inertia = {
        fmaxf((half_extents.y * half_extents.y +
               half_extents.z * half_extents.z) /
                  3.0F,
              k_epsilon),
        fmaxf((half_extents.x * half_extents.x +
               half_extents.z * half_extents.z) /
                  3.0F,
              k_epsilon),
        fmaxf((half_extents.x * half_extents.x +
               half_extents.y * half_extents.y) /
                  3.0F,
              k_epsilon)};
    mesh.bvh_nodes = owned_bvh_nodes;
    mesh.bvh_node_count = static_cast<std::uint32_t>(bvh_nodes.size());
    mesh.bvh_leaves = owned_bvh_leaves;
    mesh.bvh_leaf_count = static_cast<std::uint32_t>(bvh_leaves.size());
    mesh.solid_planes = owned.plan.solid_planes;
    mesh.solid_unique_plane_count = static_cast<std::uint32_t>(unique_solid_planes.size());
    mesh.shell_normals = owned.plan.shell_normals;
    owned.disarm();
    mesh.alive = true;
    ++impl_->triangle_mesh_count;
    ++impl_->revision;
    output = {slot, mesh.generation};
    // Host preparation rewrites the complete index order, not isolated bytes.
    // Publish its ranges explicitly before any graphics-backend dispatch.
    status = write_host_span(impl_->execution, owned_indices, triangle_indices.size);
    if (!status) return status;
    return write_host_objects(impl_->execution, &mesh);
}
Status SimulationRuntime::remove_triangle_mesh(TriangleMeshId mesh_id) noexcept {
    if (!impl_) {
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    }
    Status status = impl_->require_idle();
    if (!status) {
        return status;
    }
    status = impl_->validate_handle(mesh_id);
    if (!status) {
        return status;
    }
    for (std::uint32_t index = 0; index < impl_->rigid_body_count; ++index) {
        if (impl_->parameters[index].mesh == mesh_id) {
            return failure(StatusCode::invalid_argument,
                           "triangle mesh is still referenced by a rigid body");
        }
    }
    for (std::uint32_t index = 0;
         index < impl_->options.paint_field_capacity; ++index)
        if (impl_->paint_fields[index].alive &&
            impl_->paint_fields[index].options.mesh == mesh_id)
            return failure(StatusCode::invalid_argument,
                           "triangle mesh is still referenced by a paint field");
    TriangleMeshResource &mesh = impl_->meshes[mesh_id.index];
    release_managed(mesh.bvh_leaves);
    release_managed(mesh.bvh_nodes);
    release_managed(mesh.solid_planes);
    release_managed(mesh.shell_normals);
    release_managed(mesh.indices);
    release_managed(mesh.vertices);
    mesh.vertex_count = 0U;
    mesh.index_count = 0U;
    mesh.bvh_node_count = 0U;
    mesh.bvh_leaf_count = 0U;
    mesh.alive = false;
    ++mesh.generation;
    if (mesh.generation == 0U) {
        mesh.generation = 1U;
    }
    --impl_->triangle_mesh_count;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::add_paint_field(PaintFieldOptions options,
                              PaintFieldId &output,
                              device::ExecutionQueue queue) noexcept {
    const device::ExecutionQueue stream = queue;
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    const bool cloth_target = options.cloth.generation != 0U;
    std::uint32_t vertex_count = 0U;
    if (cloth_target) {
        if (options.body.generation != 0U || options.mesh.generation != 0U ||
            options.cloth.index >= impl_->cloths.size() ||
            !impl_->cloths[options.cloth.index] ||
            !impl_->cloths[options.cloth.index]->alive ||
            impl_->cloths[options.cloth.index]->generation != options.cloth.generation)
            return failure(StatusCode::invalid_handle, "paint cloth target is stale");
        const auto &cloth = *impl_->cloths[options.cloth.index];
        vertex_count = cloth.source_indices
            ? static_cast<std::uint32_t>(cloth.source_inverse_masses.size())
            : cloth.vertex_count;
    } else {
        std::uint32_t dense = 0U;
        if (!(status = impl_->validate_handle(options.body, dense)) ||
            !(status = impl_->validate_handle(options.mesh))) return status;
        vertex_count = impl_->meshes[options.mesh.index].vertex_count;
    }
    if (options.vertex_uvs.data == nullptr ||
        options.vertex_uvs.size != vertex_count ||
        options.width == 0U || options.height == 0U ||
        options.width > 4096U || options.height > 4096U ||
        static_cast<std::uint64_t>(options.width) * options.height >
            16'777'216U)
        return failure(StatusCode::invalid_argument,
                       "paint field UV count or dimensions are invalid");
    std::uint32_t slot = 0U;
    for (; slot < impl_->options.paint_field_capacity; ++slot)
        if (!impl_->paint_fields[slot].alive) break;
    if (slot == impl_->options.paint_field_capacity)
        return failure(StatusCode::capacity_exceeded,
                       "paint field capacity exhausted");
    Vec2 *uvs = nullptr;
    std::uint32_t *pixels = nullptr;
    status = allocate_managed(uvs, vertex_count);
    if (!status) return status;
    status = allocate_managed(pixels,
        static_cast<std::size_t>(options.width) * options.height);
    if (!status) { release_managed(uvs); return status; }
    PmResult error = copy_bytes(uvs, options.vertex_uvs.data,
        vertex_count * sizeof(Vec2), device::CopyDirection::automatic, stream);
    if (error == PM_SUCCESS)
        error = clear_bytes(pixels, 0,
            static_cast<std::size_t>(options.width) * options.height *
                sizeof(std::uint32_t), stream);
    if (error == PM_SUCCESS) error = wait_queue(stream);
    if (error != PM_SUCCESS) {
        release_managed(pixels);
        release_managed(uvs);
        return backend_failure(error, "paint field upload failed");
    }
    for (std::uint32_t i = 0; i < vertex_count; ++i)
        if (!finite(uvs[i].x) || !finite(uvs[i].y)) {
            release_managed(pixels);
            release_managed(uvs);
            return failure(StatusCode::invalid_argument,
                           "paint field contains non-finite UVs");
        }
    PaintFieldResource &field = impl_->paint_fields[slot];
    options.vertex_uvs = {uvs, vertex_count};
    field.options = options;
    field.uvs = uvs;
    field.pixels = pixels;
    field.alive = true;
    output = {slot, field.generation};
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_paint_field(PaintFieldId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->options.paint_field_capacity ||
        !impl_->paint_fields[id.index].alive ||
        impl_->paint_fields[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "paint field handle is stale");
    for (std::uint32_t i = 0; i < impl_->options.paint_rule_capacity; ++i)
        if (impl_->paint_rules[i].alive &&
            impl_->paint_rules[i].options.target == id)
            return failure(StatusCode::invalid_argument,
                           "paint field is still referenced by a paint rule");
    PaintFieldResource &field = impl_->paint_fields[id.index];
    release_managed(field.pixels);
    release_managed(field.uvs);
    field.alive = false;
    ++field.generation;
    if (field.generation == 0U) field.generation = 1U;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::clear_paint_field(
    PaintFieldId id, device::ExecutionQueue queue) noexcept {
    const device::ExecutionQueue stream = queue;
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->options.paint_field_capacity ||
        !impl_->paint_fields[id.index].alive ||
        impl_->paint_fields[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "paint field handle is stale");
    const PaintFieldResource &field = impl_->paint_fields[id.index];
    PmResult error = clear_bytes(field.pixels, 0,
        static_cast<std::size_t>(field.options.width) *
        field.options.height * sizeof(std::uint32_t), stream);
    if (error == PM_SUCCESS) error = wait_queue(stream);
    if (error != PM_SUCCESS)
        return backend_failure(error, "paint field clear failed");
    return success();
}

Status SimulationRuntime::paint_field_view(PaintFieldId id,
                               PaintFieldDeviceView &output) const noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->options.paint_field_capacity ||
        impl_->paint_fields == nullptr)
        return failure(StatusCode::invalid_handle, "paint field handle is stale");
    PaintFieldResource field{};
    const PmResult error = copy_bytes_wait(
        &field, &impl_->paint_fields[id.index], sizeof(field),
        device::CopyDirection::device_to_host, impl_->execution);
    if (error != PM_SUCCESS)
        return backend_failure(error, "paint field view readback failed");
    if (!field.alive || field.generation != id.generation)
        return failure(StatusCode::invalid_handle, "paint field handle is stale");
    output = {{field.pixels, static_cast<std::uint64_t>(field.options.width) *
                                 field.options.height},
              field.options.width, field.options.height, impl_->revision};
    return success();
}

Status SimulationRuntime::add_paint_rule(PaintRuleOptions options,
                             PaintRuleId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    const bool fluid_source = options.source.generation != 0U;
    const bool rigid_source = options.rigid_source.generation != 0U;
    if (fluid_source == rigid_source)
        return failure(StatusCode::invalid_argument,
                       "paint rule needs exactly one source");
    if (fluid_source) {
        FluidStorage *fluid = nullptr;
        if (!(status = impl_->validate_handle(options.source, fluid))) return status;
    } else {
        std::uint32_t dense = 0U;
        if (!(status = impl_->validate_handle(options.rigid_source, dense)))
            return status;
    }
    if (options.target.index >= impl_->options.paint_field_capacity ||
        !impl_->paint_fields[options.target.index].alive ||
        impl_->paint_fields[options.target.index].generation !=
            options.target.generation)
        return failure(StatusCode::invalid_handle, "paint target field is stale");
    const PaintFieldResource &target =
        impl_->paint_fields[options.target.index];
    if ((fluid_source && target.options.cloth.generation != 0U) ||
        (rigid_source && target.options.cloth.generation == 0U))
        return failure(StatusCode::not_supported,
                       "paint source and target systems do not match");
    if (!finite(options.reach) || options.reach < 0.0F ||
        options.reach > 10.0F)
        return failure(StatusCode::invalid_argument, "paint reach is invalid");
    if (!finite(options.brush_radius) || options.brush_radius <= 0.0F ||
        options.brush_radius > 10.0F)
        return failure(StatusCode::invalid_argument,
                       "paint brush radius is invalid");
    std::uint32_t slot = 0U;
    for (; slot < impl_->options.paint_rule_capacity; ++slot)
        if (!impl_->paint_rules[slot].alive) break;
    if (slot == impl_->options.paint_rule_capacity)
        return failure(StatusCode::capacity_exceeded,
                       "paint rule capacity exhausted");
    PaintRuleResource &rule = impl_->paint_rules[slot];
    rule.options = options;
    rule.alive = true;
    ++impl_->paint_rule_count;
    output = {slot, rule.generation};
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_paint_rule(PaintRuleId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->options.paint_rule_capacity ||
        !impl_->paint_rules[id.index].alive ||
        impl_->paint_rules[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "paint rule handle is stale");
    PaintRuleResource &rule = impl_->paint_rules[id.index];
    rule.alive = false;
    --impl_->paint_rule_count;
    ++rule.generation;
    if (rule.generation == 0U) rule.generation = 1U;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::add_cloth(ClothOptions options, ClothId &output,
                        device::ExecutionQueue queue) noexcept {
    (void)queue;
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.vertices.data == nullptr || options.vertices.size < 3U ||
        options.vertices.size > UINT32_MAX ||
        options.triangle_indices.data == nullptr ||
        options.triangle_indices.size == 0U ||
        options.triangle_indices.size % 3U != 0U ||
        options.triangle_indices.size > UINT32_MAX ||
        (options.inverse_masses.size != 0U &&
         (options.inverse_masses.data == nullptr ||
          options.inverse_masses.size != options.vertices.size)) ||
        !finite(options.vertex_mass) || options.vertex_mass <= 0.0F ||
        !finite(options.thickness) || options.thickness <= 0.0F ||
        !finite(options.stretch_compliance) || options.stretch_compliance < 0.0F ||
        !finite(options.bending_compliance) || options.bending_compliance < 0.0F ||
        !finite(options.velocity_damping) || options.velocity_damping < 0.0F ||
        !finite(options.contact_friction) || options.contact_friction < 0.0F ||
        !finite(options.break_strain) || options.break_strain < 0.0F ||
        options.break_strain > 9.0F ||
        !finite(options.impact_break_impulse) ||
        options.impact_break_impulse < 0.0F ||
        !finite(options.target_volume) || options.target_volume < 0.0F ||
        !finite(options.volume_compliance) ||
        options.volume_compliance < 0.0F ||
        (options.preserve_volume &&
         (options.break_strain > 0.0F || options.impact_break_impulse > 0.0F)) ||
        options.fracture_persistence_substeps == 0U ||
        options.fracture_persistence_substeps > 64U ||
        options.solver_iterations == 0U || options.solver_iterations > 64U) {
        return failure(StatusCode::invalid_argument, "invalid cloth geometry or solver options");
    }
    for (std::uint64_t vertex = 0U; vertex < options.vertices.size; ++vertex) {
        if (!finite(options.vertices.data[vertex]) ||
            (options.inverse_masses.size != 0U &&
             (!finite(options.inverse_masses.data[vertex]) ||
              options.inverse_masses.data[vertex] < 0.0F)))
            return failure(StatusCode::invalid_argument, "invalid cloth vertex or inverse mass");
    }
    std::uint32_t slot = k_invalid_dense;
    for (std::uint32_t index = 0U; index < impl_->cloths.size(); ++index) {
        if (!impl_->cloths[index] || !impl_->cloths[index]->alive) {
            slot = index;
            break;
        }
    }
    if (slot == k_invalid_dense)
        return failure(StatusCode::capacity_exceeded, "cloth capacity is exhausted");

    const std::uint32_t count = static_cast<std::uint32_t>(options.vertices.size);
    const bool fracture_enabled = options.break_strain > 0.0F ||
        options.impact_break_impulse > 0.0F;
    if (fracture_enabled && options.vertices.size + options.triangle_indices.size > UINT32_MAX)
        return failure(StatusCode::capacity_exceeded, "cloth split capacity exceeds uint32 range");
    const auto capacity = count + (fracture_enabled
        ? static_cast<std::uint32_t>(options.triangle_indices.size) : 0U);
    std::vector<std::vector<DeformableNeighbor>> adjacency;
    std::vector<std::uint32_t> offsets;
    std::vector<DeformableNeighbor> neighbors;
    std::vector<ClothBond> bonds;
    std::vector<Vec3> surface_rest;
    std::vector<std::uint32_t> surface_indices;
    std::vector<std::uint32_t> triangle_bonds;
    std::vector<std::uint32_t> volume_corner_offsets;
    std::vector<std::uint32_t> volume_corner_indices;
    float initial_signed_volume = 0.0F;
    try {
        adjacency.resize(count);
        std::unordered_map<std::uint64_t, std::uint32_t> bond_ids;
        std::unordered_map<std::uint64_t, std::uint32_t> opposite;
        std::unordered_map<std::uint64_t, std::uint32_t> edge_counts;
        const auto key = [](std::uint32_t a, std::uint32_t b) {
            if (a > b) std::swap(a, b);
            return (static_cast<std::uint64_t>(a) << 32U) | b;
        };
        const auto link = [&](std::uint32_t a, std::uint32_t b,
                              float compliance, bool bending) {
            if (a == b || bond_ids.find(key(a, b)) != bond_ids.end()) return;
            const float rest = vector_length(subtract(
                options.vertices.data[a], options.vertices.data[b]));
            if (rest <= 1.0e-6F) return;
            const auto id = static_cast<std::uint32_t>(bonds.size());
            bond_ids.emplace(key(a, b), id);
            bonds.push_back({a, b, rest, bending});
            adjacency[a].push_back({b, rest, compliance, id});
            adjacency[b].push_back({a, rest, compliance, id});
        };
        for (std::uint64_t triangle = 0U;
             triangle < options.triangle_indices.size; triangle += 3U) {
            const std::uint32_t a = options.triangle_indices.data[triangle];
            const std::uint32_t b = options.triangle_indices.data[triangle + 1U];
            const std::uint32_t c = options.triangle_indices.data[triangle + 2U];
            if (a >= count || b >= count || c >= count || a == b || b == c || c == a)
                return failure(StatusCode::invalid_argument, "invalid cloth triangle index");
            if (vector_length(subtract(options.vertices.data[a],
                                       options.vertices.data[b])) <= 1.0e-6F ||
                vector_length(subtract(options.vertices.data[b],
                                       options.vertices.data[c])) <= 1.0e-6F ||
                vector_length(subtract(options.vertices.data[c],
                                       options.vertices.data[a])) <= 1.0e-6F)
                return failure(StatusCode::invalid_argument,
                               "cloth triangle has a zero-length edge");
            initial_signed_volume += dot(options.vertices.data[a], cross(
                options.vertices.data[b], options.vertices.data[c])) / 6.0F;
            const std::array<std::array<std::uint32_t, 3>, 3> edges{{
                {a, b, c}, {b, c, a}, {c, a, b}}};
            for (const auto &edge : edges) {
                const std::uint64_t edge_key = key(edge[0], edge[1]);
                ++edge_counts[edge_key];
                const auto previous = opposite.find(edge_key);
                if (previous == opposite.end()) {
                    opposite.emplace(edge_key, edge[2]);
                    link(edge[0], edge[1], options.stretch_compliance, false);
                } else {
                    link(previous->second, edge[2], options.bending_compliance,
                         true);
                }
            }
        }
        if (options.preserve_volume) {
            if (fabsf(initial_signed_volume) <= 1.0e-8F ||
                std::any_of(edge_counts.begin(), edge_counts.end(),
                    [](const auto &edge) { return edge.second != 2U; }))
                return failure(StatusCode::invalid_argument,
                    "volume-preserving cloth must be a closed manifold");
            std::vector<std::vector<std::uint32_t>> corners(count);
            for (std::uint64_t corner = 0U;
                 corner < options.triangle_indices.size; ++corner)
                corners[options.triangle_indices.data[corner]].push_back(
                    static_cast<std::uint32_t>(corner));
            volume_corner_offsets.reserve(static_cast<std::size_t>(count) + 1U);
            volume_corner_indices.reserve(options.triangle_indices.size);
            volume_corner_offsets.push_back(0U);
            for (const auto &list : corners) {
                volume_corner_indices.insert(volume_corner_indices.end(),
                    list.begin(), list.end());
                volume_corner_offsets.push_back(
                    static_cast<std::uint32_t>(volume_corner_indices.size()));
            }
        }
        for (std::uint64_t triangle = 0U;
             triangle < options.triangle_indices.size; triangle += 3U) {
            const std::uint32_t corners[3]{
                options.triangle_indices.data[triangle],
                options.triangle_indices.data[triangle + 1U],
                options.triangle_indices.data[triangle + 2U]};
            for (std::uint32_t corner = 0U; corner < 3U; ++corner) {
                surface_rest.push_back(options.vertices.data[corners[corner]]);
                surface_indices.push_back(static_cast<std::uint32_t>(triangle + corner));
                triangle_bonds.push_back(bond_ids.at(key(corners[corner],
                    corners[(corner + 1U) % 3U])));
            }
        }
        offsets.reserve(static_cast<std::size_t>(count) + 1U);
        offsets.push_back(0U);
        for (const auto &list : adjacency) {
            if (neighbors.size() + list.size() > UINT32_MAX)
                return failure(StatusCode::capacity_exceeded, "cloth links exceed uint32 range");
            neighbors.insert(neighbors.end(), list.begin(), list.end());
            offsets.push_back(static_cast<std::uint32_t>(neighbors.size()));
        }
    } catch (...) {
        return failure(StatusCode::out_of_memory, "failed to build cloth links");
    }
    std::unique_ptr<ClothStorage> cloth;
    try { cloth = std::make_unique<ClothStorage>(); }
    catch (...) { return failure(StatusCode::out_of_memory, "failed to allocate cloth"); }
    cloth->generation = impl_->cloths[slot] ? impl_->cloths[slot]->generation : 1U;
    cloth->vertex_count = count;
    cloth->vertex_capacity = capacity;
    cloth->stretch_compliance = options.stretch_compliance;
    cloth->bending_compliance = options.bending_compliance;
    cloth->index_count = static_cast<std::uint32_t>(options.triangle_indices.size);
    cloth->neighbor_count = static_cast<std::uint32_t>(neighbors.size());
    cloth->neighbor_capacity = fracture_enabled
        ? 2U * (options.triangle_indices.size + bonds.size()) : neighbors.size();
    if (cloth->neighbor_capacity > UINT32_MAX)
        return failure(StatusCode::capacity_exceeded, "cloth split links exceed uint32 range");
    if (bonds.size() > UINT32_MAX)
        return failure(StatusCode::capacity_exceeded, "cloth bonds exceed uint32 range");
    cloth->bond_count = static_cast<std::uint32_t>(bonds.size());
    cloth->thickness = options.thickness;
    cloth->velocity_damping = options.velocity_damping;
    cloth->contact_friction = options.contact_friction;
    cloth->break_strain = options.break_strain;
    cloth->fracture_persistence_substeps =
        options.fracture_persistence_substeps;
    cloth->impact_break_impulse = options.impact_break_impulse;
    cloth->solver_iterations = options.solver_iterations;
    cloth->preserve_volume = options.preserve_volume;
    cloth->target_volume = options.target_volume > 0.0F
        ? options.target_volume : fabsf(initial_signed_volume);
    cloth->volume_compliance = options.volume_compliance;
    cloth->orientation = initial_signed_volume < 0.0F ? -1.0F : 1.0F;
    StorageAllocator allocator;
    status = simulation::visit_cloth_storage(*cloth, impl_->options.rigid_body_capacity,
        fracture_enabled, options.preserve_volume,
        impl_->options.rope_cloth_coupling_capacity != 0,
        impl_->options.fluid_cloth_coupling_capacity != 0,
        impl_->options.smoke_cloth_coupling_capacity != 0, allocator);
    if (!status) return status;
    if (options.preserve_volume) {
        *cloth->volume_lambda = 0.0F;
        std::copy(volume_corner_offsets.begin(), volume_corner_offsets.end(), cloth->volume_corner_offsets);
        std::copy(volume_corner_indices.begin(), volume_corner_indices.end(), cloth->volume_corner_indices);
    }
    for (std::uint32_t index = 0U; index < count; ++index) {
        cloth->positions[index] = options.vertices.data[index];
        cloth->scratch[index] = options.vertices.data[index];
        cloth->previous[index] = options.vertices.data[index];
        cloth->velocities[index] = {};
        cloth->rigid_contact_forces[index] = {};
        cloth->soft_body_forces[index] = {};
        if (cloth->rope_forces != nullptr) cloth->rope_forces[index] = {};
        if (cloth->fluid_forces != nullptr) cloth->fluid_forces[index] = {};
        if (cloth->smoke_forces != nullptr) cloth->smoke_forces[index] = {};
        cloth->inverse_masses[index] = options.inverse_masses.size != 0U
            ? options.inverse_masses.data[index] : 1.0F / options.vertex_mass;
    }
    std::copy_n(options.triangle_indices.data, cloth->index_count, cloth->indices);
    if (fracture_enabled) {
        std::copy_n(options.triangle_indices.data, cloth->index_count, cloth->source_indices);
        for (std::uint32_t i = 0; i < count; ++i) cloth->vertex_sources[i] = i;
        std::copy(surface_rest.begin(), surface_rest.end(), cloth->surface_positions);
        std::copy(surface_indices.begin(), surface_indices.end(),
                  cloth->surface_triangle_indices);
        std::copy(triangle_bonds.begin(), triangle_bonds.end(),
                  cloth->triangle_bonds);
    }
    std::copy(bonds.begin(), bonds.end(), cloth->bonds);
    std::fill_n(cloth->bond_active, bonds.size(), std::uint8_t{1U});
    std::fill_n(cloth->bond_damage, bonds.size(), std::uint8_t{0U});
    std::copy(offsets.begin(), offsets.end(), cloth->offsets);
    std::copy(neighbors.begin(), neighbors.end(), cloth->neighbors);
    *cloth->count = count;
    if (fracture_enabled) {
        try {
            cloth->source_inverse_masses.assign(cloth->inverse_masses, cloth->inverse_masses + count);
            cloth->source_degrees.resize(count);
            cloth->bond_corners.resize(bonds.size());
            std::unordered_map<std::uint64_t, std::uint32_t> ids, first_corners;
            const auto key = [](std::uint32_t a, std::uint32_t b) {
                return (static_cast<std::uint64_t>(std::min(a,b)) << 32U) | std::max(a,b);
            };
            for (std::uint32_t i = 0; i < bonds.size(); ++i)
                ids[key(bonds[i].first, bonds[i].second)] = i;
            for (std::uint32_t c = 0; c < cloth->index_count; ++c) {
                ++cloth->source_degrees[cloth->source_indices[c]];
                const auto next = c / 3U * 3U + (c + 1U) % 3U;
                const auto other = c / 3U * 3U + (c + 2U) % 3U;
                const auto a = cloth->source_indices[c], b = cloth->source_indices[next];
                const auto edge_key = key(a,b);
                auto [it, fresh] = first_corners.emplace(edge_key,c);
                if (fresh) cloth->bond_corners[ids.at(edge_key)] = {c,next};
                else {
                    const auto previous = it->second;
                    const auto previous_next = previous / 3U * 3U + (previous + 1U) % 3U;
                    const auto opposite = previous / 3U * 3U + (previous + 2U) % 3U;
                    const auto bending = ids.find(key(cloth->source_indices[opposite],
                                                     cloth->source_indices[other]));
                    const bool has_bend = bending != ids.end() && bonds[bending->second].bending;
                    if (has_bend) cloth->bond_corners[bending->second] = {opposite,other};
                    const bool same = cloth->source_indices[previous] == a;
                    cloth->seams.push_back({{previous,previous_next,
                        same ? c : next, same ? next : c}, ids.at(edge_key),
                        has_bend ? bending->second : k_invalid_dense});
                }
            }
        } catch (...) { return failure(StatusCode::out_of_memory, "failed to build cloth seams"); }
        status = rebuild_cloth_topology(*cloth, impl_->execution, true);
        if (!status) return status;
    }
    cloth->alive = true;
    output = {slot, cloth->generation};
    impl_->cloths[slot] = std::move(cloth);
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_cloth(ClothId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->cloths.size() || !impl_->cloths[id.index] ||
        !impl_->cloths[id.index]->alive ||
        impl_->cloths[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "cloth handle is stale");
    for (std::uint32_t field = 0U;
         field < impl_->options.paint_field_capacity; ++field)
        if (impl_->paint_fields[field].alive &&
            impl_->paint_fields[field].options.cloth == id)
            return failure(StatusCode::invalid_argument,
                           "cloth is still referenced by a paint field");
    for (const FluidClothCouplingResource &coupling :
         impl_->fluid_cloth_couplings)
        if (coupling.alive && coupling.options.cloth == id)
            return failure(StatusCode::invalid_argument,
                "cloth is still referenced by a fluid coupling");
    for (const auto &coupling : impl_->rope_cloth_couplings)
        if (coupling.alive && coupling.options.cloth == id)
            return failure(StatusCode::invalid_argument,
                "cloth is still referenced by a rope coupling");
    ClothStorage &cloth = *impl_->cloths[id.index];
    for (const auto &coupling : impl_->soft_cloth_couplings)
        if (coupling && coupling->alive && coupling->options.cloth == id)
            return failure(StatusCode::invalid_argument,
                           "cloth is still referenced by a soft-body coupling");
    for (const auto &coupling : impl_->smoke_cloth_couplings)
        if (coupling && coupling->alive && coupling->options.cloth == id)
            return failure(StatusCode::invalid_argument,
                           "cloth is still referenced by a smoke coupling");
    cloth.alive = false;
    cloth.release();
    ++cloth.generation;
    if (cloth.generation == 0U) cloth.generation = 1U;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::cloth_view(ClothId id, ClothDeviceView &output) const noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (id.index >= impl_->cloths.size() || !impl_->cloths[id.index] ||
        !impl_->cloths[id.index]->alive ||
        impl_->cloths[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "cloth handle is stale");
    const ClothStorage &cloth = *impl_->cloths[id.index];
    output.positions = {cloth.positions, cloth.vertex_count};
    output.velocities = {cloth.velocities, cloth.vertex_count};
    output.triangle_indices = {cloth.indices, cloth.index_count};
    output.vertex_count = cloth.vertex_count;
    if (cloth.surface_positions != nullptr) {
        output.surface_positions = {cloth.surface_positions, cloth.index_count};
        output.surface_triangle_indices = {
            cloth.surface_triangle_indices, cloth.index_count};
        output.surface_source_indices = {cloth.source_indices, cloth.index_count};
        output.vertex_source_indices = {cloth.vertex_sources, cloth.vertex_count};
        output.inverse_masses = {cloth.inverse_masses, cloth.vertex_count};
    }
    output.bonds = {cloth.bonds, cloth.bond_count};
    output.active_bonds = {cloth.bond_active, cloth.bond_count};
    output.rigid_contact_forces = {
        cloth.rigid_contact_forces, cloth.vertex_count};
    output.soft_body_contact_forces = {cloth.soft_body_forces, cloth.vertex_count};
    if (cloth.rope_forces != nullptr)
        output.rope_contact_forces = {cloth.rope_forces, cloth.vertex_count};
    if (cloth.fluid_forces != nullptr)
        output.fluid_contact_forces = {
            cloth.fluid_forces, cloth.vertex_count};
    return success();
}

Status SimulationRuntime::add_soft_body(SoftBodyOptions options, SoftBodyId &output,
                            device::ExecutionQueue queue) noexcept {
    (void)queue;
    if (!impl_)
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (options.nodes.data == nullptr || options.nodes.size < 4U ||
        options.nodes.size > UINT32_MAX || options.bonds.data == nullptr ||
        options.bonds.size == 0U || options.bonds.size > UINT32_MAX ||
        options.surface_vertices.data == nullptr ||
        options.surface_vertices.size < 3U ||
        options.surface_vertices.size > UINT32_MAX ||
        options.surface_triangle_indices.data == nullptr ||
        options.surface_triangle_indices.size == 0U ||
        options.surface_triangle_indices.size % 3U != 0U ||
        options.surface_triangle_indices.size > UINT32_MAX ||
        options.surface_bindings.data == nullptr ||
        options.surface_bindings.size != options.surface_vertices.size ||
        (options.inverse_masses.size != 0U &&
         (options.inverse_masses.data == nullptr ||
          options.inverse_masses.size != options.nodes.size)) ||
        !finite(options.node_mass) || options.node_mass <= 0.0F ||
        !finite(options.node_radius) || options.node_radius <= 0.0F ||
        !finite(options.stretch_compliance) ||
        options.stretch_compliance < 0.0F ||
        !finite(options.velocity_damping) || options.velocity_damping < 0.0F ||
        !finite(options.spring_damping) || options.spring_damping < 0.0F ||
        options.spring_damping > 1.0F ||
        !finite(options.contact_friction) || options.contact_friction < 0.0F ||
        !finite(options.shape_matching_stiffness) ||
        options.shape_matching_stiffness < 0.0F ||
        options.shape_matching_stiffness > 1.0F ||
        !finite(options.maximum_projection_fraction) ||
        options.maximum_projection_fraction <= 0.0F ||
        options.maximum_projection_fraction > 1.0F ||
        !finite(options.constraint_velocity_response) ||
        options.constraint_velocity_response < 0.0F ||
        options.constraint_velocity_response > 1.0F ||
        !finite(options.maximum_speed) || options.maximum_speed <= 0.0F ||
        options.solver_iterations == 0U || options.solver_iterations > 64U) {
        return failure(StatusCode::invalid_argument,
                       "invalid soft-body geometry or solver options");
    }
    const std::uint32_t node_count =
        static_cast<std::uint32_t>(options.nodes.size);
    float movable_mass = 0.0F;
    for (std::uint32_t node = 0U; node < node_count; ++node) {
        if (!finite(options.nodes.data[node]) ||
            (options.inverse_masses.size != 0U &&
             (!finite(options.inverse_masses.data[node]) ||
              options.inverse_masses.data[node] < 0.0F)))
            return failure(StatusCode::invalid_argument,
                           "invalid soft-body node or inverse mass");
        const float inverse_mass = options.inverse_masses.size != 0U
            ? options.inverse_masses.data[node] : 1.0F / options.node_mass;
        if (inverse_mass > 0.0F) movable_mass += 1.0F / inverse_mass;
    }
    if (!finite(movable_mass) || movable_mass <= 0.0F)
        return failure(StatusCode::invalid_argument,
                       "soft body needs at least one movable node");
    for (std::uint64_t index = 0U;
         index < options.surface_triangle_indices.size; ++index) {
        if (options.surface_triangle_indices.data[index] >=
            options.surface_vertices.size)
            return failure(StatusCode::invalid_argument,
                           "soft-body surface index is out of range");
    }
    for (std::uint64_t vertex = 0U;
         vertex < options.surface_vertices.size; ++vertex) {
        if (!finite(options.surface_vertices.data[vertex]))
            return failure(StatusCode::invalid_argument,
                           "soft-body surface vertex is invalid");
        const SoftBodySurfaceBinding &binding =
            options.surface_bindings.data[vertex];
        float weight_sum = 0.0F;
        for (std::uint32_t slot = 0U; slot < 4U; ++slot) {
            if (binding.nodes[slot] >= node_count ||
                !finite(binding.weights[slot]) || binding.weights[slot] < 0.0F)
                return failure(StatusCode::invalid_argument,
                               "soft-body surface binding is invalid");
            weight_sum += binding.weights[slot];
        }
        if (fabsf(weight_sum - 1.0F) > 1.0e-4F)
            return failure(StatusCode::invalid_argument,
                           "soft-body surface binding weights must sum to one");
    }
    std::uint32_t slot = k_invalid_dense;
    for (std::uint32_t index = 0U; index < impl_->soft_bodies.size(); ++index) {
        if (!impl_->soft_bodies[index] || !impl_->soft_bodies[index]->alive) {
            slot = index;
            break;
        }
    }
    if (slot == k_invalid_dense)
        return failure(StatusCode::capacity_exceeded,
                       "soft-body capacity is exhausted");

    std::vector<std::vector<DeformableNeighbor>> adjacency;
    std::vector<std::uint32_t> offsets;
    std::vector<DeformableNeighbor> neighbors;
    std::vector<SoftBodyNeighbor> warp_neighbors;
    std::vector<float> minimum_rest_lengths;
    float minimum_bond_length = FLT_MAX;
    try {
        adjacency.resize(node_count);
        std::unordered_set<std::uint64_t> unique;
        for (std::uint32_t bond_index = 0U;
             bond_index < options.bonds.size; ++bond_index) {
            const SoftBodyBond bond = options.bonds.data[bond_index];
            if (bond.first >= node_count || bond.second >= node_count ||
                bond.first == bond.second || !finite(bond.rest_length) ||
                bond.rest_length <= 1.0e-6F)
                return failure(StatusCode::invalid_argument,
                               "soft-body bond is invalid");
            const std::uint32_t first = std::min(bond.first, bond.second);
            const std::uint32_t second = std::max(bond.first, bond.second);
            const std::uint64_t key =
                (static_cast<std::uint64_t>(first) << 32U) | second;
            if (!unique.insert(key).second)
                return failure(StatusCode::invalid_argument,
                               "soft-body bond is duplicated");
            minimum_bond_length = std::min(
                minimum_bond_length, bond.rest_length);
            adjacency[bond.first].push_back({bond.second, bond.rest_length,
                                             options.stretch_compliance,
                                             bond_index});
            adjacency[bond.second].push_back({bond.first, bond.rest_length,
                                              options.stretch_compliance,
                                              bond_index});
        }
        offsets.reserve(static_cast<std::size_t>(node_count) + 1U);
        offsets.push_back(0U);
        for (const auto &list : adjacency) {
            if (neighbors.size() + list.size() > UINT32_MAX)
                return failure(StatusCode::capacity_exceeded,
                               "soft-body neighbors exceed uint32 range");
            neighbors.insert(neighbors.end(), list.begin(), list.end());
            offsets.push_back(static_cast<std::uint32_t>(neighbors.size()));
        }
        if (std::any_of(adjacency.begin(), adjacency.end(),
                        [](const auto &list) { return list.empty(); }))
            return failure(StatusCode::invalid_argument,
                           "every soft-body node needs at least one bond");
        std::size_t maximum_degree=0U;
        for(const auto &list:adjacency)
            maximum_degree=std::max(maximum_degree,list.size());
        const std::size_t ell_count=maximum_degree*node_count;
        // The transpose helps dense near-uniform graphs, but sparse hubs
        // should not pay for mostly empty rows or an unbounded extra buffer.
        if(node_count>=128U && maximum_degree>=64U &&
            maximum_degree<=UINT32_MAX/node_count &&
            ell_count<=2U*neighbors.size() && ell_count<=4'194'304U) {
            warp_neighbors.reserve(neighbors.size());
            for(const DeformableNeighbor neighbor:neighbors)
                warp_neighbors.push_back({neighbor.index,neighbor.rest_length});
            minimum_rest_lengths.reserve(node_count);
            for(const auto &list:adjacency) {
                float shortest=FLT_MAX;
                for(const DeformableNeighbor neighbor:list)
                    shortest=std::min(shortest,neighbor.rest_length);
                minimum_rest_lengths.push_back(shortest);
            }
        }
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to build soft-body adjacency");
    }

    Vec3 shape_rest_center{};
    ShapeMatrix shape_inverse_rest{};
    if (options.shape_matching_stiffness > 0.0F) {
        for (std::uint32_t node = 0U; node < node_count; ++node) {
            const float inverse_mass = options.inverse_masses.size != 0U
                ? options.inverse_masses.data[node] : 1.0F / options.node_mass;
            if (inverse_mass > 0.0F)
                shape_rest_center = add(shape_rest_center,
                    multiply(options.nodes.data[node], 1.0F / inverse_mass));
        }
        shape_rest_center = multiply(shape_rest_center, 1.0F / movable_mass);
        ShapeMatrix rest_covariance{};
        for (std::uint32_t node = 0U; node < node_count; ++node) {
            const float inverse_mass = options.inverse_masses.size != 0U
                ? options.inverse_masses.data[node] : 1.0F / options.node_mass;
            if (inverse_mass <= 0.0F) continue;
            const Vec3 rest = subtract(
                options.nodes.data[node], shape_rest_center);
            const float mass = 1.0F / inverse_mass;
            rest_covariance.columns[0] = add(
                rest_covariance.columns[0], multiply(rest, mass * rest.x));
            rest_covariance.columns[1] = add(
                rest_covariance.columns[1], multiply(rest, mass * rest.y));
            rest_covariance.columns[2] = add(
                rest_covariance.columns[2], multiply(rest, mass * rest.z));
        }
        const Vec3 row0 = cross(rest_covariance.columns[1],
                                rest_covariance.columns[2]);
        const Vec3 row1 = cross(rest_covariance.columns[2],
                                rest_covariance.columns[0]);
        const Vec3 row2 = cross(rest_covariance.columns[0],
                                rest_covariance.columns[1]);
        const float determinant = dot(rest_covariance.columns[0], row0);
        if (!finite(determinant) || fabsf(determinant) <= 1.0e-10F)
            return failure(StatusCode::invalid_argument,
                "shape-matched soft body needs a volumetric rest lattice");
        const float inverse_determinant = 1.0F / determinant;
        shape_inverse_rest.columns[0] = multiply(
            {row0.x, row1.x, row2.x}, inverse_determinant);
        shape_inverse_rest.columns[1] = multiply(
            {row0.y, row1.y, row2.y}, inverse_determinant);
        shape_inverse_rest.columns[2] = multiply(
            {row0.z, row1.z, row2.z}, inverse_determinant);
    }

    std::vector<std::uint32_t> surface_offsets;
    std::vector<SoftSurfaceInfluence> surface_influences;
    try {
        std::vector<std::vector<SoftSurfaceInfluence>> per_node(node_count);
        for (std::uint32_t corner = 0U;
             corner < options.surface_triangle_indices.size; ++corner) {
            const SoftBodySurfaceBinding binding = options.surface_bindings.data[
                options.surface_triangle_indices.data[corner]];
            float denominator = 0.0F;
            for (std::uint32_t a = 0U; a < 4U; ++a) {
                const float inverse = options.inverse_masses.size != 0U
                    ? options.inverse_masses.data[binding.nodes[a]]
                    : 1.0F / options.node_mass;
                for (std::uint32_t b = 0U; b < 4U; ++b)
                    if (binding.nodes[a] == binding.nodes[b])
                        denominator += binding.weights[a] * binding.weights[b] * inverse;
            }
            if (denominator <= 0.0F) continue;
            for (std::uint32_t slot = 0U; slot < 4U; ++slot) {
                const std::uint32_t node = binding.nodes[slot];
                bool duplicate = false;
                for (std::uint32_t earlier = 0U; earlier < slot; ++earlier)
                    duplicate |= binding.nodes[earlier] == node;
                if (duplicate) continue;
                float weight = 0.0F;
                for (std::uint32_t other = slot; other < 4U; ++other)
                    if (binding.nodes[other] == node) weight += binding.weights[other];
                const float inverse = options.inverse_masses.size != 0U
                    ? options.inverse_masses.data[node] : 1.0F / options.node_mass;
                if (weight * inverse > 0.0F)
                    per_node[node].push_back({corner, weight * inverse / denominator});
            }
        }
        surface_offsets.push_back(0U);
        for (const auto &list : per_node) {
            if (surface_influences.size() + list.size() > UINT32_MAX)
                return failure(StatusCode::capacity_exceeded,
                               "soft-body surface influences exceed uint32 range");
            surface_influences.insert(surface_influences.end(), list.begin(), list.end());
            surface_offsets.push_back(static_cast<std::uint32_t>(surface_influences.size()));
        }
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to build soft-body surface contact bindings");
    }

    std::unique_ptr<SoftBodyStorage> body;
    try {
        body = std::make_unique<SoftBodyStorage>();
    } catch (...) {
        return failure(StatusCode::out_of_memory,
                       "failed to allocate soft-body storage");
    }
    body->generation = impl_->soft_bodies[slot]
        ? impl_->soft_bodies[slot]->generation : 1U;
    body->node_count = node_count;
    body->bond_count = static_cast<std::uint32_t>(options.bonds.size);
    body->neighbor_count = static_cast<std::uint32_t>(neighbors.size());
    body->surface_vertex_count =
        static_cast<std::uint32_t>(options.surface_vertices.size);
    body->surface_index_count =
        static_cast<std::uint32_t>(options.surface_triangle_indices.size);
    body->node_radius = options.node_radius;
    body->stretch_compliance = options.stretch_compliance;
    body->velocity_damping = options.velocity_damping;
    body->spring_damping = options.spring_damping;
    body->contact_friction = options.contact_friction;
    body->shape_matching_stiffness = options.shape_matching_stiffness;
    body->shape_maximum_projection = options.maximum_projection_fraction *
                                     minimum_bond_length;
    body->maximum_projection_fraction = options.maximum_projection_fraction;
    body->constraint_velocity_response = options.constraint_velocity_response;
    body->maximum_speed = options.maximum_speed;
    body->movable_mass = movable_mass;
    body->shape_rest_center = shape_rest_center;
    body->shape_inverse_rest = shape_inverse_rest;
    body->solver_iterations = options.solver_iterations;
    StorageAllocator allocator;
    status = simulation::visit_soft_body_storage(*body, neighbors.size(),
        warp_neighbors.size(), surface_influences.size(), allocator);
    if (!status) return status;
    for (std::uint32_t node = 0U; node < body->node_count; ++node) {
        body->positions[node] = options.nodes.data[node];
        body->rest_positions[node] = options.nodes.data[node];
        body->scratch[node] = options.nodes.data[node];
        body->previous[node] = options.nodes.data[node];
        body->velocities[node] = {};
        body->velocity_scratch[node] = {};
        body->inverse_masses[node] = options.inverse_masses.size != 0U
            ? options.inverse_masses.data[node] : 1.0F / options.node_mass;
        body->rigid_contact_forces[node] = {};
        body->cloth_forces[node] = {};
        body->fluid_forces[node] = {};
        body->rope_forces[node] = {};
        body->contact_normals[node] = {};
        body->contact_arms[node] = {};
        body->contact_momentum_delta[node] = {};
        body->contact_friction_delta[node] = {};
        body->contact_normal_delta[node] = 0.0F;
    }
    std::copy_n(options.bonds.data, body->bond_count, body->bonds);
    std::fill_n(body->bond_active, body->bond_count, std::uint8_t{1U});
    std::copy(offsets.begin(), offsets.end(), body->offsets);
    std::copy(neighbors.begin(), neighbors.end(), body->neighbors);
    if(!warp_neighbors.empty())
        std::copy(warp_neighbors.begin(),warp_neighbors.end(),body->warp_neighbors);
    if(!minimum_rest_lengths.empty())
        std::copy(minimum_rest_lengths.begin(),minimum_rest_lengths.end(),
            body->minimum_rest_lengths);
    std::copy_n(options.surface_vertices.data, body->surface_vertex_count,
                body->surface_rest_positions);
    std::copy_n(options.surface_vertices.data, body->surface_vertex_count,
                body->surface_positions);
    std::copy_n(options.surface_triangle_indices.data,
                body->surface_index_count, body->surface_indices);
    std::copy_n(options.surface_bindings.data, body->surface_vertex_count,
                body->surface_bindings);
    std::copy(surface_offsets.begin(), surface_offsets.end(), body->surface_node_offsets);
    std::copy(surface_influences.begin(), surface_influences.end(), body->surface_node_influences);
    *body->count = body->node_count;
    *body->contact_count = 0U;
    *body->predicted_momentum = {};
    *body->shape_orientation = {0.0F, 0.0F, 0.0F, 1.0F};
    *body->dynamic_contact_flag = 0U;
    body->alive = true;
    output = {slot, body->generation};
    impl_->soft_bodies[slot] = std::move(body);
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::add_rope(RopeOptions options, RopeId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument,"simulation is not initialized");
    Status status=impl_->require_idle();
    if(!status)return status;
    if(!finite(options.radius)||options.radius<=0 || !finite(options.node_spacing)||
       options.node_spacing<=0 || options.node_spacing>2*options.radius ||
       !finite(options.mass)||options.mass<=0 || !finite(options.stretch_compliance)||options.stretch_compliance<0 ||
       !finite(options.velocity_damping)||options.velocity_damping<0 ||
       !finite(options.maximum_substep_timestep)||options.maximum_substep_timestep<=0 ||
       !finite(options.friction)||options.friction<0 ||
       !finite(options.maximum_speed)||options.maximum_speed<=0 || options.solver_iterations<1 || options.solver_iterations>128)
        return failure(StatusCode::invalid_argument,"invalid rope material or resolution");
    for(const auto anchor:{options.first,options.last}) {
        if(!finite(anchor.local_anchor))return failure(StatusCode::invalid_argument,"invalid rope anchor");
        if(anchor.enabled){unsigned dense;if(!(status=impl_->validate_handle(anchor.body,dense)))return status;}
    }
    if (options.first.enabled && options.last.enabled && options.first.body == options.last.body)
        return failure(StatusCode::invalid_argument,"rope endpoints must attach to distinct bodies");
    std::uint32_t slot=0;
    while(slot<impl_->ropes.size() && impl_->ropes[slot] && impl_->ropes[slot]->alive)++slot;
    if(slot==impl_->ropes.size())return failure(StatusCode::capacity_exceeded,"rope capacity exhausted");
    std::vector<Vec3> nodes;
    if(!(status=sample_rope_centerline(options.centerline,options.node_spacing,nodes)))return status;
    int attached[2]{-1, -1};
    // Rest curve endpoints define anchors, not an initial teleport/impulse.
    for(unsigned end=0;end<2;++end) {
        const auto anchor=end?options.last:options.first;
        if(!anchor.enabled)continue;
        unsigned dense;if(!(status=impl_->validate_handle(anchor.body,dense)))return status;
        attached[end] = static_cast<int>(dense);
        const Vec3 target=transform_point(impl_->states[impl_->current_state][dense],anchor.local_anchor);
        if(vector_length(subtract(target,end?nodes.back():nodes.front()))>1e-3F)
            return failure(StatusCode::invalid_argument,"rope endpoint must match its body-local attachment");
    }
    const unsigned first_skip=rope_attachment_contact_skip(nodes,true,attached[0],
        impl_->parameters,impl_->states[impl_->current_state],impl_->meshes);
    const unsigned last_skip=rope_attachment_contact_skip(nodes,false,attached[1],
        impl_->parameters,impl_->states[impl_->current_state],impl_->meshes);
    try {
        if (rope_rest_crosses_collider(nodes, attached[0], attached[1],
                first_skip,last_skip,impl_->parameters,
                impl_->states[impl_->current_state], impl_->meshes, impl_->rigid_body_count))
            return failure(StatusCode::invalid_argument,"rope rest centerline crosses a rigid collider");
    } catch (...) {
        return failure(StatusCode::out_of_memory,"rope rest collision validation allocation failed");
    }
    std::unique_ptr<RopeStorage> owner(new(std::nothrow)RopeStorage());
    if(!owner)return failure(StatusCode::out_of_memory,"rope allocation failed");
    owner->generation=impl_->ropes[slot]?impl_->ropes[slot]->generation:1;
    auto &r=owner->data;
    r.options=options;r.options.centerline={};r.count=static_cast<unsigned>(nodes.size());
    r.first_attachment_contact_skip=first_skip;
    r.last_attachment_contact_skip=last_skip;
    r.body_capacity=impl_->options.rigid_body_capacity;
    StorageAllocator allocator;
    status = simulation::visit_rope_storage(r, r.count, r.body_capacity, allocator);
    if (!status) return status;
    r.soft_anchor_inverse_masses[0] = r.soft_anchor_inverse_masses[1] = 0.0F;
    std::fill_n(r.solid_hint,std::size_t(r.count)*r.body_capacity,~0U);
    for(unsigned i=0;i<r.count;++i){
        r.positions[i]=r.previous[i]=nodes[i];
        r.velocities[i]=r.constraint_forces[i]=r.contact_forces[i]=r.fluid_contact_forces[i]=r.soft_body_contact_forces[i]={};
        if(i+1<r.count){r.rest[i]=vector_length(subtract(nodes[i+1],nodes[i]));
            if(!finite(r.rest[i]) || r.rest[i]<1e-6F)return failure(StatusCode::invalid_argument,"rope contains an invalid segment");}
    }
    owner->alive=true;output={slot,owner->generation};impl_->ropes[slot]=std::move(owner);++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_rope(RopeId id) noexcept {
    if(!impl_)return failure(StatusCode::invalid_argument,"simulation is not initialized");
    auto status=impl_->require_idle();if(!status)return status;
    if(id.index>=impl_->ropes.size() || !impl_->ropes[id.index] || !impl_->ropes[id.index]->alive || impl_->ropes[id.index]->generation!=id.generation)
        return failure(StatusCode::invalid_handle,"rope handle is stale");
    for (const auto &coupling : impl_->fluid_rope_couplings)
        if (coupling && coupling->alive && coupling->options.rope == id)
            return failure(StatusCode::invalid_argument,
                           "rope is still referenced by a fluid coupling");
    for (const auto &coupling : impl_->smoke_rope_couplings)
        if (coupling && coupling->alive && coupling->options.rope == id)
            return failure(StatusCode::invalid_argument,
                           "rope is still referenced by a smoke coupling");
    for (const auto &coupling : impl_->rope_soft_body_couplings)
        if (coupling && coupling->alive && coupling->options.rope == id)
            return failure(StatusCode::invalid_argument,
                           "rope is still referenced by a soft-body coupling");
    for (const auto &coupling : impl_->rope_cloth_couplings)
        if (coupling.alive && coupling.options.rope == id)
            return failure(StatusCode::invalid_argument,
                           "rope is still referenced by a cloth coupling");
    auto &rope=*impl_->ropes[id.index];rope.alive=false;rope.release();++rope.generation;
    if(rope.generation==0)rope.generation=1;
    ++impl_->revision;return success();
}

Status SimulationRuntime::rope_view(RopeId id, RopeDeviceView &output) const noexcept {
    output={};
    if(!impl_)return failure(StatusCode::invalid_argument,"simulation is not initialized");
    auto status=impl_->require_current_device();if(!status)return status;
    if(impl_->frame && !impl_->frame->acknowledged)return failure(StatusCode::busy,"rope view requires completed frame");
    if(id.index>=impl_->ropes.size() || !impl_->ropes[id.index] || !impl_->ropes[id.index]->alive || impl_->ropes[id.index]->generation!=id.generation)
        return failure(StatusCode::invalid_handle,"rope handle is stale");
    const auto &r=impl_->ropes[id.index]->data;
    output={{r.positions,r.count},{r.velocities,r.count},{r.constraint_forces,r.count},
        {r.contact_forces,r.count},{r.fluid_contact_forces,r.count},
        {r.soft_body_contact_forces,r.count},
        {r.rest,r.count-1},r.options.radius};
    return success();
}

static bool valid_fluid_rope_options(FluidRopeCouplingOptions options) noexcept {
    return finite(options.contact_distance) && options.contact_distance >= 0.0F &&
        finite(options.friction) && options.friction >= 0.0F && options.friction <= 1.0F &&
        finite(options.maximum_rope_acceleration) && options.maximum_rope_acceleration > 0.0F;
}

Status SimulationRuntime::add_fluid_rope_coupling(
    FluidRopeCouplingOptions options, FluidRopeCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (options.rope.index >= impl_->ropes.size() ||
        !impl_->ropes[options.rope.index] || !impl_->ropes[options.rope.index]->alive ||
        impl_->ropes[options.rope.index]->generation != options.rope.generation)
        return failure(StatusCode::invalid_handle, "rope handle is stale");
    if (!valid_fluid_rope_options(options))
        return failure(StatusCode::invalid_argument, "invalid fluid rope options");
    for (const auto &item : impl_->fluid_rope_couplings)
        if (item && item->alive && item->options.fluid == options.fluid &&
            item->options.rope == options.rope)
            return failure(StatusCode::invalid_argument, "fluid and rope are already coupled");
    std::uint32_t slot = 0;
    while (slot < impl_->fluid_rope_couplings.size() &&
           impl_->fluid_rope_couplings[slot] && impl_->fluid_rope_couplings[slot]->alive)
        ++slot;
    if (slot == impl_->fluid_rope_couplings.size())
        return failure(StatusCode::capacity_exceeded, "fluid rope coupling capacity exhausted");
    std::unique_ptr<FluidRopeCouplingStorage> owner(
        new (std::nothrow) FluidRopeCouplingStorage());
    if (!owner) return failure(StatusCode::out_of_memory, "fluid rope coupling allocation failed");
    StorageAllocator allocator;
    status = simulation::visit_fluid_rope_storage(*owner,
        impl_->ropes[options.rope.index]->data.count, allocator);
    if (!status) return status;
    *owner->contact_count = 0U;
    *owner->maximum_penetration = 0.0F;
    owner->options = options;
    owner->generation = impl_->fluid_rope_couplings[slot]
        ? impl_->fluid_rope_couplings[slot]->generation : 1U;
    owner->alive = true;
    output = {slot, owner->generation};
    impl_->fluid_rope_couplings[slot] = std::move(owner);
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::update_fluid_rope_coupling(
    FluidRopeCouplingId id, FluidRopeCouplingOptions options) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_rope_couplings.size() ||
        !impl_->fluid_rope_couplings[id.index] ||
        !impl_->fluid_rope_couplings[id.index]->alive ||
        impl_->fluid_rope_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid rope coupling is stale");
    auto &item = *impl_->fluid_rope_couplings[id.index];
    if (!valid_fluid_rope_options(options) ||
        !(options.fluid == item.options.fluid) || !(options.rope == item.options.rope))
        return failure(StatusCode::invalid_argument, "invalid options or changed fluid rope endpoints");
    item.options = options;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_fluid_rope_coupling(FluidRopeCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_rope_couplings.size() ||
        !impl_->fluid_rope_couplings[id.index] ||
        !impl_->fluid_rope_couplings[id.index]->alive ||
        impl_->fluid_rope_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid rope coupling is stale");
    auto &item = *impl_->fluid_rope_couplings[id.index];
    item.release();
    item.alive = false;
    if (++item.generation == 0U) item.generation = 1U;
    ++impl_->revision;
    return success();
}

static bool valid_rope_soft_options(RopeSoftBodyCouplingOptions options) noexcept {
    return finite(options.contact_distance) && options.contact_distance >= 0.0F &&
        finite(options.friction) && options.friction >= 0.0F &&
        finite(options.maximum_soft_body_acceleration) &&
        options.maximum_soft_body_acceleration > 0.0F &&
        finite(options.anchor_support_radius_scale) &&
        options.anchor_support_radius_scale >= 0.0F &&
        finite(options.anchor_contact_support_radius_scale) &&
        options.anchor_contact_support_radius_scale >= 0.0F &&
        options.anchor_contact_support_radius_scale <=
            options.anchor_support_radius_scale;
}

Status SimulationRuntime::add_rope_soft_body_coupling(RopeSoftBodyCouplingOptions options,
    RopeSoftBodyCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument,"simulation is not initialized");
    auto status=impl_->require_idle();if(!status)return status;
    if(options.rope.index>=impl_->ropes.size() || !impl_->ropes[options.rope.index] ||
       !impl_->ropes[options.rope.index]->alive ||
       impl_->ropes[options.rope.index]->generation!=options.rope.generation ||
       options.soft_body.index>=impl_->soft_bodies.size() ||
       !impl_->soft_bodies[options.soft_body.index] ||
       !impl_->soft_bodies[options.soft_body.index]->alive ||
       impl_->soft_bodies[options.soft_body.index]->generation!=options.soft_body.generation)
        return failure(StatusCode::invalid_handle,"rope or soft body handle is stale");
    if(!valid_rope_soft_options(options))
        return failure(StatusCode::invalid_argument,"invalid rope soft-body options");
    auto &rope=impl_->ropes[options.rope.index]->data;
    auto &body=*impl_->soft_bodies[options.soft_body.index];
    if((options.attach_first && rope.options.first.enabled) ||
       (options.attach_last && rope.options.last.enabled))
        return failure(StatusCode::invalid_argument,"rope endpoint already has an attachment");
    unsigned existing_targets=0;
    for(const auto &item:impl_->rope_soft_body_couplings)
        if(item && item->alive && item->options.rope==options.rope) {
            ++existing_targets;
            if(item->options.soft_body==options.soft_body)
                return failure(StatusCode::invalid_argument,"rope and soft body are already coupled");
            if((options.attach_first && item->options.attach_first) ||
               (options.attach_last && item->options.attach_last))
                return failure(StatusCode::invalid_argument,"rope endpoint already has an attachment");
        }
    for(const auto &item:impl_->rope_cloth_couplings)
        if(item.alive && item.options.rope==options.rope &&
           ((options.attach_first && item.options.first_vertex!=UINT32_MAX) ||
            (options.attach_last && item.options.last_vertex!=UINT32_MAX)))
            return failure(StatusCode::invalid_argument,
                "rope endpoint already has a cloth attachment");
    if(existing_targets>=2U)
        return failure(StatusCode::capacity_exceeded,
            "a rope supports at most two soft-body contact targets");
    unsigned slot=0;
    while(slot<impl_->rope_soft_body_couplings.size() &&
          impl_->rope_soft_body_couplings[slot] &&
          impl_->rope_soft_body_couplings[slot]->alive)++slot;
    if(slot==impl_->rope_soft_body_couplings.size())
        return failure(StatusCode::capacity_exceeded,"rope soft-body coupling capacity exhausted");
    std::unique_ptr<RopeSoftBodyCouplingStorage> owner(
        new(std::nothrow)RopeSoftBodyCouplingStorage());
    if(!owner)return failure(StatusCode::out_of_memory,"rope soft-body coupling allocation failed");
    std::vector<BvhNode> tree;
    std::vector<std::uint32_t> order,parents;
    double signed_volume=0.0;
    const Vec3 volume_origin=body.surface_rest_positions[0];
    for(unsigned t=0;t<body.surface_index_count/3U;++t) {
        const auto *idx=body.surface_indices+3U*t;
        signed_volume+=dot(subtract(body.surface_rest_positions[idx[0]],volume_origin),
            cross(subtract(body.surface_rest_positions[idx[1]],volume_origin),
                  subtract(body.surface_rest_positions[idx[2]],volume_origin)))/6.0;
    }
    if(std::abs(signed_volume)<1e-10)
        return failure(StatusCode::invalid_argument,"rope soft-body skin has zero volume");
    owner->orientation=signed_volume>=0.0?1.0F:-1.0F;
    try {
        order.resize(body.surface_index_count/3U);
        std::iota(order.begin(),order.end(),0U);
        tree.reserve(order.size()*2U);
        parents.reserve(order.size()*2U);
        const auto centroid=[&](unsigned triangle) {
            const auto *idx=body.surface_indices+triangle*3U;
            return multiply(add(add(body.surface_rest_positions[idx[0]],
                body.surface_rest_positions[idx[1]]),
                body.surface_rest_positions[idx[2]]),1.0F/3.0F);
        };
        std::function<unsigned(unsigned,unsigned,unsigned)> build=
            [&](unsigned begin,unsigned end,unsigned parent) {
                const unsigned index=static_cast<unsigned>(tree.size());
                BvhNode node{};
                node.minimum={FLT_MAX,FLT_MAX,FLT_MAX};
                node.maximum={-FLT_MAX,-FLT_MAX,-FLT_MAX};
                for(unsigned item=begin;item<end;++item) {
                    const auto point=centroid(order[item]);
                    node.minimum=component_min(node.minimum,point);
                    node.maximum=component_max(node.maximum,point);
                }
                tree.push_back(node);parents.push_back(parent);
                if(end-begin<=4U) {
                    tree[index].first_triangle=begin;
                    tree[index].triangle_count=end-begin;
                } else {
                    const auto extent=subtract(node.maximum,node.minimum);
                    const unsigned axis=extent.x>=extent.y && extent.x>=extent.z?0U:
                        extent.y>=extent.z?1U:2U;
                    const unsigned middle=begin+(end-begin)/2U;
                    std::nth_element(order.begin()+begin,order.begin()+middle,
                        order.begin()+end,[&](unsigned left,unsigned right) {
                            const auto a=centroid(left),b=centroid(right);
                            const float x=axis==0?a.x:axis==1?a.y:a.z;
                            const float y=axis==0?b.x:axis==1?b.y:b.z;
                            return x<y || (x==y && left<right);
                        });
                    const unsigned left=build(begin,middle,index);
                    const unsigned right=build(middle,end,index);
                    tree[index].left=left;tree[index].right=right;
                }
                return index;
            };
        build(0U,static_cast<unsigned>(order.size()),k_invalid_dense);
    } catch (...) {
        return failure(StatusCode::out_of_memory,"rope soft-body BVH build failed");
    }
    owner->tree_count=static_cast<unsigned>(tree.size());
    StorageAllocator allocator;
    status = simulation::visit_rope_soft_tree_storage(*owner, tree.size(), order.size(), allocator);
    if (!status) return status;
    std::copy(tree.begin(),tree.end(),owner->tree);
    std::copy(order.begin(),order.end(),owner->order);
    std::copy(parents.begin(),parents.end(),owner->parents);
    for(unsigned end=0;end<2;++end) {
        if(!(end?options.attach_last:options.attach_first))continue;
        const Vec3 point=rope.positions[end?rope.count-1U:0U];
        float best=FLT_MAX;
        for(unsigned t=0;t<body.surface_index_count/3U;++t) {
            const auto *idx=body.surface_indices+3U*t;
            const Vec3 a=body.surface_rest_positions[idx[0]],
                       b=body.surface_rest_positions[idx[1]],
                       c=body.surface_rest_positions[idx[2]];
            const Vec3 ab=subtract(b,a),ac=subtract(c,a);
            const float d00=dot(ab,ab),d01=dot(ab,ac),d11=dot(ac,ac);
            const float denominator=d00*d11-d01*d01;
            if(denominator<=1e-14F)continue;
            const Vec3 nearest=closest_on_triangle(point,a,b,c);
            const float distance=length_squared(subtract(point,nearest));
            if(distance>=best)continue;
            const Vec3 ap=subtract(nearest,a);
            const float d20=dot(ap,ab),d21=dot(ap,ac);
            const float v=(d11*d20-d01*d21)/denominator;
            const float w=(d00*d21-d01*d20)/denominator;
            owner->anchor_triangle[end]=t;
            owner->anchor_weights[end]={1.0F-v-w,v,w};
            owner->anchor_offset[end]=subtract(point,nearest);
            best=distance;
        }
        if(owner->anchor_triangle[end]==~0U)
            return failure(StatusCode::invalid_argument,"soft body has no usable anchor triangle");
    }
    status = simulation::visit_rope_soft_data_storage(*owner, body.node_count, body.surface_vertex_count, allocator);
    if (!status) return status;
    std::fill_n(owner->node_impulses,body.node_count,Vec3{});
    std::copy_n(body.surface_positions,body.surface_vertex_count,
        owner->previous_surface);
    *owner->contact_count=0U;*owner->maximum_penetration=0.0F;
    owner->options=options;
    owner->generation=impl_->rope_soft_body_couplings[slot]
        ? impl_->rope_soft_body_couplings[slot]->generation:1U;
    owner->alive=true;
    if(options.enabled) {
        rope.soft_first|=options.attach_first;
        rope.soft_last|=options.attach_last;
    }
    output={slot,owner->generation};
    impl_->rope_soft_body_couplings[slot]=std::move(owner);
    ++impl_->revision;return success();
}

Status SimulationRuntime::update_rope_soft_body_coupling(RopeSoftBodyCouplingId id,
    RopeSoftBodyCouplingOptions options) noexcept {
    if(!impl_)return failure(StatusCode::invalid_argument,"simulation is not initialized");
    auto status=impl_->require_idle();if(!status)return status;
    if(id.index>=impl_->rope_soft_body_couplings.size() ||
       !impl_->rope_soft_body_couplings[id.index] ||
       !impl_->rope_soft_body_couplings[id.index]->alive ||
       impl_->rope_soft_body_couplings[id.index]->generation!=id.generation)
        return failure(StatusCode::invalid_handle,"rope soft-body coupling is stale");
    auto &item=*impl_->rope_soft_body_couplings[id.index];
    if(!valid_rope_soft_options(options) || !(options.rope==item.options.rope) ||
       !(options.soft_body==item.options.soft_body) ||
       options.attach_first!=item.options.attach_first ||
       options.attach_last!=item.options.attach_last)
        return failure(StatusCode::invalid_argument,"cannot change rope soft-body endpoints");
    item.options=options;
    auto &rope=impl_->ropes[options.rope.index]->data;
    rope.soft_first=rope.soft_last=false;
    for(const auto &coupling:impl_->rope_soft_body_couplings)
        if(coupling && coupling->alive && coupling->options.enabled &&
           coupling->options.rope==options.rope) {
            rope.soft_first|=coupling->options.attach_first;
            rope.soft_last|=coupling->options.attach_last;
        }
    for(const auto &coupling:impl_->rope_cloth_couplings)
        if(coupling.alive && coupling.options.enabled &&
           coupling.options.rope==options.rope) {
            rope.soft_first|=coupling.options.first_vertex!=UINT32_MAX;
            rope.soft_last|=coupling.options.last_vertex!=UINT32_MAX;
        }
    ++impl_->revision;return success();
}

Status SimulationRuntime::remove_rope_soft_body_coupling(RopeSoftBodyCouplingId id) noexcept {
    if(!impl_)return failure(StatusCode::invalid_argument,"simulation is not initialized");
    auto status=impl_->require_idle();if(!status)return status;
    if(id.index>=impl_->rope_soft_body_couplings.size() ||
       !impl_->rope_soft_body_couplings[id.index] ||
       !impl_->rope_soft_body_couplings[id.index]->alive ||
       impl_->rope_soft_body_couplings[id.index]->generation!=id.generation)
        return failure(StatusCode::invalid_handle,"rope soft-body coupling is stale");
    auto &item=*impl_->rope_soft_body_couplings[id.index];
    const auto rope_id=item.options.rope;
    item.release();item.alive=false;
    if(++item.generation==0U)item.generation=1U;
    auto &rope=impl_->ropes[rope_id.index]->data;
    rope.soft_first=rope.soft_last=false;
    for(const auto &coupling:impl_->rope_soft_body_couplings)
        if(coupling && coupling->alive && coupling->options.enabled &&
           coupling->options.rope==rope_id) {
            rope.soft_first|=coupling->options.attach_first;
            rope.soft_last|=coupling->options.attach_last;
        }
    for(const auto &coupling:impl_->rope_cloth_couplings)
        if(coupling.alive && coupling.options.enabled &&
           coupling.options.rope==rope_id) {
            rope.soft_first|=coupling.options.first_vertex!=UINT32_MAX;
            rope.soft_last|=coupling.options.last_vertex!=UINT32_MAX;
        }
    ++impl_->revision;return success();
}

Status SimulationRuntime::add_rope_cloth_coupling(RopeClothCouplingOptions options,
    RopeClothCouplingId &output) noexcept {
    output = {};
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    auto status = impl_->require_idle(); if (!status) return status;
    if (options.rope.index >= impl_->ropes.size() ||
        !impl_->ropes[options.rope.index] ||
        !impl_->ropes[options.rope.index]->alive ||
        impl_->ropes[options.rope.index]->generation != options.rope.generation ||
        options.cloth.index >= impl_->cloths.size() ||
        !impl_->cloths[options.cloth.index] ||
        !impl_->cloths[options.cloth.index]->alive ||
        impl_->cloths[options.cloth.index]->generation != options.cloth.generation)
        return failure(StatusCode::invalid_handle, "rope or cloth handle is stale");
    if (!finite(options.anchor_effective_mass) ||
        options.anchor_effective_mass <= 0.0F ||
        !finite(options.maximum_cloth_acceleration) ||
        options.maximum_cloth_acceleration <= 0.0F ||
        (options.first_vertex == UINT32_MAX && options.last_vertex == UINT32_MAX))
        return failure(StatusCode::invalid_argument, "invalid rope cloth options");
    auto &rope = impl_->ropes[options.rope.index]->data;
    auto &cloth = *impl_->cloths[options.cloth.index];
    const std::uint32_t vertices[2]{options.first_vertex, options.last_vertex};
    for (unsigned end = 0; end < 2; ++end) {
        if (vertices[end] == UINT32_MAX) continue;
        if (vertices[end] >= cloth.vertex_count ||
            (end ? rope.options.last.enabled : rope.options.first.enabled))
            return failure(StatusCode::invalid_argument,
                "rope cloth vertex is invalid or endpoint has a rigid attachment");
        const Vec3 delta = subtract(rope.positions[end ? rope.count - 1U : 0U],
                                    cloth.positions[vertices[end]]);
        if (length_squared(delta) > 1.0e-6F)
            return failure(StatusCode::invalid_argument,
                "rope endpoint must coincide with its cloth vertex");
    }
    for (const auto &item : impl_->rope_soft_body_couplings)
        if (item && item->alive && item->options.rope == options.rope &&
            ((options.first_vertex != UINT32_MAX && item->options.attach_first) ||
             (options.last_vertex != UINT32_MAX && item->options.attach_last)))
            return failure(StatusCode::invalid_argument,
                "rope endpoint already has a soft-body attachment");
    for (const auto &item : impl_->rope_cloth_couplings)
        if (item.alive && item.options.rope == options.rope &&
            (item.options.cloth == options.cloth ||
             (options.first_vertex != UINT32_MAX && item.options.first_vertex != UINT32_MAX) ||
             (options.last_vertex != UINT32_MAX && item.options.last_vertex != UINT32_MAX)))
            return failure(StatusCode::invalid_argument,
                "rope endpoint already has a cloth attachment");
    unsigned slot = 0U;
    while (slot < impl_->rope_cloth_couplings.size() &&
           impl_->rope_cloth_couplings[slot].alive) ++slot;
    if (slot == impl_->rope_cloth_couplings.size())
        return failure(StatusCode::capacity_exceeded,
            "rope cloth coupling capacity exhausted");
    auto &item = impl_->rope_cloth_couplings[slot];
    item.options = options;
    item.alive = true;
    if (options.enabled) {
        rope.soft_first |= options.first_vertex != UINT32_MAX;
        rope.soft_last |= options.last_vertex != UINT32_MAX;
    }
    output = {slot, item.generation};
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::update_rope_cloth_coupling(RopeClothCouplingId id,
    RopeClothCouplingOptions options) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    auto status = impl_->require_idle(); if (!status) return status;
    if (id.index >= impl_->rope_cloth_couplings.size() ||
        !impl_->rope_cloth_couplings[id.index].alive ||
        impl_->rope_cloth_couplings[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "rope cloth coupling is stale");
    auto &item = impl_->rope_cloth_couplings[id.index];
    if (!(item.options.rope == options.rope) ||
        !(item.options.cloth == options.cloth) ||
        item.options.first_vertex != options.first_vertex ||
        item.options.last_vertex != options.last_vertex ||
        !finite(options.anchor_effective_mass) ||
        options.anchor_effective_mass <= 0.0F ||
        !finite(options.maximum_cloth_acceleration) ||
        options.maximum_cloth_acceleration <= 0.0F)
        return failure(StatusCode::invalid_argument,
            "cannot change rope cloth endpoints or use invalid acceleration");
    item.options = options;
    auto &rope = impl_->ropes[options.rope.index]->data;
    rope.soft_first = rope.soft_last = false;
    for (const auto &soft : impl_->rope_soft_body_couplings)
        if (soft && soft->alive && soft->options.enabled &&
            soft->options.rope == options.rope) {
            rope.soft_first |= soft->options.attach_first;
            rope.soft_last |= soft->options.attach_last;
        }
    for (const auto &cloth : impl_->rope_cloth_couplings)
        if (cloth.alive && cloth.options.enabled && cloth.options.rope == options.rope) {
            rope.soft_first |= cloth.options.first_vertex != UINT32_MAX;
            rope.soft_last |= cloth.options.last_vertex != UINT32_MAX;
        }
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_rope_cloth_coupling(RopeClothCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    auto status = impl_->require_idle(); if (!status) return status;
    if (id.index >= impl_->rope_cloth_couplings.size() ||
        !impl_->rope_cloth_couplings[id.index].alive ||
        impl_->rope_cloth_couplings[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle, "rope cloth coupling is stale");
    auto &item = impl_->rope_cloth_couplings[id.index];
    const RopeId rope_id = item.options.rope;
    item.alive = false;
    if (++item.generation == 0U) item.generation = 1U;
    auto &rope = impl_->ropes[rope_id.index]->data;
    rope.soft_first = rope.soft_last = false;
    for (const auto &soft : impl_->rope_soft_body_couplings)
        if (soft && soft->alive && soft->options.enabled &&
            soft->options.rope == rope_id) {
            rope.soft_first |= soft->options.attach_first;
            rope.soft_last |= soft->options.attach_last;
        }
    for (const auto &cloth : impl_->rope_cloth_couplings)
        if (cloth.alive && cloth.options.enabled && cloth.options.rope == rope_id) {
            rope.soft_first |= cloth.options.first_vertex != UINT32_MAX;
            rope.soft_last |= cloth.options.last_vertex != UINT32_MAX;
        }
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_soft_body(SoftBodyId id) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->soft_bodies.size() || !impl_->soft_bodies[id.index] ||
        !impl_->soft_bodies[id.index]->alive ||
        impl_->soft_bodies[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "soft-body handle is stale");
    SoftBodyStorage &body = *impl_->soft_bodies[id.index];
    for (const auto &coupling : impl_->fluid_soft_couplings)
        if (coupling && coupling->alive && coupling->options.soft_body == id)
            return failure(StatusCode::invalid_argument,
                           "soft body is still referenced by a fluid coupling");
    for (const auto &coupling : impl_->smoke_soft_body_couplings)
        if (coupling && coupling->alive && coupling->options.soft_body == id)
            return failure(StatusCode::invalid_argument,
                           "soft body is still referenced by a smoke coupling");
    for (const auto &coupling : impl_->soft_cloth_couplings)
        if (coupling && coupling->alive && coupling->options.soft_body == id)
            return failure(StatusCode::invalid_argument,
                           "soft body is still referenced by a cloth coupling");
    for (const auto &coupling : impl_->rope_soft_body_couplings)
        if (coupling && coupling->alive && coupling->options.soft_body == id)
            return failure(StatusCode::invalid_argument,
                           "soft body is still referenced by a rope coupling");
    body.alive = false;
    body.release();
    ++body.generation;
    if (body.generation == 0U) body.generation = 1U;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::soft_body_view(SoftBodyId id,
                            SoftBodyDeviceView &output) const noexcept {
    output = {};
    if (!impl_)
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_current_device();
    if (!status) return status;
    if (id.index >= impl_->soft_bodies.size() || !impl_->soft_bodies[id.index] ||
        !impl_->soft_bodies[id.index]->alive ||
        impl_->soft_bodies[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "soft-body handle is stale");
    const SoftBodyStorage &body = *impl_->soft_bodies[id.index];
    output.positions = {body.positions, body.node_count};
    output.velocities = {body.velocities, body.node_count};
    output.bonds = {body.bonds, body.bond_count};
    output.surface_positions = {
        body.surface_positions, body.surface_vertex_count};
    output.surface_triangle_indices = {
        body.surface_indices, body.surface_index_count};
    output.rigid_contact_forces = {
        body.rigid_contact_forces, body.node_count};
    output.cloth_contact_forces = {body.cloth_forces, body.node_count};
    output.fluid_contact_forces = {body.fluid_forces, body.node_count};
    output.rope_contact_forces = {body.rope_forces, body.node_count};
    output.node_count = body.node_count;
    output.surface_vertex_count = body.surface_vertex_count;
    return success();
}

static bool valid_fluid_soft_options(FluidSoftBodyCouplingOptions options) noexcept {
    return finite(options.contact_distance) && options.contact_distance >= 0.0F &&
        finite(options.friction) && options.friction >= 0.0F &&
        options.solver_iterations > 0U && options.solver_iterations <= 16U;
}

Status SimulationRuntime::add_fluid_soft_body_coupling(
    FluidSoftBodyCouplingOptions options, FluidSoftBodyCouplingId &output) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    SoftBodyDeviceView view{};
    if (!(status = impl_->validate_handle(options.fluid, fluid)) ||
        !(status = soft_body_view(options.soft_body, view))) return status;
    if (!valid_fluid_soft_options(options))
        return failure(StatusCode::invalid_argument, "invalid fluid soft-body options");
    for (const auto &coupling : impl_->fluid_soft_couplings)
        if (coupling && coupling->alive && coupling->options.fluid == options.fluid &&
            coupling->options.soft_body == options.soft_body)
            return failure(StatusCode::invalid_argument, "fluid and soft body are already coupled");
    std::uint32_t slot = 0;
    for (; slot < impl_->fluid_soft_couplings.size(); ++slot)
        if (!impl_->fluid_soft_couplings[slot] || !impl_->fluid_soft_couplings[slot]->alive) break;
    if (slot == impl_->fluid_soft_couplings.size())
        return failure(StatusCode::capacity_exceeded, "fluid soft-body coupling capacity exhausted");
    auto &body = *impl_->soft_bodies[options.soft_body.index];
    std::unique_ptr<FluidSoftCouplingStorage> coupling;
    std::vector<BvhNode> tree;
    std::vector<std::uint32_t> order, parents;
    double volume = 0;
    try {
        // glTF can duplicate a position at normal/UV seams. Validate geometric
        // edges, not rendering indices, without changing the skin bindings.
        std::map<std::array<float, 3>, std::uint32_t> vertices;
        std::vector<std::uint32_t> canonical(body.surface_vertex_count);
        for (std::uint32_t i = 0; i < body.surface_vertex_count; ++i) {
            const auto p = body.surface_rest_positions[i];
            canonical[i] = vertices.emplace(std::array{p.x, p.y, p.z}, vertices.size()).first->second;
        }
        struct Edge { std::uint32_t count{}; int winding{}; };
        std::map<std::pair<std::uint32_t, std::uint32_t>, Edge> edges;
        const Vec3 origin = body.surface_rest_positions[0];
        for (std::uint32_t base = 0; base < body.surface_index_count; base += 3) {
            const auto *tri = body.surface_indices + base;
            volume += dot(subtract(body.surface_rest_positions[tri[0]], origin),
                cross(subtract(body.surface_rest_positions[tri[1]], origin),
                      subtract(body.surface_rest_positions[tri[2]], origin))) / 6.0;
            for (std::uint32_t e = 0; e < 3; ++e) {
                auto a = canonical[tri[e]], b = canonical[tri[(e + 1) % 3]];
                auto &edge = edges[std::minmax(a, b)];
                ++edge.count;
                edge.winding += a < b ? 1 : -1;
            }
        }
        if (std::abs(volume) < 1.0e-10 || std::any_of(edges.begin(), edges.end(),
            [](const auto &edge) { return edge.second.count != 2 || edge.second.winding != 0; }))
            return failure(StatusCode::invalid_argument,
                           "fluid soft-body coupling needs a closed consistently wound surface");
        coupling = std::make_unique<FluidSoftCouplingStorage>();
        order.resize(body.surface_index_count / 3U);
        std::iota(order.begin(), order.end(), 0U);
        const auto centroid = [&](std::uint32_t triangle) {
            const auto *indices = body.surface_indices + triangle * 3U;
            return multiply(add(add(body.surface_rest_positions[indices[0]],
                body.surface_rest_positions[indices[1]]),
                body.surface_rest_positions[indices[2]]), 1.0F / 3.0F);
        };
        std::function<std::uint32_t(std::uint32_t, std::uint32_t, std::uint32_t)> build =
            [&](std::uint32_t first, std::uint32_t end, std::uint32_t parent) {
                const auto index = static_cast<std::uint32_t>(tree.size());
                BvhNode node{};
                node.minimum = {FLT_MAX, FLT_MAX, FLT_MAX};
                node.maximum = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
                for (auto item = first; item < end; ++item) {
                    const Vec3 center = centroid(order[item]);
                    node.minimum = component_min(node.minimum, center);
                    node.maximum = component_max(node.maximum, center);
                }
                tree.push_back(node);
                parents.push_back(parent);
                if (end - first <= 4U) {
                    tree[index].first_triangle = first;
                    tree[index].triangle_count = end - first;
                } else {
                    const Vec3 extent = subtract(node.maximum, node.minimum);
                    const int axis = extent.x >= extent.y && extent.x >= extent.z ? 0 :
                        (extent.y >= extent.z ? 1 : 2);
                    const auto middle = first + (end - first) / 2U;
                    std::nth_element(order.begin() + first, order.begin() + middle, order.begin() + end,
                        [&](auto a, auto b) {
                            const Vec3 ca = centroid(a), cb = centroid(b);
                            const float x = axis == 0 ? ca.x : (axis == 1 ? ca.y : ca.z);
                            const float y = axis == 0 ? cb.x : (axis == 1 ? cb.y : cb.z);
                            return x < y || (x == y && a < b);
                        });
                    const auto left = build(first, middle, index);
                    const auto right = build(middle, end, index);
                    tree[index].left = left;
                    tree[index].right = right;
                }
                return index;
            };
        build(0U, static_cast<std::uint32_t>(order.size()), k_invalid_dense);
    } catch (...) { return failure(StatusCode::out_of_memory, "failed to build fluid soft-body coupling"); }
    StorageAllocator allocator;
    status = simulation::visit_fluid_soft_storage(*coupling, fluid->options.capacity, body.node_count,
        body.surface_vertex_count, tree.size(), order.size(), allocator);
    if (!status) return status;
    std::copy(tree.begin(), tree.end(), coupling->tree);
    std::copy(parents.begin(), parents.end(), coupling->parents);
    std::copy(order.begin(), order.end(), coupling->triangle_order);
    coupling->tree_count = static_cast<std::uint32_t>(tree.size());
    std::copy_n(body.surface_positions, body.surface_vertex_count, coupling->previous_surface);
    *coupling->contact_count = 0;
    *coupling->maximum_penetration = 0;
    coupling->orientation = volume < 0 ? -1.0F : 1.0F;
    if (impl_->fluid_soft_couplings[slot])
        coupling->generation = impl_->fluid_soft_couplings[slot]->generation;
    coupling->options = options;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->fluid_soft_couplings[slot] = std::move(coupling);
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::update_fluid_soft_body_coupling(
    FluidSoftBodyCouplingId id, FluidSoftBodyCouplingOptions options) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_soft_couplings.size() || !impl_->fluid_soft_couplings[id.index] ||
        !impl_->fluid_soft_couplings[id.index]->alive ||
        impl_->fluid_soft_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid soft-body coupling is stale");
    auto &coupling = *impl_->fluid_soft_couplings[id.index];
    if (!valid_fluid_soft_options(options) || !(options.fluid == coupling.options.fluid) ||
        !(options.soft_body == coupling.options.soft_body))
        return failure(StatusCode::invalid_argument, "invalid options or changed fluid soft-body endpoints");
    coupling.options = options;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_fluid_soft_body_coupling(FluidSoftBodyCouplingId id) noexcept {
    if (!impl_) return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_soft_couplings.size() || !impl_->fluid_soft_couplings[id.index] ||
        !impl_->fluid_soft_couplings[id.index]->alive ||
        impl_->fluid_soft_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "fluid soft-body coupling is stale");
    auto &coupling = *impl_->fluid_soft_couplings[id.index];
    coupling.release();
    coupling.alive = false;
    if (++coupling.generation == 0) coupling.generation = 1;
    ++impl_->revision;
    return success();
}

static bool valid_soft_cloth_options(SoftBodyClothCouplingOptions options) noexcept {
    return finite(options.contact_distance) && options.contact_distance >= 0.0F &&
        finite(options.friction) && options.friction >= 0.0F &&
        options.solver_iterations > 0U && options.solver_iterations <= 16U;
}

Status SimulationRuntime::add_soft_body_cloth_coupling(
    SoftBodyClothCouplingOptions options,
    SoftBodyClothCouplingId &output) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    SoftBodyDeviceView soft_view{};
    ClothDeviceView cloth_view_result{};
    if (!(status = soft_body_view(options.soft_body, soft_view)) ||
        !(status = cloth_view(options.cloth, cloth_view_result))) return status;
    if (!valid_soft_cloth_options(options))
        return failure(StatusCode::invalid_argument, "invalid soft-body cloth options");
    for (const auto &coupling : impl_->soft_cloth_couplings)
        if (coupling && coupling->alive &&
            coupling->options.soft_body == options.soft_body &&
            coupling->options.cloth == options.cloth)
            return failure(StatusCode::invalid_argument,
                           "soft body and cloth are already coupled");
    std::uint32_t slot = 0U;
    for (; slot < impl_->soft_cloth_couplings.size(); ++slot)
        if (!impl_->soft_cloth_couplings[slot] ||
            !impl_->soft_cloth_couplings[slot]->alive) break;
    if (slot == impl_->soft_cloth_couplings.size())
        return failure(StatusCode::capacity_exceeded,
                       "soft-body cloth coupling capacity exhausted");
    std::unique_ptr<SoftClothCouplingStorage> coupling;
    try { coupling = std::make_unique<SoftClothCouplingStorage>(); }
    catch (...) { return failure(StatusCode::out_of_memory,
                                "failed to allocate soft-body cloth coupling"); }
    StorageAllocator allocator;
    status = simulation::visit_soft_cloth_storage(*coupling, soft_view.node_count,
        impl_->cloths[options.cloth.index]->vertex_capacity, allocator);
    if (!status) return status;
    if (impl_->soft_cloth_couplings[slot])
        coupling->generation = impl_->soft_cloth_couplings[slot]->generation;
    coupling->options = options;
    coupling->alive = true;
    output = {slot, coupling->generation};
    impl_->soft_cloth_couplings[slot] = std::move(coupling);
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::update_soft_body_cloth_coupling(
    SoftBodyClothCouplingId id, SoftBodyClothCouplingOptions options) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->soft_cloth_couplings.size() ||
        !impl_->soft_cloth_couplings[id.index] ||
        !impl_->soft_cloth_couplings[id.index]->alive ||
        impl_->soft_cloth_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "soft-body cloth coupling is stale");
    auto &coupling = *impl_->soft_cloth_couplings[id.index];
    if (!valid_soft_cloth_options(options) ||
        !(options.soft_body == coupling.options.soft_body) ||
        !(options.cloth == coupling.options.cloth))
        return failure(StatusCode::invalid_argument,
                       "invalid options or changed soft-body cloth endpoints");
    coupling.options = options;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_soft_body_cloth_coupling(SoftBodyClothCouplingId id) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->soft_cloth_couplings.size() ||
        !impl_->soft_cloth_couplings[id.index] ||
        !impl_->soft_cloth_couplings[id.index]->alive ||
        impl_->soft_cloth_couplings[id.index]->generation != id.generation)
        return failure(StatusCode::invalid_handle, "soft-body cloth coupling is stale");
    auto &coupling = *impl_->soft_cloth_couplings[id.index];
    coupling.alive = false;
    coupling.release();
    if (++coupling.generation == 0U) coupling.generation = 1U;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::add_fluid_cloth_coupling(
    FluidClothCouplingOptions options,
    FluidClothCouplingId &output) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (options.cloth.index >= impl_->cloths.size() ||
        !impl_->cloths[options.cloth.index] ||
        !impl_->cloths[options.cloth.index]->alive ||
        impl_->cloths[options.cloth.index]->generation !=
            options.cloth.generation)
        return failure(StatusCode::invalid_handle,
                       "cloth coupling handle is invalid or stale");
    const ClothStorage &cloth = *impl_->cloths[options.cloth.index];
    if (!cloth.preserve_volume)
        return failure(StatusCode::invalid_argument,
            "contained fluid requires closed volume-preserving cloth");
    if (!finite(options.contact_distance) || options.contact_distance < 0.0F ||
        !finite(options.interaction_radius) || options.interaction_radius < 0.0F ||
        !finite(options.stiffness) || options.stiffness <= 0.0F ||
        !finite(options.damping) || options.damping < 0.0F ||
        !finite(options.tangential_drag) || options.tangential_drag < 0.0F ||
        !finite(options.maximum_force) || options.maximum_force <= 0.0F)
        return failure(StatusCode::invalid_argument,
                       "invalid fluid-cloth coupling options");
    for (const FluidClothCouplingResource &coupling :
         impl_->fluid_cloth_couplings)
        if (coupling.alive && coupling.options.fluid == options.fluid &&
            coupling.options.cloth == options.cloth)
            return failure(StatusCode::invalid_argument,
                "fluid and cloth are already coupled");
    std::uint32_t slot = 0U;
    for (; slot < impl_->fluid_cloth_couplings.size(); ++slot)
        if (!impl_->fluid_cloth_couplings[slot].alive) break;
    if (slot == impl_->fluid_cloth_couplings.size())
        return failure(StatusCode::capacity_exceeded,
                       "fluid-cloth coupling capacity exhausted");
    FluidClothCouplingResource &coupling =
        impl_->fluid_cloth_couplings[slot];
    coupling.options = options;
    coupling.alive = true;
    output = {slot, coupling.generation};
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::update_fluid_cloth_coupling(
    FluidClothCouplingId id,
    FluidClothCouplingOptions options) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_cloth_couplings.size() ||
        !impl_->fluid_cloth_couplings[id.index].alive ||
        impl_->fluid_cloth_couplings[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle,
                       "fluid-cloth coupling handle is stale");
    FluidStorage *fluid = nullptr;
    if (!(status = impl_->validate_handle(options.fluid, fluid))) return status;
    if (options.cloth.index >= impl_->cloths.size() ||
        !impl_->cloths[options.cloth.index] ||
        !impl_->cloths[options.cloth.index]->alive ||
        impl_->cloths[options.cloth.index]->generation !=
            options.cloth.generation ||
        !impl_->cloths[options.cloth.index]->preserve_volume)
        return failure(StatusCode::invalid_handle,
                       "fluid-cloth coupling cloth is invalid or open");
    if (!finite(options.contact_distance) || options.contact_distance < 0.0F ||
        !finite(options.interaction_radius) || options.interaction_radius < 0.0F ||
        !finite(options.stiffness) || options.stiffness <= 0.0F ||
        !finite(options.damping) || options.damping < 0.0F ||
        !finite(options.tangential_drag) || options.tangential_drag < 0.0F ||
        !finite(options.maximum_force) || options.maximum_force <= 0.0F)
        return failure(StatusCode::invalid_argument,
                       "invalid fluid-cloth coupling options");
    impl_->fluid_cloth_couplings[id.index].options = options;
    ++impl_->revision;
    return success();
}

Status SimulationRuntime::remove_fluid_cloth_coupling(
    FluidClothCouplingId id) noexcept {
    if (!impl_)
        return failure(StatusCode::invalid_argument, "simulation is not initialized");
    Status status = impl_->require_idle();
    if (!status) return status;
    if (id.index >= impl_->fluid_cloth_couplings.size() ||
        !impl_->fluid_cloth_couplings[id.index].alive ||
        impl_->fluid_cloth_couplings[id.index].generation != id.generation)
        return failure(StatusCode::invalid_handle,
                       "fluid-cloth coupling handle is stale");
    FluidClothCouplingResource &coupling =
        impl_->fluid_cloth_couplings[id.index];
    coupling.alive = false;
    ++coupling.generation;
    if (coupling.generation == 0U) coupling.generation = 1U;
    ++impl_->revision;
    return success();
}

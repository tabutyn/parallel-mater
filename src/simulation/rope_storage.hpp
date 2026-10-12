// SPDX-License-Identifier: MIT
#pragma once

// Host resource ownership and CPU asset validation for ropes. GPU execution
// is provided exclusively by the generated Slang rope dispatch.
using RopeData = simulation::RopeDispatchData;
using RopeSoftTarget = simulation::RopeSoftTargetDispatch;

struct RopeStorage {
    RopeData data{};
    std::uint32_t generation{1U};
    bool alive{};
    void release() {
        release_managed(data.positions);
        release_managed(data.previous);
        release_managed(data.velocities);
        release_managed(data.constraint_forces);
        release_managed(data.contact_forces);
        release_managed(data.fluid_contact_forces);
        release_managed(data.soft_body_contact_forces);
        release_managed(data.soft_anchor_positions);
        release_managed(data.soft_anchor_velocities);
        release_managed(data.soft_anchor_impulses);
        release_managed(data.soft_anchor_inverse_masses);
        release_managed(data.directions);
        release_managed(data.scratch);
        release_managed(data.body_translation);
        release_managed(data.body_rotation);
        release_managed(data.rest);
        release_managed(data.lambda);
        release_managed(data.normals);
        release_managed(data.normals2);
        release_managed(data.solid_hint);
    }
    ~RopeStorage() { release(); }
};

struct RopeSoftBodyCouplingStorage {
    RopeSoftBodyCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    unsigned anchor_triangle[2]{~0U, ~0U};
    Vec3 anchor_weights[2]{}, anchor_offset[2]{};
    Vec3 *node_impulses{};
    Vec3 *previous_surface{};
    Vec3 *bounds{};
    BvhNode *tree{};
    std::uint32_t *order{}, *parents{}, *ready{};
    std::uint32_t tree_count{};
    float orientation{1.0F};
    std::uint32_t *contact_count{};
    float *maximum_penetration{};
    void release() noexcept {
        release_managed(node_impulses);
        release_managed(previous_surface);
        release_managed(bounds);
        release_managed(tree);
        release_managed(order);
        release_managed(parents);
        release_managed(ready);
        release_managed(contact_count);
        release_managed(maximum_penetration);
    }
    ~RopeSoftBodyCouplingStorage() { release(); }
};

// Skip only the short path that begins inside an attached convex body. The
// rest of the rope continues to collide with the attachment target.
unsigned rope_attachment_contact_skip(const std::vector<Vec3> &nodes,
    bool from_first, int body, const BodyParameters *parameters,
    const RigidBodyState *states, const TriangleMeshResource *meshes) {
    if (body < 0 || nodes.empty()) return 2U;
    const auto &mesh = meshes[parameters[body].mesh.index];
    if (!mesh.solid_planes) return 2U;
    unsigned skip = 2U;
    for (unsigned index = 0U; index < nodes.size() / 2U; ++index) {
        const Vec3 point = nodes[from_first ? index : nodes.size() - 1U - index];
        const Vec3 local = inverse_rotate(states[body].orientation,
            subtract(point, states[body].position));
        bool inside = true;
        for (unsigned triangle = 0U; triangle < mesh.index_count / 3U;
             ++triangle) {
            const auto plane = mesh.solid_planes[triangle];
            if (dot(plane.normal, local) - plane.offset > 0.0F) {
                inside = false;
                break;
            }
        }
        if (!inside) break;
        skip = std::max(skip, index + 2U);
    }
    return skip;
}

// Reject a rest pose threaded through an unrelated collider before allocating
// solver resources; stretch constraints cannot repair invalid topology.
bool rope_rest_crosses_collider(const std::vector<Vec3> &nodes,
    int first, int last, unsigned first_skip, unsigned last_skip,
    const BodyParameters *parameters, const RigidBodyState *states,
    const TriangleMeshResource *meshes, unsigned body_count) {
    for (unsigned body = 0U; body < body_count; ++body) {
        const auto &mesh = meshes[parameters[body].mesh.index];
        const auto &state = states[body];
        for (unsigned index = 0U; index + 1U < nodes.size(); ++index) {
            if ((int(body) == first && index < first_skip) ||
                (int(body) == last &&
                    index + last_skip + 1U >= nodes.size())) continue;
            const Vec3 a = inverse_rotate(state.orientation,
                subtract(nodes[index], state.position));
            const Vec3 b = inverse_rotate(state.orientation,
                subtract(nodes[index + 1U], state.position));
            const Vec3 low = component_min(a, b);
            const Vec3 high = component_max(a, b);
            if (!bounds_overlap(low, high, mesh.minimum, mesh.maximum))
                continue;
            std::vector<unsigned> pending{0U};
            while (!pending.empty()) {
                const auto node = mesh.bvh_nodes[pending.back()];
                pending.pop_back();
                if (!bounds_overlap(low, high, node.minimum, node.maximum))
                    continue;
                if (!node.triangle_count) {
                    pending.push_back(node.left);
                    pending.push_back(node.right);
                    continue;
                }
                for (unsigned triangle = node.first_triangle;
                     triangle < node.first_triangle + node.triangle_count;
                     ++triangle) {
                    const Vec3 x = mesh.vertices[mesh.indices[3U * triangle]];
                    const Vec3 y =
                        mesh.vertices[mesh.indices[3U * triangle + 1U]];
                    const Vec3 z =
                        mesh.vertices[mesh.indices[3U * triangle + 2U]];
                    const Vec3 normal = normalized_or(
                        cross(subtract(y, x), subtract(z, x)), {});
                    const float before = dot(normal, subtract(a, x));
                    const float after = dot(normal, subtract(b, x));
                    if (before * after > 0.0F ||
                        std::abs(before - after) < 1.0e-7F) continue;
                    const Vec3 point = add(a, multiply(subtract(b, a),
                        before / (before - after)));
                    if (length_squared(subtract(point,
                        closest_on_triangle(point, x, y, z))) < 1.0e-12F)
                        return true;
                }
            }
        }
    }
    return false;
}

// SPDX-License-Identifier: MIT
// Included by rope.cuh after the shared contact and soft-surface helpers.
struct RopeSoftBodyCouplingStorage {
    RopeSoftBodyCouplingOptions options{};
    std::uint32_t generation{1U};
    bool alive{};
    unsigned anchor_triangle[2]{~0U, ~0U};
    Vec3 anchor_weights[2]{}, anchor_offset[2]{};
    Vec3 *node_impulses{};
    Vec3 *bounds{};
    BvhNode *tree{};
    std::uint32_t *order{}, *parents{}, *ready{};
    std::uint32_t tree_count{};
    float orientation{1.0F};
    std::uint32_t *contact_count{};
    float *maximum_penetration{};
    void release() noexcept {
        release_managed(node_impulses);
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

struct RopeSoftTarget {
    const Vec3 *surface{}, *velocities{};
    const Vec3 *bounds{};
    const BvhNode *tree{};
    const std::uint32_t *order{};
    const std::uint32_t *indices{};
    const SoftBodySurfaceBinding *bindings{};
    const float *inverse_masses{};
    Vec3 *node_impulses{};
    std::uint32_t *contact_count{};
    float *maximum_penetration{};
    unsigned triangle_count{}, surface_vertex_count{};
    float distance{}, friction{};
    float orientation{1.0F};
    bool first_anchor{}, last_anchor{};
};

__global__ void rope_soft_sample_anchor(RopeData rope, unsigned end,
    const Vec3 *surface, const Vec3 *velocities,
    const std::uint32_t *indices, const SoftBodySurfaceBinding *bindings,
    unsigned triangle, Vec3 weights, Vec3 offset) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const unsigned base = 3U * triangle;
    const float w[3]{weights.x, weights.y, weights.z};
    Vec3 position = offset, velocity{};
    for (unsigned corner = 0; corner < 3; ++corner) {
        const unsigned vertex = indices[base + corner];
        position = add(position, multiply(surface[vertex], w[corner]));
        const auto binding = bindings[vertex];
        for (unsigned slot = 0; slot < 4; ++slot)
            velocity = add(velocity, multiply(velocities[binding.nodes[slot]],
                w[corner] * binding.weights[slot]));
    }
    rope.soft_anchor_positions[end] = position;
    rope.soft_anchor_velocities[end] = velocity;
    rope.soft_anchor_impulses[end] = {};
}

__global__ void rope_soft_scatter_anchor(RopeData rope, unsigned end,
    const std::uint32_t *indices, const SoftBodySurfaceBinding *bindings,
    unsigned triangle, Vec3 weights, Vec3 *node_impulses) {
    if(threadIdx.x!=0 || blockIdx.x!=0)return;
    const float w[3]{weights.x,weights.y,weights.z};
    for(unsigned corner=0;corner<3;++corner) {
        const auto binding=bindings[indices[3U*triangle+corner]];
        for(unsigned slot=0;slot<4;++slot) {
            const float weight=w[corner]*binding.weights[slot];
            if(weight>0)atomic_add(node_impulses+binding.nodes[slot],
                multiply(rope.soft_anchor_impulses[end],weight));
        }
    }
}

struct RopeSoftHit {
    float depth{}, fraction{};
    unsigned triangle{~0U};
    Vec3 normal{}, weights{}, point{};
};

__device__ RopeSoftHit rope_soft_find_contact(RopeData rope, RopeSoftTarget target,
    unsigned i, bool segment) {
    RopeSoftHit best{};
    if (!target.surface ||
        (target.first_anchor && i == 0U) ||
        (target.last_anchor && i + (segment ? 2U : 1U) == rope.count)) return best;
    const unsigned j = segment ? i + 1U : i;
    const Vec3 a = rope.positions[i], b = rope.positions[j], old = rope.previous[i];
    const float radius = target.distance;
    if(!segment) {
        // Share the closed-skin nearest/swept query with fluid contacts. A
        // local maximum-penetration search can select the back wall of a thin
        // soft solid; nearest-surface classification keeps the rope outside.
        const auto hit=fluid_soft_nearest(a,old,target.surface,target.surface,
            target.indices,target.triangle_count*3U,target.orientation,radius,
            target.bounds,true,target.tree,target.order);
        if(hit.triangle!=k_invalid_dense)
            best={hit.penetration,0,hit.triangle/3U,hit.normal,hit.weights,hit.point};
        return best;
    }
    const Vec3 margin{radius, radius, radius};
    const Vec3 low = subtract(component_min(a, segment ? b : old), margin);
    const Vec3 high = add(component_max(a, segment ? b : old), margin);
    if(!bounds_overlap(low,high,target.bounds[0],target.bounds[1]))return best;
    unsigned stack[64]{0};unsigned pending=1;
    while(pending) {
        const auto node=target.tree[stack[--pending]];
        if(!bounds_overlap(low,high,node.minimum,node.maximum))continue;
        if(!node.triangle_count) {
            if(pending+2U<=64U) {
                stack[pending++]=node.left;
                stack[pending++]=node.right;
            }
            continue;
        }
        for(unsigned leaf=0;leaf<node.triangle_count;++leaf) {
        const unsigned t=target.order[node.first_triangle+leaf];
        const unsigned base = 3U * t;
        const Vec3 x = target.surface[target.indices[base]],
                   y = target.surface[target.indices[base + 1U]],
                   z = target.surface[target.indices[base + 2U]];
        if (!bounds_overlap(low, high, component_min(x, component_min(y,z)),
                component_max(x, component_max(y,z)))) continue;
        Vec3 p = a, q{};
        if (segment) rope_closest_segment_triangle(a,b,x,y,z,p,q);
        else q = closest_on_triangle(a,x,y,z);
        const Vec3 delta = subtract(p,q);
        const float distance = vector_length(delta);
        if (distance >= radius && segment) continue;
        const Vec3 face = multiply(normalized_or(cross(subtract(y,x),subtract(z,x)),{}),
            target.orientation);
        if (length_squared(face) < 0.5F) continue;
        const float side = dot(delta,face);
        Vec3 normal = distance > 1e-7F ? multiply(delta,1.0F/distance) : face;
        float depth = radius-distance;
        if (side < -1e-7F) { normal=face; depth=radius+distance; }
        if (!segment) {
            const float before = dot(subtract(old,x),face);
            const float after = dot(subtract(a,x),face);
            if (before > 0 && after < 0) {
                const Vec3 crossing = add(old,multiply(subtract(a,old),
                    before/fmaxf(before-after,1e-12F)));
                if (length_squared(subtract(crossing,
                        closest_on_triangle(crossing,x,y,z))) <= radius*radius) {
                    normal=face;
                    depth=fmaxf(depth,radius-after);
                }
            }
        }
        if (depth <= best.depth) continue;
        const Vec3 edge = subtract(b,a);
        const float fraction = segment ? clamp_scalar(
            dot(subtract(p,a),edge)/fmaxf(length_squared(edge),1e-12F),0,1) : 0;
        Vec3 weights{};
        q=fluid_closest_triangle_barycentric(q,x,y,z,weights);
        best={depth,fraction,t,normal,weights,q};
        }
    }
    return best;
}

__device__ void rope_soft_contact(RopeData rope, RopeSoftTarget target,
    unsigned i, bool segment, float dt, int first, int last,
    const BodyParameters *parameters, const RigidBodyState *states) {
    const auto hit=rope_soft_find_contact(rope,target,i,segment);
    if (hit.triangle==~0U) return;
    const unsigned j=segment?i+1U:i;
    const float a=1.0F-hit.fraction,b=hit.fraction;
    const float wa=rope_contact_weight(rope,i,hit.normal,first,last,parameters,states);
    const float wb=segment?rope_contact_weight(rope,j,hit.normal,first,last,parameters,states):0;
    const unsigned base=3U*hit.triangle;
    const float weights[3]{hit.weights.x,hit.weights.y,hit.weights.z};
    float soft_weight=0;
    Vec3 soft_velocity{};
    for (unsigned corner=0;corner<3;++corner) {
        const auto binding=target.bindings[target.indices[base+corner]];
        for(unsigned slot=0;slot<4;++slot) {
            const unsigned node=binding.nodes[slot];
            const float weight=weights[corner]*binding.weights[slot];
            soft_weight+=weight*weight*target.inverse_masses[node];
            soft_velocity=add(soft_velocity,multiply(target.velocities[node],weight));
        }
    }
    // The soft solve applies a capped reaction later and its shape constraints
    // can restore this surface. Do not count that unrealized displacement when
    // projecting the rope out of the current skin.
    const float sum=wa*a*a+wb*b*b;
    if(sum<=1e-12F)return;
    const float correction=hit.depth/sum;
    Vec3 impulse=multiply(hit.normal,correction);
    const Vec3 motion=subtract(add(multiply(subtract(rope.positions[i],rope.previous[i]),a),
        multiply(subtract(rope.positions[j],rope.previous[j]),b)),multiply(soft_velocity,dt));
    const Vec3 tangent=subtract(motion,multiply(hit.normal,dot(motion,hit.normal)));
    const float speed=vector_length(tangent);
    if(speed>1e-8F) {
        const Vec3 direction=multiply(tangent,1.0F/speed);
        const float tw=rope_contact_weight(rope,i,direction,first,last,parameters,states)*a*a+
            (segment?rope_contact_weight(rope,j,direction,first,last,parameters,states)*b*b:0);
        impulse=subtract(impulse,multiply(direction,
            fminf(speed/fmaxf(tw,1e-12F),target.friction*correction)));
    }
    rope_contact_move(rope,i,multiply(impulse,a),first,last,parameters,states);
    rope.soft_body_contact_forces[i]=add(rope.soft_body_contact_forces[i],
        multiply(impulse,a/(dt*dt)));
    if(segment) {
        rope_contact_move(rope,j,multiply(impulse,b),first,last,parameters,states);
        rope.soft_body_contact_forces[j]=add(rope.soft_body_contact_forces[j],
            multiply(impulse,b/(dt*dt)));
    }
    if(soft_weight<1e-9F) {
        for(unsigned node=i;node<=j;++node) {
            if(length_squared(rope.normals[node])<0.5F ||
               dot(rope.normals[node],hit.normal)>0.95F)rope.normals[node]=hit.normal;
            else rope.normals2[node]=hit.normal;
        }
    }
    for(unsigned corner=0;corner<3;++corner) {
        const auto binding=target.bindings[target.indices[base+corner]];
        for(unsigned slot=0;slot<4;++slot) {
            const float weight=weights[corner]*binding.weights[slot];
            if(weight>0)atomic_add(target.node_impulses+binding.nodes[slot],
                multiply(impulse,-weight/dt));
        }
    }
    atomicAdd(target.contact_count,1U);
    atomicMax(reinterpret_cast<int *>(target.maximum_penetration),
        __float_as_int(hit.depth));
}

__global__ void rope_soft_apply(Vec3 *positions, Vec3 *velocities,
    Vec3 *forces, const float *inverse_masses, const Vec3 *impulses,
    unsigned count, float maximum_acceleration, float maximum_speed,
    float dt, float frame_inverse_dt) {
    const unsigned node=blockIdx.x*blockDim.x+threadIdx.x;
    if(node>=count || inverse_masses[node]==0)return;
    const Vec3 impulse=clamp_length(multiply(impulses[node],inverse_masses[node]),
        maximum_acceleration*dt);
    velocities[node]=clamp_length(add(velocities[node],impulse),maximum_speed);
    positions[node]=add(positions[node],multiply(impulse,dt));
    forces[node]=add(forces[node],multiply(impulse,frame_inverse_dt/inverse_masses[node]));
}

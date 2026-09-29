// SPDX-License-Identifier: MIT
// Included after the shared triangle/contact helpers in world.cu.
struct RopeData {
    RopeOptions options{};
    std::uint32_t count{};
    Vec3 *positions{}, *previous{}, *velocities{}, *constraint_forces{}, *contact_forces{}, *fluid_contact_forces{};
    Vec3 *directions{}, *scratch{}, *body_translation{}, *body_rotation{};
    Vec3 *normals{}, *normals2{};
    float *rest{}, *lambda{};
    unsigned *solid_hint{};
    unsigned body_capacity{};
};
struct RopeStorage {
    RopeData data{};
    std::uint32_t generation{1};
    bool alive{};
    void release() {
        release_managed(data.positions); release_managed(data.previous);
        release_managed(data.velocities); release_managed(data.constraint_forces);
        release_managed(data.contact_forces); release_managed(data.fluid_contact_forces);
        release_managed(data.directions);
        release_managed(data.scratch); release_managed(data.body_translation);
        release_managed(data.body_rotation); release_managed(data.rest);
        release_managed(data.lambda);
        release_managed(data.normals); release_managed(data.normals2);
        release_managed(data.solid_hint);
    }
    ~RopeStorage() { release(); }
};

// Reject topologically invalid rest poses before allocating solver storage.
// Stretch constraints cannot repair a rope threaded through an unrelated wall.
bool rope_rest_crosses_collider(const std::vector<Vec3> &nodes, int first, int last,
    const BodyParameters *parameters, const RigidBodyState *states,
    const TriangleMeshResource *meshes, unsigned body_count) {
    for (unsigned body = 0; body < body_count; ++body) {
        const auto &mesh = meshes[parameters[body].mesh.index];
        const auto &state = states[body];
        for (unsigned i = 0; i + 1 < nodes.size(); ++i) {
            if ((int(body) == first && i < 2) ||
                (int(body) == last && i + 3 >= nodes.size())) continue;
            const auto a = inverse_rotate(state.orientation, subtract(nodes[i], state.position));
            const auto b = inverse_rotate(state.orientation, subtract(nodes[i + 1], state.position));
            const auto low = component_min(a, b), high = component_max(a, b);
            if (!bounds_overlap(low, high, mesh.minimum, mesh.maximum)) continue;
            std::vector<unsigned> pending{0};
            while (!pending.empty()) {
                const auto node = mesh.bvh_nodes[pending.back()];
                pending.pop_back();
                if (!bounds_overlap(low, high, node.minimum, node.maximum)) continue;
                if (!node.triangle_count) {
                    pending.push_back(node.left); pending.push_back(node.right);
                    continue;
                }
                for (unsigned t = node.first_triangle; t < node.first_triangle + node.triangle_count; ++t) {
                    const auto x = mesh.vertices[mesh.indices[3*t]],
                               y = mesh.vertices[mesh.indices[3*t+1]],
                               z = mesh.vertices[mesh.indices[3*t+2]];
                    const auto normal = normalized_or(cross(subtract(y,x), subtract(z,x)), {});
                    const float before = dot(normal, subtract(a,x)), after = dot(normal, subtract(b,x));
                    if (before * after > 0 || std::abs(before - after) < 1e-7F) continue;
                    const auto point = add(a, multiply(subtract(b,a), before / (before - after)));
                    if (length_squared(subtract(point, closest_on_triangle(point,x,y,z))) < 1e-12F)
                        return true;
                }
            }
        }
    }
    return false;
}

__device__ int rope_anchor_body(RopeData r, unsigned node, int first, int last) {
    return node==0 ? first : (node+1==r.count ? last : -1);
}
__device__ Vec3 rope_anchor_local(RopeData r, unsigned node) {
    return node==0 ? r.options.first.local_anchor : r.options.last.local_anchor;
}
__device__ float rope_node_weight(RopeData r, unsigned node, int first, int last) {
    return rope_anchor_body(r,node,first,last)>=0 ? 0.0F : float(r.count)/r.options.mass;
}
__device__ Vec3 rope_mass(RopeData r,unsigned node,Vec3 vector,int first,int last) {
    const auto n=length_squared(r.normals[node])>0.5F?r.normals[node]:r.normals2[node];
    if(length_squared(n)>0.5F) {
        // Build positive-semidefinite projectors from orthogonal tangents.
        // Subtracting two almost-parallel normals can produce an indefinite
        // inverse mass matrix after float cancellation at triangle seams.
        const auto edge=cross(n,r.normals2[node]);
        if(length_squared(edge)>1e-4F) {
            const auto tangent=normalized_or(edge,{});
            vector=multiply(tangent,dot(tangent,vector));
        } else vector=subtract(vector,multiply(n,dot(n,vector)));
    }
    return multiply(vector,rope_node_weight(r,node,first,last));
}
__device__ float rope_direction_weight(RopeData r, unsigned node, Vec3 direction,
    int first, int last, const BodyParameters *parameters, const RigidBodyState *states) {
    const int body=rope_anchor_body(r,node,first,last);
    if(body<0) return dot(direction,rope_mass(r,node,direction,first,last));
    const Vec3 arm=rotate(states[body].orientation,rope_anchor_local(r,node));
    const Vec3 torque=cross(arm,direction);
    return parameters[body].inverse_mass + dot(torque,inverse_inertia_world(parameters[body],states[body],torque));
}
__device__ void rope_move_body(RigidBodyState &state, Vec3 translation, Vec3 rotation, float dt) {
    state.position=add(state.position,translation);
    const Quaternion spin=quaternion_multiply({rotation.x,rotation.y,rotation.z,0},state.orientation);
    state.orientation=normalized_quaternion({state.orientation.x+0.5F*spin.x,
        state.orientation.y+0.5F*spin.y,state.orientation.z+0.5F*spin.z,state.orientation.w+0.5F*spin.w});
    state.linear_velocity=add(state.linear_velocity,multiply(translation,1.0F/dt));
    state.angular_velocity=add(state.angular_velocity,multiply(rotation,1.0F/dt));
}
__device__ void rope_sync_anchors(RopeData r,int first,int last,const RigidBodyState *states) {
    if(first>=0) r.positions[0]=transform_point(states[first],r.options.first.local_anchor);
    if(last>=0) r.positions[r.count-1]=transform_point(states[last],r.options.last.local_anchor);
}

__device__ void rope_solve_tridiagonal(unsigned edges, float *diagonal,
    const float *upper, float *rhs) {
    for (unsigned e=1;e<edges;++e) {
        const float factor=upper[e-1]/fmaxf(diagonal[e-1],1e-10F);
        diagonal[e]-=factor*upper[e-1];
        rhs[e]-=factor*rhs[e-1];
    }
    rhs[edges-1]/=fmaxf(diagonal[edges-1],1e-10F);
    for (int e=int(edges)-2;e>=0;--e)
        rhs[e]=(rhs[e]-upper[e]*rhs[e+1])/fmaxf(diagonal[e],1e-10F);
}

// The open chain's linearized distance system is tridiagonal. A direct solve
// propagates endpoint tension through the entire rope in one iteration instead
// of requiring O(node_count^2) Jacobi sweeps for a stiff, heavy-ended chain.
__device__ void rope_project_stretch(RopeData r,float dt,int first,int last,
    const BodyParameters *parameters,RigidBodyState *states) {
    const unsigned tid=threadIdx.x;
    const unsigned edges=r.count-1;
    __shared__ float maximum;
    // Forward/back substitution is serial, so dependent reads belong in
    // block-local memory rather than paying device-memory latency per edge.
    __shared__ float diagonal[1024], upper[1024], rhs[1024];
    __shared__ unsigned released;
    if(tid==0)maximum=0;
    // Active contact planes can remove an entire directional degree of
    // freedom. Keep the reduced system positive definite at float precision.
    const float alpha=fmaxf(r.options.stretch_compliance/(dt*dt),1e-5F*float(r.count)/r.options.mass);
    for(unsigned e=tid;e<edges;e+=blockDim.x) {
        const Vec3 d=subtract(r.positions[e+1],r.positions[e]);
        r.directions[e]=normalized_or(d,{1,0,0});
    }
    __syncthreads();
    for(unsigned active_pass=0;active_pass<4;++active_pass) {
    if(tid==0)released=0;
    for(unsigned e=tid;e<edges;e+=blockDim.x) {
        rhs[e]=-(vector_length(subtract(r.positions[e+1],r.positions[e]))-r.rest[e])-alpha*r.lambda[e];
        diagonal[e]=rope_direction_weight(r,e,r.directions[e],first,last,parameters,states)+
            rope_direction_weight(r,e+1,r.directions[e],first,last,parameters,states)+alpha;
        if(e)upper[e-1]=-dot(r.directions[e-1],rope_mass(r,e,r.directions[e],first,last));
    }
    __syncthreads();
    if(tid==0)rope_solve_tridiagonal(edges,diagonal,upper,rhs);
    __syncthreads();
    for(unsigned i=tid;i<r.count;i+=blockDim.x) {
        Vec3 impulse{};
        if(i) impulse=add(impulse,multiply(r.directions[i-1],rhs[i-1]));
        if(i<edges) impulse=subtract(impulse,multiply(r.directions[i],rhs[i]));
        r.scratch[i]=impulse;
        const int body=rope_anchor_body(r,i,first,last);
        // Release unilateral contacts, then rebuild the matrix before moving
        // anything. Applying a solution with a different inverse mass than
        // the one used to solve it injected energy at resting contacts.
        if(body<0 && active_pass<3) {
            const float release_impulse=1e-6F/rope_node_weight(r,i,first,last);
            if(dot(impulse,r.normals[i])>release_impulse){r.normals[i]={};atomicExch(&released,1U);}
            if(dot(impulse,r.normals2[i])>release_impulse){r.normals2[i]={};atomicExch(&released,1U);}
        }
    }
    __syncthreads();
    // Every warp must consume this pass's flag before thread zero can clear
    // it for the next pass. Otherwise warps can take different loop counts.
    const bool active_set_unchanged=released==0;
    __syncthreads();
    if(active_set_unchanged)break;
    }
    for(unsigned i=tid;i<r.count;i+=blockDim.x) {
        const Vec3 impulse=r.scratch[i];
        const int body=rope_anchor_body(r,i,first,last);
        Vec3 movement=rope_mass(r,i,impulse,first,last);
        if(body>=0) {
            const auto arm=rotate(states[body].orientation,rope_anchor_local(r,i));
            movement=add(multiply(impulse,parameters[body].inverse_mass),
                cross(inverse_inertia_world(parameters[body],states[body],cross(arm,impulse)),arm));
        }
        atomicMax(reinterpret_cast<unsigned *>(&maximum),__float_as_uint(vector_length(movement)));
    }
    __syncthreads();
    // Bound each direct solve to one radius so a loaded endpoint can move
    // without a single pass jumping across nearby triangles.
    const float scale=fminf(1.0F,r.options.radius/fmaxf(maximum,1e-10F));
    for(unsigned e=tid;e<edges;e+=blockDim.x) r.lambda[e]+=scale*rhs[e];
    for(unsigned i=tid;i<r.count;i+=blockDim.x) {
        const Vec3 impulse=multiply(r.scratch[i],scale);
        r.constraint_forces[i]=add(r.constraint_forces[i],multiply(impulse,1/(dt*dt)));
        const int body=rope_anchor_body(r,i,first,last);
        if(body<0) r.positions[i]=add(r.positions[i],rope_mass(r,i,impulse,first,last));
    }
    __syncthreads();
    if(tid==0) {
        for(unsigned i:{0U,r.count-1}) {
            const int body=rope_anchor_body(r,i,first,last);
            if(body<0)continue;
            const Vec3 impulse=multiply(r.scratch[i],scale);
            const Vec3 arm=rotate(states[body].orientation,rope_anchor_local(r,i));
            rope_move_body(states[body],multiply(impulse,parameters[body].inverse_mass),
                inverse_inertia_world(parameters[body],states[body],cross(arm,impulse)),dt);
        }
        rope_sync_anchors(r,first,last,states);
    }
    __syncthreads();
}

struct RopeHit { float depth{}; Vec3 normal{},point{}; int body{-1}; float fraction{}; };
__device__ void rope_closest_segment_triangle(Vec3 a,Vec3 b,Vec3 x,Vec3 y,Vec3 z,Vec3 &p,Vec3 &q) {
    if(segment_hits_triangle(a,b,x,y,z,p)) {q=p;return;}
    float best=FLT_MAX;
    consider_closest_pair(a,closest_on_triangle(a,x,y,z),best,p,q);
    consider_closest_pair(b,closest_on_triangle(b,x,y,z),best,p,q);
    const Vec3 corners[3]{x,y,z};
    for(unsigned i=0;i<3;++i) {
        Vec3 u,v;closest_segments(a,b,corners[i],corners[(i+1)%3],u,v);
        consider_closest_pair(u,v,best,p,q);
    }
}

__device__ RopeHit rope_find_contact(RopeData r,unsigned i,bool segment,int first,int last,
    const BodyParameters *parameters,const RigidBodyState *states,
    const RigidBodyState *old_states,const TriangleMeshResource *meshes,unsigned body_count) {
    RopeHit best;
    const unsigned j=segment?i+1:i;
    for(unsigned body=0;body<body_count;++body) {
        // Only the endpoint neighbourhood is exempt from its own attachment
        // target. Distant sections still collide with that same body.
        if((int(body)==first && i<2) || (int(body)==last && j+2>=r.count)) continue;
        const auto state=states[body];
        const auto old=old_states?old_states[body]:state;
        const auto mesh=meshes[parameters[body].mesh.index];
        const float radius=r.options.radius+parameters[body].collision_margin;
        const Vec3 a=inverse_rotate(state.orientation,subtract(r.positions[i],state.position));
        const Vec3 b=inverse_rotate(state.orientation,subtract(r.positions[j],state.position));
        const Vec3 origin=inverse_rotate(old.orientation,subtract(r.previous[i],old.position));
        const Vec3 lo=subtract(component_min(a,segment?b:origin),{radius,radius,radius});
        const Vec3 hi=add(component_max(a,segment?b:origin),{radius,radius,radius});
        if(!bounds_overlap(lo,hi,mesh.minimum,mesh.maximum))continue;
        if(mesh.solid_planes) {
            // A convex solid's separating plane is reusable across contact
            // passes. Verify it at the current position before using it;
            // winding changes which face separates the rope from the post.
            unsigned &hint=r.solid_hint[i*r.body_capacity+body];
            const unsigned triangles=mesh.index_count/3;
            const bool stationary=state.position.x==old.position.x &&
                state.position.y==old.position.y && state.position.z==old.position.z &&
                state.orientation.x==old.orientation.x && state.orientation.y==old.orientation.y &&
                state.orientation.z==old.orientation.z && state.orientation.w==old.orientation.w;
            bool outside=false, separated=false;
            if(hint<triangles) {
                const auto plane=mesh.solid_planes[hint];
                const float side=dot(plane.normal,a)-plane.offset;
                if(side>0) {
                    outside=true;
                    separated=stationary && side>radius &&
                        (segment ? dot(plane.normal,b)-plane.offset>radius :
                         dot(plane.normal,origin)-plane.offset>radius);
                }
            }
            float side=-FLT_MAX,entry_time=-1;CollisionPlane nearest{},entry{};
            for(unsigned t=0;!outside && t<triangles;++t) {
                const auto plane=mesh.solid_planes[t];
                const float s=dot(plane.normal,a)-plane.offset;
                if(s>side){side=s;nearest=plane;}
                if(!segment) {
                    const float before=dot(plane.normal,origin)-plane.offset;
                    if(before>0 && s<=0) {
                        const float time=before/(before-s);
                        if(time>entry_time){entry_time=time;entry=plane;}
                    }
                }
                if(s>0) {
                    hint=t;outside=true;
                    separated=stationary && s>radius &&
                        (segment ? dot(plane.normal,b)-plane.offset>radius :
                         dot(plane.normal,origin)-plane.offset>radius);
                }
            }
            if(separated)continue;
            if(!outside && !segment) {
                hint=~0U;
                // Recover on the entry side, not the closest exit face. A
                // rolling body can otherwise expel adjacent rope nodes to
                // opposite sides and trap the segment through its interior.
                if(entry_time>=0){nearest=entry;side=dot(entry.normal,a)-entry.offset;}
                if(radius-side>best.depth)
                    best={radius-side,rotate(state.orientation,nearest.normal),
                        transform_point(state,subtract(a,multiply(nearest.normal,side))),int(body),0};
                continue;
            }
        }
        unsigned stack[64]{0};int pending=mesh.bvh_node_count?1:0;
        while(pending) {
            const auto node=mesh.bvh_nodes[stack[--pending]];
            if(!bounds_overlap(lo,hi,node.minimum,node.maximum))continue;
            if(!node.triangle_count){if(pending+2<=64){stack[pending++]=node.left;stack[pending++]=node.right;}continue;}
            for(unsigned t=node.first_triangle;t<node.first_triangle+node.triangle_count;++t) {
                const Vec3 x=mesh.vertices[mesh.indices[3*t]],y=mesh.vertices[mesh.indices[3*t+1]],z=mesh.vertices[mesh.indices[3*t+2]];
                if(!bounds_overlap(lo,hi,component_min(x,component_min(y,z)),
                        component_max(x,component_max(y,z))))continue;
                Vec3 p=a,q;
                if(segment)rope_closest_segment_triangle(a,b,x,y,z,p,q);
                else q=closest_on_triangle(a,x,y,z);
                const Vec3 delta=subtract(p,q);
                const float distance=vector_length(delta);
                Vec3 face=normalized_or(cross(subtract(y,x),subtract(z,x)),{0,1,0});
                if(mesh.solid_planes)face=mesh.solid_planes[t].normal;
                const float before=dot(subtract(origin,x),face),after=dot(subtract(a,x),face);
                Vec3 normal=distance>1e-7F?multiply(delta,1/distance):multiply(face,before>=0?1.0F:-1.0F);
                float depth=radius-distance;
                if(segment && mesh.solid_planes && distance<radius && dot(delta,face)<=1e-7F) {
                    // A closed solid is not a two-sided sheet. Segment
                    // recovery must agree with node recovery, otherwise a
                    // capsule crossing a face is pushed back into the body.
                    normal=face;
                    depth=radius+distance;
                }
                if(!segment && before*after<0) {
                    const Vec3 crossing=add(origin,multiply(subtract(a,origin),before/(before-after)));
                    if(length_squared(subtract(closest_on_triangle(crossing,x,y,z),crossing))<=radius*radius) {
                        normal=multiply(face,before>=0?1.0F:-1.0F);
                        depth=fmaxf(depth,radius+fabsf(after));
                    }
                }
                if(depth>best.depth) {
                    const Vec3 edge=subtract(b,a);
                    const float fraction=segment?clamp_scalar(dot(subtract(p,a),edge)/fmaxf(length_squared(edge),1e-12F),0,1):0;
                    best={depth,rotate(state.orientation,normal),transform_point(state,q),int(body),fraction};
                }
            }
        }
    }
    return best;
}
__device__ void rope_accumulate_body(RopeData r,unsigned body,Vec3 impulse,Vec3 arm,
    const BodyParameters *parameters,const RigidBodyState *states) {
    const Vec3 translation=multiply(impulse,parameters[body].inverse_mass);
    const Vec3 rotation=inverse_inertia_world(parameters[body],states[body],cross(arm,impulse));
    atomicAdd(&r.body_translation[body].x,translation.x);atomicAdd(&r.body_translation[body].y,translation.y);atomicAdd(&r.body_translation[body].z,translation.z);
    atomicAdd(&r.body_rotation[body].x,rotation.x);atomicAdd(&r.body_rotation[body].y,rotation.y);atomicAdd(&r.body_rotation[body].z,rotation.z);
}
__device__ float rope_contact_weight(RopeData r,unsigned node,Vec3 direction,int first,int last,
    const BodyParameters *parameters,const RigidBodyState *states) {
    return rope_anchor_body(r,node,first,last)<0?float(r.count)/r.options.mass:
        rope_direction_weight(r,node,direction,first,last,parameters,states);
}
__device__ void rope_contact_move(RopeData r,unsigned node,Vec3 impulse,int first,int last,
    const BodyParameters *parameters,const RigidBodyState *states) {
    const int body=rope_anchor_body(r,node,first,last);
    if(body<0)r.positions[node]=add(r.positions[node],multiply(impulse,float(r.count)/r.options.mass));
    else rope_accumulate_body(r,body,impulse,rotate(states[body].orientation,rope_anchor_local(r,node)),parameters,states);
}
__device__ void rope_contact(RopeData r,unsigned i,bool segment,float dt,int first,int last,
    const BodyParameters *parameters,const RigidBodyState *states,
    const RigidBodyState *old_states,const TriangleMeshResource *meshes,unsigned body_count) {
    const RopeHit hit=rope_find_contact(r,i,segment,first,last,parameters,states,old_states,meshes,body_count);
    if(hit.body<0)return;
    const unsigned j=segment?i+1:i;
    const float a=1-hit.fraction,b=hit.fraction;
    const float wa=rope_contact_weight(r,i,hit.normal,first,last,parameters,states);
    const float wb=segment?rope_contact_weight(r,j,hit.normal,first,last,parameters,states):0;
    const auto body=parameters[hit.body];const auto state=states[hit.body];
    const Vec3 arm=subtract(hit.point,state.position),torque=cross(arm,hit.normal);
    const float body_weight=body.inverse_mass+dot(torque,inverse_inertia_world(body,state,torque));
    const float sum=wa*a*a+wb*b*b+body_weight;
    if(sum<=1e-12F)return;
    Vec3 impulse=multiply(hit.normal,hit.depth/sum);
    const Vec3 movement=subtract(add(multiply(subtract(r.positions[i],r.previous[i]),a),
        multiply(subtract(r.positions[j],r.previous[j]),b)),
        multiply(add(state.linear_velocity,cross(state.angular_velocity,arm)),dt));
    const Vec3 tangent=subtract(movement,multiply(hit.normal,dot(movement,hit.normal)));
    const float length=vector_length(tangent);
    if(length>1e-8F) {
        const Vec3 direction=multiply(tangent,1/length),axis=cross(arm,direction);
        const float denom=rope_contact_weight(r,i,direction,first,last,parameters,states)*a*a+
            (segment?rope_contact_weight(r,j,direction,first,last,parameters,states)*b*b:0)+
            body.inverse_mass+dot(axis,inverse_inertia_world(body,state,axis));
        impulse=subtract(impulse,multiply(direction,fminf(length/fmaxf(denom,1e-12F),r.options.friction*hit.depth/sum)));
    }
    rope_contact_move(r,i,multiply(impulse,a),first,last,parameters,states);
    r.contact_forces[i]=add(r.contact_forces[i],multiply(impulse,a/(dt*dt)));
    if(segment){rope_contact_move(r,j,multiply(impulse,b),first,last,parameters,states);r.contact_forces[j]=add(r.contact_forces[j],multiply(impulse,b/(dt*dt)));}
    if(body.inverse_mass==0) {
        for(unsigned node=i;node<=j;++node) {
            if(length_squared(r.normals[node])<0.5F || dot(r.normals[node],hit.normal)>0.95F)
                r.normals[node]=hit.normal;
            else r.normals2[node]=hit.normal;
        }
    }
    rope_accumulate_body(r,hit.body,multiply(impulse,-1),arm,parameters,states);
}


// Project axial velocity using the full mass matrix. Contact-reduced masses
// are valid for positional support, but not for an arbitrary incoming velocity:
// an unsatisfiable normal component can otherwise amplify endpoint impulses.
__device__ void rope_project_velocities(RopeData r,float dt,int first,int last,
    const BodyParameters *parameters,RigidBodyState *states) {
    const unsigned tid=threadIdx.x,edges=r.count-1;
    __shared__ float diagonal[1024],upper[1024],rhs[1024];
    const float alpha=fmaxf(r.options.stretch_compliance/(dt*dt),1e-5F*float(r.count)/r.options.mass);
    for(unsigned e=tid;e<edges;e+=blockDim.x) {
        r.directions[e]=normalized_or(subtract(r.positions[e+1],r.positions[e]),{1,0,0});
        rhs[e]=-dot(r.directions[e],subtract(r.velocities[e+1],r.velocities[e]));
    }
    __syncthreads();
    for(unsigned e=tid;e<edges;e+=blockDim.x) {
        diagonal[e]=rope_contact_weight(r,e,r.directions[e],first,last,parameters,states)+
            rope_contact_weight(r,e+1,r.directions[e],first,last,parameters,states)+alpha;
        if(e)upper[e-1]=-dot(r.directions[e-1],r.directions[e])*rope_node_weight(r,e,first,last);
    }
    __syncthreads();
    if(tid==0)rope_solve_tridiagonal(edges,diagonal,upper,rhs);
    __syncthreads();
    for(unsigned i=tid;i<r.count;i+=blockDim.x) {
        Vec3 impulse{};
        if(i)impulse=add(impulse,multiply(r.directions[i-1],rhs[i-1]));
        if(i<edges)impulse=subtract(impulse,multiply(r.directions[i],rhs[i]));
        r.scratch[i]=impulse;
        r.constraint_forces[i]=add(r.constraint_forces[i],multiply(impulse,1/dt));
        if(rope_anchor_body(r,i,first,last)<0)
            r.velocities[i]=add(r.velocities[i],multiply(impulse,rope_node_weight(r,i,first,last)));
    }
    __syncthreads();
    if(tid==0)for(unsigned i:{0U,r.count-1}) {
        const int body=rope_anchor_body(r,i,first,last);
        if(body<0)continue;
        const auto arm=rotate(states[body].orientation,rope_anchor_local(r,i));
        states[body].linear_velocity=add(states[body].linear_velocity,multiply(r.scratch[i],parameters[body].inverse_mass));
        states[body].angular_velocity=add(states[body].angular_velocity,
            inverse_inertia_world(parameters[body],states[body],cross(arm,r.scratch[i])));
        r.velocities[i]=add(states[body].linear_velocity,cross(states[body].angular_velocity,arm));
    }
    __syncthreads();
}

__global__ void rope_advance(RopeData r,float dt,Vec3 gravity,int first,int last,
    const BodyParameters *parameters,RigidBodyState *states,const RigidBodyState *old_states,
    const TriangleMeshResource *meshes,unsigned body_count,bool first_substep) {
    const unsigned tid=threadIdx.x;
    __shared__ bool converged;
    if(tid==0)rope_sync_anchors(r,first,last,old_states?old_states:states);
    __syncthreads();
    for(unsigned i=tid;i<r.count;i+=blockDim.x) {
        r.previous[i]=r.positions[i];
        r.normals[i]=r.normals2[i]={};
        if(first_substep){r.constraint_forces[i]={};r.contact_forces[i]={};r.fluid_contact_forces[i]={};}
        if(rope_anchor_body(r,i,first,last)<0) {
            r.velocities[i]=clamp_length(multiply(add(r.velocities[i],multiply(gravity,dt)),expf(-r.options.velocity_damping*dt)),r.options.maximum_speed);
            r.positions[i]=add(r.positions[i],multiply(r.velocities[i],dt));
        }
        if(i+1<r.count)r.lambda[i]=0;
    }
    __syncthreads();
    if(tid==0)rope_sync_anchors(r,first,last,states);
    __syncthreads();
    // A sharp contact can need more nonlinear sweeps than a resting chain.
    // Spend the recovery budget only while strain is still above 0.5%; never
    // carry a large residual into the next step as an artificial velocity.
    const unsigned maximum_iterations=min(32U,4U*r.options.solver_iterations);
    for(unsigned iteration=0;iteration<maximum_iterations;++iteration) {
        rope_project_stretch(r,dt,first,last,parameters,states);
        __syncthreads();
        // Node sweeps catch travel through thin triangles. Segment contacts then
        // close the gaps between nodes, including when winding around a post.
        for(unsigned phase=0;phase<3;++phase) {
            for(unsigned body=tid;body<body_count;body+=blockDim.x){r.body_translation[body]={};r.body_rotation[body]={};}
            __syncthreads();
            if(phase==0) {
                for(unsigned i=tid;i<r.count;i+=blockDim.x)
                    rope_contact(r,i,false,dt,first,last,parameters,states,old_states,meshes,body_count);
            } else {
                for(unsigned i=2*tid+phase-1;i+1<r.count;i+=2*blockDim.x)
                    rope_contact(r,i,true,dt,first,last,parameters,states,old_states,meshes,body_count);
            }
            __syncthreads();
            if(tid==0) {
                for(unsigned body=0;body<body_count;++body)rope_move_body(states[body],r.body_translation[body],r.body_rotation[body],dt);
                rope_sync_anchors(r,first,last,states);
            }
            __syncthreads();
        }
        if(r.options.self_collision) {
            for(unsigned i=tid;i<r.count;i+=blockDim.x) {
                Vec3 correction{};unsigned hits=0;
                const float wi=rope_node_weight(r,i,first,last);
                for(unsigned j=0;j<r.count && wi>0;++j) {
                    if(abs(int(i)-int(j))<=2)continue;
                    const Vec3 d=subtract(r.positions[i],r.positions[j]);
                    const float length=vector_length(d),wj=rope_node_weight(r,j,first,last);
                    if(length>=2*r.options.radius || wi+wj<=0)continue;
                    correction=add(correction,multiply(normalized_or(d,{1,0,0}),
                        (2*r.options.radius-length)*wi/(wi+wj)));++hits;
                }
                r.scratch[i]=hits?multiply(correction,1.0F/hits):Vec3{};
            }
            __syncthreads();
            for(unsigned i=tid;i<r.count;i+=blockDim.x)r.positions[i]=add(r.positions[i],r.scratch[i]);
            __syncthreads();
        }
        if(tid==0) {
            float strain=0;
            for(unsigned i=0;i+1<r.count;++i)
                strain=fmaxf(strain,fabsf(vector_length(subtract(r.positions[i+1],r.positions[i]))/r.rest[i]-1));
            converged=(iteration>=1 && strain<1e-3F) ||
                (iteration+1>=r.options.solver_iterations && strain<0.005F);
        }
        __syncthreads();
        if(converged)break;
    }
    if(!converged) {
        // A taut wrap can exhaust contact passes with stale support planes.
        // Finish the substep by removing the remaining length error.
        for(unsigned i=tid;i<r.count;i+=blockDim.x)r.normals[i]=r.normals2[i]={};
        __syncthreads();
        for(unsigned pass=0;pass<8;++pass)rope_project_stretch(r,dt,first,last,parameters,states);
    }
    for(unsigned i=tid;i<r.count;i+=blockDim.x) {
        const int body=rope_anchor_body(r,i,first,last);
        r.velocities[i]=body<0?clamp_length(multiply(subtract(r.positions[i],r.previous[i]),1/dt),r.options.maximum_speed):
            add(states[body].linear_velocity,cross(states[body].angular_velocity,rotate(states[body].orientation,rope_anchor_local(r,i))));
    }
    __syncthreads();
    rope_project_velocities(r,dt,first,last,parameters,states);
}

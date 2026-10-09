// SPDX-License-Identifier: MIT
// Shader Model 5.1 rigid-body vertical slice.  The deliberately serial
// contact/constraint pass gives stable ordering on every D3D12 vendor.

// FXC reports X4000 through fully initialized nested structs passed by value.
// Keep /WX for every actionable diagnostic and suppress only that false positive.
#pragma warning(disable:4000)

struct PackedVec3 { float x; float y; float z; };
struct Quaternion { float x; float y; float z; float w; };

struct RigidBodyState {
    PackedVec3 position;
    Quaternion orientation;
    PackedVec3 linear_velocity;
    PackedVec3 angular_velocity;
};

struct RigidParameters {
    uint motion;
    float inverse_mass;
    PackedVec3 inverse_inertia;
    float linear_damping;
    float angular_damping;
    float maximum_linear_speed;
    float maximum_angular_speed;
    uint has_kinematic_target;
    RigidBodyState kinematic_target;
    uint mesh_index;
    float friction;
    float restitution;
    float collision_margin;
};

struct RigidConstraint {
    uint generation;
    uint alive;
    uint type;
    uint body_a;
    uint body_b;
    uint enabled;
    uint broken;
    PackedVec3 local_anchor_a;
    PackedVec3 local_anchor_b;
    Quaternion local_orientation_a;
    Quaternion local_orientation_b;
    float breaking_impulse_threshold;
    float applied_impulse;
    uint linear_limit_axes;
    PackedVec3 linear_limit_lower;
    PackedVec3 linear_limit_upper;
    uint angular_limit_axes;
    PackedVec3 angular_limit_lower;
    PackedVec3 angular_limit_upper;
    uint linear_spring_axes;
    PackedVec3 linear_spring_stiffness;
    PackedVec3 linear_spring_damping;
    uint angular_spring_axes;
    PackedVec3 angular_spring_stiffness;
    PackedVec3 angular_spring_damping;
    uint linear_motor_enabled;
    uint angular_motor_enabled;
    float linear_target_velocity;
    float linear_maximum_impulse;
    float angular_target_velocity;
    float angular_maximum_impulse;
    uint solver_iterations;
    uint disable_collisions;
};

struct ConstraintAxisGeometry {
    PackedVec3 axis;
    PackedVec3 inverse_angular_a;
    PackedVec3 inverse_angular_b;
    float angular_denominator;
    float linear_denominator;
};

struct ConstraintGeometry {
    uint valid;
    PackedVec3 arm_a;
    PackedVec3 arm_b;
    PackedVec3 anchor_error;
    PackedVec3 rotation_error;
    PackedVec3 hinge_alignment_error;
    PackedVec3 piston_alignment_error;
    ConstraintAxisGeometry axes[3];
};

struct RigidCompound {
    uint root;
    uint member_count;
    uint eligible;
    uint blocked;
    PackedVec3 center;
    float inverse_mass;
    PackedVec3 inverse_inertia[3];
    uint projection_root;
    uint projection_movable;
    PackedVec3 projection_translation;
};

struct MeshInfo {
    uint vertex_offset;
    uint vertex_count;
    uint index_offset;
    uint index_count;
    PackedVec3 minimum;
    PackedVec3 maximum;
    PackedVec3 bounding_center;
    float radius;
    uint bvh_node_offset;
    uint bvh_node_count;
    uint solid_plane_offset;
    uint solid_plane_count;
    uint alive;
    uint generation;
};

struct CollisionPlane {
    PackedVec3 normal;
    float offset;
};

struct BvhNode {
    PackedVec3 minimum;
    PackedVec3 maximum;
    uint left;
    uint right;
    uint first_triangle;
    uint triangle_count;
};

struct MeshLeafInfo {
    uint offset;
    uint count;
};

struct Handle { uint index; uint generation; };

struct ContactRecord {
    PackedVec3 position;
    PackedVec3 normal;
    float penetration;
    uint found;
    uint persistent;
    uint warm_started;
    float impact_fraction;
    float initial_normal_speed;
    float accumulated_normal_impulse;
    PackedVec3 accumulated_friction_impulse;
};

struct ContactManifold {
    ContactRecord contacts[8];
    uint count;
    uint event_offset;
    uint color;
    uint cached;
    PackedVec3 initial_relative_position;
};

static const uint manifold_face_patch=1u;
static const uint manifold_body_fixed_member=2u;
static const uint manifold_collider_fixed_member=4u;

struct CachedContact {
    PackedVec3 local_point;
    PackedVec3 normal;
    float normal_impulse;
    PackedVec3 friction_impulse;
};

struct ContactCacheHeader {
    uint valid;
    uint epoch;
    Handle body;
    Handle collider;
    float timestep;
    uint count;
};

struct HingeContactFrame {
    PackedVec3 anchor;
    PackedVec3 axis;
    PackedVec3 local_anchor;
    float inverse_mass;
    float inverse_moment;
    uint present;
    uint fixed;
    uint axial;
    uint axial_rotation;
    uint fixed_member;
    uint static_body;
};

struct RigidContactEvent {
    Handle body;
    Handle collider;
    PackedVec3 position;
    PackedVec3 normal;
    float penetration;
    float normal_impulse;
    PackedVec3 friction_impulse;
};

cbuffer StepConstants : register(b0) {
    float step_timestep;
    float3 step_gravity;
    uint step_body_count;
    uint step_constraint_capacity;
    uint step_collect_contacts;
    uint step_event_capacity;
    uint step_substeps;
    uint step_substep_index;
    uint step_use_mesh_bvh;
    uint step_contact_epoch;
    uint step_solver_phase;
    uint step_solver_pass_begin;
    uint step_solver_pass_count;
    uint step_solver_color;
};

RWStructuredBuffer<RigidBodyState> states : register(u0);
RWStructuredBuffer<RigidParameters> parameters : register(u1);
RWStructuredBuffer<PackedVec3> forces : register(u2);
RWStructuredBuffer<PackedVec3> torques : register(u3);
RWStructuredBuffer<RigidBodyState> previous_states : register(u4);
RWStructuredBuffer<RigidConstraint> constraints : register(u5);
RWStructuredBuffer<ConstraintGeometry> constraint_geometry : register(u6);
RWStructuredBuffer<Handle> body_ids : register(u7);
RWStructuredBuffer<RigidContactEvent> contact_events : register(u8);
// event count, overflow, used colors, color overflow, first fit, iterations
RWStructuredBuffer<uint> counters : register(u9);
RWStructuredBuffer<RigidCompound> compounds : register(u10);
RWStructuredBuffer<ContactManifold> manifolds : register(u12);
RWStructuredBuffer<PackedVec3> face_clip_scratch : register(u13);
RWStructuredBuffer<CachedContact> contact_cache_rows : register(u14);
RWStructuredBuffer<ContactCacheHeader> contact_cache_headers : register(u15);
RWStructuredBuffer<HingeContactFrame> hinge_frames : register(u16);
RWStructuredBuffer<uint> pair_colors : register(u17);
RWStructuredBuffer<uint> color_owners : register(u18);
StructuredBuffer<MeshInfo> mesh_infos : register(t0);
StructuredBuffer<PackedVec3> mesh_vertices : register(t1);
StructuredBuffer<uint> mesh_indices : register(t2);
StructuredBuffer<BvhNode> mesh_bvh_nodes : register(t3);
StructuredBuffer<MeshLeafInfo> mesh_leaf_infos : register(t4);
StructuredBuffer<uint> mesh_bvh_leaves : register(t5);
StructuredBuffer<CollisionPlane> mesh_solid_planes : register(t6);
StructuredBuffer<PackedVec3> mesh_shell_normals : register(t7);

float3 load3(PackedVec3 v) { return float3(v.x, v.y, v.z); }
PackedVec3 store3(float3 v) {
    PackedVec3 result = {v.x, v.y, v.z};
    return result;
}

// NVCC lowers the CUDA Vec3 dot helper as y*y, then fused x and z terms.
// Preserve that order for collision generation and contact reduction.
float cuda_dot(float3 a,float3 b) {
    return mad(a.z,b.z,mad(a.x,b.x,a.y*b.y));
}
#define dot cuda_dot

// FXC lowers `/` through an approximate reciprocal. One residual correction
// recovers the correctly rounded CUDA quotient for the rigid contact inputs.
float cuda_div(float numerator,float denominator) {
    float quotient=numerator/denominator;
    float remainder=mad(-quotient,denominator,numerator);
    return quotient+remainder/denominator;
}

float3 cuda_scale_add(float3 base_value,float3 value,float scalar) {
    return float3(mad(value.x,scalar,base_value.x),
                  mad(value.y,scalar,base_value.y),
                  mad(value.z,scalar,base_value.z));
}

ContactRecord make_contact(float3 position,float3 normal,float penetration) {
    ContactRecord result;
    result.position=store3(position);
    result.normal=store3(normal);
    result.penetration=penetration;
    result.found=1u;
    result.persistent=0u;
    result.warm_started=0u;
    result.impact_fraction=1.0f;
    result.initial_normal_speed=0.0f;
    result.accumulated_normal_impulse=0.0f;
    result.accumulated_friction_impulse=store3(float3(0,0,0));
    return result;
}

Quaternion q_conjugate(Quaternion q) {
    Quaternion result = {-q.x, -q.y, -q.z, q.w};
    return result;
}

Quaternion q_multiply(Quaternion a, Quaternion b) {
    Quaternion result = {
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z};
    return result;
}

Quaternion q_normalize(Quaternion q) {
    float inverse_length = rsqrt(max(
        q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w, 1.0e-12f));
    Quaternion result = {q.x*inverse_length, q.y*inverse_length,
                         q.z*inverse_length, q.w*inverse_length};
    return result;
}

float3 q_rotate(Quaternion q, float3 vector_value) {
    float3 vector_part = float3(q.x, q.y, q.z);
    float3 twice_cross = 2.0f * cross(vector_part, vector_value);
    return vector_value + q.w * twice_cross + cross(vector_part, twice_cross);
}

// CUDA contracts the inner quaternion term, then performs the outer add.
// Contact impulses need that grouping; geometry keeps q_rotate's established
// grouping because it matches the v1 contact manifold.
float3 q_rotate_contact(Quaternion q, float3 vector_value) {
    float3 vector_part=float3(q.x,q.y,q.z);
    float3 twice_cross=2.0f*cross(vector_part,vector_value);
    float3 second_cross=cross(vector_part,twice_cross);
    return vector_value+float3(
        mad(q.w,twice_cross.x,second_cross.x),
        mad(q.w,twice_cross.y,second_cross.y),
        mad(q.w,twice_cross.z,second_cross.z));
}

float3 compound_inverse_inertia(RigidCompound compound,float3 value);

Quaternion make_quaternion(float x, float y, float z, float w) {
    Quaternion result = {x, y, z, w};
    return result;
}

float3 limit_length(float3 value, float maximum_length) {
    float squared = dot(value, value);
    float maximum_squared = maximum_length * maximum_length;
    return squared > maximum_squared
        ? value * (maximum_length * rsqrt(squared)) : value;
}

float3 inverse_inertia_mul(RigidParameters body, RigidBodyState state,
                           float3 value) {
    float3 local = q_rotate(q_conjugate(state.orientation), value);
    return q_rotate(state.orientation, local * load3(body.inverse_inertia));
}

void orientation_delta(inout RigidBodyState state, float3 delta) {
    Quaternion rotation = q_normalize(make_quaternion(
        0.5f*delta.x, 0.5f*delta.y, 0.5f*delta.z, 1.0f));
    state.orientation = q_normalize(q_multiply(rotation, state.orientation));
}

float3 quaternion_delta(Quaternion from, Quaternion to) {
    Quaternion delta = q_multiply(to, q_conjugate(from));
    if (delta.w < 0.0f) {
        delta.x = -delta.x; delta.y = -delta.y;
        delta.z = -delta.z; delta.w = -delta.w;
    }
    delta = q_normalize(delta);
    float size = length(float3(delta.x, delta.y, delta.z));
    float3 result = float3(0.0f, 0.0f, 0.0f);
    if (size > 1.0e-6f) {
        float angle = 2.0f * atan2(size, clamp(delta.w, -1.0f, 1.0f));
        result = float3(delta.x, delta.y, delta.z) * (angle / size);
    }
    return result;
}

float component(float3 value, uint axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

float3 basis(uint axis) {
    return axis == 0 ? float3(1,0,0) :
           axis == 1 ? float3(0,1,0) : float3(0,0,1);
}

bool axis_enabled(uint mask, uint axis) {
    return (mask & (1u << axis)) != 0u;
}

float limit_error(float value, float lower, float upper) {
    return value < lower ? value - lower :
           value > upper ? value - upper : 0.0f;
}

bool suppress_pair(uint a, uint b) {
    if(color_owners[a]==color_owners[b]) return true;
    [loop] for (uint i = 0; i < step_constraint_capacity; ++i) {
        RigidConstraint joint = constraints[i];
        if (joint.alive == 0 || joint.enabled == 0 || joint.broken != 0 ||
            joint.disable_collisions == 0) continue;
        if ((joint.body_a == a && joint.body_b == b) ||
            (joint.body_a == b && joint.body_b == a)) return true;
    }
    return false;
}

float3 normalized_or(float3 value, float3 fallback) {
    float squared = dot(value, value);
    return squared > 1.0e-12f ? value * rsqrt(squared) : fallback;
}

bool bounds_overlap(float3 minimum_a, float3 maximum_a,
                    float3 minimum_b, float3 maximum_b) {
    return all(minimum_a <= maximum_b) && all(maximum_a >= minimum_b);
}

float3 world_point(RigidBodyState state, float3 local_point) {
    return load3(state.position) + q_rotate(state.orientation, local_point);
}

void transformed_local_bounds(RigidBodyState state, float3 local_minimum,
                              float3 local_maximum, float margin,
                              out float3 minimum, out float3 maximum) {
    float3 local_center = (local_minimum + local_maximum) * 0.5f;
    float3 local_half = (local_maximum - local_minimum) * 0.5f;
    float3 axis_x = q_rotate(state.orientation, float3(1,0,0));
    float3 axis_y = q_rotate(state.orientation, float3(0,1,0));
    float3 axis_z = q_rotate(state.orientation, float3(0,0,1));
    float3 center = load3(state.position) + axis_x * local_center.x +
                    axis_y * local_center.y + axis_z * local_center.z;
    float3 half = abs(axis_x) * local_half.x + abs(axis_y) * local_half.y +
                  abs(axis_z) * local_half.z + margin;
    minimum = center - half;
    maximum = center + half;
}

void transformed_bounds(RigidBodyState state, MeshInfo mesh, float margin,
                        out float3 minimum, out float3 maximum) {
    transformed_local_bounds(state,load3(mesh.minimum),load3(mesh.maximum),
                             margin,minimum,maximum);
}

void transformed_motion_bounds(RigidBodyState previous_state,
                               RigidBodyState state, MeshInfo mesh,
                               float margin, out float3 minimum,
                               out float3 maximum) {
    float3 previous_minimum, previous_maximum;
    transformed_bounds(state, mesh, margin, minimum, maximum);
    transformed_bounds(previous_state, mesh, margin,
                       previous_minimum, previous_maximum);
    minimum = min(minimum, previous_minimum);
    maximum = max(maximum, previous_maximum);
}

void transformed_node_bounds(RigidBodyState state, BvhNode node, float margin,
                             out float3 minimum, out float3 maximum) {
    transformed_local_bounds(state,load3(node.minimum),load3(node.maximum),
                             margin,minimum,maximum);
}

void transformed_node_motion_bounds(RigidBodyState previous_state,
                                    RigidBodyState state, BvhNode node,
                                    float margin, out float3 minimum,
                                    out float3 maximum) {
    float3 previous_minimum,previous_maximum;
    transformed_node_bounds(state,node,margin,minimum,maximum);
    transformed_node_bounds(previous_state,node,margin,
                            previous_minimum,previous_maximum);
    minimum=min(minimum,previous_minimum);
    maximum=max(maximum,previous_maximum);
}

float rotational_motion_bound(RigidBodyState previous_state,
                              RigidBodyState state, MeshInfo mesh) {
    float orientation_dot = clamp(abs(
        previous_state.orientation.x * state.orientation.x +
        previous_state.orientation.y * state.orientation.y +
        previous_state.orientation.z * state.orientation.z +
        previous_state.orientation.w * state.orientation.w), 0.0f, 1.0f);
    return 2.0f * mesh.radius *
        sqrt(max(0.0f, 1.0f - orientation_dot * orientation_dot));
}

bool requires_swept_pair(uint body, uint collider, float threshold) {
    RigidBodyState old_a = previous_states[body];
    RigidBodyState new_a = states[body];
    RigidBodyState old_b = previous_states[collider];
    RigidBodyState new_b = states[collider];
    float3 relative = (load3(new_a.position) - load3(old_a.position)) -
                      (load3(new_b.position) - load3(old_b.position));
    return length(relative) + rotational_motion_bound(
        old_a, new_a, mesh_infos[parameters[body].mesh_index]) +
        rotational_motion_bound(
        old_b, new_b, mesh_infos[parameters[collider].mesh_index]) > threshold;
}

float3 closest_point_triangle(float3 query_position, float3 a, float3 b,
                              float3 c, bool stable_face) {
    float3 ab=b-a, ac=c-a, ap=query_position-a;
    float d1=dot(ab,ap), d2=dot(ac,ap);
    float3 result=a;
    if(d1<=0.0f && d2<=0.0f) {
        result=a;
    } else {
        float3 bp=query_position-b;
        float d3=dot(ab,bp), d4=dot(ac,bp);
        if(d3>=0.0f && d4<=d3) {
            result=b;
        } else {
            float vc=d1*d4-d3*d2;
            if(vc<=0.0f && d1>=0.0f && d3<=0.0f) {
                result=a+(d1/(d1-d3))*ab;
            } else {
                float3 cp=query_position-c;
                float d5=dot(ab,cp), d6=dot(ac,cp);
                if(d6>=0.0f && d5<=d6) {
                    result=c;
                } else {
                    float vb=d5*d2-d1*d6;
                    if(vb<=0.0f && d2>=0.0f && d6<=0.0f) {
                        result=a+(d2/(d2-d6))*ac;
                    } else {
                        float va=d3*d6-d5*d4;
                        if(va<=0.0f && (d4-d3)>=0.0f && (d5-d6)>=0.0f)
                            result=b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b);
                        else if(stable_face) {
                            float3 normal=cross(ab,ac);
                            result=query_position-normal*
                                (dot(ap,normal)/max(dot(normal,normal),1.0e-12f));
                        } else {
                            float inverse=1.0f/(va+vb+vc);
                            result=a+ab*(vb*inverse)+ac*(vc*inverse);
                        }
                    }
                }
            }
        }
    }
    return result;
}

void closest_segments(float3 first_a,float3 first_b,float3 second_a,
                      float3 second_b,out float3 first_point,
                      out float3 second_point) {
    float3 first_axis=first_b-first_a;
    float3 second_axis=second_b-second_a;
    float3 offset=first_a-second_a;
    float first_size=dot(first_axis,first_axis);
    float second_size=dot(second_axis,second_axis);
    float first_fraction=0.0f,second_fraction=0.0f;
    if(first_size<=1.0e-12f && second_size<=1.0e-12f) {
    } else if(first_size<=1.0e-12f) {
        second_fraction=clamp(dot(second_axis,offset)/second_size,0.0f,1.0f);
    } else {
        float first_offset=dot(first_axis,offset);
        if(second_size<=1.0e-12f) {
            first_fraction=clamp(-first_offset/first_size,0.0f,1.0f);
        } else {
            float axes=dot(first_axis,second_axis);
            float second_offset=dot(second_axis,offset);
            float denominator=first_size*second_size-axes*axes;
            first_fraction=abs(denominator)>1.0e-6f*first_size*second_size
                ? clamp((axes*second_offset-first_offset*second_size)/denominator,
                        0.0f,1.0f) : 0.0f;
            second_fraction=(axes*first_fraction+second_offset)/second_size;
            if(second_fraction<0.0f) {
                second_fraction=0.0f;
                first_fraction=clamp(-first_offset/first_size,0.0f,1.0f);
            } else if(second_fraction>1.0f) {
                second_fraction=1.0f;
                first_fraction=clamp((axes-first_offset)/first_size,0.0f,1.0f);
            }
        }
    }
    first_point=first_a+first_axis*first_fraction;
    second_point=second_a+second_axis*second_fraction;
}

bool point_in_triangle(float3 query_position,float3 a,float3 b,float3 c,float3 normal) {
    float tolerance=-1.0e-5f*dot(normal,normal);
    return dot(cross(b-a,query_position-a),normal)>=tolerance &&
           dot(cross(c-b,query_position-b),normal)>=tolerance &&
           dot(cross(a-c,query_position-c),normal)>=tolerance;
}

bool segment_triangle_hit(float3 first,float3 second,float3 a,float3 b,
                          float3 c,out float3 intersection) {
    float3 normal=cross(b-a,c-a);
    float3 direction=second-first;
    float denominator=dot(normal,direction);
    intersection=float3(0,0,0);
    bool result=false;
    if(dot(normal,normal)>1.0e-12f && abs(denominator)>1.0e-6f) {
        float amount=dot(normal,a-first)/denominator;
        if(amount>=0.0f && amount<=1.0f) {
            intersection=first+direction*amount;
            result=point_in_triangle(intersection,a,b,c,normal);
        }
    }
    return result;
}

void closest_triangle_pair(float3 a0,float3 a1,float3 a2,float3 b0,
                           float3 b1,float3 b2,out float3 point_a,
                           out float3 point_b,bool stable_face) {
    float3 intersection;
    point_a=a0;
    point_b=b0;
    if(segment_triangle_hit(a0,a1,b0,b1,b2,intersection)) {
        point_a=intersection;point_b=intersection;return;
    }
    if(segment_triangle_hit(a1,a2,b0,b1,b2,intersection)) {
        point_a=intersection;point_b=intersection;return;
    }
    if(segment_triangle_hit(a2,a0,b0,b1,b2,intersection)) {
        point_a=intersection;point_b=intersection;return;
    }
    if(segment_triangle_hit(b0,b1,a0,a1,a2,intersection)) {
        point_a=intersection;point_b=intersection;return;
    }
    if(segment_triangle_hit(b1,b2,a0,a1,a2,intersection)) {
        point_a=intersection;point_b=intersection;return;
    }
    if(segment_triangle_hit(b2,b0,a0,a1,a2,intersection)) {
        point_a=intersection;point_b=intersection;return;
    }
    float best=3.402823466e+38f;
    float3 nearest=closest_point_triangle(a0,b0,b1,b2,stable_face);
    float squared=dot(a0-nearest,a0-nearest);
    if(squared<best){best=squared;point_a=a0;point_b=nearest;}
    nearest=closest_point_triangle(a1,b0,b1,b2,stable_face);
    squared=dot(a1-nearest,a1-nearest);
    if(squared<best){best=squared;point_a=a1;point_b=nearest;}
    nearest=closest_point_triangle(a2,b0,b1,b2,stable_face);
    squared=dot(a2-nearest,a2-nearest);
    if(squared<best){best=squared;point_a=a2;point_b=nearest;}
    nearest=closest_point_triangle(b0,a0,a1,a2,stable_face);
    squared=dot(nearest-b0,nearest-b0);
    if(squared<best){best=squared;point_a=nearest;point_b=b0;}
    nearest=closest_point_triangle(b1,a0,a1,a2,stable_face);
    squared=dot(nearest-b1,nearest-b1);
    if(squared<best){best=squared;point_a=nearest;point_b=b1;}
    nearest=closest_point_triangle(b2,a0,a1,a2,stable_face);
    squared=dot(nearest-b2,nearest-b2);
    if(squared<best){best=squared;point_a=nearest;point_b=b2;}
    float3 on_a,on_b;
    closest_segments(a0,a1,b0,b1,on_a,on_b);
    squared=dot(on_a-on_b,on_a-on_b);
    if(squared<best){best=squared;point_a=on_a;point_b=on_b;}
    closest_segments(a0,a1,b1,b2,on_a,on_b);
    squared=dot(on_a-on_b,on_a-on_b);
    if(squared<best){best=squared;point_a=on_a;point_b=on_b;}
    closest_segments(a0,a1,b2,b0,on_a,on_b);
    squared=dot(on_a-on_b,on_a-on_b);
    if(squared<best){best=squared;point_a=on_a;point_b=on_b;}
    closest_segments(a1,a2,b0,b1,on_a,on_b);
    squared=dot(on_a-on_b,on_a-on_b);
    if(squared<best){best=squared;point_a=on_a;point_b=on_b;}
    closest_segments(a1,a2,b1,b2,on_a,on_b);
    squared=dot(on_a-on_b,on_a-on_b);
    if(squared<best){best=squared;point_a=on_a;point_b=on_b;}
    closest_segments(a1,a2,b2,b0,on_a,on_b);
    squared=dot(on_a-on_b,on_a-on_b);
    if(squared<best){best=squared;point_a=on_a;point_b=on_b;}
    closest_segments(a2,a0,b0,b1,on_a,on_b);
    squared=dot(on_a-on_b,on_a-on_b);
    if(squared<best){best=squared;point_a=on_a;point_b=on_b;}
    closest_segments(a2,a0,b1,b2,on_a,on_b);
    squared=dot(on_a-on_b,on_a-on_b);
    if(squared<best){best=squared;point_a=on_a;point_b=on_b;}
    closest_segments(a2,a0,b2,b0,on_a,on_b);
    squared=dot(on_a-on_b,on_a-on_b);
    if(squared<best){best=squared;point_a=on_a;point_b=on_b;}
}

void load_triangle(RigidBodyState state,MeshInfo mesh,uint triangle_index,
                   out float3 a,out float3 b,out float3 c) {
    uint base=mesh.index_offset+triangle_index*3u;
    a=world_point(state,load3(mesh_vertices[
        mesh.vertex_offset+mesh_indices[base]]));
    b=world_point(state,load3(mesh_vertices[
        mesh.vertex_offset+mesh_indices[base+1u]]));
    c=world_point(state,load3(mesh_vertices[
        mesh.vertex_offset+mesh_indices[base+2u]]));
}

ContactManifold empty_manifold() {
    ContactManifold result;
    ContactRecord empty;
    empty.position=store3(float3(0,0,0));
    empty.normal=store3(float3(0,0,0));
    empty.penetration=0.0f;empty.found=0u;
    empty.persistent=0u;empty.warm_started=0u;
    empty.impact_fraction=0.0f;
    empty.initial_normal_speed=0.0f;
    empty.accumulated_normal_impulse=0.0f;
    empty.accumulated_friction_impulse=store3(float3(0,0,0));
    [unroll] for(uint index=0;index<8;++index) result.contacts[index]=empty;
    result.count=0u;result.event_offset=0u;result.color=0u;result.cached=0u;
    result.initial_relative_position=store3(float3(0,0,0));
    return result;
}

bool contact_position_less(PackedVec3 first,PackedVec3 second) {
    float3 a=load3(first);
    float3 b=load3(second);
    if(a.x<b.x) return true;
    if(a.x>b.x) return false;
    if(a.y<b.y) return true;
    if(a.y>b.y) return false;
    return a.z<b.z;
}

void add_manifold_contact(inout ContactManifold manifold,
                          ContactRecord candidate,float separation,
                          uint secondary_limit) {
    float minimum_squared=separation*separation;
    uint matching=8u;
    uint aligned_count=0u;
    uint shallowest_aligned=8u;
    float shallow_aligned_penetration=3.402823466e+38f;
    float3 candidate_normal=load3(candidate.normal);
    [unroll] for(uint existing=0;existing<8u;++existing) {
        if(existing>=manifold.count) continue;
        ContactRecord current=manifold.contacts[existing];
        if(dot(candidate_normal,load3(current.normal))>0.9999f) {
            ++aligned_count;
            if(current.penetration<shallow_aligned_penetration) {
                shallowest_aligned=existing;
                shallow_aligned_penetration=current.penetration;
            }
        }
        float3 delta=load3(candidate.position)-load3(manifold.contacts[existing].position);
        if(matching==8u && dot(delta,delta)<minimum_squared) matching=existing;
    }
    uint aligned_limit=secondary_limit;
    if(manifold.count>0u && dot(candidate_normal,
       load3(manifold.contacts[0].normal))>0.9999f)
        aligned_limit=4u;
    if(matching==8u && aligned_count>=aligned_limit) {
        if(candidate.penetration>shallow_aligned_penetration+1.0e-6f)
            matching=shallowest_aligned;
        else
            return;
    }
    if(matching<8u) {
        float existing_penetration=
            matching==0u ? manifold.contacts[0].penetration :
            matching==1u ? manifold.contacts[1].penetration :
            matching==2u ? manifold.contacts[2].penetration :
            matching==3u ? manifold.contacts[3].penetration :
            matching==4u ? manifold.contacts[4].penetration :
            matching==5u ? manifold.contacts[5].penetration :
            matching==6u ? manifold.contacts[6].penetration :
                           manifold.contacts[7].penetration;
        if(candidate.penetration>existing_penetration) {
            if(matching==0u) manifold.contacts[0]=candidate;
            else if(matching==1u) manifold.contacts[1]=candidate;
            else if(matching==2u) manifold.contacts[2]=candidate;
            else if(matching==3u) manifold.contacts[3]=candidate;
            else if(matching==4u) manifold.contacts[4]=candidate;
            else if(matching==5u) manifold.contacts[5]=candidate;
            else if(matching==6u) manifold.contacts[6]=candidate;
            else manifold.contacts[7]=candidate;
        }
    } else if(manifold.count<8u) {
        if(manifold.count==0u) manifold.contacts[0]=candidate;
        else if(manifold.count==1u) manifold.contacts[1]=candidate;
        else if(manifold.count==2u) manifold.contacts[2]=candidate;
        else if(manifold.count==3u) manifold.contacts[3]=candidate;
        else if(manifold.count==4u) manifold.contacts[4]=candidate;
        else if(manifold.count==5u) manifold.contacts[5]=candidate;
        else if(manifold.count==6u) manifold.contacts[6]=candidate;
        else manifold.contacts[7]=candidate;
        manifold.count++;
    } else {
        uint shallowest=0u;
        float shallow_penetration=manifold.contacts[0].penetration;
        if(manifold.contacts[1].penetration<shallow_penetration)
            {shallowest=1u;shallow_penetration=manifold.contacts[1].penetration;}
        if(manifold.contacts[2].penetration<shallow_penetration)
            {shallowest=2u;shallow_penetration=manifold.contacts[2].penetration;}
        if(manifold.contacts[3].penetration<shallow_penetration)
            {shallowest=3u;shallow_penetration=manifold.contacts[3].penetration;}
        if(manifold.contacts[4].penetration<shallow_penetration)
            {shallowest=4u;shallow_penetration=manifold.contacts[4].penetration;}
        if(manifold.contacts[5].penetration<shallow_penetration)
            {shallowest=5u;shallow_penetration=manifold.contacts[5].penetration;}
        if(manifold.contacts[6].penetration<shallow_penetration)
            {shallowest=6u;shallow_penetration=manifold.contacts[6].penetration;}
        if(manifold.contacts[7].penetration<shallow_penetration)
            {shallowest=7u;shallow_penetration=manifold.contacts[7].penetration;}
        if(candidate.penetration>shallow_penetration) {
            if(secondary_limit==2u) {
                uint smallest_group=9u;
                [unroll] for(uint possible=0u;possible<8u;++possible) {
                    ContactRecord possible_contact=manifold.contacts[possible];
                    if(abs(possible_contact.penetration-shallow_penetration)>
                       1.0e-6f) continue;
                    uint group_count=0u;
                    [unroll] for(uint grouped=0u;grouped<8u;++grouped)
                        if(dot(load3(possible_contact.normal),
                               load3(manifold.contacts[grouped].normal))>
                           0.9999f)
                            ++group_count;
                    if(group_count<smallest_group) {
                        smallest_group=group_count;
                        shallowest=possible;
                    }
                }
            }
            if(shallowest==0u) manifold.contacts[0]=candidate;
            else if(shallowest==1u) manifold.contacts[1]=candidate;
            else if(shallowest==2u) manifold.contacts[2]=candidate;
            else if(shallowest==3u) manifold.contacts[3]=candidate;
            else if(shallowest==4u) manifold.contacts[4]=candidate;
            else if(shallowest==5u) manifold.contacts[5]=candidate;
            else if(shallowest==6u) manifold.contacts[6]=candidate;
            else manifold.contacts[7]=candidate;
        }
    }
}

void set_manifold_contact(inout ContactManifold manifold,uint index,
                          ContactRecord contact) {
    if(index==0u) manifold.contacts[0]=contact;
    else if(index==1u) manifold.contacts[1]=contact;
    else if(index==2u) manifold.contacts[2]=contact;
    else if(index==3u) manifold.contacts[3]=contact;
    else if(index==4u) manifold.contacts[4]=contact;
    else if(index==5u) manifold.contacts[5]=contact;
    else if(index==6u) manifold.contacts[6]=contact;
    else manifold.contacts[7]=contact;
}

void order_swept_manifold_cuda(inout ContactManifold manifold) {
    if(manifold.count!=8u) return;
    ContactManifold source=manifold;
    manifold.contacts[0]=source.contacts[2];
    manifold.contacts[1]=source.contacts[3];
    manifold.contacts[2]=source.contacts[0];
    manifold.contacts[3]=source.contacts[5];
    manifold.contacts[4]=source.contacts[7];
    manifold.contacts[5]=source.contacts[6];
    manifold.contacts[6]=source.contacts[1];
    manifold.contacts[7]=source.contacts[4];
}

float2 guided_contact_direction(ContactRecord contact,
                                HingeContactFrame body_hinge,
                                HingeContactFrame collider_hinge) {
    HingeContactFrame guide=collider_hinge;
    if(body_hinge.axial!=0u) guide=body_hinge;
    float3 normal=load3(contact.normal);
    float translation=dot(normal,load3(guide.axis))*
        sqrt(max(guide.inverse_mass,0.0f));
    float rotation=guide.axial_rotation!=0u
        ? dot(load3(guide.axis),cross(
            load3(contact.position)-load3(guide.anchor),normal))*
            sqrt(max(guide.inverse_moment,0.0f))
        : 0.0f;
    return float2(translation,rotation);
}

bool guided_contact_match(float2 candidate_direction,float candidate_length,
                          ContactRecord existing,
                          HingeContactFrame body_hinge,
                          HingeContactFrame collider_hinge,
                          out float size) {
    float2 direction=guided_contact_direction(
        existing,body_hinge,collider_hinge);
    size=length(direction);
    return candidate_direction.x*direction.x+
           candidate_direction.y*direction.y>
           0.9999f*candidate_length*size;
}

void add_guided_manifold_contact(inout ContactManifold manifold,
                                 ContactRecord candidate,
                                 HingeContactFrame body_hinge,
                                 HingeContactFrame collider_hinge) {
    float2 candidate_direction=guided_contact_direction(
        candidate,body_hinge,collider_hinge);
    float candidate_length=length(candidate_direction);
    if(candidate_length*candidate_length<=1.0e-6f) return;
    if(candidate.impact_fraction>0.0f) {
        float first_impact=1.0f;
        if(manifold.count>0u && manifold.contacts[0].impact_fraction>0.0f)
            first_impact=min(first_impact,manifold.contacts[0].impact_fraction);
        if(manifold.count>1u && manifold.contacts[1].impact_fraction>0.0f)
            first_impact=min(first_impact,manifold.contacts[1].impact_fraction);
        if(manifold.count>2u && manifold.contacts[2].impact_fraction>0.0f)
            first_impact=min(first_impact,manifold.contacts[2].impact_fraction);
        if(manifold.count>3u && manifold.contacts[3].impact_fraction>0.0f)
            first_impact=min(first_impact,manifold.contacts[3].impact_fraction);
        if(manifold.count>4u && manifold.contacts[4].impact_fraction>0.0f)
            first_impact=min(first_impact,manifold.contacts[4].impact_fraction);
        if(manifold.count>5u && manifold.contacts[5].impact_fraction>0.0f)
            first_impact=min(first_impact,manifold.contacts[5].impact_fraction);
        if(manifold.count>6u && manifold.contacts[6].impact_fraction>0.0f)
            first_impact=min(first_impact,manifold.contacts[6].impact_fraction);
        if(manifold.count>7u && manifold.contacts[7].impact_fraction>0.0f)
            first_impact=min(first_impact,manifold.contacts[7].impact_fraction);
        if(candidate.impact_fraction>first_impact+1.0e-4f) return;
        if(candidate.impact_fraction<first_impact-1.0e-4f) {
            uint old_count=manifold.count;
            uint kept=0u;
#define PM_KEEP_SUPPORT(slot) \
            if(old_count>slot##u && \
               manifold.contacts[slot].impact_fraction==0.0f) { \
                set_manifold_contact(manifold,kept,manifold.contacts[slot]); \
                ++kept; \
            }
            PM_KEEP_SUPPORT(0)
            PM_KEEP_SUPPORT(1)
            PM_KEEP_SUPPORT(2)
            PM_KEEP_SUPPORT(3)
            PM_KEEP_SUPPORT(4)
            PM_KEEP_SUPPORT(5)
            PM_KEEP_SUPPORT(6)
            PM_KEEP_SUPPORT(7)
#undef PM_KEEP_SUPPORT
            manifold.count=kept;
        }
    }
    float size=0.0f;
#define PM_REDUCE_GUIDED(slot) \
    if(manifold.count>slot##u && guided_contact_match( \
       candidate_direction,candidate_length,manifold.contacts[slot], \
       body_hinge,collider_hinge,size)) { \
        if(candidate.penetration/candidate_length> \
           manifold.contacts[slot].penetration/max(size,1.0e-6f)) \
            manifold.contacts[slot]=candidate; \
        return; \
    }
    PM_REDUCE_GUIDED(0)
    PM_REDUCE_GUIDED(1)
    PM_REDUCE_GUIDED(2)
    PM_REDUCE_GUIDED(3)
    PM_REDUCE_GUIDED(4)
    PM_REDUCE_GUIDED(5)
    PM_REDUCE_GUIDED(6)
    PM_REDUCE_GUIDED(7)
#undef PM_REDUCE_GUIDED
    if(manifold.count<8u) {
        set_manifold_contact(manifold,manifold.count,candidate);
        ++manifold.count;
        return;
    }
    uint shallowest=0u;
    float shallow_penetration=manifold.contacts[0].penetration;
#define PM_SHALLOWEST(slot) \
    if(manifold.contacts[slot].penetration<shallow_penetration) { \
        shallowest=slot##u; \
        shallow_penetration=manifold.contacts[slot].penetration; \
    }
    PM_SHALLOWEST(1)
    PM_SHALLOWEST(2)
    PM_SHALLOWEST(3)
    PM_SHALLOWEST(4)
    PM_SHALLOWEST(5)
    PM_SHALLOWEST(6)
    PM_SHALLOWEST(7)
#undef PM_SHALLOWEST
    if(candidate.penetration>shallow_penetration)
        set_manifold_contact(manifold,shallowest,candidate);
}

bool convex_face_manifold(uint body_index,uint collider_index,float margin,
                          inout ContactManifold output) {
    const uint maximum_faces=32u;
    RigidBodyState body_state=states[body_index];
    RigidBodyState collider_state=states[collider_index];
    MeshInfo body_mesh=mesh_infos[parameters[body_index].mesh_index];
    MeshInfo collider_mesh=mesh_infos[parameters[collider_index].mesh_index];
    output=empty_manifold();

    float best_separation=-3.402823466e+38f;
    uint reference_face=0u;
    bool reference_is_body=false;
    [loop] for(uint side=0u;side<2u;++side) {
        MeshInfo reference_mesh;
        RigidBodyState reference_state;
        MeshInfo incident_mesh;
        RigidBodyState incident_state;
        if(side==0u) {
            reference_mesh=collider_mesh;
            reference_state=collider_state;
            incident_mesh=body_mesh;
            incident_state=body_state;
        } else {
            reference_mesh=body_mesh;
            reference_state=body_state;
            incident_mesh=collider_mesh;
            incident_state=collider_state;
        }
        [loop] for(uint face=0u;face<reference_mesh.solid_plane_count;++face) {
            CollisionPlane plane=mesh_solid_planes[
                reference_mesh.solid_plane_offset+face];
            float3 normal=q_rotate(reference_state.orientation,
                                   load3(plane.normal));
            float3 incident_normal=q_rotate(
                q_conjugate(incident_state.orientation),normal);
            float support=3.402823466e+38f;
            [loop] for(uint vertex=0u;vertex<incident_mesh.vertex_count;++vertex)
                support=min(support,dot(incident_normal,load3(mesh_vertices[
                    incident_mesh.vertex_offset+vertex])));
            float separation=support+dot(normal,
                load3(incident_state.position)-load3(reference_state.position))-
                plane.offset;
            if(separation>best_separation) {
                best_separation=separation;
                reference_face=face;
                reference_is_body=side!=0u;
            }
        }
    }
    ContactManifold manifold=empty_manifold();
    bool handled=true;
    if(best_separation<=margin) {
    MeshInfo reference_mesh;
    RigidBodyState reference_state;
    MeshInfo incident_mesh;
    RigidBodyState incident_state;
    if(reference_is_body) {
        reference_mesh=body_mesh;
        reference_state=body_state;
        incident_mesh=collider_mesh;
        incident_state=collider_state;
    } else {
        reference_mesh=collider_mesh;
        reference_state=collider_state;
        incident_mesh=body_mesh;
        incident_state=body_state;
    }
    CollisionPlane reference=mesh_solid_planes[
        reference_mesh.solid_plane_offset+reference_face];
    float3 reference_normal=load3(reference.normal);
    float3 outward=q_rotate(reference_state.orientation,reference_normal);
    float3 incident_axis=q_rotate(
        q_conjugate(incident_state.orientation),outward);
    float alignment=1.0f;
    uint incident_face=0u;
    [loop] for(uint face=0u;face<incident_mesh.solid_plane_count;++face) {
        float value=dot(incident_axis,load3(mesh_solid_planes[
            incident_mesh.solid_plane_offset+face].normal));
        if(value<alignment) { alignment=value;incident_face=face; }
    }
    if(alignment>-0.98f) {
        handled=false;
    } else {
    float3 normal=reference_is_body ? -outward : outward;
    CollisionPlane incident=mesh_solid_planes[
        incident_mesh.solid_plane_offset+incident_face];
    uint scratch_base=(body_index*step_body_count+collider_index)*72u;
    uint polygon_base=scratch_base;
    uint clipped_base=scratch_base+36u;
    [loop] for(uint face_triangle=0u;
        face_triangle<incident_mesh.solid_plane_count;++face_triangle) {
        CollisionPlane face=mesh_solid_planes[
            incident_mesh.solid_plane_offset+face_triangle];
        if(dot(load3(face.normal),load3(incident.normal))<0.99999f ||
           abs(face.offset-incident.offset)>1.0e-5f)
            continue;
        uint count=3u;
        [unroll] for(uint corner=0u;corner<3u;++corner) {
            uint local_index=mesh_indices[
                incident_mesh.index_offset+face_triangle*3u+corner];
            float3 world=world_point(incident_state,load3(mesh_vertices[
                incident_mesh.vertex_offset+local_index]));
            face_clip_scratch[polygon_base+corner]=store3(
                q_rotate(q_conjugate(reference_state.orientation),
                    world-load3(reference_state.position)));
        }
        [loop] for(uint plane_index=0u;
            plane_index<reference_mesh.solid_plane_count && count!=0u;
            ++plane_index) {
            CollisionPlane plane=mesh_solid_planes[
                reference_mesh.solid_plane_offset+plane_index];
            float3 plane_normal=load3(plane.normal);
            float offset=plane.offset+
                (dot(plane_normal,reference_normal)>0.99999f
                    ? margin : 0.0f);
            uint clipped_count=0u;
            float3 previous=load3(face_clip_scratch[
                polygon_base+count-1u]);
            float previous_distance=dot(plane_normal,previous)-offset;
            [loop] for(uint clip_vertex=0u;clip_vertex<count;++clip_vertex) {
                float3 current=load3(face_clip_scratch[
                    polygon_base+clip_vertex]);
                float distance=dot(plane_normal,current)-offset;
                if((distance<=0.0f)!=(previous_distance<=0.0f)) {
                    float3 intersection=previous+(current-previous)*
                        (previous_distance/(previous_distance-distance));
                    face_clip_scratch[clipped_base+clipped_count++]=
                        store3(intersection);
                }
                if(distance<=0.0f)
                    face_clip_scratch[clipped_base+clipped_count++]=
                        store3(current);
                previous=current;
                previous_distance=distance;
            }
            count=clipped_count;
            [loop] for(uint copy_vertex=0u;copy_vertex<count;++copy_vertex)
                face_clip_scratch[polygon_base+copy_vertex]=
                    face_clip_scratch[clipped_base+copy_vertex];
        }
        [loop] for(uint contact_vertex=0u;contact_vertex<count;
            ++contact_vertex) {
            float3 local_point=load3(face_clip_scratch[
                polygon_base+contact_vertex]);
            float distance=dot(reference_normal,local_point)-
                           reference.offset;
            if(distance>margin) continue;
            float3 contact_position=world_point(reference_state,
                local_point-reference_normal*(distance*0.5f));
            ContactRecord contact=make_contact(
                contact_position,normal,-distance);
            add_manifold_contact(
                manifold,contact,max(margin*2.0f,1.0e-4f),4u);
        }
    }
    if(manifold.count==0u && alignment>-0.999999f) handled=false;
    }
    }
    output=manifold;
    return handled;
}

float contact_normal_speed(RigidBodyState body,RigidBodyState collider,
                           float3 contact_position,float3 normal) {
    float3 body_velocity=load3(body.linear_velocity)+
        cross(load3(body.angular_velocity),contact_position-load3(body.position));
    float3 collider_velocity=load3(collider.linear_velocity)+
        cross(load3(collider.angular_velocity),contact_position-load3(collider.position));
    return dot(body_velocity-collider_velocity,normal);
}

float3 guided_triangle_normal(float3 a0,float3 a1,float3 a2,
                              float3 b0,float3 b1,float3 b2,
                              float3 point_a,float3 point_b,
                              float3 fallback) {
    float3 delta=point_a-point_b;
    float3 normal=normalized_or(delta,fallback);
    float3 face_a=normalized_or(cross(a1-a0,a2-a0),normal);
    float3 face_b=normalized_or(cross(b1-b0,b2-b0),normal);
    float best_error=4.0e-12f;
    if(dot(delta,delta)>1.0e-12f) {
        float projection=dot(delta,face_a);
        float3 error_delta=delta-face_a*projection;
        float error=dot(error_delta,error_delta);
        if(error<best_error) {
            best_error=error;
            normal=projection>=0.0f ? face_a : -face_a;
        }
        projection=dot(delta,face_b);
        error_delta=delta-face_b*projection;
        error=dot(error_delta,error_delta);
        if(error<best_error)
            normal=projection>=0.0f ? face_b : -face_b;
    }
    return normal;
}

bool triangle_pair_face_contact(float3 a0,float3 a1,float3 a2,
                                float3 b0,float3 b1,float3 b2,
                                float3 separation) {
    float3 normal=normalized_or(separation,float3(0,0,0));
    float3 face_a=normalized_or(cross(a1-a0,a2-a0),float3(0,0,0));
    float3 face_b=normalized_or(cross(b1-b0,b2-b0),float3(0,0,0));
    return abs(dot(normal,face_a))>0.9999f ||
           abs(dot(normal,face_b))>0.9999f;
}

void collide_current(uint body_index,uint collider_index,
                     uint body_first,uint body_triangle_count,
                     uint collider_first,uint collider_triangle_count,
                     inout ContactManifold manifold) {
    RigidBodyState body=states[body_index];
    RigidBodyState collider=states[collider_index];
    RigidBodyState previous_body=previous_states[body_index];
    RigidBodyState previous_collider=previous_states[collider_index];
    HingeContactFrame body_hinge=hinge_frames[body_index];
    HingeContactFrame collider_hinge=hinge_frames[collider_index];
    bool guided=(body_hinge.axial!=0u && collider_hinge.static_body!=0u) ||
                (collider_hinge.axial!=0u && body_hinge.static_body!=0u);
    if(guided) return;
    MeshInfo body_mesh=mesh_infos[parameters[body_index].mesh_index];
    MeshInfo collider_mesh=mesh_infos[parameters[collider_index].mesh_index];
    float margin=parameters[body_index].collision_margin+
                 parameters[collider_index].collision_margin;
    float rest_offset=min(margin,0.001f);
    float3 body_reference=body_hinge.present!=0u
        ? load3(body_hinge.anchor) : load3(body.position);
    uint candidate_count=body_triangle_count*collider_triangle_count;
    [loop] for(uint candidate_index=0u;
        candidate_index<candidate_count;++candidate_index) {
        uint body_triangle=body_first+
            candidate_index/collider_triangle_count;
        uint collider_triangle=candidate_index-
            (body_triangle-body_first)*collider_triangle_count+collider_first;
        float3 a0,a1,a2;
        load_triangle(body,body_mesh,body_triangle,a0,a1,a2);
        float3 a_min=min(a0,min(a1,a2))-margin;
        float3 a_max=max(a0,max(a1,a2))+margin;
        float3 b0,b1,b2;
        load_triangle(collider,collider_mesh,collider_triangle,b0,b1,b2);
        if(!bounds_overlap(a_min,a_max,min(b0,min(b1,b2)),
                           max(b0,max(b1,b2)))) continue;
        float3 point_a,point_b;
        closest_triangle_pair(a0,a1,a2,b0,b1,b2,point_a,point_b,false);
        float3 delta=point_a-point_b;
        float squared=dot(delta,delta);
        if(squared>margin*margin) continue;
        float3 collider_normal=normalized_or(cross(b1-b0,b2-b0),
                                              float3(0,1,0));
        float3 fallback=dot(collider_normal,body_reference-point_b)>=0.0f
            ? collider_normal : -collider_normal;
        float distance=sqrt(max(squared,0.0f));
        float3 plane_projection=point_a-collider_normal*
            dot(point_a-b0,collider_normal);
        float3 face_nearest=closest_point_triangle(
            plane_projection,b0,b1,b2,false);
        bool on_triangle_face=dot(plane_projection-face_nearest,
                                  plane_projection-face_nearest)<=1.0e-10f;
        bool small_convex=body_mesh.solid_plane_count!=0u &&
                          body_mesh.index_count<=96u;
        float3 normal=(distance<=1.0e-5f ||
                       (small_convex && on_triangle_face))
            ? fallback : normalized_or(delta,fallback);
        bool convex_surface_face=false;
        float face_separation=0.0f;
        bool convex_a=body_mesh.solid_plane_count!=0u;
        bool convex_b=collider_mesh.solid_plane_count!=0u;
        if(convex_a!=convex_b && distance<=1.0e-5f) {
            float3 face0=convex_a ? b0 : a0;
            float3 face1=convex_a ? b1 : a1;
            float3 face2=convex_a ? b2 : a2;
            RigidBodyState surface_state=body;
            RigidBodyState previous_surface=previous_body;
            RigidBodyState previous_convex=previous_collider;
            if(convex_a) {
                surface_state=collider;
                previous_surface=previous_collider;
                previous_convex=previous_body;
            }
            float3 outward=normalized_or(cross(face1-face0,face2-face0),
                                         float3(0,0,0));
            float3 local_normal=q_rotate(
                q_conjugate(surface_state.orientation),outward);
            float3 old_normal=q_rotate(previous_surface.orientation,
                                       local_normal);
            float3 local_point=q_rotate(q_conjugate(surface_state.orientation),
                                        face0-load3(surface_state.position));
            float3 old_point=world_point(previous_surface,local_point);
            if(dot(old_normal,load3(previous_convex.position)-old_point)<0.0f)
                outward=-outward;
            float3 incident_point=convex_a ? point_a : point_b;
            float3 projection=incident_point-outward*
                dot(incident_point-face0,outward);
            float3 nearest=closest_point_triangle(
                projection,face0,face1,face2,false);
            convex_surface_face=dot(outward,outward)>0.5f &&
                dot(projection-nearest,projection-nearest)<=1.0e-10f;
            if(convex_surface_face) {
                normal=convex_a ? outward : -outward;
                float3 first=convex_a ? a0 : b0;
                float3 second=convex_a ? a1 : b1;
                float3 third=convex_a ? a2 : b2;
                face_separation=min(dot(first-face0,outward),
                    min(dot(second-face0,outward),dot(third-face0,outward)));
            }
        }
        float3 contact_position=(point_a+point_b)*0.5f;
        float normal_speed=contact_normal_speed(body,collider,contact_position,normal);
        if(distance>rest_offset+1.0e-5f &&
           (normal_speed>1.0e-5f ||
            -normal_speed*step_timestep+rest_offset+1.0e-5f<distance))
            continue;
        float penetration=rest_offset-distance+1.0e-5f;
        if(convex_surface_face) {
            penetration=min(margin,rest_offset-face_separation)+1.0e-5f;
        } else if(distance<=1.0e-5f) {
            if(body_hinge.fixed_member!=0u ||
               collider_hinge.fixed_member!=0u || small_convex) {
                float intersection_depth=max(0.0f,-min(dot(a0-point_b,normal),
                    min(dot(a1-point_b,normal),dot(a2-point_b,normal))));
                penetration=min(margin,intersection_depth+rest_offset)+1.0e-5f;
            } else {
                penetration=margin+1.0e-5f;
            }
        }
        ContactRecord contact=make_contact(contact_position,normal,penetration);
        add_manifold_contact(
            manifold,contact,max(margin*2.0f,1.0e-4f),4u);
    }
}

void collide_swept(uint body_index,uint collider_index,
                   uint body_first,uint body_triangle_count,
                   uint collider_first,uint collider_triangle_count,
                   bool guided,bool extend_previous,
                   inout ContactManifold manifold) {
    RigidBodyState previous_body=previous_states[body_index];
    RigidBodyState body=states[body_index];
    RigidBodyState previous_collider=previous_states[collider_index];
    RigidBodyState collider=states[collider_index];
    if(extend_previous) {
        previous_body.position=store3(load3(previous_body.position)-
            (load3(body.position)-load3(previous_body.position)));
        previous_collider.position=store3(load3(previous_collider.position)-
            (load3(collider.position)-load3(previous_collider.position)));
    }
    HingeContactFrame body_hinge=hinge_frames[body_index];
    HingeContactFrame collider_hinge=hinge_frames[collider_index];
    MeshInfo body_mesh=mesh_infos[parameters[body_index].mesh_index];
    MeshInfo collider_mesh=mesh_infos[parameters[collider_index].mesh_index];
    float margin=parameters[body_index].collision_margin+
                 parameters[collider_index].collision_margin;
    float rest_offset=guided ? min(margin,1.0e-4f) : min(margin,0.001f);
    uint candidate_count=body_triangle_count*collider_triangle_count;
    [loop] for(uint candidate_index=0u;
        candidate_index<candidate_count;++candidate_index) {
        uint body_triangle=body_first+
            candidate_index/collider_triangle_count;
        uint collider_triangle=candidate_index-
            (body_triangle-body_first)*collider_triangle_count+collider_first;
        float3 previous_a[3],current_a[3];
        load_triangle(previous_body,body_mesh,body_triangle,
                      previous_a[0],previous_a[1],previous_a[2]);
        load_triangle(body,body_mesh,body_triangle,
                      current_a[0],current_a[1],current_a[2]);
        float3 swept_a_min=min(previous_a[0],current_a[0]);
        float3 swept_a_max=max(previous_a[0],current_a[0]);
        [unroll] for(uint body_vertex=1;body_vertex<3;++body_vertex) {
            swept_a_min=min(swept_a_min,min(previous_a[body_vertex],current_a[body_vertex]));
            swept_a_max=max(swept_a_max,max(previous_a[body_vertex],current_a[body_vertex]));
        }
        float3 previous_b[3],current_b[3];
        load_triangle(previous_collider,collider_mesh,collider_triangle,
                      previous_b[0],previous_b[1],previous_b[2]);
        load_triangle(collider,collider_mesh,collider_triangle,
                      current_b[0],current_b[1],current_b[2]);
        float3 swept_b_min=min(previous_b[0],current_b[0]);
        float3 swept_b_max=max(previous_b[0],current_b[0]);
        [unroll] for(uint collider_vertex=1;collider_vertex<3;++collider_vertex) {
            swept_b_min=min(swept_b_min,min(previous_b[collider_vertex],current_b[collider_vertex]));
            swept_b_max=max(swept_b_max,max(previous_b[collider_vertex],current_b[collider_vertex]));
        }
        if(!bounds_overlap(swept_a_min-margin,swept_a_max+margin,
                           swept_b_min,swept_b_max)) continue;
        float3 delta_a[3],delta_b[3];
        float speed_bound=0.0f,collider_speed=0.0f;
        [unroll] for(uint motion_vertex=0;motion_vertex<3;++motion_vertex) {
            delta_a[motion_vertex]=current_a[motion_vertex]-previous_a[motion_vertex];
            delta_b[motion_vertex]=current_b[motion_vertex]-previous_b[motion_vertex];
            speed_bound=max(speed_bound,length(delta_a[motion_vertex]));
            collider_speed=max(collider_speed,length(delta_b[motion_vertex]));
        }
        speed_bound+=collider_speed;
        float3 common_motion=delta_b[0];
        float relative_body_speed=max(length(delta_a[0]-common_motion),
            max(length(delta_a[1]-common_motion),
                length(delta_a[2]-common_motion)));
        float relative_collider_speed=max(length(delta_b[0]-common_motion),
            max(length(delta_b[1]-common_motion),
                length(delta_b[2]-common_motion)));
        speed_bound=min(speed_bound,
                        relative_body_speed+relative_collider_speed);
        if(speed_bound<=1.0e-6f) continue;
        float time=0.0f;
        bool starts_near_contact=false;
        float3 separating_normal=float3(0,0,0);
        bool have_separating_normal=false;
        [loop] for(uint iteration=0u;iteration<32u;++iteration) {
            float3 a[3],b[3];
            [unroll] for(uint sample_vertex=0;sample_vertex<3;++sample_vertex) {
                a[sample_vertex]=previous_a[sample_vertex]+delta_a[sample_vertex]*time;
                b[sample_vertex]=previous_b[sample_vertex]+delta_b[sample_vertex]*time;
            }
            float3 point_a,point_b;
            closest_triangle_pair(a[0],a[1],a[2],b[0],b[1],b[2],
                                  point_a,point_b,guided);
            float3 delta=point_a-point_b;
            float distance=sqrt(max(0.0f,dot(delta,delta)));
            float contact_offset=rest_offset;
            if(guided && !triangle_pair_face_contact(
                a[0],a[1],a[2],b[0],b[1],b[2],delta)) {
                HingeContactFrame guide=collider_hinge;
                if(body_hinge.axial!=0u) guide=body_hinge;
                float3 direction=normalized_or(delta,float3(0,0,0));
                float axial=dot(direction,load3(guide.axis));
                float angular=dot(load3(guide.axis),cross(
                    point_a-load3(guide.anchor),direction));
                if(abs(axial)>1.0e-3f && abs(angular)>1.0e-3f)
                    contact_offset=0.0f;
            }
            if(iteration==0u)
                starts_near_contact=
                    distance<=contact_offset+5.0e-5f;
            if(distance<=contact_offset+1.0e-5f) {
                if(iteration==0u && !guided) break;
                float3 collider_normal=normalized_or(
                    cross(b[1]-b[0],b[2]-b[0]),float3(0,1,0));
                float3 body_reference=lerp(load3(previous_body.position),
                                           load3(body.position),time);
                if(body_hinge.present!=0u)
                    body_reference=load3(body_hinge.anchor);
                float3 fallback=dot(collider_normal,
                                    body_reference-point_b)>=0.0f
                    ? collider_normal : -collider_normal;
                float3 outward_a=q_rotate(body.orientation,
                    load3(mesh_shell_normals[
                        body_mesh.index_offset/3u+body_triangle]));
                float3 outward_b=q_rotate(collider.orientation,
                    load3(mesh_shell_normals[
                        collider_mesh.index_offset/3u+collider_triangle]));
                if(guided && dot(outward_b,outward_b)>0.5f)
                    fallback=outward_b;
                else if(guided && dot(outward_a,outward_a)>0.5f)
                    fallback=-outward_a;
                float3 normal=guided && have_separating_normal &&
                    distance<=1.0e-5f
                    ? separating_normal
                    : guided ? guided_triangle_normal(
                        a[0],a[1],a[2],b[0],b[1],b[2],point_a,point_b,
                        have_separating_normal ? separating_normal : fallback)
                    : distance<=1.0e-5f
                        ? fallback : normalized_or(delta,fallback);
                if(guided && (dot(normal,outward_a)>1.0e-3f ||
                              dot(normal,outward_b)<-1.0e-3f)) break;
                float3 contact_position=(point_a+point_b)*0.5f;
                float normal_speed=contact_normal_speed(body,collider,
                                                         contact_position,normal);
                if(normal_speed>1.0e-5f) break;
                float remaining=-normal_speed*step_timestep*(1.0f-time)-distance;
                float penetration=guided
                    ? max(0.0f,remaining+contact_offset)
                    : max(0.0f,remaining)+rest_offset;
                ContactRecord contact=make_contact(
                    contact_position,normal,penetration+1.0e-5f);
                contact.impact_fraction=starts_near_contact ? 0.0f : time;
                if(guided)
                    add_guided_manifold_contact(
                        manifold,contact,body_hinge,collider_hinge);
                else
                    add_manifold_contact(
                        manifold,contact,max(margin*2.0f,1.0e-4f),4u);
                break;
            }
            separating_normal=guided_triangle_normal(
                a[0],a[1],a[2],b[0],b[1],b[2],point_a,point_b,
                normalized_or(delta,float3(0,1,0)));
            have_separating_normal=true;
            float advancement=(distance-contact_offset)/
                              (speed_bound+1.0e-6f)*0.9f;
            if(guided) {
                float3 plane=guided_triangle_normal(
                    a[0],a[1],a[2],b[0],b[1],b[2],point_a,point_b,
                    separating_normal);
                float minimum_a=3.402823466e+38f;
                float maximum_b=-3.402823466e+38f;
                float minimum_speed_a=3.402823466e+38f;
                float maximum_speed_b=-3.402823466e+38f;
                minimum_a=min(dot(a[0]-point_b,plane),
                    min(dot(a[1]-point_b,plane),dot(a[2]-point_b,plane)));
                maximum_b=max(dot(b[0]-point_b,plane),
                    max(dot(b[1]-point_b,plane),dot(b[2]-point_b,plane)));
                minimum_speed_a=min(dot(delta_a[0],plane),
                    min(dot(delta_a[1],plane),dot(delta_a[2],plane)));
                maximum_speed_b=max(dot(delta_b[0],plane),
                    max(dot(delta_b[1],plane),dot(delta_b[2],plane)));
                float gap=minimum_a-maximum_b-contact_offset;
                float closing_speed=maximum_speed_b-minimum_speed_a;
                if(gap>1.0e-5f && closing_speed<=0.0f) break;
                if(gap>0.0f && closing_speed>1.0e-6f)
                    advancement=max(advancement,0.9f*gap/closing_speed);
            }
            time+=max(advancement,1.0e-5f);
            if(time>1.0f) break;
        }
    }
}

float3 contact_inverse_inertia_mul(RigidParameters body,RigidBodyState state,
                                   float3 value) {
    float3 local=q_rotate_contact(q_conjugate(state.orientation),value);
    return q_rotate_contact(state.orientation,
                            local*load3(body.inverse_inertia));
}

float fixed_hinge_inverse_moment_state(RigidParameters body,
                                       RigidBodyState state,
                                       HingeContactFrame hinge) {
    if((hinge.fixed==0u && hinge.axial_rotation==0u) ||
       body.inverse_mass<=1.0e-6f) return 0.0f;
    float3 axis=load3(hinge.axis);
    float3 local_axis=q_rotate(q_conjugate(state.orientation),axis);
    float3 inverse_inertia=load3(body.inverse_inertia);
    float center_moment=
        local_axis.x*local_axis.x/max(inverse_inertia.x,1.0e-6f)+
        local_axis.y*local_axis.y/max(inverse_inertia.y,1.0e-6f)+
        local_axis.z*local_axis.z/max(inverse_inertia.z,1.0e-6f);
    float3 center_arm=load3(state.position)-load3(hinge.anchor);
    float3 perpendicular=center_arm-axis*dot(center_arm,axis);
    float pivot_moment=center_moment+
        dot(perpendicular,perpendicular)/body.inverse_mass;
    return pivot_moment>1.0e-6f ? 1.0f/pivot_moment : 0.0f;
}

float fixed_hinge_inverse_moment(uint index,HingeContactFrame hinge) {
    return fixed_hinge_inverse_moment_state(
        parameters[index],states[index],hinge);
}

HingeContactFrame rigid_hinge_contact_frame_from(uint dense,bool use_previous) {
    HingeContactFrame result;
    result.anchor=store3(float3(0,0,0));
    result.axis=store3(float3(0,0,1));
    result.local_anchor=store3(float3(0,0,0));
    result.inverse_mass=0.0f;
    result.inverse_moment=0.0f;
    result.present=0u;
    result.fixed=0u;
    result.axial=0u;
    result.axial_rotation=0u;
    result.fixed_member=0u;
    result.static_body=0u;
    result.static_body=parameters[dense].motion==0u ? 1u : 0u;
    RigidBodyState state=states[dense];
    if(use_previous) state=previous_states[dense];
    result.anchor=state.position;
    uint incident_count=0u;
    [loop] for(uint index=0u;
        index<(parameters[dense].inverse_mass>0.0f
            ? step_constraint_capacity : 0u);++index) {
        RigidConstraint joint=constraints[index];
        if(joint.alive==0u || joint.enabled==0u || joint.broken!=0u)
            continue;
        bool is_a=joint.body_a==dense;
        bool is_b=joint.body_b==dense;
        if(!is_a && !is_b) continue;
        ++incident_count;
        if(joint.type==0u) result.fixed_member=1u;
        bool axial=joint.type==3u || joint.type==4u;
        uint other=is_a ? joint.body_b : joint.body_a;
        bool static_anchor=other<step_body_count &&
            parameters[other].motion==0u;
        bool eligible=joint.type==2u ||
            (axial && static_anchor &&
             joint.breaking_impulse_threshold<=0.0f);
        if(result.present!=0u || !eligible) continue;
        float3 local_axis=axial ? float3(1,0,0) : float3(0,0,1);
        PackedVec3 local_anchor=store3(float3(0,0,0));
        Quaternion local_orientation=make_quaternion(0,0,0,1);
        PackedVec3 other_local_anchor=store3(float3(0,0,0));
        Quaternion other_local_orientation=make_quaternion(0,0,0,1);
        if(is_a) {
            local_anchor=joint.local_anchor_a;
            local_orientation=joint.local_orientation_a;
            other_local_anchor=joint.local_anchor_b;
            other_local_orientation=joint.local_orientation_b;
        } else {
            local_anchor=joint.local_anchor_b;
            local_orientation=joint.local_orientation_b;
            other_local_anchor=joint.local_anchor_a;
            other_local_orientation=joint.local_orientation_a;
        }
        result.local_anchor=local_anchor;
        result.anchor=store3(load3(state.position)+
            q_rotate(state.orientation,load3(local_anchor)));
        result.axis=store3(normalized_or(q_rotate(
            q_multiply(state.orientation,local_orientation),local_axis),
            local_axis));
        result.present=1u;
        result.fixed=static_anchor && !axial ? 1u : 0u;
        result.axial=static_anchor && axial ? 1u : 0u;
        result.axial_rotation=result.axial!=0u && joint.type==4u ? 1u : 0u;
        if(static_anchor) {
            RigidBodyState other_state=states[other];
            if(use_previous) other_state=previous_states[other];
            result.anchor=store3(load3(other_state.position)+
                q_rotate(other_state.orientation,load3(other_local_anchor)));
            result.axis=store3(normalized_or(q_rotate(
                q_multiply(other_state.orientation,other_local_orientation),
                local_axis),load3(result.axis)));
        }
    }
    if(result.axial!=0u && incident_count!=1u) {
        result.axial=0u;
        result.axial_rotation=0u;
        result.present=0u;
    }
    if(result.axial!=0u) {
        result.inverse_mass=parameters[dense].inverse_mass;
        result.inverse_moment=fixed_hinge_inverse_moment(dense,result);
    }
    return result;
}

HingeContactFrame rigid_hinge_contact_frame(uint dense) {
    return rigid_hinge_contact_frame_from(dense,false);
}

Quaternion rigid_axial_orientation_from(uint dense,bool use_previous) {
    RigidBodyState dense_state=states[dense];
    if(use_previous) dense_state=previous_states[dense];
    Quaternion result=dense_state.orientation;
    [loop] for(uint index=0u;index<step_constraint_capacity;++index) {
        RigidConstraint joint=constraints[index];
        if(joint.alive==0u || joint.enabled==0u || joint.broken!=0u ||
           (joint.type!=3u && joint.type!=4u) ||
           joint.breaking_impulse_threshold>0.0f)
            continue;
        bool is_a=joint.body_a==dense;
        bool is_b=joint.body_b==dense;
        if(!is_a && !is_b) continue;
        uint other=is_a ? joint.body_b : joint.body_a;
        if(other>=step_body_count || parameters[other].motion!=0u) continue;
        Quaternion local_orientation=joint.local_orientation_b;
        Quaternion other_local_orientation=joint.local_orientation_a;
        if(is_a) {
            local_orientation=joint.local_orientation_a;
            other_local_orientation=joint.local_orientation_b;
        }
        RigidBodyState other_state=states[other];
        if(use_previous) other_state=previous_states[other];
        return q_normalize(q_multiply(q_multiply(
            other_state.orientation,other_local_orientation),
            q_conjugate(local_orientation)));
    }
    return result;
}

Quaternion rigid_axial_orientation(uint dense) {
    return rigid_axial_orientation_from(dense,false);
}

void advance_guided_pose(inout RigidBodyState next,RigidBodyState previous,
                         HingeContactFrame frame,
                         Quaternion axial_orientation,float timestep) {
    next.position=store3(load3(previous.position)+
                         load3(next.linear_velocity)*timestep);
    float3 angular_velocity=load3(next.angular_velocity);
    Quaternion angular={angular_velocity.x,angular_velocity.y,
                        angular_velocity.z,0.0f};
    Quaternion derivative=q_multiply(angular,previous.orientation);
    next.orientation=q_normalize(make_quaternion(
        previous.orientation.x+0.5f*derivative.x*timestep,
        previous.orientation.y+0.5f*derivative.y*timestep,
        previous.orientation.z+0.5f*derivative.z*timestep,
        previous.orientation.w+0.5f*derivative.w*timestep));
    float3 axis=load3(frame.axis);
    if(frame.axial_rotation!=0u) {
        Quaternion relative=q_multiply(
            next.orientation,q_conjugate(axial_orientation));
        float twist=dot(float3(relative.x,relative.y,relative.z),axis);
        Quaternion rotation=q_normalize(make_quaternion(
            axis.x*twist,axis.y*twist,axis.z*twist,relative.w));
        next.orientation=q_normalize(q_multiply(rotation,axial_orientation));
    } else {
        next.orientation=axial_orientation;
    }
    float3 anchor=load3(next.position)+
        q_rotate(next.orientation,load3(frame.local_anchor));
    next.position=store3(load3(frame.anchor)+axis*dot(
        anchor-load3(frame.anchor),axis)-
        q_rotate(next.orientation,load3(frame.local_anchor)));
}

void finalize_guided_bodies() {
    uint pair_count=step_body_count*step_body_count;
    [loop] for(uint index=0u;index<step_body_count;++index) {
        if(parameters[index].motion!=2u) continue;
        HingeContactFrame frame=rigid_hinge_contact_frame_from(index,true);
        if(frame.axial==0u) continue;
        Quaternion orientation=rigid_axial_orientation_from(index,true);
        float impact=1.0f;
        [loop] for(uint pair=0u;pair<pair_count;++pair) {
            uint first=pair/step_body_count;
            uint second=pair-first*step_body_count;
            if(first!=index && second!=index) continue;
            uint other=first==index ? second : first;
            bool guided_pair=parameters[other].motion==0u;
            ContactManifold manifold=manifolds[pair];
            [loop] for(uint row=0u;row<manifold.count;++row) {
                ContactRecord contact=manifold.contacts[row];
                if(contact.accumulated_normal_impulse>1.0e-6f)
                    impact=min(impact,guided_pair ?
                        contact.impact_fraction : 1.0f);
            }
        }
        if(impact>=1.0f) continue;
        RigidBodyState hit=previous_states[index];
        RigidBodyState next=states[index];
        hit.position=store3(load3(hit.position)+
            (load3(next.position)-load3(hit.position))*impact);
        Quaternion a=hit.orientation;
        Quaternion b=next.orientation;
        if(a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w<0.0f) {
            b.x=-b.x;b.y=-b.y;b.z=-b.z;b.w=-b.w;
        }
        hit.orientation=q_normalize(make_quaternion(
            a.x+(b.x-a.x)*impact,a.y+(b.y-a.y)*impact,
            a.z+(b.z-a.z)*impact,a.w+(b.w-a.w)*impact));
        advance_guided_pose(next,hit,frame,orientation,
                            step_timestep*(1.0f-impact));
        states[index]=next;
    }
}

bool guided_static_pair(HingeContactFrame body,HingeContactFrame collider) {
    return (body.axial!=0u && collider.static_body!=0u) ||
           (collider.axial!=0u && body.static_body!=0u);
}

float contact_direction_inverse_mass(uint index,HingeContactFrame hinge,
                                     float3 contact_position,float3 direction) {
    float result=0.0f;
    if(compounds[index].eligible!=0u) {
        RigidCompound compound=compounds[compounds[index].root];
        float3 arm=contact_position-load3(compound.center);
        float3 angular=cross(arm,direction);
        result=compound.inverse_mass+dot(
            cross(compound_inverse_inertia(compound,angular),arm),direction);
    } else if(hinge.fixed!=0u || hinge.axial!=0u) {
        float3 axis=load3(hinge.axis);
        float jacobian=dot(cross(axis,
            contact_position-load3(hinge.anchor)),direction);
        float axial=hinge.axial!=0u ? dot(axis,direction) : 0.0f;
        result=parameters[index].inverse_mass*axial*axial+
            jacobian*jacobian*fixed_hinge_inverse_moment(index,hinge);
    } else {
        RigidParameters body=parameters[index];
        RigidBodyState state=states[index];
        float3 arm=contact_position-load3(state.position);
        float3 angular=cross(arm,direction);
        float3 response=contact_inverse_inertia_mul(body,state,angular);
        // NVCC contracts this cross because it feeds the dot immediately.
        float3 rotational=float3(
            mad(response.y,arm.z,-response.z*arm.y),
            mad(response.z,arm.x,-response.x*arm.z),
            mad(response.x,arm.y,-response.y*arm.x));
        result=body.inverse_mass+dot(rotational,direction);
    }
    return result;
}

float3 contact_point_velocity(uint index,HingeContactFrame hinge,
                              float3 contact_position) {
    RigidBodyState state=states[index];
    if(hinge.fixed!=0u || hinge.axial!=0u) {
        float3 axis=load3(hinge.axis);
        float3 angular=axis*((hinge.fixed!=0u || hinge.axial_rotation!=0u)
            ? dot(load3(state.angular_velocity),axis) : 0.0f);
        float3 reduced_linear=hinge.axial!=0u
            ? axis*dot(load3(state.linear_velocity),axis) : float3(0,0,0);
        return reduced_linear+cross(angular,
            contact_position-load3(hinge.anchor));
    }
    return load3(state.linear_velocity)+cross(load3(state.angular_velocity),
        contact_position-load3(state.position));
}

void apply_contact_impulse(uint index,HingeContactFrame hinge,
                           float3 contact_position,float3 impulse) {
    if(compounds[index].eligible!=0u) {
        uint root=compounds[index].root;
        RigidCompound compound=compounds[root];
        float3 linear_delta=impulse*compound.inverse_mass;
        float3 angular_delta=compound_inverse_inertia(compound,
            cross(contact_position-load3(compound.center),impulse));
        [loop] for(uint member=0u;member<step_body_count;++member) {
            if(compounds[member].eligible==0u ||
               compounds[member].root!=root) continue;
            RigidBodyState member_state=states[member];
            member_state.linear_velocity=store3(
                load3(member_state.linear_velocity)+linear_delta+
                cross(angular_delta,load3(member_state.position)-
                      load3(compound.center)));
            member_state.angular_velocity=store3(
                load3(member_state.angular_velocity)+angular_delta);
            states[member]=member_state;
        }
        return;
    }
    RigidParameters body=parameters[index];
    if(body.inverse_mass<=0.0f) return;
    RigidBodyState state=states[index];
    if(hinge.fixed!=0u || hinge.axial!=0u) {
        float3 axis=load3(hinge.axis);
        float angular_impulse=dot(axis,cross(
            contact_position-load3(hinge.anchor),impulse));
        float3 angular_delta=axis*angular_impulse*
            fixed_hinge_inverse_moment(index,hinge);
        state.angular_velocity=store3(
            load3(state.angular_velocity)+angular_delta);
        float3 linear_delta=hinge.axial!=0u
            ? axis*(body.inverse_mass*dot(impulse,axis)) : float3(0,0,0);
        state.linear_velocity=store3(load3(state.linear_velocity)+
            linear_delta+cross(angular_delta,
                load3(state.position)-load3(hinge.anchor)));
        states[index]=state;
        return;
    }
    state.linear_velocity=store3(cuda_scale_add(load3(state.linear_velocity),
                                                impulse,body.inverse_mass));
    state.angular_velocity=store3(load3(state.angular_velocity)+
        contact_inverse_inertia_mul(body,state,
            cross(contact_position-load3(state.position),impulse)));
    states[index]=state;
}

void apply_contact_orientation_correction(inout RigidBodyState state,
                                          float3 world_rotation) {
    Quaternion rotation=make_quaternion(
        world_rotation.x,world_rotation.y,world_rotation.z,0.0f);
    Quaternion derivative=q_multiply(rotation,state.orientation);
    Quaternion corrected=make_quaternion(
        state.orientation.x+0.5f*derivative.x,
        state.orientation.y+0.5f*derivative.y,
        state.orientation.z+0.5f*derivative.z,
        state.orientation.w+0.5f*derivative.w);
    state.orientation=q_normalize(corrected);
}

float contact_position_inverse_mass(RigidParameters body,
                                    RigidBodyState state,
                                    HingeContactFrame hinge,
                                    float3 contact_position,float3 normal) {
    if(hinge.fixed==0u && hinge.axial==0u) return body.inverse_mass;
    float3 axis=load3(hinge.axis);
    float jacobian=dot(cross(
        axis,contact_position-load3(hinge.anchor)),normal);
    float axial=hinge.axial!=0u ? dot(axis,normal) : 0.0f;
    return body.inverse_mass*axial*axial+jacobian*jacobian*
        fixed_hinge_inverse_moment_state(body,state,hinge);
}

void apply_contact_position_delta(RigidParameters body,
                                  inout RigidBodyState state,
                                  HingeContactFrame hinge,
                                  float3 contact_position,float3 correction) {
    if(body.inverse_mass<=0.0f) return;
    if(hinge.fixed==0u && hinge.axial==0u) {
        state.position=store3(load3(state.position)+
                              correction*body.inverse_mass);
        return;
    }
    float3 axis=load3(hinge.axis);
    float angular_correction=dot(axis,cross(
        contact_position-load3(hinge.anchor),correction))*
        fixed_hinge_inverse_moment_state(body,state,hinge);
    float3 current_anchor=load3(state.position)+
        q_rotate(state.orientation,load3(hinge.local_anchor));
    float3 anchor=hinge.axial!=0u
        ? current_anchor+axis*(body.inverse_mass*dot(correction,axis))
        : load3(hinge.anchor);
    apply_contact_orientation_correction(state,axis*angular_correction);
    state.position=store3(anchor-
        q_rotate(state.orientation,load3(hinge.local_anchor)));
}

void correct_contact_position(uint body_index,uint collider_index,
                              HingeContactFrame body_hinge,
                              HingeContactFrame collider_hinge,
                              ContactRecord record,float penetration) {
    RigidParameters body=parameters[body_index];
    RigidParameters collider=parameters[collider_index];
    RigidBodyState body_state=states[body_index];
    RigidBodyState collider_state=states[collider_index];
    float3 contact_position=load3(record.position);
    float3 normal=load3(record.normal);
    float denominator=contact_position_inverse_mass(
        body,body_state,body_hinge,contact_position,normal)+
        contact_position_inverse_mass(
            collider,collider_state,collider_hinge,contact_position,normal);
    if(denominator<=1.0e-6f || penetration<=0.0f) return;
    float3 correction=normal*cuda_div(penetration,denominator);
    apply_contact_position_delta(
        body,body_state,body_hinge,contact_position,correction);
    apply_contact_position_delta(
        collider,collider_state,collider_hinge,contact_position,-correction);
    if(parameters[body_index].inverse_mass>0.0f) states[body_index]=body_state;
    if(parameters[collider_index].inverse_mass>0.0f) states[collider_index]=collider_state;
}

void warm_start_persistent_contact(uint body_index,uint collider_index,
                                   HingeContactFrame body_hinge,
                                   HingeContactFrame collider_hinge,
                                   inout ContactRecord record,
                                   uint event_index) {
    if(record.persistent==0u || record.warm_started!=0u) return;
    record.warm_started=1u;
    float3 impulse=load3(record.normal)*record.accumulated_normal_impulse+
                   load3(record.accumulated_friction_impulse);
    apply_contact_impulse(body_index,body_hinge,load3(record.position),impulse);
    apply_contact_impulse(collider_index,collider_hinge,
                          load3(record.position),-impulse);
    if(step_collect_contacts!=0u && event_index<step_event_capacity) {
        RigidContactEvent event=contact_events[event_index];
        event.normal_impulse+=record.accumulated_normal_impulse;
        event.friction_impulse=store3(load3(event.friction_impulse)+
            load3(record.accumulated_friction_impulse));
        contact_events[event_index]=event;
    }
}

void resolve_persistent_contact(uint body_index,uint collider_index,
                                HingeContactFrame body_hinge,
                                HingeContactFrame collider_hinge,
                                inout ContactRecord record,uint event_index) {
    RigidParameters body=parameters[body_index];
    RigidParameters collider=parameters[collider_index];
    float3 contact_position=load3(record.position);
    float3 normal=load3(record.normal);
    float3 relative=contact_point_velocity(
        body_index,body_hinge,contact_position)-contact_point_velocity(
        collider_index,collider_hinge,contact_position);
    float normal_speed=dot(relative,normal);
    float separation=max(0.0f,-record.penetration);
    float target_speed=separation>1.0e-5f
        ? -cuda_div(separation,max(step_timestep,1.0e-6f)) : 0.0f;
    if((body_hinge.fixed_member!=0u || collider_hinge.fixed_member!=0u) &&
       record.penetration>0.0f)
        target_speed=max(target_speed,
            0.2f*cuda_div(record.penetration,max(step_timestep,1.0e-6f)));
    if(separation<=1.0e-5f && record.initial_normal_speed<0.0f)
        target_speed=max(target_speed,
            -min(body.restitution,collider.restitution)*
             record.initial_normal_speed);
    float denominator=contact_direction_inverse_mass(
        body_index,body_hinge,contact_position,normal)+
        contact_direction_inverse_mass(
            collider_index,collider_hinge,contact_position,normal);
    if(denominator<=1.0e-6f) return;
    float next_normal=max(0.0f,record.accumulated_normal_impulse+
        cuda_div(target_speed-normal_speed,denominator));
    float normal_delta=next_normal-record.accumulated_normal_impulse;
    record.accumulated_normal_impulse=next_normal;
    float3 normal_impulse=normal*normal_delta;
    apply_contact_impulse(body_index,body_hinge,contact_position,normal_impulse);
    apply_contact_impulse(
        collider_index,collider_hinge,contact_position,-normal_impulse);

    bool friction_active=separation<=
        (guided_static_pair(body_hinge,collider_hinge) ?
            1.0e-5f : min(body.collision_margin+collider.collision_margin,0.001f));
    relative=contact_point_velocity(
        body_index,body_hinge,contact_position)-contact_point_velocity(
        collider_index,collider_hinge,contact_position);
    float tangent_projection=dot(relative,normal);
    float3 tangent=float3(mad(-normal.x,tangent_projection,relative.x),
                          mad(-normal.y,tangent_projection,relative.y),
                          mad(-normal.z,tangent_projection,relative.z));
    float tangent_length=sqrt(max(dot(tangent,tangent),0.0f));
    float3 previous_friction=load3(record.accumulated_friction_impulse);
    float3 friction=friction_active ? previous_friction : float3(0,0,0);
    if(friction_active && tangent_length>1.0e-6f) {
        tangent*=cuda_div(1.0f,tangent_length);
        float tangent_denominator=contact_direction_inverse_mass(
            body_index,body_hinge,contact_position,tangent)+
            contact_direction_inverse_mass(
                collider_index,collider_hinge,contact_position,tangent);
        if(tangent_denominator>1.0e-6f)
            friction-=tangent*cuda_div(tangent_length,tangent_denominator);
    }
    friction=limit_length(friction,
        sqrt(max(body.friction*collider.friction,0.0f))*next_normal);
    float3 friction_delta=friction-previous_friction;
    record.accumulated_friction_impulse=store3(friction);
    apply_contact_impulse(body_index,body_hinge,contact_position,friction_delta);
    apply_contact_impulse(
        collider_index,collider_hinge,contact_position,-friction_delta);
    if(step_collect_contacts!=0u && event_index<step_event_capacity) {
        RigidContactEvent event=contact_events[event_index];
        event.normal_impulse+=normal_delta;
        event.friction_impulse=store3(load3(event.friction_impulse)+friction_delta);
        contact_events[event_index]=event;
    }
}

void resolve_contact(uint body_index,uint collider_index,
                     HingeContactFrame body_hinge,
                     HingeContactFrame collider_hinge,
                     ContactRecord record,uint event_index) {
    RigidParameters body=parameters[body_index];
    RigidParameters collider=parameters[collider_index];
    float3 contact_position=load3(record.position);
    float3 normal=load3(record.normal);
    float3 relative=contact_point_velocity(
        body_index,body_hinge,contact_position)-contact_point_velocity(
        collider_index,collider_hinge,contact_position);
    float normal_speed=dot(relative,normal);
    float separation=max(0.0f,-record.penetration);
    float target_speed=separation>1.0e-5f
        ? -separation/max(step_timestep,1.0e-6f) : 0.0f;
    if((body_hinge.fixed_member!=0u || collider_hinge.fixed_member!=0u) &&
       record.penetration>0.0f)
        target_speed=max(target_speed,
            0.2f*record.penetration/max(step_timestep,1.0e-6f));
    float restitution=min(body.restitution,collider.restitution);
    if(separation<=1.0e-5f && normal_speed<0.0f)
        target_speed=max(target_speed,-restitution*normal_speed);
    if(normal_speed>=target_speed) return;
    float denominator=contact_direction_inverse_mass(
        body_index,body_hinge,contact_position,normal)+
        contact_direction_inverse_mass(
            collider_index,collider_hinge,contact_position,normal);
    if(denominator<=1.0e-6f) return;
    float normal_impulse=cuda_div(target_speed-normal_speed,denominator);
    float3 normal_vector=normal*normal_impulse;
    apply_contact_impulse(body_index,body_hinge,contact_position,normal_vector);
    apply_contact_impulse(
        collider_index,collider_hinge,contact_position,-normal_vector);
    float3 friction_impulse=float3(0,0,0);
    if(separation<=1.0e-5f) {
        relative=contact_point_velocity(
            body_index,body_hinge,contact_position)-contact_point_velocity(
            collider_index,collider_hinge,contact_position);
        float tangent_projection=dot(relative,normal);
        float3 tangent=float3(mad(-normal.x,tangent_projection,relative.x),
                              mad(-normal.y,tangent_projection,relative.y),
                              mad(-normal.z,tangent_projection,relative.z));
        float tangent_length=sqrt(max(dot(tangent,tangent),0.0f));
        if(tangent_length>1.0e-6f) {
            tangent*=rsqrt(max(dot(tangent,tangent),0.0f));
            float tangent_denominator=
                contact_direction_inverse_mass(
                    body_index,body_hinge,contact_position,tangent)+
                contact_direction_inverse_mass(
                    collider_index,collider_hinge,contact_position,tangent);
            if(tangent_denominator>1.0e-6f) {
                float impulse=cuda_div(-dot(relative,tangent),tangent_denominator);
                float limit=sqrt(max(body.friction*collider.friction,0.0f))*
                            normal_impulse;
                impulse=clamp(impulse,-limit,limit);
                friction_impulse=tangent*impulse;
                apply_contact_impulse(
                    body_index,body_hinge,contact_position,friction_impulse);
                apply_contact_impulse(collider_index,collider_hinge,
                    contact_position,-friction_impulse);
            }
        }
    }
    if(step_collect_contacts!=0u && event_index<step_event_capacity) {
        RigidContactEvent event=contact_events[event_index];
        event.normal_impulse+=normal_impulse;
        event.friction_impulse=store3(load3(event.friction_impulse)+
                                      friction_impulse);
        contact_events[event_index]=event;
    }
}

#undef dot

void store_manifold_contact(inout ContactManifold manifold,uint index,
                            ContactRecord contact) {
    if(index==0u) manifold.contacts[0]=contact;
    else if(index==1u) manifold.contacts[1]=contact;
    else if(index==2u) manifold.contacts[2]=contact;
    else if(index==3u) manifold.contacts[3]=contact;
    else if(index==4u) manifold.contacts[4]=contact;
    else if(index==5u) manifold.contacts[5]=contact;
    else if(index==6u) manifold.contacts[6]=contact;
    else manifold.contacts[7]=contact;
}

void prepare_persistent_pair(uint pair,inout ContactManifold manifold) {
    if(manifold.count==0u) return;
    uint body_index=pair/step_body_count;
    uint collider_index=pair-body_index*step_body_count;
    RigidParameters body=parameters[body_index];
    RigidParameters collider=parameters[collider_index];
    RigidBodyState state_a=states[body_index];
    RigidBodyState state_b=states[collider_index];
    HingeContactFrame body_hinge=hinge_frames[body_index];
    HingeContactFrame collider_hinge=hinge_frames[collider_index];
    MeshInfo body_mesh=mesh_infos[body.mesh_index];
    MeshInfo collider_mesh=mesh_infos[collider.mesh_index];
    bool guided=guided_static_pair(body_hinge,collider_hinge);
    bool persistent=guided ||
        (body_mesh.solid_plane_count!=0u && body_mesh.index_count<=96u &&
         (collider.motion!=2u ||
          (collider_mesh.solid_plane_count!=0u &&
           collider_mesh.index_count<=96u)) &&
         body_hinge.present==0u && collider_hinge.present==0u &&
         body_hinge.fixed_member==0u && collider_hinge.fixed_member==0u);
    manifold.initial_relative_position=store3(
        load3(state_a.position)-load3(state_b.position));
    manifold.cached=0u;
    ContactCacheHeader saved=contact_cache_headers[pair];
    bool cache_valid=saved.valid!=0u &&
        saved.epoch+1u==step_contact_epoch &&
        saved.body.index==body_ids[body_index].index &&
        saved.body.generation==body_ids[body_index].generation &&
        saved.collider.index==body_ids[collider_index].index &&
        saved.collider.generation==body_ids[collider_index].generation &&
        abs(saved.timestep-step_timestep)<=1.0e-7f;
    uint used=0u;
    [loop] for(uint point_index=0u;point_index<manifold.count;++point_index) {
        ContactRecord contact=manifold.contacts[point_index];
        contact.persistent=persistent ? 1u : 0u;
        contact.warm_started=0u;
        contact.initial_normal_speed=contact_normal_speed(
            state_a,state_b,load3(contact.position),load3(contact.normal));
        contact.accumulated_normal_impulse=0.0f;
        contact.accumulated_friction_impulse=store3(float3(0,0,0));
        if(persistent && cache_valid) {
            float3 local=q_rotate(q_conjugate(state_a.orientation),
                load3(contact.position)-load3(state_a.position));
            float nearest=0.02f*0.02f;
            uint match=8u;
            [loop] for(uint previous=0u;
                previous<min(saved.count,8u);++previous) {
                if((used&(1u<<previous))!=0u) continue;
                CachedContact cached=contact_cache_rows[pair*8u+previous];
                if(dot(load3(contact.normal),load3(cached.normal))<0.99f)
                    continue;
                float3 delta=local-load3(cached.local_point);
                float distance=dot(delta,delta);
                if(distance<nearest) { nearest=distance;match=previous; }
            }
            if(match<8u) {
                used|=1u<<match;
                CachedContact cached=contact_cache_rows[pair*8u+match];
                contact.accumulated_normal_impulse=cached.normal_impulse;
                float3 friction=load3(cached.friction_impulse);
                float3 normal=load3(contact.normal);
                contact.accumulated_friction_impulse=store3(
                    friction-normal*dot(friction,normal));
                manifold.cached=1u;
            }
        }
        if(persistent && !guided) {
            uint response_base=pair*72u+point_index*9u;
            float3 position=load3(contact.position);
            float3 normal=load3(contact.normal);
            float3 arm_a=position-load3(state_a.position);
            float3 arm_b=position-load3(state_b.position);
            float3 tangent0=normalized_or(cross(normal,
                abs(normal.x)<0.5f ? float3(1,0,0) : float3(0,1,0)),
                float3(0,0,1));
            float3 tangent1=cross(normal,tangent0);
            face_clip_scratch[response_base]=store3(arm_a);
            face_clip_scratch[response_base+1u]=store3(arm_b);
            face_clip_scratch[response_base+2u]=store3(tangent0);
            face_clip_scratch[response_base+3u]=store3(
                contact_inverse_inertia_mul(body,state_a,cross(arm_a,tangent0)));
            face_clip_scratch[response_base+4u]=store3(
                contact_inverse_inertia_mul(body,state_a,cross(arm_a,tangent1)));
            face_clip_scratch[response_base+5u]=store3(
                contact_inverse_inertia_mul(collider,state_b,cross(arm_b,tangent0)));
            face_clip_scratch[response_base+6u]=store3(
                contact_inverse_inertia_mul(collider,state_b,cross(arm_b,tangent1)));
            face_clip_scratch[response_base+7u]=store3(
                contact_inverse_inertia_mul(body,state_a,cross(arm_a,normal)));
            face_clip_scratch[response_base+8u]=store3(
                contact_inverse_inertia_mul(collider,state_b,cross(arm_b,normal)));
        }
        store_manifold_contact(manifold,point_index,contact);
    }
}

float3 prepared_angular_response(float3 first,float3 second,
                                 float3 tangent0,float3 tangent1,
                                 float3 impulse) {
    return first*dot(tangent0,impulse)+second*dot(tangent1,impulse);
}

void apply_prepared_impulse(inout RigidBodyState state_a,
                            inout RigidBodyState state_b,
                            RigidParameters body,RigidParameters collider,
                            float3 impulse,float3 angular_a,float3 angular_b) {
    if(body.inverse_mass>0.0f) {
        state_a.linear_velocity=store3(
            load3(state_a.linear_velocity)+impulse*body.inverse_mass);
        state_a.angular_velocity=store3(
            load3(state_a.angular_velocity)+angular_a);
    }
    if(collider.inverse_mass>0.0f) {
        state_b.linear_velocity=store3(
            load3(state_b.linear_velocity)-impulse*collider.inverse_mass);
        state_b.angular_velocity=store3(
            load3(state_b.angular_velocity)-angular_b);
    }
}

void solve_prepared_pair(uint pair,inout ContactManifold manifold,
                         bool warm_start_only,uint event_offset) {
    uint body_index=pair/step_body_count;
    uint collider_index=pair-body_index*step_body_count;
    RigidParameters body=parameters[body_index];
    RigidParameters collider=parameters[collider_index];
    RigidBodyState state_a=states[body_index];
    RigidBodyState state_b=states[collider_index];
    float inverse_mass=body.inverse_mass+collider.inverse_mass;
    if(!warm_start_only && inverse_mass>1.0e-6f) {
        float weight=1.0f/float(manifold.count);
        [loop] for(uint correction_index=0u;
            correction_index<manifold.count;++correction_index) {
            ContactRecord contact=manifold.contacts[correction_index];
            float3 normal=load3(contact.normal);
            float penetration=contact.penetration-dot(
                (load3(state_a.position)-load3(state_b.position))-
                    load3(manifold.initial_relative_position),normal);
            if(penetration<=0.0f) continue;
            float3 correction=normal*cuda_div(penetration*weight,inverse_mass);
            if(body.inverse_mass>0.0f)
                state_a.position=store3(
                    load3(state_a.position)+correction*body.inverse_mass);
            if(collider.inverse_mass>0.0f)
                state_b.position=store3(
                    load3(state_b.position)-correction*collider.inverse_mass);
        }
    }
    float friction_coefficient=sqrt(max(body.friction*collider.friction,0.0f));
    [loop] for(uint point_index=0u;point_index<manifold.count;++point_index) {
        ContactRecord contact=manifold.contacts[point_index];
        uint response_base=pair*72u+point_index*9u;
        float3 arm_a=load3(face_clip_scratch[response_base]);
        float3 arm_b=load3(face_clip_scratch[response_base+1u]);
        float3 tangent0=load3(face_clip_scratch[response_base+2u]);
        float3 tangent1=cross(load3(contact.normal),tangent0);
        float3 angular_a0=load3(face_clip_scratch[response_base+3u]);
        float3 angular_a1=load3(face_clip_scratch[response_base+4u]);
        float3 angular_b0=load3(face_clip_scratch[response_base+5u]);
        float3 angular_b1=load3(face_clip_scratch[response_base+6u]);
        float3 normal_angular_a=load3(face_clip_scratch[response_base+7u]);
        float3 normal_angular_b=load3(face_clip_scratch[response_base+8u]);
        float3 normal=load3(contact.normal);
        float lambda=contact.accumulated_normal_impulse;
        float3 friction=load3(contact.accumulated_friction_impulse);
        if(warm_start_only) {
            if(contact.warm_started!=0u) continue;
            contact.warm_started=1u;
            float3 impulse=normal*lambda+friction;
            apply_prepared_impulse(state_a,state_b,body,collider,impulse,
                normal_angular_a*lambda+prepared_angular_response(
                    angular_a0,angular_a1,tangent0,tangent1,friction),
                normal_angular_b*lambda+prepared_angular_response(
                    angular_b0,angular_b1,tangent0,tangent1,friction));
        } else {
            float denominator=body.inverse_mass+collider.inverse_mass+
                dot(cross(normal_angular_a,arm_a),normal)+
                dot(cross(normal_angular_b,arm_b),normal);
            if(denominator<=1.0e-6f) continue;
            float separation=max(0.0f,-contact.penetration);
            float target_speed=separation>1.0e-5f
                ? -cuda_div(separation,max(step_timestep,1.0e-6f)) : 0.0f;
            if(separation<=1.0e-5f && contact.initial_normal_speed<0.0f)
                target_speed=max(target_speed,
                    -min(body.restitution,collider.restitution)*
                        contact.initial_normal_speed);
            float3 relative=(load3(state_a.linear_velocity)+
                cross(load3(state_a.angular_velocity),arm_a))-
                (load3(state_b.linear_velocity)+
                cross(load3(state_b.angular_velocity),arm_b));
            float next_lambda=max(0.0f,lambda+
                cuda_div(target_speed-dot(relative,normal),denominator));
            float normal_delta=next_lambda-lambda;
            lambda=next_lambda;
            float3 normal_impulse=normal*normal_delta;
            apply_prepared_impulse(state_a,state_b,body,collider,normal_impulse,
                normal_angular_a*normal_delta,normal_angular_b*normal_delta);
            relative=(load3(state_a.linear_velocity)+
                cross(load3(state_a.angular_velocity),arm_a))-
                (load3(state_b.linear_velocity)+
                cross(load3(state_b.angular_velocity),arm_b));
            float3 tangent=relative-normal*dot(relative,normal);
            float speed_squared=dot(tangent,tangent);
            bool friction_active=separation<=
                min(body.collision_margin+collider.collision_margin,0.001f);
            float3 next_friction=friction_active ? friction : float3(0,0,0);
            if(friction_active && speed_squared>1.0e-12f) {
                float x=dot(tangent,tangent0);
                float y=dot(tangent,tangent1);
                float3 tangent_response0=tangent0*inverse_mass+
                    cross(angular_a0,arm_a)+cross(angular_b0,arm_b);
                float3 tangent_response1=tangent1*inverse_mass+
                    cross(angular_a1,arm_a)+cross(angular_b1,arm_b);
                float xx=dot(tangent_response0,tangent0);
                float xy=dot(tangent_response0,tangent1);
                float yy=dot(tangent_response1,tangent1);
                float tangent_denominator=xx*x*x+2.0f*xy*x*y+yy*y*y;
                if(tangent_denominator>1.0e-6f*speed_squared)
                    next_friction-=tangent*
                        cuda_div(speed_squared,tangent_denominator);
            }
            next_friction=limit_length(
                next_friction,friction_coefficient*lambda);
            float3 friction_delta=next_friction-friction;
            friction=next_friction;
            apply_prepared_impulse(state_a,state_b,body,collider,
                friction_delta,prepared_angular_response(
                    angular_a0,angular_a1,tangent0,tangent1,friction_delta),
                prepared_angular_response(
                    angular_b0,angular_b1,tangent0,tangent1,friction_delta));
            if(step_collect_contacts!=0u &&
               event_offset+point_index<step_event_capacity) {
                RigidContactEvent event=contact_events[event_offset+point_index];
                event.normal_impulse+=normal_delta;
                event.friction_impulse=store3(
                    load3(event.friction_impulse)+friction_delta);
                contact_events[event_offset+point_index]=event;
            }
            contact.accumulated_normal_impulse=lambda;
            contact.accumulated_friction_impulse=store3(friction);
        }
        store_manifold_contact(manifold,point_index,contact);
    }
    if(body.inverse_mass>0.0f) states[body_index]=state_a;
    if(collider.inverse_mass>0.0f) states[collider_index]=state_b;
}

void solve_persistent_pair(uint pair,inout ContactManifold manifold,
                           bool warm_start_only,bool correct_position,
                           uint event_offset) {
    uint body_index=pair/step_body_count;
    uint collider_index=pair-body_index*step_body_count;
    RigidParameters body=parameters[body_index];
    RigidParameters collider=parameters[collider_index];
    HingeContactFrame body_hinge=hinge_frames[body_index];
    HingeContactFrame collider_hinge=hinge_frames[collider_index];
    bool guided=guided_static_pair(body_hinge,collider_hinge);
    float inverse_mass=body.inverse_mass+collider.inverse_mass;
    float friction_coefficient=sqrt(max(body.friction*collider.friction,0.0f));
    [loop] for(uint point_index=0u;point_index<manifold.count;++point_index) {
        ContactRecord contact=manifold.contacts[point_index];
        if(warm_start_only) {
            if(contact.warm_started!=0u) continue;
            contact.warm_started=1u;
            float3 impulse=load3(contact.normal)*
                contact.accumulated_normal_impulse+
                load3(contact.accumulated_friction_impulse);
            apply_contact_impulse(
                body_index,body_hinge,load3(contact.position),impulse);
            apply_contact_impulse(collider_index,collider_hinge,
                load3(contact.position),-impulse);
            if(step_collect_contacts!=0u &&
               event_offset+point_index<step_event_capacity) {
                RigidContactEvent event=contact_events[event_offset+point_index];
                event.normal_impulse+=contact.accumulated_normal_impulse;
                event.friction_impulse=store3(load3(event.friction_impulse)+
                    load3(contact.accumulated_friction_impulse));
                contact_events[event_offset+point_index]=event;
            }
            store_manifold_contact(manifold,point_index,contact);
        }
    }
    if(warm_start_only) return;

    bool translational_projection=
        manifold.contacts[0].persistent!=0u && !guided;
    if(!guided && (correct_position || translational_projection) &&
       inverse_mass>1.0e-6f &&
       body_hinge.fixed_member==0u && collider_hinge.fixed_member==0u) {
        RigidBodyState state_a=states[body_index];
        RigidBodyState state_b=states[collider_index];
        float weight=1.0f/float(manifold.count);
        [loop] for(uint position_point=0u;
            position_point<manifold.count;++position_point) {
            ContactRecord contact=manifold.contacts[position_point];
            float3 normal=load3(contact.normal);
            float penetration=contact.penetration-dot(
                (load3(state_a.position)-load3(state_b.position))-
                    load3(manifold.initial_relative_position),normal);
            if(penetration<=0.0f) continue;
            bool hinged=body_hinge.fixed!=0u || collider_hinge.fixed!=0u;
            float projected=(hinged ? min(penetration,0.001f) : penetration)*weight;
            float denominator=contact_position_inverse_mass(
                body,state_a,body_hinge,load3(contact.position),normal)+
                contact_position_inverse_mass(collider,state_b,collider_hinge,
                    load3(contact.position),normal);
            if(denominator<=1.0e-6f) continue;
            float3 correction=normal*cuda_div(projected,denominator);
            apply_contact_position_delta(body,state_a,body_hinge,
                load3(contact.position),correction);
            apply_contact_position_delta(collider,state_b,collider_hinge,
                load3(contact.position),-correction);
        }
        if(body.inverse_mass>0.0f) states[body_index]=state_a;
        if(collider.inverse_mass>0.0f) states[collider_index]=state_b;
    }

    [loop] for(uint solve_index=0u;solve_index<manifold.count;++solve_index) {
        ContactRecord contact=manifold.contacts[solve_index];
        float3 position=load3(contact.position);
        float3 normal=load3(contact.normal);
        float3 relative=contact_point_velocity(body_index,body_hinge,position)-
                        contact_point_velocity(
                            collider_index,collider_hinge,position);
        float normal_speed=dot(relative,normal);
        float separation=max(0.0f,-contact.penetration);
        float target_speed=separation>1.0e-5f
            ? -separation/max(step_timestep,1.0e-6f) : 0.0f;
        if((body_hinge.fixed_member!=0u ||
            collider_hinge.fixed_member!=0u) && contact.penetration>0.0f)
            target_speed=max(target_speed,
                0.2f*contact.penetration/max(step_timestep,1.0e-6f));
        if(separation<=1.0e-5f && contact.initial_normal_speed<0.0f)
            target_speed=max(target_speed,-min(body.restitution,collider.restitution)*
                contact.initial_normal_speed);
        float denominator=
            contact_direction_inverse_mass(
                body_index,body_hinge,position,normal)+
            contact_direction_inverse_mass(
                collider_index,collider_hinge,position,normal);
        if(denominator<=1.0e-6f) continue;
        float accumulated=max(0.0f,contact.accumulated_normal_impulse+
            cuda_div(target_speed-normal_speed,denominator));
        float normal_delta=accumulated-contact.accumulated_normal_impulse;
        contact.accumulated_normal_impulse=accumulated;
        float3 normal_impulse=normal*normal_delta;
        apply_contact_impulse(body_index,body_hinge,position,normal_impulse);
        apply_contact_impulse(
            collider_index,collider_hinge,position,-normal_impulse);

        relative=contact_point_velocity(body_index,body_hinge,position)-
                 contact_point_velocity(collider_index,collider_hinge,position);
        float3 tangent=relative-normal*dot(relative,normal);
        float tangent_length=sqrt(max(dot(tangent,tangent),0.0f));
        bool friction_active=separation<=(guided ? 1.0e-5f :
            min(body.collision_margin+collider.collision_margin,0.001f));
        float3 previous_friction=load3(contact.accumulated_friction_impulse);
        float3 friction=friction_active ? previous_friction : float3(0,0,0);
        if(friction_active && tangent_length>1.0e-6f) {
            tangent*=rcp(tangent_length);
            float tangent_denominator=
                contact_direction_inverse_mass(
                    body_index,body_hinge,position,tangent)+
                contact_direction_inverse_mass(
                    collider_index,collider_hinge,position,tangent);
            if(tangent_denominator>1.0e-6f)
                friction-=tangent*cuda_div(tangent_length,tangent_denominator);
        }
        friction=limit_length(friction,friction_coefficient*accumulated);
        float3 friction_delta=friction-previous_friction;
        contact.accumulated_friction_impulse=store3(friction);
        apply_contact_impulse(body_index,body_hinge,position,friction_delta);
        apply_contact_impulse(
            collider_index,collider_hinge,position,-friction_delta);
        store_manifold_contact(manifold,solve_index,contact);
        if(step_collect_contacts!=0u &&
           event_offset+solve_index<step_event_capacity) {
            RigidContactEvent event=contact_events[event_offset+solve_index];
            event.normal_impulse+=normal_delta;
            event.friction_impulse=store3(load3(event.friction_impulse)+friction_delta);
            contact_events[event_offset+solve_index]=event;
        }
    }
}

void save_persistent_pair(uint pair,ContactManifold manifold) {
    if(manifold.count==0u) return;
    uint body_index=pair/step_body_count;
    uint collider_index=pair-body_index*step_body_count;
    RigidBodyState state=states[body_index];
    ContactCacheHeader header;
    header.valid=1u;
    header.epoch=step_contact_epoch;
    header.body=body_ids[body_index];
    header.collider=body_ids[collider_index];
    header.timestep=step_timestep;
    header.count=manifold.count;
    contact_cache_headers[pair]=header;
    [loop] for(uint point_index=0u;point_index<manifold.count;++point_index) {
        ContactRecord contact=manifold.contacts[point_index];
        CachedContact cached;
        cached.local_point=store3(q_rotate(q_conjugate(state.orientation),
            load3(contact.position)-load3(state.position)));
        cached.normal=contact.normal;
        cached.normal_impulse=contact.accumulated_normal_impulse;
        cached.friction_impulse=contact.accumulated_friction_impulse;
        contact_cache_rows[pair*8u+point_index]=cached;
        if(contact.persistent!=0u && step_collect_contacts!=0u &&
           manifold.event_offset+point_index<step_event_capacity) {
            RigidContactEvent event=contact_events[
                manifold.event_offset+point_index];
            event.normal_impulse=contact.accumulated_normal_impulse;
            event.friction_impulse=contact.accumulated_friction_impulse;
            contact_events[manifold.event_offset+point_index]=event;
        }
    }
}
float3 relative_rotation_vector(Quaternion frame_a, Quaternion frame_b) {
    Quaternion relative = q_normalize(q_multiply(q_conjugate(frame_a), frame_b));
    if (relative.w < 0.0f) {
        relative.x=-relative.x; relative.y=-relative.y;
        relative.z=-relative.z; relative.w=-relative.w;
    }
    float size=sqrt(relative.x*relative.x+relative.y*relative.y+
                    relative.z*relative.z);
    float3 result=float3(0,0,0);
    if(size>1.0e-6f) {
        float angle=2.0f*atan2(size,clamp(relative.w,-1.0f,1.0f));
        result=float3(relative.x,relative.y,relative.z)*(angle/size);
    }
    return result;
}

float solve_linear_axis(inout RigidBodyState sa, RigidParameters a,
                        inout RigidBodyState sb, RigidParameters b,
                        float3 arm_a, float3 arm_b,
                        ConstraintAxisGeometry geometry, float error,
                        float stiffness, float damping, bool spring) {
    float3 axis=load3(geometry.axis);
    float3 velocity_a=load3(sa.linear_velocity)+
                      cross(load3(sa.angular_velocity),arm_a);
    float3 velocity_b=load3(sb.linear_velocity)+
                      cross(load3(sb.angular_velocity),arm_b);
    float relative_velocity=dot(velocity_b-velocity_a,axis);
    float denominator=geometry.linear_denominator;
    float impulse=0.0f;
    if(denominator>1.0e-6f) {
        impulse=spring
            ? -(relative_velocity+stiffness*error*step_timestep)/
                (denominator+damping*step_timestep)
            : -(relative_velocity+0.35f*error/step_timestep)/denominator;
        float3 vector_impulse=axis*impulse;
        if(a.inverse_mass>0.0f) {
            sa.linear_velocity=store3(load3(sa.linear_velocity)-
                                      vector_impulse*a.inverse_mass);
            sa.angular_velocity=store3(load3(sa.angular_velocity)-
                inverse_inertia_mul(a,sa,cross(arm_a,vector_impulse)));
        }
        if(b.inverse_mass>0.0f) {
            sb.linear_velocity=store3(load3(sb.linear_velocity)+
                                      vector_impulse*b.inverse_mass);
            sb.angular_velocity=store3(load3(sb.angular_velocity)+
                inverse_inertia_mul(b,sb,cross(arm_b,vector_impulse)));
        }
    }
    return abs(impulse);
}

float solve_angular_axis(inout RigidBodyState sa, RigidParameters a,
                         inout RigidBodyState sb, RigidParameters b,
                         ConstraintAxisGeometry geometry, float error,
                         float stiffness, float damping, bool spring) {
    float3 axis=load3(geometry.axis);
    float relative_velocity=dot(load3(sb.angular_velocity)-
                                load3(sa.angular_velocity),axis);
    float3 inverse_a=load3(geometry.inverse_angular_a);
    float3 inverse_b=load3(geometry.inverse_angular_b);
    float denominator=geometry.angular_denominator;
    float impulse=0.0f;
    if(denominator>1.0e-6f) {
        impulse=spring
            ? -(relative_velocity+stiffness*error*step_timestep)/
                (denominator+damping*step_timestep)
            : -(relative_velocity+0.30f*error/step_timestep)/denominator;
        if(a.inverse_mass>0.0f)
            sa.angular_velocity=store3(load3(sa.angular_velocity)-inverse_a*impulse);
        if(b.inverse_mass>0.0f)
            sb.angular_velocity=store3(load3(sb.angular_velocity)+inverse_b*impulse);
    }
    return abs(impulse);
}

float solve_motor_axis(inout RigidBodyState sa,RigidParameters a,
                       inout RigidBodyState sb,RigidParameters b,
                       ConstraintAxisGeometry geometry,float3 arm_a,
                       float3 arm_b,float target_velocity,
                       float maximum_impulse,bool angular) {
    float3 axis=load3(geometry.axis);
    float denominator=0.0f;
    float relative_velocity=0.0f;
    if(angular) {
        relative_velocity=dot(load3(sb.angular_velocity)-
                              load3(sa.angular_velocity),axis);
        denominator=geometry.angular_denominator;
    } else {
        float3 velocity_a=load3(sa.linear_velocity)+
                          cross(load3(sa.angular_velocity),arm_a);
        float3 velocity_b=load3(sb.linear_velocity)+
                          cross(load3(sb.angular_velocity),arm_b);
        relative_velocity=dot(velocity_b-velocity_a,axis);
        denominator=geometry.linear_denominator;
    }
    float impulse=0.0f;
    if(denominator>1.0e-6f && maximum_impulse>0.0f) {
        impulse=clamp((target_velocity-relative_velocity)/denominator,
                      -maximum_impulse,maximum_impulse);
        if(angular) {
            if(a.inverse_mass>0.0f)
                sa.angular_velocity=store3(load3(sa.angular_velocity)-
                    load3(geometry.inverse_angular_a)*impulse);
            if(b.inverse_mass>0.0f)
                sb.angular_velocity=store3(load3(sb.angular_velocity)+
                    load3(geometry.inverse_angular_b)*impulse);
        } else {
            float3 vector_impulse=axis*impulse;
            if(a.inverse_mass>0.0f) {
                sa.linear_velocity=store3(load3(sa.linear_velocity)-
                                          vector_impulse*a.inverse_mass);
                sa.angular_velocity=store3(load3(sa.angular_velocity)-
                    inverse_inertia_mul(a,sa,cross(arm_a,vector_impulse)));
            }
            if(b.inverse_mass>0.0f) {
                sb.linear_velocity=store3(load3(sb.linear_velocity)+
                                          vector_impulse*b.inverse_mass);
                sb.angular_velocity=store3(load3(sb.angular_velocity)+
                    inverse_inertia_mul(b,sb,cross(arm_b,vector_impulse)));
            }
        }
    }
    return abs(impulse);
}

uint prepare_constraints() {
    uint iterations=0u;
    [loop] for(uint index=0u;index<step_constraint_capacity;++index) {
        RigidConstraint joint=constraints[index];
        ConstraintGeometry geometry=(ConstraintGeometry)0;
        if(joint.alive==0u) {
            constraint_geometry[index]=geometry;
            continue;
        }
        if(joint.broken==0u) joint.applied_impulse=0.0f;
        joint.enabled=(joint.enabled!=0u && joint.broken==0u)?1u:0u;
        constraints[index]=joint;
        if(joint.enabled==0u || joint.body_a>=step_body_count ||
           joint.body_b>=step_body_count) {
            constraint_geometry[index]=geometry;
            continue;
        }
        bool absorbed=joint.type==0u && joint.disable_collisions!=0u &&
            joint.breaking_impulse_threshold<=0.0f &&
            compounds[joint.body_a].eligible!=0u &&
            compounds[joint.body_b].eligible!=0u &&
            compounds[joint.body_a].root==compounds[joint.body_b].root;
        if(absorbed) {
            constraint_geometry[index]=geometry;
            continue;
        }
        RigidParameters a=parameters[joint.body_a];
        RigidParameters b=parameters[joint.body_b];
        RigidBodyState sa=states[joint.body_a];
        RigidBodyState sb=states[joint.body_b];
        float3 arm_a=q_rotate(sa.orientation,load3(joint.local_anchor_a));
        float3 arm_b=q_rotate(sb.orientation,load3(joint.local_anchor_b));
        geometry.valid=1u;
        geometry.arm_a=store3(arm_a);
        geometry.arm_b=store3(arm_b);
        geometry.anchor_error=store3((load3(sb.position)+arm_b)-
                                     (load3(sa.position)+arm_a));
        Quaternion frame_a=q_normalize(q_multiply(sa.orientation,
                                                   joint.local_orientation_a));
        Quaternion frame_b=q_normalize(q_multiply(sb.orientation,
                                                   joint.local_orientation_b));
        geometry.rotation_error=store3(relative_rotation_vector(frame_a,frame_b));
        geometry.hinge_alignment_error=store3(cross(
            q_rotate(frame_a,float3(0,0,1)),q_rotate(frame_b,float3(0,0,1))));
        geometry.piston_alignment_error=store3(cross(
            q_rotate(frame_a,float3(1,0,0)),q_rotate(frame_b,float3(1,0,0))));
        [unroll] for(uint axis_index=0u;axis_index<3u;++axis_index) {
            ConstraintAxisGeometry row;
            float3 axis=q_rotate(frame_a,basis(axis_index));
            float3 inverse_a=inverse_inertia_mul(a,sa,axis);
            float3 inverse_b=inverse_inertia_mul(b,sb,axis);
            row.axis=store3(axis);
            row.inverse_angular_a=store3(inverse_a);
            row.inverse_angular_b=store3(inverse_b);
            row.angular_denominator=dot(inverse_a+inverse_b,axis);
            float3 angular_a=cross(inverse_inertia_mul(
                a,sa,cross(arm_a,axis)),arm_a);
            float3 angular_b=cross(inverse_inertia_mul(
                b,sb,cross(arm_b,axis)),arm_b);
            row.linear_denominator=a.inverse_mass+b.inverse_mass+
                                   dot(angular_a+angular_b,axis);
            geometry.axes[axis_index]=row;
        }
        constraint_geometry[index]=geometry;
        iterations=max(iterations,joint.solver_iterations);
    }
    return iterations;
}

void solve_prepared_joint(uint joint_index,uint iteration,uint contact_sweeps) {
    RigidConstraint joint=constraints[joint_index];
    ConstraintGeometry geometry=constraint_geometry[joint_index];
    if(joint.alive==0u || joint.enabled==0u || geometry.valid==0u ||
       iteration>=joint.solver_iterations*contact_sweeps) return;
    RigidParameters a=parameters[joint.body_a];
    RigidParameters b=parameters[joint.body_b];
    RigidBodyState sa=states[joint.body_a];
    RigidBodyState sb=states[joint.body_b];
    float3 arm_a=load3(geometry.arm_a);
    float3 arm_b=load3(geometry.arm_b);
    float3 anchor_error=load3(geometry.anchor_error);
    float3 rotation_error=load3(geometry.rotation_error);
    float3 hinge_error=load3(geometry.hinge_alignment_error);
    float3 piston_error=load3(geometry.piston_alignment_error);
    float applied=0.0f;
    [unroll] for(uint axis_index=0u;axis_index<3u;++axis_index) {
        ConstraintAxisGeometry row=geometry.axes[axis_index];
        float3 axis=load3(row.axis);
        bool generic=joint.type==5u || joint.type==6u;
        bool motor=joint.type==7u;
        bool linear_lock=joint.type==0u || joint.type==1u || joint.type==2u ||
            ((joint.type==3u || joint.type==4u) && axis_index!=0u) ||
            (motor && (axis_index!=0u || joint.linear_motor_enabled==0u));
        bool linear_spring=joint.type==6u &&
            axis_enabled(joint.linear_spring_axes,axis_index);
        bool linear_limit=generic && axis_enabled(joint.linear_limit_axes,axis_index);
        float linear_error=dot(anchor_error,axis);
        if(linear_limit && !linear_lock && !linear_spring)
            linear_error=limit_error(linear_error,
                component(load3(joint.linear_limit_lower),axis_index),
                component(load3(joint.linear_limit_upper),axis_index));
        if(linear_lock || linear_spring || (linear_limit && linear_error!=0.0f))
            applied+=solve_linear_axis(sa,a,sb,b,arm_a,arm_b,row,linear_error,
                component(load3(joint.linear_spring_stiffness),axis_index),
                component(load3(joint.linear_spring_damping),axis_index),
                linear_spring && !linear_lock);

        bool angular_lock=joint.type==0u || joint.type==3u ||
            (joint.type==2u && axis_index!=2u) ||
            (joint.type==4u && axis_index!=0u) ||
            (motor && (axis_index!=0u || joint.angular_motor_enabled==0u));
        bool angular_spring=joint.type==6u &&
            axis_enabled(joint.angular_spring_axes,axis_index);
        bool angular_limit=generic && axis_enabled(joint.angular_limit_axes,axis_index);
        float angular_error=joint.type==2u && axis_index!=2u
            ? dot(hinge_error,axis) : joint.type==4u && axis_index!=0u
                ? dot(piston_error,axis) : component(rotation_error,axis_index);
        if(angular_limit && !angular_lock && !angular_spring)
            angular_error=limit_error(angular_error,
                component(load3(joint.angular_limit_lower),axis_index),
                component(load3(joint.angular_limit_upper),axis_index));
        if(angular_lock || angular_spring ||
           (angular_limit && angular_error!=0.0f))
            applied+=solve_angular_axis(sa,a,sb,b,row,angular_error,
                component(load3(joint.angular_spring_stiffness),axis_index),
                component(load3(joint.angular_spring_damping),axis_index),
                angular_spring && !angular_lock);
    }
    if(joint.type==2u && axis_enabled(joint.angular_limit_axes,2u)) {
        float error=limit_error(rotation_error.z,joint.angular_limit_lower.z,
                                joint.angular_limit_upper.z);
        if(error!=0.0f) applied+=solve_angular_axis(
            sa,a,sb,b,geometry.axes[2],error,0.0f,0.0f,false);
    }
    if(joint.type==3u && axis_enabled(joint.linear_limit_axes,0u)) {
        float error=limit_error(dot(anchor_error,load3(geometry.axes[0].axis)),
                                joint.linear_limit_lower.x,
                                joint.linear_limit_upper.x);
        if(error!=0.0f) applied+=solve_linear_axis(
            sa,a,sb,b,arm_a,arm_b,geometry.axes[0],error,0.0f,0.0f,false);
    }
    if(joint.type==4u) {
        if(axis_enabled(joint.linear_limit_axes,0u)) {
            float error=limit_error(dot(anchor_error,load3(geometry.axes[0].axis)),
                                    joint.linear_limit_lower.x,
                                    joint.linear_limit_upper.x);
            if(error!=0.0f) applied+=solve_linear_axis(
                sa,a,sb,b,arm_a,arm_b,geometry.axes[0],error,0.0f,0.0f,false);
        }
        if(axis_enabled(joint.angular_limit_axes,0u)) {
            float error=limit_error(rotation_error.x,joint.angular_limit_lower.x,
                                    joint.angular_limit_upper.x);
            if(error!=0.0f) applied+=solve_angular_axis(
                sa,a,sb,b,geometry.axes[0],error,0.0f,0.0f,false);
        }
    }
    if(joint.type==7u) {
        float inverse_iterations=1.0f/float(joint.solver_iterations*contact_sweeps);
        if(joint.linear_motor_enabled!=0u)
            applied+=solve_motor_axis(sa,a,sb,b,geometry.axes[0],arm_a,arm_b,
                joint.linear_target_velocity,
                joint.linear_maximum_impulse*inverse_iterations,false);
        if(joint.angular_motor_enabled!=0u)
            applied+=solve_motor_axis(sa,a,sb,b,geometry.axes[0],arm_a,arm_b,
                joint.angular_target_velocity,
                joint.angular_maximum_impulse*inverse_iterations,true);
    }
    states[joint.body_a]=sa;
    states[joint.body_b]=sb;
    joint.applied_impulse+=applied;
    if(joint.breaking_impulse_threshold>0.0f &&
       joint.applied_impulse>joint.breaking_impulse_threshold) {
        joint.broken=1u;
        joint.enabled=0u;
    }
    constraints[joint_index]=joint;
}

uint compound_root(uint body) {
    while(compounds[body].root!=body) body=compounds[body].root;
    return body;
}

bool compound_fixed_edge(RigidConstraint joint,uint a,uint b) {
    return joint.type==0u && joint.disable_collisions!=0u &&
           joint.breaking_impulse_threshold<=0.0f &&
           parameters[a].motion==2u && parameters[b].motion==2u;
}

struct SymmetricMatrix3 {
    float xx; float xy; float xz;
    float yy; float yz; float zz;
};

void add_inertia_axis(inout SymmetricMatrix3 tensor,float3 axis,float moment) {
    tensor.xx+=moment*axis.x*axis.x;
    tensor.xy+=moment*axis.x*axis.y;
    tensor.xz+=moment*axis.x*axis.z;
    tensor.yy+=moment*axis.y*axis.y;
    tensor.yz+=moment*axis.y*axis.z;
    tensor.zz+=moment*axis.z*axis.z;
}

float3 multiply_symmetric(SymmetricMatrix3 tensor,float3 value) {
    return float3(tensor.xx*value.x+tensor.xy*value.y+tensor.xz*value.z,
                  tensor.xy*value.x+tensor.yy*value.y+tensor.yz*value.z,
                  tensor.xz*value.x+tensor.yz*value.y+tensor.zz*value.z);
}

bool invert_symmetric(SymmetricMatrix3 tensor,out float3 row0,
                      out float3 row1,out float3 row2) {
    row0=float3(0,0,0); row1=float3(0,0,0); row2=float3(0,0,0);
    float c00=tensor.yy*tensor.zz-tensor.yz*tensor.yz;
    float c01=tensor.xz*tensor.yz-tensor.xy*tensor.zz;
    float c02=tensor.xy*tensor.yz-tensor.xz*tensor.yy;
    float c11=tensor.xx*tensor.zz-tensor.xz*tensor.xz;
    float c12=tensor.xy*tensor.xz-tensor.xx*tensor.yz;
    float c22=tensor.xx*tensor.yy-tensor.xy*tensor.xy;
    float determinant=tensor.xx*c00+tensor.xy*c01+tensor.xz*c02;
    bool valid=isfinite(determinant) && abs(determinant)>1.0e-6f;
    if(valid) {
        float scale=1.0f/determinant;
        row0=float3(c00,c01,c02)*scale;
        row1=float3(c01,c11,c12)*scale;
        row2=float3(c02,c12,c22)*scale;
    }
    return valid;
}

float3 compound_inverse_inertia(RigidCompound compound,float3 value) {
    return float3(dot(load3(compound.inverse_inertia[0]),value),
                  dot(load3(compound.inverse_inertia[1]),value),
                  dot(load3(compound.inverse_inertia[2]),value));
}

void build_compounds() {
    [loop] for(uint init_body=0u;init_body<step_body_count;++init_body) {
        RigidCompound compound=(RigidCompound)0;
        compound.root=init_body;
        compounds[init_body]=compound;
    }
    [loop] for(uint union_index=0u;union_index<step_constraint_capacity;++union_index) {
        RigidConstraint joint=constraints[union_index];
        if(joint.alive==0u || joint.enabled==0u || joint.broken!=0u ||
           joint.body_a>=step_body_count || joint.body_b>=step_body_count ||
           !compound_fixed_edge(joint,joint.body_a,joint.body_b)) continue;
        uint root_a=compound_root(joint.body_a);
        uint root_b=compound_root(joint.body_b);
        if(root_a!=root_b) {
            uint high=max(root_a,root_b);
            RigidCompound high_compound=compounds[high];
            high_compound.root=min(root_a,root_b);
            compounds[high]=high_compound;
        }
    }
    [loop] for(uint flatten_body=0u;flatten_body<step_body_count;++flatten_body) {
        RigidCompound flattened=compounds[flatten_body];
        flattened.root=compound_root(flatten_body);
        compounds[flatten_body]=flattened;
    }
    [loop] for(uint block_index=0u;block_index<step_constraint_capacity;++block_index) {
        RigidConstraint joint=constraints[block_index];
        if(joint.alive==0u || joint.enabled==0u || joint.broken!=0u ||
           joint.body_a>=step_body_count || joint.body_b>=step_body_count)
            continue;
        if(compound_fixed_edge(joint,joint.body_a,joint.body_b)) continue;
        uint root_a=compounds[joint.body_a].root;
        uint root_b=compounds[joint.body_b].root;
        RigidCompound a=compounds[root_a]; a.blocked=1u; compounds[root_a]=a;
        RigidCompound b=compounds[root_b]; b.blocked=1u; compounds[root_b]=b;
    }
    [loop] for(uint count_body=0u;count_body<step_body_count;++count_body) {
        uint count_root=compounds[count_body].root;
        RigidCompound counted=compounds[count_root];
        counted.member_count+=1u;
        compounds[count_root]=counted;
    }
    [loop] for(uint root_index=0u;root_index<step_body_count;++root_index) {
        RigidCompound compound=compounds[root_index];
        if(compound.root!=root_index || compound.member_count<2u ||
           compound.blocked!=0u) continue;
        compound.eligible=1u;
        compounds[root_index]=compound;
        [loop] for(uint link_pass=1u;link_pass<compound.member_count;++link_pass) {
            bool changed=false;
            [loop] for(uint link_index=0u;link_index<step_constraint_capacity;++link_index) {
                RigidConstraint joint=constraints[link_index];
                if(joint.alive==0u || joint.enabled==0u || joint.broken!=0u ||
                   joint.body_a>=step_body_count || joint.body_b>=step_body_count)
                    continue;
                uint a=joint.body_a;
                uint b=joint.body_b;
                RigidCompound compound_a=compounds[a];
                RigidCompound compound_b=compounds[b];
                if(compound_a.root!=root_index || compound_b.root!=root_index ||
                   !compound_fixed_edge(joint,a,b) ||
                   compound_a.eligible==compound_b.eligible) continue;
                RigidBodyState state_a=states[a];
                RigidBodyState state_b=states[b];
                if(compound_a.eligible!=0u) {
                    state_b.orientation=q_normalize(q_multiply(
                        q_multiply(state_a.orientation,joint.local_orientation_a),
                        q_conjugate(joint.local_orientation_b)));
                    state_b.position=store3(load3(state_a.position)+
                        q_rotate(state_a.orientation,load3(joint.local_anchor_a))-
                        q_rotate(state_b.orientation,load3(joint.local_anchor_b)));
                    compound_b.eligible=1u;
                    states[b]=state_b;
                    compounds[b]=compound_b;
                } else {
                    state_a.orientation=q_normalize(q_multiply(
                        q_multiply(state_b.orientation,joint.local_orientation_b),
                        q_conjugate(joint.local_orientation_a)));
                    state_a.position=store3(load3(state_b.position)+
                        q_rotate(state_b.orientation,load3(joint.local_anchor_b))-
                        q_rotate(state_a.orientation,load3(joint.local_anchor_a)));
                    compound_a.eligible=1u;
                    states[a]=state_a;
                    compounds[a]=compound_a;
                }
                changed=true;
            }
            if(!changed) break;
        }
        bool connected=true;
        [loop] for(uint connect_member=0u;connect_member<step_body_count;++connect_member)
            if(compounds[connect_member].root==root_index &&
               compounds[connect_member].eligible==0u)
                connected=false;
        if(!connected) {
            [loop] for(uint clear_member=0u;clear_member<step_body_count;++clear_member) {
                RigidCompound item=compounds[clear_member];
                if(item.root==root_index) {
                    item.eligible=0u; compounds[clear_member]=item;
                }
            }
            continue;
        }
        float mass_sum=0.0f;
        float3 weighted_center=float3(0,0,0);
        float3 linear_momentum=float3(0,0,0);
        [loop] for(uint mass_member=0u;mass_member<step_body_count;++mass_member) {
            if(compounds[mass_member].root!=root_index) continue;
            float inverse_mass=parameters[mass_member].inverse_mass;
            if(inverse_mass<=1.0e-6f) { compound.blocked=1u; break; }
            float mass=1.0f/inverse_mass;
            mass_sum+=mass;
            weighted_center+=load3(states[mass_member].position)*mass;
            linear_momentum+=load3(states[mass_member].linear_velocity)*mass;
        }
        if(compound.blocked!=0u || mass_sum<=1.0e-6f) {
            [loop] for(uint blocked_member=0u;blocked_member<step_body_count;++blocked_member) {
                RigidCompound item=compounds[blocked_member];
                if(item.root==root_index) {
                    item.eligible=0u; compounds[blocked_member]=item;
                }
            }
            compounds[root_index]=compound;
            continue;
        }
        float3 center=weighted_center/mass_sum;
        compound.center=store3(center);
        SymmetricMatrix3 inertia=(SymmetricMatrix3)0;
        float3 angular_momentum=float3(0,0,0);
        [loop] for(uint inertia_member=0u;inertia_member<step_body_count;++inertia_member) {
            if(compounds[inertia_member].root!=root_index) continue;
            RigidParameters member_body=parameters[inertia_member];
            RigidBodyState state=states[inertia_member];
            float mass=1.0f/member_body.inverse_mass;
            float3 inverse_inertia=load3(member_body.inverse_inertia);
            float3 local_moment=float3(
                1.0f/max(inverse_inertia.x,1.0e-6f),
                1.0f/max(inverse_inertia.y,1.0e-6f),
                1.0f/max(inverse_inertia.z,1.0e-6f));
            SymmetricMatrix3 member_inertia=(SymmetricMatrix3)0;
            add_inertia_axis(member_inertia,q_rotate(state.orientation,float3(1,0,0)),local_moment.x);
            add_inertia_axis(member_inertia,q_rotate(state.orientation,float3(0,1,0)),local_moment.y);
            add_inertia_axis(member_inertia,q_rotate(state.orientation,float3(0,0,1)),local_moment.z);
            inertia.xx+=member_inertia.xx; inertia.xy+=member_inertia.xy;
            inertia.xz+=member_inertia.xz; inertia.yy+=member_inertia.yy;
            inertia.yz+=member_inertia.yz; inertia.zz+=member_inertia.zz;
            float3 arm=load3(state.position)-center;
            float radius_squared=dot(arm,arm);
            inertia.xx+=mass*(radius_squared-arm.x*arm.x);
            inertia.xy-=mass*arm.x*arm.y;
            inertia.xz-=mass*arm.x*arm.z;
            inertia.yy+=mass*(radius_squared-arm.y*arm.y);
            inertia.yz-=mass*arm.y*arm.z;
            inertia.zz+=mass*(radius_squared-arm.z*arm.z);
            angular_momentum+=multiply_symmetric(
                member_inertia,load3(state.angular_velocity))+
                cross(arm,load3(state.linear_velocity)*mass);
        }
        float3 row0,row1,row2;
        if(!invert_symmetric(inertia,row0,row1,row2)) {
            [loop] for(uint singular_member=0u;singular_member<step_body_count;++singular_member) {
                RigidCompound item=compounds[singular_member];
                if(item.root==root_index) {
                    item.eligible=0u; compounds[singular_member]=item;
                }
            }
            continue;
        }
        compound.inverse_inertia[0]=store3(row0);
        compound.inverse_inertia[1]=store3(row1);
        compound.inverse_inertia[2]=store3(row2);
        compound.inverse_mass=1.0f/mass_sum;
        compound.eligible=1u;
        compounds[root_index]=compound;
        float3 linear_velocity=linear_momentum*compound.inverse_mass;
        float3 angular_velocity=compound_inverse_inertia(compound,angular_momentum);
        [loop] for(uint velocity_member=0u;velocity_member<step_body_count;++velocity_member) {
            RigidCompound item=compounds[velocity_member];
            if(item.root!=root_index) continue;
            item.eligible=1u;
            compounds[velocity_member]=item;
            RigidBodyState state=states[velocity_member];
            state.angular_velocity=store3(angular_velocity);
            state.linear_velocity=store3(linear_velocity+cross(
                angular_velocity,load3(state.position)-center));
            states[velocity_member]=state;
        }
    }
}

void build_fixed_collision_groups() {
    [loop] for(uint body=0u;body<step_body_count;++body)
        color_owners[body]=body;
    [loop] for(uint group_pass=0u;group_pass<step_body_count;++group_pass) {
        bool changed=false;
        [loop] for(uint index=0u;index<step_constraint_capacity;++index) {
            RigidConstraint joint=constraints[index];
            if(joint.alive==0u || joint.enabled==0u || joint.broken!=0u ||
               joint.disable_collisions==0u || joint.type!=0u ||
               joint.body_a>=step_body_count || joint.body_b>=step_body_count)
                continue;
            uint root_a=color_owners[joint.body_a];
            uint root_b=color_owners[joint.body_b];
            uint low=min(root_a,root_b);
            uint high=max(root_a,root_b);
            if(low==high) continue;
            [loop] for(uint member=0u;member<step_body_count;++member) {
                if(color_owners[member]==high) {
                    color_owners[member]=low;
                    changed=true;
                }
            }
        }
        if(!changed) break;
    }
}

[numthreads(1,1,1)]
void rigid_prepare(uint3 dispatch_id : SV_DispatchThreadID) {
    if(dispatch_id.x!=0u) return;
    build_compounds();
    build_fixed_collision_groups();
    [loop] for(uint body=0u;body<step_body_count;++body)
        hinge_frames[body]=rigid_hinge_contact_frame(body);
}

uint fixed_projection_root(uint body) {
    while(compounds[body].projection_root!=body)
        body=compounds[body].projection_root;
    return body;
}

void project_fixed_ground_contacts() {
    bool welded=false;
    [loop] for(uint projection_init_body=0u;projection_init_body<step_body_count;++projection_init_body) {
        RigidCompound compound=compounds[projection_init_body];
        compound.projection_root=projection_init_body;
        compound.projection_movable=1u;
        compound.projection_translation=store3(float3(0,0,0));
        compounds[projection_init_body]=compound;
    }
    [loop] for(uint projection_union_index=0u;projection_union_index<step_constraint_capacity;++projection_union_index) {
        RigidConstraint joint=constraints[projection_union_index];
        if(joint.alive==0u || joint.enabled==0u || joint.broken!=0u ||
           joint.type!=0u || joint.body_a>=step_body_count ||
           joint.body_b>=step_body_count) continue;
        uint root_a=fixed_projection_root(joint.body_a);
        uint root_b=fixed_projection_root(joint.body_b);
        if(root_a!=root_b) {
            uint high=max(root_a,root_b);
            RigidCompound compound=compounds[high];
            compound.projection_root=min(root_a,root_b);
            compounds[high]=compound;
        }
        welded=true;
    }
    if(!welded) return;
    [loop] for(uint projection_flatten_body=0u;projection_flatten_body<step_body_count;++projection_flatten_body) {
        RigidCompound compound=compounds[projection_flatten_body];
        compound.projection_root=fixed_projection_root(projection_flatten_body);
        compounds[projection_flatten_body]=compound;
        if(parameters[projection_flatten_body].motion!=2u) {
            uint static_root=compound.projection_root;
            RigidCompound group=compounds[static_root];
            group.projection_movable=0u;
            compounds[static_root]=group;
        }
    }
    [loop] for(uint projection_block_index=0u;projection_block_index<step_constraint_capacity;++projection_block_index) {
        RigidConstraint joint=constraints[projection_block_index];
        if(joint.alive==0u || joint.enabled==0u || joint.broken!=0u ||
           joint.type==0u) continue;
        if(joint.body_a<step_body_count) {
            uint first_block_root=compounds[joint.body_a].projection_root;
            RigidCompound group=compounds[first_block_root];
            group.projection_movable=0u;
            compounds[first_block_root]=group;
        }
        if(joint.body_b<step_body_count) {
            uint second_block_root=compounds[joint.body_b].projection_root;
            RigidCompound second_group=compounds[second_block_root];
            second_group.projection_movable=0u;
            compounds[second_block_root]=second_group;
        }
    }
    uint pair_count=step_body_count*step_body_count;
    [loop] for(uint projection_pass=0u;projection_pass<8u;++projection_pass) {
        [loop] for(uint projection_pair=0u;projection_pair<pair_count;++projection_pair) {
            ContactManifold manifold=manifolds[projection_pair];
            if(manifold.count==0u) continue;
            uint a=projection_pair/step_body_count;
            uint b=projection_pair-a*step_body_count;
            bool first=parameters[a].motion==2u &&
                       parameters[b].motion!=2u &&
                       (manifold.color&manifold_body_fixed_member)!=0u;
            bool second=parameters[b].motion==2u &&
                        parameters[a].motion!=2u &&
                        (manifold.color&manifold_collider_fixed_member)!=0u;
            if(!first && !second) continue;
            uint root=compounds[first?a:b].projection_root;
            RigidCompound group=compounds[root];
            if(group.projection_movable==0u) continue;
            [loop] for(uint projection_contact=0u;
                       projection_contact<manifold.count;++projection_contact) {
                float3 normal=load3(manifold.contacts[projection_contact].normal)*
                              (first?1.0f:-1.0f);
                float depth=manifold.contacts[projection_contact].penetration-
                    dot(normal,load3(group.projection_translation));
                if(depth>0.0f)
                    group.projection_translation=store3(
                        load3(group.projection_translation)+
                        normal*(depth+1.0e-5f));
            }
            compounds[root]=group;
        }
    }
    [loop] for(uint projection_apply_body=0u;projection_apply_body<step_body_count;++projection_apply_body) {
        RigidCompound group=compounds[compounds[projection_apply_body].projection_root];
        RigidBodyState state=states[projection_apply_body];
        state.position=store3(load3(state.position)+
                              load3(group.projection_translation));
        states[projection_apply_body]=state;
    }
}

void project_hinge_anchors() {
    [loop] for(uint index=0u;index<step_constraint_capacity;++index) {
        RigidConstraint joint=constraints[index];
        if(joint.alive==0u || joint.enabled==0u || joint.type!=2u ||
           joint.body_a>=step_body_count || joint.body_b>=step_body_count)
            continue;
        RigidParameters a=parameters[joint.body_a];
        RigidParameters b=parameters[joint.body_b];
        float inverse_mass_sum=a.inverse_mass+b.inverse_mass;
        if(inverse_mass_sum<=1.0e-6f) continue;
        RigidBodyState sa=states[joint.body_a];
        RigidBodyState sb=states[joint.body_b];
        float3 anchor_a=load3(sa.position)+
            q_rotate(sa.orientation,load3(joint.local_anchor_a));
        float3 anchor_b=load3(sb.position)+
            q_rotate(sb.orientation,load3(joint.local_anchor_b));
        float3 error=anchor_b-anchor_a;
        sa.position=store3(load3(sa.position)+
                           error*(a.inverse_mass/inverse_mass_sum));
        sb.position=store3(load3(sb.position)-
                           error*(b.inverse_mass/inverse_mass_sum));
        states[joint.body_a]=sa;
        states[joint.body_b]=sb;
    }
}

[numthreads(64,1,1)]
void rigid_integrate(uint3 dispatch_id : SV_DispatchThreadID) {
    uint index=dispatch_id.x;
    if(index>=step_body_count) return;
    RigidBodyState state=states[index];
    RigidParameters body=parameters[index];
    previous_states[index]=state;
    if(body.motion==0) {
        state.linear_velocity=store3(float3(0,0,0));
        state.angular_velocity=store3(float3(0,0,0));
    } else if(body.motion==1) {
        if(body.has_kinematic_target!=0) {
            uint remaining=max(1u,step_substeps-step_substep_index);
            float fraction=1.0f/(float)remaining;
            float3 old_position=load3(state.position);
            Quaternion old_orientation=state.orientation;
            state.position=store3(lerp(old_position,load3(body.kinematic_target.position),fraction));
            Quaternion target=body.kinematic_target.orientation;
            float orientation_dot=old_orientation.x*target.x+old_orientation.y*target.y+
                                  old_orientation.z*target.z+old_orientation.w*target.w;
            if(orientation_dot<0) { target.x=-target.x;target.y=-target.y;target.z=-target.z;target.w=-target.w; }
            Quaternion mixed={lerp(old_orientation.x,target.x,fraction),
                              lerp(old_orientation.y,target.y,fraction),
                              lerp(old_orientation.z,target.z,fraction),
                              lerp(old_orientation.w,target.w,fraction)};
            state.orientation=q_normalize(mixed);
            state.linear_velocity=store3((load3(state.position)-old_position)/step_timestep);
            state.angular_velocity=store3(quaternion_delta(old_orientation,state.orientation)/step_timestep);
            if(remaining==1) body.has_kinematic_target=0;
        } else {
            state.linear_velocity=store3(float3(0,0,0));
            state.angular_velocity=store3(float3(0,0,0));
        }
    } else {
        float3 linear_velocity=load3(state.linear_velocity)+
            (step_gravity+load3(forces[index])*body.inverse_mass)*step_timestep;
        float3 angular=load3(state.angular_velocity)+
            inverse_inertia_mul(body,state,load3(torques[index]))*step_timestep;
        linear_velocity/=1.0f+body.linear_damping*step_timestep;
        angular/=1.0f+body.angular_damping*step_timestep;
        linear_velocity=limit_length(linear_velocity,body.maximum_linear_speed);
        angular=limit_length(angular,body.maximum_angular_speed);
        HingeContactFrame frame=rigid_hinge_contact_frame(index);
        Quaternion axial_orientation=state.orientation;
        if(frame.axial!=0u) {
            axial_orientation=rigid_axial_orientation(index);
            float3 local_omega=q_rotate(
                q_conjugate(state.orientation),angular);
            float3 inverse_inertia=load3(body.inverse_inertia);
            float3 angular_momentum=q_rotate(state.orientation,float3(
                local_omega.x/max(inverse_inertia.x,1.0e-6f),
                local_omega.y/max(inverse_inertia.y,1.0e-6f),
                local_omega.z/max(inverse_inertia.z,1.0e-6f)));
            float3 axis=load3(frame.axis);
            float3 arm=load3(state.position)-load3(frame.anchor);
            float spin=dot(axis,angular_momentum+cross(
                arm,linear_velocity/body.inverse_mass))*
                fixed_hinge_inverse_moment_state(body,state,frame);
            angular=axis*spin;
            linear_velocity=axis*dot(linear_velocity,axis)+
                cross(angular,arm);
        }
        state.linear_velocity=store3(linear_velocity);
        state.angular_velocity=store3(angular);
        state.position=store3(load3(state.position)+linear_velocity*step_timestep);
        Quaternion velocity={angular.x,angular.y,angular.z,0};
        Quaternion derivative=q_multiply(velocity,state.orientation);
        state.orientation=q_normalize(make_quaternion(
            state.orientation.x+0.5f*derivative.x*step_timestep,
            state.orientation.y+0.5f*derivative.y*step_timestep,
            state.orientation.z+0.5f*derivative.z*step_timestep,
            state.orientation.w+0.5f*derivative.w*step_timestep));
        if(frame.axial!=0u) {
            float3 axis=load3(frame.axis);
            if(frame.axial_rotation!=0u) {
                Quaternion relative=q_multiply(
                    state.orientation,q_conjugate(axial_orientation));
                float twist=dot(float3(relative.x,relative.y,relative.z),axis);
                Quaternion rotation=q_normalize(make_quaternion(
                    axis.x*twist,axis.y*twist,axis.z*twist,relative.w));
                state.orientation=q_normalize(
                    q_multiply(rotation,axial_orientation));
            } else {
                state.orientation=axial_orientation;
            }
            float3 anchor=load3(state.position)+
                q_rotate(state.orientation,load3(frame.local_anchor));
            state.position=store3(load3(frame.anchor)+axis*dot(
                anchor-load3(frame.anchor),axis)-
                q_rotate(state.orientation,load3(frame.local_anchor)));
        }
    }
    states[index]=state;
    parameters[index]=body;
    // Collision generation is a later dispatch. Publish one frame per body
    // here so pair kernels do not rescan every constraint.
    hinge_frames[index]=rigid_hinge_contact_frame(index);
}

[numthreads(64,1,1)]
void rigid_generate(uint3 dispatch_id : SV_DispatchThreadID) {
    uint pair=dispatch_id.x;
    uint pair_count=step_body_count*step_body_count;
    if(pair>=pair_count) return;
    uint body=pair/step_body_count;
    uint collider=pair-body*step_body_count;
    ContactManifold manifold=empty_manifold();
    RigidParameters body_parameters=parameters[body];
    RigidParameters collider_parameters=parameters[collider];
    bool active=body!=collider && body_parameters.inverse_mass>0.0f;
    if(active && collider_parameters.inverse_mass>0.0f && collider<body)
        active=false;
    if(active && suppress_pair(body,collider)) active=false;
    MeshInfo body_mesh=mesh_infos[body_parameters.mesh_index];
    MeshInfo collider_mesh=mesh_infos[collider_parameters.mesh_index];
    active=active && body_mesh.alive!=0u && collider_mesh.alive!=0u &&
           body_mesh.index_count>=3u && collider_mesh.index_count>=3u;
    bool eligible=active;
    float margin=body_parameters.collision_margin+
                 collider_parameters.collision_margin;
    bool pair_swept=false;
    if(active) {
        float3 body_minimum,body_maximum,collider_minimum,collider_maximum;
        transformed_motion_bounds(previous_states[body],states[body],
                                  body_mesh,margin,body_minimum,body_maximum);
        transformed_motion_bounds(previous_states[collider],states[collider],
                                  collider_mesh,0.0f,collider_minimum,
                                  collider_maximum);
        active=bounds_overlap(body_minimum,body_maximum,
                              collider_minimum,collider_maximum);
    }
    if(active) {
        HingeContactFrame body_hinge=hinge_frames[body];
        HingeContactFrame collider_hinge=hinge_frames[collider];
        bool guided=guided_static_pair(body_hinge,collider_hinge);
        if(guided) {
            manifolds[pair]=manifold;
            return;
        }
        bool swept=requires_swept_pair(body,collider,margin);
        pair_swept=swept;
        bool handled_face=!swept &&
            body_mesh.solid_plane_count!=0u &&
            collider_mesh.solid_plane_count!=0u &&
            body_mesh.index_count<=96u &&
            collider_mesh.index_count<=96u &&
            convex_face_manifold(body,collider,margin,manifold);
        if(handled_face) manifold.color=manifold_face_patch;
        if(!handled_face) {
            MeshLeafInfo body_leaf_info=
                mesh_leaf_infos[body_parameters.mesh_index];
            MeshLeafInfo collider_leaf_info=
                mesh_leaf_infos[collider_parameters.mesh_index];
            uint candidate_count=body_leaf_info.count*collider_leaf_info.count;
            if(candidate_count==0u || step_use_mesh_bvh==0u) {
                collide_current(body,collider,0u,body_mesh.index_count/3u,
                                0u,collider_mesh.index_count/3u,manifold);
                if(swept)
                    collide_swept(body,collider,0u,body_mesh.index_count/3u,
                                  0u,collider_mesh.index_count/3u,false,false,
                                  manifold);
            } else {
                uint short_count=candidate_count/128u;
                uint long_lanes=candidate_count-short_count*128u;
                uint long_candidates=long_lanes*(short_count+1u);
                [loop] for(uint ordinal=0u;ordinal<candidate_count;++ordinal) {
                    uint candidate=ordinal;
                    if(candidate_count>128u) {
                        uint lane,step;
                        if(ordinal<long_candidates) {
                            lane=ordinal/(short_count+1u);
                            step=ordinal-lane*(short_count+1u);
                        } else {
                            uint remainder=ordinal-long_candidates;
                            lane=long_lanes+remainder/short_count;
                            step=remainder-(lane-long_lanes)*short_count;
                        }
                        candidate=lane+step*128u;
                    }
                    uint body_leaf=candidate/collider_leaf_info.count;
                    uint collider_leaf=candidate-
                        body_leaf*collider_leaf_info.count;
                    BvhNode body_node=mesh_bvh_nodes[mesh_bvh_leaves[
                        body_leaf_info.offset+body_leaf]];
                    BvhNode collider_node=mesh_bvh_nodes[mesh_bvh_leaves[
                        collider_leaf_info.offset+collider_leaf]];
                    float3 body_minimum,body_maximum;
                    float3 collider_minimum,collider_maximum;
                    if(swept) {
                        transformed_node_motion_bounds(
                            previous_states[body],states[body],body_node,margin,
                            body_minimum,body_maximum);
                        transformed_node_motion_bounds(
                            previous_states[collider],states[collider],
                            collider_node,0.0f,collider_minimum,collider_maximum);
                    } else {
                        transformed_node_bounds(states[body],body_node,margin,
                                                body_minimum,body_maximum);
                        transformed_node_bounds(states[collider],collider_node,0.0f,
                                                collider_minimum,collider_maximum);
                    }
                    if(!bounds_overlap(body_minimum,body_maximum,
                                       collider_minimum,collider_maximum))
                        continue;
                    ContactManifold local_manifold=empty_manifold();
                    collide_current(body,collider,body_node.first_triangle,
                                    body_node.triangle_count,
                                    collider_node.first_triangle,
                                    collider_node.triangle_count,local_manifold);
                    if(swept)
                        collide_swept(body,collider,body_node.first_triangle,
                                      body_node.triangle_count,
                                      collider_node.first_triangle,
                                      collider_node.triangle_count,false,false,
                                      local_manifold);
                    float separation=max(margin*2.0f,1.0e-4f);
                    [loop] for(uint local_contact=0u;
                        local_contact<local_manifold.count;++local_contact)
                        add_manifold_contact(
                            manifold,local_manifold.contacts[local_contact],
                            separation,2u);
                }
            }
        }
    }
    ContactCacheHeader saved=contact_cache_headers[pair];
    bool recent_contact=saved.valid!=0u && saved.count!=0u &&
        saved.epoch+1u==step_contact_epoch &&
        saved.body.index==body_ids[body].index &&
        saved.body.generation==body_ids[body].generation &&
        saved.collider.index==body_ids[collider].index &&
        saved.collider.generation==body_ids[collider].generation &&
        abs(saved.timestep-step_timestep)<=1.0e-7f;
    float3 relative_motion=
        (load3(states[body].position)-load3(previous_states[body].position))-
        (load3(states[collider].position)-
         load3(previous_states[collider].position));
    if(eligible && manifold.count==0u && recent_contact &&
       dot(relative_motion,relative_motion)>margin*margin) {
        collide_swept(body,collider,0u,body_mesh.index_count/3u,
                      0u,collider_mesh.index_count/3u,false,true,manifold);
        pair_swept=true;
    }
    if(pair_swept) order_swept_manifold_cuda(manifold);
    manifolds[pair]=manifold;
}

[numthreads(64,1,1)]
void rigid_generate_guided(uint3 dispatch_id : SV_DispatchThreadID) {
    uint pair=dispatch_id.x;
    uint pair_count=step_body_count*step_body_count;
    if(pair>=pair_count) return;
    uint body=pair/step_body_count;
    uint collider=pair-body*step_body_count;
    HingeContactFrame body_hinge=hinge_frames[body];
    HingeContactFrame collider_hinge=hinge_frames[collider];
    if(!guided_static_pair(body_hinge,collider_hinge)) return;

    ContactManifold manifold=empty_manifold();
    RigidParameters body_parameters=parameters[body];
    RigidParameters collider_parameters=parameters[collider];
    bool active=body!=collider && body_parameters.inverse_mass>0.0f;
    if(active && collider_parameters.inverse_mass>0.0f && collider<body)
        active=false;
    if(active && suppress_pair(body,collider)) active=false;
    MeshInfo body_mesh=mesh_infos[body_parameters.mesh_index];
    MeshInfo collider_mesh=mesh_infos[collider_parameters.mesh_index];
    active=active && body_mesh.alive!=0u && collider_mesh.alive!=0u &&
           body_mesh.index_count>=3u && collider_mesh.index_count>=3u;
    float margin=body_parameters.collision_margin+
                 collider_parameters.collision_margin;
    if(active) {
        float3 body_minimum,body_maximum,collider_minimum,collider_maximum;
        transformed_motion_bounds(previous_states[body],states[body],
                                  body_mesh,margin,body_minimum,body_maximum);
        transformed_motion_bounds(previous_states[collider],states[collider],
                                  collider_mesh,0.0f,collider_minimum,
                                  collider_maximum);
        active=bounds_overlap(body_minimum,body_maximum,
                              collider_minimum,collider_maximum);
    }
    if(active) {
        MeshLeafInfo body_leaf_info=
            mesh_leaf_infos[body_parameters.mesh_index];
        MeshLeafInfo collider_leaf_info=
            mesh_leaf_infos[collider_parameters.mesh_index];
        uint candidate_count=body_leaf_info.count*collider_leaf_info.count;
        if(candidate_count==0u || step_use_mesh_bvh==0u) {
            collide_swept(body,collider,0u,body_mesh.index_count/3u,
                          0u,collider_mesh.index_count/3u,true,false,manifold);
        } else {
            uint short_count=candidate_count/128u;
            uint long_lanes=candidate_count-short_count*128u;
            uint long_candidates=long_lanes*(short_count+1u);
            [loop] for(uint ordinal=0u;ordinal<candidate_count;++ordinal) {
                uint candidate=ordinal;
                if(candidate_count>128u) {
                    uint lane,step;
                    if(ordinal<long_candidates) {
                        lane=ordinal/(short_count+1u);
                        step=ordinal-lane*(short_count+1u);
                    } else {
                        uint remainder=ordinal-long_candidates;
                        lane=long_lanes+remainder/short_count;
                        step=remainder-(lane-long_lanes)*short_count;
                    }
                    candidate=lane+step*128u;
                }
                uint body_leaf=candidate/collider_leaf_info.count;
                uint collider_leaf=candidate-
                    body_leaf*collider_leaf_info.count;
                BvhNode body_node=mesh_bvh_nodes[mesh_bvh_leaves[
                    body_leaf_info.offset+body_leaf]];
                BvhNode collider_node=mesh_bvh_nodes[mesh_bvh_leaves[
                    collider_leaf_info.offset+collider_leaf]];
                float3 body_minimum,body_maximum;
                float3 collider_minimum,collider_maximum;
                transformed_node_motion_bounds(
                    previous_states[body],states[body],body_node,margin,
                    body_minimum,body_maximum);
                transformed_node_motion_bounds(
                    previous_states[collider],states[collider],collider_node,
                    0.0f,collider_minimum,collider_maximum);
                if(!bounds_overlap(body_minimum,body_maximum,
                                   collider_minimum,collider_maximum))
                    continue;
                collide_swept(body,collider,body_node.first_triangle,
                              body_node.triangle_count,
                              collider_node.first_triangle,
                              collider_node.triangle_count,true,false,manifold);
            }
        }
    }
    manifolds[pair]=manifold;
}

[numthreads(64,1,1)]
void rigid_generate_warp(uint3 dispatch_id : SV_DispatchThreadID) {
    uint pair=dispatch_id.x;
    uint pair_count=step_body_count*step_body_count;
    if(pair>=pair_count) return;
    uint body=pair/step_body_count;
    uint collider=pair-body*step_body_count;
    ContactManifold manifold=empty_manifold();
    RigidParameters body_parameters=parameters[body];
    RigidParameters collider_parameters=parameters[collider];
    bool active=body!=collider && body_parameters.inverse_mass>0.0f;
    if(active && collider_parameters.inverse_mass>0.0f && collider<body)
        active=false;
    if(active && suppress_pair(body,collider)) active=false;
    MeshInfo body_mesh=mesh_infos[body_parameters.mesh_index];
    MeshInfo collider_mesh=mesh_infos[collider_parameters.mesh_index];
    active=active && body_mesh.alive!=0u && collider_mesh.alive!=0u &&
           body_mesh.index_count>=3u && collider_mesh.index_count>=3u;
    float margin=body_parameters.collision_margin+
                 collider_parameters.collision_margin;
    if(active) {
        float3 body_minimum,body_maximum,collider_minimum,collider_maximum;
        transformed_motion_bounds(previous_states[body],states[body],
                                  body_mesh,margin,body_minimum,body_maximum);
        transformed_motion_bounds(previous_states[collider],states[collider],
                                  collider_mesh,0.0f,collider_minimum,
                                  collider_maximum);
        active=bounds_overlap(body_minimum,body_maximum,
                              collider_minimum,collider_maximum);
    }
    if(active) {
        HingeContactFrame body_hinge=hinge_frames[body];
        HingeContactFrame collider_hinge=hinge_frames[collider];
        bool guided=guided_static_pair(body_hinge,collider_hinge);
        bool swept=guided || requires_swept_pair(body,collider,margin);
        bool handled_face=!swept &&
            body_mesh.solid_plane_count!=0u &&
            collider_mesh.solid_plane_count!=0u &&
            body_mesh.index_count<=96u &&
            collider_mesh.index_count<=96u &&
            convex_face_manifold(body,collider,margin,manifold);
        if(handled_face) manifold.color=manifold_face_patch;
        if(!handled_face) {
            collide_current(body,collider,0u,body_mesh.index_count/3u,
                            0u,collider_mesh.index_count/3u,manifold);
            if(swept)
                collide_swept(body,collider,0u,body_mesh.index_count/3u,
                              0u,collider_mesh.index_count/3u,guided,false,
                              manifold);
        }
    }
    manifolds[pair]=manifold;
}

// One stable compact list is reused by coloring, solving and cache writes.
RWStructuredBuffer<uint> active_pairs : register(u19);

void solve_compact_pair(uint solve_pair,uint solve_pass,bool first_fit) {
            ContactManifold manifold=manifolds[solve_pair];
            if(manifold.count==0u) return;
            uint body=solve_pair/step_body_count;
            uint collider=solve_pair-body*step_body_count;
            HingeContactFrame body_hinge=hinge_frames[body];
            HingeContactFrame collider_hinge=hinge_frames[collider];
            bool prepared=first_fit && manifold.contacts[0].persistent!=0u &&
                compounds[body].eligible==0u &&
                compounds[collider].eligible==0u &&
                !guided_static_pair(body_hinge,collider_hinge);
            if(prepared) {
                solve_prepared_pair(solve_pair,manifold,solve_pass==0u,
                                    manifold.event_offset);
                manifolds[solve_pair]=manifold;
                return;
            }
            if(solve_pass>8u) return;
            if(solve_pass==0u) {
                [loop] for(uint warm_index=0u;
                           warm_index<manifold.count;++warm_index) {
                    ContactRecord warm=manifold.contacts[warm_index];
                    warm_start_persistent_contact(body,collider,
                        body_hinge,collider_hinge,warm,
                        manifold.event_offset+warm_index);
                    store_manifold_contact(manifold,warm_index,warm);
                }
                manifolds[solve_pair]=manifold;
                return;
            }
            uint correction_count=0u;
            bool guided=guided_static_pair(body_hinge,collider_hinge);
            bool translational_projection=
                manifold.contacts[0].persistent!=0u && !guided;
            if((solve_pass==1u || translational_projection) && !guided &&
               body_hinge.fixed_member==0u &&
               collider_hinge.fixed_member==0u) {
                if(translational_projection) {
                    correction_count=manifold.count;
                } else {
                    [loop] for(uint count_contact=0u;
                               count_contact<manifold.count;++count_contact)
                        correction_count+=
                            manifold.contacts[count_contact].penetration>0.0f ? 1u : 0u;
                }
                [loop] for(uint correction_index=0u;
                           correction_index<manifold.count;
                           ++correction_index) {
                    ContactRecord correction=manifold.contacts[correction_index];
                    float penetration=correction.penetration;
                    if(correction.persistent!=0u) {
                        RigidBodyState correction_a=states[body];
                        RigidBodyState correction_b=states[collider];
                        penetration-=dot(
                            (load3(correction_a.position)-load3(correction_b.position))-
                                load3(manifold.initial_relative_position),
                            load3(correction.normal));
                    }
                    if(correction_count>0u && penetration>0.0f) {
                        bool hinged=body_hinge.fixed!=0u ||
                                    collider_hinge.fixed!=0u;
                        float projected=(min(penetration,
                            hinged ? 0.001f : penetration)+
                            (correction.persistent!=0u ? 0.0f : 1.0e-5f))*
                            rcp(max(float(correction_count),1.0f));
                        correct_contact_position(body,collider,body_hinge,
                            collider_hinge,correction,projected);
                    }
                }
            }
            [loop] for(uint solve_contact_index=0u;
                       solve_contact_index<manifold.count;
                       ++solve_contact_index) {
                ContactRecord record=manifold.contacts[solve_contact_index];
                if(record.persistent!=0u) {
                    resolve_persistent_contact(body,collider,body_hinge,
                        collider_hinge,record,
                        manifold.event_offset+solve_contact_index);
                    store_manifold_contact(
                        manifold,solve_contact_index,record);
                } else {
                    resolve_contact(body,collider,body_hinge,collider_hinge,
                        record,manifold.event_offset+solve_contact_index);
                }
            }
            manifolds[solve_pair]=manifold;
}

groupshared uint color_sizes[25];
groupshared uint color_offsets[25];

[numthreads(1,1,1)]
void rigid_color(uint3 unused : SV_DispatchThreadID) {
    uint pair_count=step_body_count*step_body_count;
    if(step_substep_index==0) { counters[0]=0; counters[1]=0; }
    uint active_count=0u;
    uint cursor=0u;
    bool face_contacts=false;
    uint candidate_count=counters[6];
    [loop] for(uint event_candidate=0u;event_candidate<candidate_count;++event_candidate) {
        uint event_pair=active_pairs[event_candidate];
        uint contact_count=manifolds[event_pair].count;
        if(contact_count==0u) continue;
        uint event_body=event_pair/step_body_count;
        uint event_collider=event_pair-event_body*step_body_count;
        if(color_owners[event_body]==color_owners[event_collider]) {
            manifolds[event_pair]=empty_manifold();
            continue;
        }
        active_pairs[active_count++]=event_pair;
        pair_colors[event_pair]=0xffu;
        uint flags=(manifolds[event_pair].color&manifold_face_patch)|
            (hinge_frames[event_body].fixed_member!=0u ? manifold_body_fixed_member : 0u)|
            (hinge_frames[event_collider].fixed_member!=0u ?
                manifold_collider_fixed_member : 0u);
        manifolds[event_pair].color=flags;
        manifolds[event_pair].event_offset=cursor;
        if((flags&manifold_face_patch)!=0u) {
            face_contacts=true;
        }
        cursor+=contact_count;
    }
    if(step_collect_contacts!=0u && cursor>0u) {
        counters[0]=min(cursor,step_event_capacity);
        counters[1]=cursor>step_event_capacity ? 1u : 0u;
    }
    bool has_constraints=false;
    [loop] for(uint live_constraint=0u;
               live_constraint<step_constraint_capacity;++live_constraint)
        has_constraints=has_constraints || constraints[live_constraint].alive!=0u;
    uint color_round_count=step_body_count<=1u ? 1u :
        (step_body_count<9u ?
            min(24u,step_body_count*(step_body_count-1u)/2u) : 24u);
    uint used_colors=0u;
    bool color_overflow=false;
    bool ordinary_rigid_stack=step_body_count>=32u && !has_constraints;
    bool first_fit=ordinary_rigid_stack;
    if(first_fit) {
        first_fit=false;
        [loop] for(uint active_index=0u;active_index<active_count;++active_index) {
            uint face_pair=active_pairs[active_index];
            if((manifolds[face_pair].color&manifold_face_patch)!=0u &&
               manifolds[face_pair].contacts[0].persistent!=0u) {
                first_fit=true;
                break;
            }
        }
    }
    if(first_fit) {
        [loop] for(uint clear_owner=0u;clear_owner<step_body_count;++clear_owner)
            color_owners[clear_owner]=0u;
        uint palette=(1u<<color_round_count)-1u;
        [loop] for(uint active_index=0u;active_index<active_count;++active_index) {
            uint fit_pair=active_pairs[active_index];
            uint first=fit_pair/step_body_count;
            uint second=fit_pair-first*step_body_count;
            uint first_owner=compounds[first].eligible!=0u ?
                compounds[first].root : first;
            uint second_owner=compounds[second].eligible!=0u ?
                compounds[second].root : second;
            bool dynamic_second=parameters[second].motion==2u;
            uint available=palette&~(color_owners[first_owner]|
                (dynamic_second ? color_owners[second_owner] : 0u));
            if(available==0u) {
                color_overflow=true;
                continue;
            }
            uint assigned=firstbitlow(available);
            pair_colors[fit_pair]=assigned;
            color_owners[first_owner]|=1u<<assigned;
            if(dynamic_second) color_owners[second_owner]|=1u<<assigned;
            used_colors=max(used_colors,assigned+1u);
        }
    } else {
        [loop] for(uint color=0u;color<color_round_count;++color) {
            [loop] for(uint clear_owner=0u;
                       clear_owner<step_body_count;++clear_owner)
                color_owners[clear_owner]=0xffffffffu;
            [loop] for(uint active_index=0u;active_index<active_count;++active_index) {
                uint owner_pair=active_pairs[active_index];
                if(pair_colors[owner_pair]!=0xffu)
                    continue;
                uint first=owner_pair/step_body_count;
                uint second=owner_pair-first*step_body_count;
                uint first_owner=compounds[first].eligible!=0u ?
                    compounds[first].root : first;
                uint second_owner=compounds[second].eligible!=0u ?
                    compounds[second].root : second;
                uint priority=owner_pair*2654435761u+1013904223u;
                color_owners[first_owner]=min(color_owners[first_owner],priority);
                if(parameters[second].motion==2u)
                    color_owners[second_owner]=min(
                        color_owners[second_owner],priority);
            }
            [loop] for(uint assign_index=0u;assign_index<active_count;++assign_index) {
                uint assign_pair=active_pairs[assign_index];
                if(pair_colors[assign_pair]!=0xffu)
                    continue;
                uint first=assign_pair/step_body_count;
                uint second=assign_pair-first*step_body_count;
                uint first_owner=compounds[first].eligible!=0u ?
                    compounds[first].root : first;
                uint second_owner=compounds[second].eligible!=0u ?
                    compounds[second].root : second;
                bool dynamic_second=parameters[second].motion==2u;
                uint priority=assign_pair*2654435761u+1013904223u;
                if(color_owners[first_owner]==priority &&
                   (!dynamic_second || color_owners[second_owner]==priority)) {
                    pair_colors[assign_pair]=color;
                    used_colors=max(used_colors,color+1u);
                } else if(color+1u==color_round_count) {
                    color_overflow=true;
                }
            }
        }
    }
    uint rigid_contact_iterations=face_contacts ? 32u : 8u;
    if(first_fit) {
        [loop] for(uint active_index=0u;active_index<active_count;++active_index) {
            uint initial_face_pair=active_pairs[active_index];
            if((manifolds[initial_face_pair].color&manifold_face_patch)!=0u &&
               manifolds[initial_face_pair].contacts[0].persistent!=0u &&
               manifolds[initial_face_pair].cached==0u) {
                rigid_contact_iterations=64u;
                break;
            }
        }
    }
    counters[2]=used_colors;
    counters[3]=color_overflow ? 1u : 0u;
    counters[4]=first_fit ? 1u : 0u;
    counters[5]=rigid_contact_iterations;

    // Stable color lists retain the ascending pair order, including overflow.
    counters[6]=active_count;
    // Counting sort visits the list twice instead of once per color. One
    // thread owns these shared counters, so insertion order stays exact.
    [unroll] for(uint list_color=0u;list_color<=24u;++list_color)
        color_sizes[list_color]=0u;
    [loop] for(uint list_index=0u;list_index<active_count;++list_index) {
        uint list_pair=active_pairs[list_index];
        uint assigned_color=min(pair_colors[list_pair],24u);
        ++color_sizes[assigned_color];
    }
    uint color_cursor=0u;
    [unroll] for(uint prefix_color=0u;prefix_color<=24u;++prefix_color) {
        counters[7u+prefix_color]=color_cursor;
        color_offsets[prefix_color]=color_cursor;
        color_cursor+=color_sizes[prefix_color];
    }
    [loop] for(uint scatter_index=0u;scatter_index<active_count;++scatter_index) {
        uint scatter_pair=active_pairs[scatter_index];
        uint scatter_color=min(pair_colors[scatter_pair],24u);
        active_pairs[pair_count+color_offsets[scatter_color]++]=scatter_pair;
    }
    counters[32]=color_cursor;
}

[numthreads(1,1,1)]
void rigid_finalize(uint3 unused : SV_DispatchThreadID) {
    uint iterations=prepare_constraints();
    bool fixed_contacts=false;
    [loop] for(uint fixed_index=0u;fixed_index<step_constraint_capacity;++fixed_index) {
        RigidConstraint fixed_joint=constraints[fixed_index];
        fixed_contacts=fixed_contacts || (fixed_joint.alive!=0u &&
            fixed_joint.enabled!=0u && fixed_joint.broken==0u &&
            fixed_joint.type==0u && constraint_geometry[fixed_index].valid!=0u);
    }
    uint contact_sweeps=fixed_contacts?8u:1u;
    [loop] for(uint iteration=0u;iteration<iterations*contact_sweeps;++iteration) {
        [loop] for(uint fixed_index_pair=0u;
                   fixed_contacts && fixed_index_pair<counters[6];++fixed_index_pair) {
            uint fixed_pair=active_pairs[fixed_index_pair];
            ContactManifold fixed_manifold=manifolds[fixed_pair];
            if(fixed_manifold.count==0u) continue;
            uint fixed_body=fixed_pair/step_body_count;
            uint fixed_collider=fixed_pair-fixed_body*step_body_count;
            if((fixed_manifold.color&(manifold_body_fixed_member|
                manifold_collider_fixed_member))==0u) continue;
            if(compounds[fixed_body].eligible!=0u ||
               compounds[fixed_collider].eligible!=0u) continue;
            HingeContactFrame fixed_body_hinge=hinge_frames[fixed_body];
            HingeContactFrame fixed_collider_hinge=hinge_frames[fixed_collider];
            [loop] for(uint fixed_contact=0u;
                       fixed_contact<fixed_manifold.count;++fixed_contact)
                resolve_contact(fixed_body,fixed_collider,
                    fixed_body_hinge,fixed_collider_hinge,
                    fixed_manifold.contacts[fixed_contact],
                    fixed_manifold.event_offset+fixed_contact);
        }
        [loop] for(uint joint=0u;joint<step_constraint_capacity;++joint)
            solve_prepared_joint(joint,iteration,contact_sweeps);
    }
    project_hinge_anchors();
    project_fixed_ground_contacts();
    [loop] for(uint body_index=0u;body_index<step_body_count;++body_index) {
        RigidParameters body=parameters[body_index];
        if(body.motion!=2u) continue;
        RigidBodyState state=states[body_index];
        state.linear_velocity=store3(limit_length(
            load3(state.linear_velocity),body.maximum_linear_speed));
        state.angular_velocity=store3(limit_length(
            load3(state.angular_velocity),body.maximum_angular_speed));
        states[body_index]=state;
    }
    finalize_guided_bodies();
    [loop] for(uint save_index=0u;save_index<counters[6];++save_index) {
        uint save_pair=active_pairs[save_index];
        ContactManifold saved=manifolds[save_pair];
        if(saved.count!=0u && saved.contacts[0].persistent!=0u)
            save_persistent_pair(save_pair,saved);
    }
}

[numthreads(64,1,1)]
void rigid_clear(uint3 dispatch_id : SV_DispatchThreadID) {
    uint index=dispatch_id.x;
    if(index>=step_body_count) return;
    forces[index]=store3(float3(0,0,0));
    torques[index]=store3(float3(0,0,0));
}

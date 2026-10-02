// SPDX-License-Identifier: MIT
// Soft-body and smoke coupling kernels.

__global__ void smoke_soft_body_wind(
    Vec3 *positions, Vec3 *velocities, const float *inverse_masses,
    std::uint32_t node_count, SmokeOptions smoke,
    const Vec3 *smoke_positions, const Vec3 *smoke_velocities,
    const std::uint64_t *keys, const std::uint32_t *indices,
    SmokeGridField grid,
    float drag, float maximum_acceleration, float dt,
    float maximum_speed) {
    const auto node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= node_count || inverse_masses[node] == 0.0F) return;
    velocities[node] = clamp_length(add(velocities[node], smoke_wind_delta(
        positions[node], velocities[node], smoke_positions, smoke_velocities,
        keys, indices, smoke, grid, drag, maximum_acceleration, dt)), maximum_speed);
}

__global__ void smoke_soft_body_contact(
    Vec3 *positions, const Vec3 *previous_positions,
    Vec3 *velocities, const float *ages,
    std::uint32_t count, float lifetime,
    const Vec3 *surface, const SoftBodySurfaceBinding *bindings,
    const std::uint32_t *indices,std::uint32_t index_count,
    const Vec3 *node_velocities,
    const Vec3 *minimum, const Vec3 *maximum, float clearance,
    SmokeGridField grid) {
    const auto particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= count || ages[particle] >= lifetime) return;
    const Vec3 point = positions[particle];
    const Vec3 before = previous_positions[particle];
    const Vec3 low = *minimum, high = *maximum;
    if(fmaxf(point.x,before.x)<low.x-clearance||
       fminf(point.x,before.x)>high.x+clearance||
       fmaxf(point.y,before.y)<low.y-clearance||
       fminf(point.y,before.y)>high.y+clearance||
       fmaxf(point.z,before.z)<low.z-clearance||
       fminf(point.z,before.z)>high.z+clearance)return;
    float nearest2=clearance*clearance,earliest=2.0F;
    std::uint32_t best=index_count;
    Vec3 nearest{},face_normal{},weights{};
    const Vec3 path=subtract(point,before);
    for(std::uint32_t base=0;base<index_count;base+=3U){
        const Vec3 a=surface[indices[base]],b=surface[indices[base+1U]];
        const Vec3 c=surface[indices[base+2U]];
        const Vec3 raw_normal=cross(subtract(b,a),subtract(c,a));
        if(length_squared(raw_normal)<1.0e-12F)continue;
        const float side_before=dot(subtract(before,a),raw_normal);
        const float side_after=dot(subtract(point,a),raw_normal);
        if(side_before*side_after<0.0F){
            const float fraction=side_before/(side_before-side_after);
            if(fraction<earliest){
                const Vec3 hit=add(before,multiply(path,fraction));
                Vec3 hit_weights{};
                const Vec3 on_face=fluid_closest_triangle_barycentric(
                    hit,a,b,c,hit_weights);
                if(length_squared(subtract(hit,on_face))<1.0e-8F){
                    earliest=fraction;best=base;nearest=on_face;
                    face_normal=normalized_or(raw_normal,{1,0,0});
                    weights=hit_weights;
                }
            }
        }
        if(earliest<=1.0F)continue;
        Vec3 barycentric{};
        const Vec3 candidate=fluid_closest_triangle_barycentric(
            point,a,b,c,barycentric);
        const float distance2=length_squared(subtract(point,candidate));
        if(distance2<nearest2){
            nearest2=distance2;best=base;nearest=candidate;
            face_normal=normalized_or(raw_normal,{1,0,0});weights=barycentric;
        }
    }
    if(best==index_count)return;
    float side=dot(subtract(before,nearest),face_normal);
    if(fabsf(side)<1.0e-5F)side=dot(subtract(point,nearest),face_normal);
    const Vec3 normal=side>=0.0F?face_normal:multiply(face_normal,-1.0F);
    const Vec3 trace_position=add(nearest,multiply(normal,
        grid.n!=0U?1.5F*grid.spacing:clearance));
    positions[particle]=trace_position;
    const Vec3 va=smoke_soft_surface_velocity(
        bindings[indices[best]],node_velocities);
    const Vec3 vb=smoke_soft_surface_velocity(
        bindings[indices[best+1U]],node_velocities);
    const Vec3 vc=smoke_soft_surface_velocity(
        bindings[indices[best+2U]],node_velocities);
    const Vec3 body_velocity=add(multiply(va,weights.x),
        add(multiply(vb,weights.y),multiply(vc,weights.z)));
    Vec3 relative=subtract(grid.n!=0U&&smoke_grid_contains(trace_position,grid)?
        smoke_grid_sample_velocity(trace_position,grid):velocities[particle],
        body_velocity);
    const float source_speed=fmaxf(vector_length(relative),vector_length(
        subtract(velocities[particle],body_velocity)));
    relative=subtract(relative,multiply(normal,
        fminf(0.0F,dot(relative,normal))));
    const float tangent_speed=vector_length(relative);
    if(grid.n!=0U&&tangent_speed>1.0e-6F&&tangent_speed<source_speed)
        relative=multiply(relative,source_speed/tangent_speed);
    velocities[particle]=add(body_velocity,relative);
}

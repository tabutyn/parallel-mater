// SPDX-License-Identifier: MIT
// Staggered-grid incompressible air used by the hybrid smoke solver.
// Particles carry visible density and heat; this field carries momentum.
struct SmokePressureLevel {
    std::uint32_t n{};
    std::uint32_t height{};
    std::uint32_t cell_count{};
    float *pressure[2]{};
    float *rhs{};
    float *residual{};
    float *open[3]{};
    ~SmokePressureLevel() {
        for (auto &field : pressure) release_managed(field);
        release_managed(rhs);
        release_managed(residual);
        for (auto &field : open) release_managed(field);
    }
};

struct SmokeGridStorage {
    std::uint32_t resolution{};
    std::uint32_t height{};
    std::uint32_t cell_count{};
    std::uint32_t face_count[3]{};
    Vec3 minimum{};
    float spacing{};
    Vec3 *velocity{};
    float *face_velocity[3][2]{};
    float *face_reverse[3]{};
    float *face_open[3]{};
    float *face_wall_velocity[3]{};
    Vec3 *face_normal[3]{};
    unsigned int *face_nearest_triangle[3]{};
    float *density{};
    float *temperature{};
    unsigned long long *density_accumulator{};
    unsigned long long *temperature_accumulator{};
    float *pressure[2]{};
    float *divergence{};
    float *residual{};
    Vec3 *vorticity{};
    float *strain{};
    Vec3 *subgrid_force{};
    unsigned int *solid{};
    float *rhs_max{};
    float *residual_max{};
    unsigned int *pressure_converged{};
    // Static triangle metadata is identical between steps.  Apertures are
    // still rebuilt each frame; this only avoids repeating the expensive
    // closest-triangle metadata pass when no coupled surface can move.
    bool static_metadata_valid{};
    SmokePressureLevel coarse[3]{};
    ~SmokeGridStorage() {
        release_managed(velocity);
        for (int axis = 0; axis < 3; ++axis) {
            for (auto &field : face_velocity[axis]) release_managed(field);
            release_managed(face_reverse[axis]);
            release_managed(face_open[axis]);
            release_managed(face_wall_velocity[axis]);
            release_managed(face_normal[axis]);
            release_managed(face_nearest_triangle[axis]);
        }
        release_managed(density);
        release_managed(temperature);
        release_managed(density_accumulator);
        release_managed(temperature_accumulator);
        for (auto &field : pressure) release_managed(field);
        release_managed(divergence);
        release_managed(residual);
        release_managed(vorticity);
        release_managed(strain);
        release_managed(subgrid_force);
        release_managed(solid);
        release_managed(rhs_max);
        release_managed(residual_max);
        release_managed(pressure_converged);
    }
};

struct SmokeGridField {
    std::uint32_t n{};
    std::uint32_t height{};
    Vec3 minimum{};
    float spacing{};
    const Vec3 *velocity{};
    const float *density{};
    const float *pressure{};
    const float *face[3]{};
    const Vec3 *vorticity{};
    const float *strain{};
};

__host__ __device__ std::uint32_t smoke_grid_index(
    int x, int y, int z, int n, int height) {
    return std::uint32_t(x + n * (y + height * z));
}

__host__ __device__ std::uint32_t smoke_grid_face_count(
    int axis, int n, int height) {
    if (axis == 0) return std::uint32_t((n + 1) * height * n);
    if (axis == 1) return std::uint32_t(n * (height + 1) * n);
    return std::uint32_t(n * height * (n + 1));
}

__host__ __device__ std::uint32_t smoke_grid_face_index(
    int axis, int x, int y, int z, int n, int height) {
    if (axis == 0) return std::uint32_t(x + (n + 1) * (y + height * z));
    if (axis == 1) return std::uint32_t(x + n * (y + (height + 1) * z));
    return std::uint32_t(x + n * (y + height * z));
}

__host__ __device__ void smoke_grid_face_coordinates(
    int axis, std::uint32_t face, int n, int height,
    int &x, int &y, int &z) {
    if (axis == 0) {
        x = int(face % std::uint32_t(n + 1));
        y = int(face / std::uint32_t(n + 1) % std::uint32_t(height));
        z = int(face / std::uint32_t((n + 1) * height));
    } else if (axis == 1) {
        x = int(face % std::uint32_t(n));
        y = int(face / std::uint32_t(n) % std::uint32_t(height + 1));
        z = int(face / std::uint32_t(n * (height + 1)));
    } else {
        x = int(face % std::uint32_t(n));
        y = int(face / std::uint32_t(n) % std::uint32_t(height));
        z = int(face / std::uint32_t(n * height));
    }
}

__host__ __device__ Vec3 smoke_grid_face_position(
    int axis, int x, int y, int z, SmokeGridField grid) {
    Vec3 offset{float(x) + 0.5F, float(y) + 0.5F, float(z) + 0.5F};
    if (axis == 0) offset.x = float(x);
    if (axis == 1) offset.y = float(y);
    if (axis == 2) offset.z = float(z);
    return add(grid.minimum, multiply(offset, grid.spacing));
}

__device__ bool smoke_grid_contains(Vec3 point, SmokeGridField grid) {
    const Vec3 p = multiply(subtract(point, grid.minimum), 1.0F / grid.spacing);
    return p.x >= 0.0F && p.y >= 0.0F && p.z >= 0.0F &&
        p.x < float(grid.n) && p.y < float(grid.height) && p.z < float(grid.n);
}

__device__ float smoke_grid_sample_scalar(const float *values, Vec3 point,
    SmokeGridField grid) {
    if (!values) return 0.0F;
    const Vec3 p = subtract(multiply(subtract(point, grid.minimum),
                                    1.0F / grid.spacing),
                            {0.5F, 0.5F, 0.5F});
    const int x0 = int(floorf(p.x)), y0 = int(floorf(p.y));
    const int z0 = int(floorf(p.z));
    const float fx = p.x-float(x0), fy = p.y-float(y0), fz = p.z-float(z0);
    float result{};
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const int x = max(0, min(int(grid.n)-1, x0+dx));
                const int y = max(0, min(int(grid.height)-1, y0+dy));
                const int z = max(0, min(int(grid.n)-1, z0+dz));
                const float weight = (dx?fx:1.0F-fx) *
                    (dy?fy:1.0F-fy) * (dz?fz:1.0F-fz);
                result += values[smoke_grid_index(x,y,z,int(grid.n),
                    int(grid.height))] * weight;
            }
    return result;
}

__device__ Vec3 smoke_grid_sample_vector(const Vec3 *values, Vec3 point,
    SmokeGridField grid) {
    if (!values) return {};
    const Vec3 p = subtract(multiply(subtract(point, grid.minimum),
                                    1.0F / grid.spacing),
                            {0.5F, 0.5F, 0.5F});
    const int x0 = int(floorf(p.x)), y0 = int(floorf(p.y));
    const int z0 = int(floorf(p.z));
    const float fx = p.x-float(x0), fy = p.y-float(y0), fz = p.z-float(z0);
    Vec3 result{};
    for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const int x = max(0, min(int(grid.n)-1, x0+dx));
                const int y = max(0, min(int(grid.height)-1, y0+dy));
                const int z = max(0, min(int(grid.n)-1, z0+dz));
                const float weight = (dx?fx:1.0F-fx) *
                    (dy?fy:1.0F-fy) * (dz?fz:1.0F-fz);
                result = add(result, multiply(values[smoke_grid_index(
                    x,y,z,int(grid.n),int(grid.height))], weight));
            }
    return result;
}

__device__ float smoke_grid_sample_face_component(
    int axis, const float *values, Vec3 point, SmokeGridField grid) {
    if (!values) return 0.0F;
    Vec3 p = multiply(subtract(point, grid.minimum), 1.0F / grid.spacing);
    int sx = int(grid.n), sy = int(grid.height), sz = int(grid.n);
    if (axis == 0) { p.y -= 0.5F; p.z -= 0.5F; ++sx; }
    if (axis == 1) { p.x -= 0.5F; p.z -= 0.5F; ++sy; }
    if (axis == 2) { p.x -= 0.5F; p.y -= 0.5F; ++sz; }
    const int x0=int(floorf(p.x)), y0=int(floorf(p.y)), z0=int(floorf(p.z));
    const float fx=p.x-float(x0), fy=p.y-float(y0), fz=p.z-float(z0);
    float result{};
    for (int dz=0;dz<2;++dz) for (int dy=0;dy<2;++dy)
        for (int dx=0;dx<2;++dx) {
            const int x=max(0,min(sx-1,x0+dx));
            const int y=max(0,min(sy-1,y0+dy));
            const int z=max(0,min(sz-1,z0+dz));
            const float weight=(dx?fx:1.0F-fx)*(dy?fy:1.0F-fy)*
                               (dz?fz:1.0F-fz);
            result += values[smoke_grid_face_index(axis,x,y,z,int(grid.n),
                                                   int(grid.height))]*weight;
        }
    return result;
}

__device__ Vec3 smoke_grid_sample_velocity(Vec3 point, SmokeGridField grid) {
    if (grid.face[0] && grid.face[1] && grid.face[2])
        return {smoke_grid_sample_face_component(0,grid.face[0],point,grid),
                smoke_grid_sample_face_component(1,grid.face[1],point,grid),
                smoke_grid_sample_face_component(2,grid.face[2],point,grid)};
    return smoke_grid_sample_vector(grid.velocity, point, grid);
}

__device__ void smoke_grid_sample(Vec3 point, SmokeGridField grid,
    Vec3 &velocity, float &density) {
    velocity = smoke_grid_sample_velocity(point, grid);
    density = smoke_grid_sample_scalar(grid.density, point, grid);
}

__device__ float smoke_grid_sample_pressure(Vec3 point, SmokeGridField grid) {
    return smoke_grid_sample_scalar(grid.pressure, point, grid);
}

__global__ void smoke_grid_initialize_cells(Vec3 *velocity, float *density,
    float *temperature, float *pressure, float *divergence,
    Vec3 *vorticity, float *strain, Vec3 *subgrid_force,
    unsigned int *solid, std::uint32_t count, Vec3 wind) {
    const auto cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=count)return;
    velocity[cell]=wind; density[cell]=temperature[cell]=pressure[cell]=0.0F;
    divergence[cell]=strain[cell]=0.0F; vorticity[cell]=subgrid_force[cell]={};
    solid[cell]=0U;
}

__global__ void smoke_grid_initialize_faces(float *values, float *reverse,
    float *open, float *wall, Vec3 *normal, unsigned int *nearest_triangle,
    std::uint32_t count, float wind) {
    const auto face=blockIdx.x*blockDim.x+threadIdx.x;
    if(face>=count)return;
    values[face]=reverse[face]=wind; open[face]=1.0F; wall[face]=0.0F;
    normal[face]={};nearest_triangle[face]=0x7fffffffU;
}

__global__ void smoke_grid_clear_obstacles(unsigned int *solid,
    std::uint32_t cell_count, float *open_x, float *open_y, float *open_z,
    float *wall_x, float *wall_y, float *wall_z,
    std::uint32_t x_count, std::uint32_t y_count, std::uint32_t z_count) {
    const auto index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<cell_count)solid[index]=0U;
    // Metadata on open faces is ignored.  Avoid rewriting the much larger
    // normal/id arrays; the triangle-resolution pass overwrites every cut
    // face that can be consumed this frame.
    if(index<x_count){open_x[index]=1.0F;wall_x[index]=0.0F;}
    if(index<y_count){open_y[index]=1.0F;wall_y[index]=0.0F;}
    if(index<z_count){open_z[index]=1.0F;wall_z[index]=0.0F;}
}

__device__ void smoke_grid_atomic_min_positive(float *address,float value) {
    atomicMin(reinterpret_cast<unsigned int *>(address),__float_as_uint(value));
}

__device__ void smoke_grid_raster_triangle(Vec3 a, Vec3 b, Vec3 c,
    Vec3 va, Vec3 vb, Vec3 vc, SmokeGridField grid, unsigned int *solid,
    float *const open[3], float *const wall[3], Vec3 *const face_normal[3],
    unsigned int *const nearest_triangle[3], unsigned int triangle_id,
    bool resolve_metadata) {
    const float radius=0.55F*grid.spacing;
    const float inverse=1.0F/grid.spacing;
    const int n=int(grid.n), h=int(grid.height);
    const Vec3 triangle_normal=normalized_or(
        cross(subtract(b,a),subtract(c,a)),{0,1,0});
    if(!resolve_metadata){
        const int cx0=max(0,int(floorf((fminf(a.x,fminf(b.x,c.x))-radius-
            grid.minimum.x)*inverse)));
        const int cy0=max(0,int(floorf((fminf(a.y,fminf(b.y,c.y))-radius-
            grid.minimum.y)*inverse)));
        const int cz0=max(0,int(floorf((fminf(a.z,fminf(b.z,c.z))-radius-
            grid.minimum.z)*inverse)));
        const int cx1=min(n-1,int(floorf((fmaxf(a.x,fmaxf(b.x,c.x))+radius-
            grid.minimum.x)*inverse)));
        const int cy1=min(h-1,int(floorf((fmaxf(a.y,fmaxf(b.y,c.y))+radius-
            grid.minimum.y)*inverse)));
        const int cz1=min(n-1,int(floorf((fmaxf(a.z,fmaxf(b.z,c.z))+radius-
            grid.minimum.z)*inverse)));
        for(int z=cz0;z<=cz1;++z)for(int y=cy0;y<=cy1;++y)
            for(int x=cx0;x<=cx1;++x){
                const Vec3 point=add(grid.minimum,multiply(
                    {float(x)+0.5F,float(y)+0.5F,float(z)+0.5F},grid.spacing));
                Vec3 weights{};
                const Vec3 nearest=fluid_closest_triangle_barycentric(
                    point,a,b,c,weights);
                if(length_squared(subtract(point,nearest))<=radius*radius)
                    atomicExch(&solid[smoke_grid_index(x,y,z,n,h)],1U);
            }
    }
    for(int axis=0;axis<3;++axis){
        Vec3 offset{0.5F,0.5F,0.5F};
        if(axis==0)offset.x=0.0F;
        if(axis==1)offset.y=0.0F;
        if(axis==2)offset.z=0.0F;
        const int sx=n+(axis==0), sy=h+(axis==1), sz=n+(axis==2);
        const int x0=max(0,int(floorf((fminf(a.x,fminf(b.x,c.x))-radius-
            grid.minimum.x)*inverse-offset.x)));
        const int y0=max(0,int(floorf((fminf(a.y,fminf(b.y,c.y))-radius-
            grid.minimum.y)*inverse-offset.y)));
        const int z0=max(0,int(floorf((fminf(a.z,fminf(b.z,c.z))-radius-
            grid.minimum.z)*inverse-offset.z)));
        const int x1=min(sx-1,int(floorf((fmaxf(a.x,fmaxf(b.x,c.x))+radius-
            grid.minimum.x)*inverse-offset.x)));
        const int y1=min(sy-1,int(floorf((fmaxf(a.y,fmaxf(b.y,c.y))+radius-
            grid.minimum.y)*inverse-offset.y)));
        const int z1=min(sz-1,int(floorf((fmaxf(a.z,fmaxf(b.z,c.z))+radius-
            grid.minimum.z)*inverse-offset.z)));
        for(int z=z0;z<=z1;++z)for(int y=y0;y<=y1;++y)
            for(int x=x0;x<=x1;++x){
                const Vec3 point=add(grid.minimum,multiply(
                    {float(x)+offset.x,float(y)+offset.y,float(z)+offset.z},
                    grid.spacing));
                Vec3 weights{};
                const Vec3 nearest=fluid_closest_triangle_barycentric(
                    point,a,b,c,weights);
                const float distance=vector_length(subtract(point,nearest));
                if(distance>radius)continue;
                const float aperture=clamp_scalar(distance/radius,0.0F,1.0F);
                const auto face=smoke_grid_face_index(axis,x,y,z,n,h);
                if(!resolve_metadata)
                    smoke_grid_atomic_min_positive(&open[axis][face],aperture);
                else if(aperture<=open[axis][face]+1.0e-7F){
                    const Vec3 velocity=add(multiply(va,weights.x),
                        add(multiply(vb,weights.y),multiply(vc,weights.z)));
                    wall[axis][face]=axis==0?velocity.x:
                        axis==1?velocity.y:velocity.z;
                    face_normal[axis][face]=triangle_normal;
                    nearest_triangle[axis][face]=triangle_id;
                }
            }
    }
}

__global__ void smoke_grid_raster_rigid(TriangleMeshResource mesh,
    const RigidBodyState *states, std::uint32_t body, SmokeGridField grid,
    unsigned int *solid, float *open_x,float *open_y,float *open_z,
    float *wall_x,float *wall_y,float *wall_z,
    Vec3 *normal_x,Vec3 *normal_y,Vec3 *normal_z,
    unsigned int *triangle_x,unsigned int *triangle_y,
    unsigned int *triangle_z,bool resolve_metadata) {
    const auto triangle=blockIdx.x*blockDim.x+threadIdx.x;
    if(triangle>=mesh.index_count/3U)return;
    const auto base=triangle*3U;
    const RigidBodyState state=states[body];
    const Vec3 a=transform_point(state,mesh.vertices[mesh.indices[base]]);
    const Vec3 b=transform_point(state,mesh.vertices[mesh.indices[base+1U]]);
    const Vec3 c=transform_point(state,mesh.vertices[mesh.indices[base+2U]]);
    const auto surface_velocity=[&](Vec3 point){return add(state.linear_velocity,
        cross(state.angular_velocity,subtract(point,state.position)));};
    float *open[3]={open_x,open_y,open_z};
    float *wall[3]={wall_x,wall_y,wall_z};
    Vec3 *normal[3]={normal_x,normal_y,normal_z};
    unsigned int *nearest[3]={triangle_x,triangle_y,triangle_z};
    smoke_grid_raster_triangle(a,b,c,surface_velocity(a),surface_velocity(b),
        surface_velocity(c),grid,solid,open,wall,normal,nearest,
        body*0x9e3779b9U+triangle,resolve_metadata);
}

__global__ void smoke_grid_raster_deformable(const Vec3 *positions,
    const Vec3 *velocities, const std::uint32_t *indices,
    std::uint32_t index_count, SmokeGridField grid, unsigned int *solid,
    float *open_x,float *open_y,float *open_z,
    float *wall_x,float *wall_y,float *wall_z,
    Vec3 *normal_x,Vec3 *normal_y,Vec3 *normal_z,
    unsigned int *triangle_x,unsigned int *triangle_y,
    unsigned int *triangle_z,bool resolve_metadata) {
    const auto triangle=blockIdx.x*blockDim.x+threadIdx.x;
    if(triangle>=index_count/3U)return;
    const auto base=triangle*3U;
    const auto ia=indices[base],ib=indices[base+1U],ic=indices[base+2U];
    float *open[3]={open_x,open_y,open_z};
    float *wall[3]={wall_x,wall_y,wall_z};
    Vec3 *normal[3]={normal_x,normal_y,normal_z};
    unsigned int *nearest[3]={triangle_x,triangle_y,triangle_z};
    smoke_grid_raster_triangle(positions[ia],positions[ib],positions[ic],
        velocities?velocities[ia]:Vec3{},velocities?velocities[ib]:Vec3{},
        velocities?velocities[ic]:Vec3{},grid,solid,open,wall,normal,nearest,
        triangle,resolve_metadata);
}

__device__ Vec3 smoke_soft_surface_velocity(SoftBodySurfaceBinding binding,
    const Vec3 *velocities) {
    Vec3 result{};
    for(int i=0;i<4;++i)result=add(result,multiply(
        velocities[binding.nodes[i]],binding.weights[i]));
    return result;
}

__global__ void smoke_grid_raster_soft(const Vec3 *positions,
    const SoftBodySurfaceBinding *bindings, const Vec3 *node_velocities,
    const std::uint32_t *indices, std::uint32_t index_count,
    SmokeGridField grid,unsigned int *solid,
    float *open_x,float *open_y,float *open_z,
    float *wall_x,float *wall_y,float *wall_z,
    Vec3 *normal_x,Vec3 *normal_y,Vec3 *normal_z,
    unsigned int *triangle_x,unsigned int *triangle_y,
    unsigned int *triangle_z,bool resolve_metadata) {
    const auto triangle=blockIdx.x*blockDim.x+threadIdx.x;
    if(triangle>=index_count/3U)return;
    const auto base=triangle*3U;
    const auto ia=indices[base],ib=indices[base+1U],ic=indices[base+2U];
    float *open[3]={open_x,open_y,open_z};
    float *wall[3]={wall_x,wall_y,wall_z};
    Vec3 *normal[3]={normal_x,normal_y,normal_z};
    unsigned int *nearest[3]={triangle_x,triangle_y,triangle_z};
    smoke_grid_raster_triangle(positions[ia],positions[ib],positions[ic],
        smoke_soft_surface_velocity(bindings[ia],node_velocities),
        smoke_soft_surface_velocity(bindings[ib],node_velocities),
        smoke_soft_surface_velocity(bindings[ic],node_velocities),
        grid,solid,open,wall,normal,nearest,triangle,resolve_metadata);
}

__global__ void smoke_grid_resolve_particle_fields(
    const unsigned long long *density_accumulator,
    const unsigned long long *temperature_accumulator,float *density,
    float *temperature,std::uint32_t count) {
    const auto cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=count)return;
    constexpr float inverse_scale=1.0F/16777216.0F;
    density[cell]=float(density_accumulator[cell])*inverse_scale;
    temperature[cell]=float(temperature_accumulator[cell])*inverse_scale;
}

__global__ void smoke_grid_splat_particles(const Vec3 *positions,
    const float *ages,const float *thermal_lift,std::uint32_t count,
    SmokeOptions options,SmokeGridField grid,
    unsigned long long *density_accumulator,
    unsigned long long *temperature_accumulator) {
    const auto particle=blockIdx.x*blockDim.x+threadIdx.x;
    if(particle>=count||ages[particle]>=options.lifetime)return;
    Vec3 g=subtract(multiply(subtract(positions[particle],grid.minimum),
                            1.0F/grid.spacing),{0.5F,0.5F,0.5F});
    const int center[3]={int(floorf(g.x+0.5F)),int(floorf(g.y+0.5F)),
                         int(floorf(g.z+0.5F))};
    const float d[3]={g.x-float(center[0]),g.y-float(center[1]),
                      g.z-float(center[2])};
    float weight[3][3]{};
    for(int axis=0;axis<3;++axis){
        weight[axis][0]=0.5F*(0.5F-d[axis])*(0.5F-d[axis]);
        weight[axis][1]=0.75F-d[axis]*d[axis];
        weight[axis][2]=0.5F*(0.5F+d[axis])*(0.5F+d[axis]);
    }
    const float normalization=1.0F/fmaxf(1.0F,options.rest_number_density);
    for(int dz=-1;dz<=1;++dz)for(int dy=-1;dy<=1;++dy)
        for(int dx=-1;dx<=1;++dx){
            const int x=center[0]+dx,y=center[1]+dy,z=center[2]+dz;
            if(x<0||y<0||z<0||x>=int(grid.n)||y>=int(grid.height)||z>=int(grid.n))
                continue;
            const float w=weight[0][dx+1]*weight[1][dy+1]*weight[2][dz+1]*
                          normalization;
            const auto cell=smoke_grid_index(x,y,z,int(grid.n),int(grid.height));
            constexpr float scale=16777216.0F;
            atomicAdd(&density_accumulator[cell],
                static_cast<unsigned long long>(w*scale+0.5F));
            atomicAdd(&temperature_accumulator[cell],
                static_cast<unsigned long long>(w*clamp_scalar(
                    thermal_lift[particle],0.0F,options.maximum_speed)*scale+0.5F));
        }
}

__global__ void smoke_grid_advect_face(int axis,SmokeGridField grid,
    const float *source,float *destination,float dt) {
    const auto face=blockIdx.x*blockDim.x+threadIdx.x;
    const auto count=smoke_grid_face_count(axis,int(grid.n),int(grid.height));
    if(face>=count)return;
    int x{},y{},z{};
    smoke_grid_face_coordinates(axis,face,int(grid.n),int(grid.height),x,y,z);
    const Vec3 point=smoke_grid_face_position(axis,x,y,z,grid);
    const Vec3 v0=smoke_grid_sample_velocity(point,grid);
    const Vec3 midpoint=subtract(point,multiply(v0,0.5F*dt));
    const Vec3 velocity=smoke_grid_sample_velocity(midpoint,grid);
    const Vec3 departure=subtract(point,multiply(velocity,dt));
    destination[face]=smoke_grid_sample_face_component(axis,source,departure,grid);
}

__device__ void smoke_grid_face_minmax(int axis,const float *values,Vec3 point,
    SmokeGridField grid,float &low,float &high) {
    Vec3 p=multiply(subtract(point,grid.minimum),1.0F/grid.spacing);
    int sx=int(grid.n),sy=int(grid.height),sz=int(grid.n);
    if(axis==0){p.y-=0.5F;p.z-=0.5F;++sx;}
    if(axis==1){p.x-=0.5F;p.z-=0.5F;++sy;}
    if(axis==2){p.x-=0.5F;p.y-=0.5F;++sz;}
    const int x0=int(floorf(p.x)),y0=int(floorf(p.y)),z0=int(floorf(p.z));
    low=FLT_MAX;high=-FLT_MAX;
    for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
        const int x=max(0,min(sx-1,x0+dx)),y=max(0,min(sy-1,y0+dy));
        const int z=max(0,min(sz-1,z0+dz));
        const float value=values[smoke_grid_face_index(axis,x,y,z,int(grid.n),
                                                       int(grid.height))];
        low=fminf(low,value);high=fmaxf(high,value);
    }
}

__global__ void smoke_grid_correct_face(int axis,SmokeGridField original,
    const float *source,const float *predicted,const float *reversed,
    float *corrected,float dt) {
    const auto face=blockIdx.x*blockDim.x+threadIdx.x;
    const auto count=smoke_grid_face_count(axis,int(original.n),int(original.height));
    if(face>=count)return;
    int x{},y{},z{};
    smoke_grid_face_coordinates(axis,face,int(original.n),int(original.height),x,y,z);
    const Vec3 point=smoke_grid_face_position(axis,x,y,z,original);
    const Vec3 midpoint=subtract(point,multiply(
        smoke_grid_sample_velocity(point,original),0.5F*dt));
    const Vec3 departure=subtract(point,multiply(
        smoke_grid_sample_velocity(midpoint,original),dt));
    float low{},high{};
    smoke_grid_face_minmax(axis,source,departure,original,low,high);
    corrected[face]=clamp_scalar(predicted[face]+0.5F*(source[face]-reversed[face]),
                                 low,high);
}

__global__ void smoke_grid_cell_diagnostics(SmokeGridField grid,
    Vec3 *cell_velocity,Vec3 *vorticity,float *strain,float *divergence) {
    const auto cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=grid.n*grid.height*grid.n)return;
    const int n=int(grid.n),h=int(grid.height);
    const int x=int(cell%grid.n),y=int(cell/grid.n%grid.height);
    const int z=int(cell/(grid.n*grid.height));
    const auto u=[&](int xx,int yy,int zz){
        xx=max(0,min(n,xx));yy=max(0,min(h-1,yy));zz=max(0,min(n-1,zz));
        return grid.face[0][smoke_grid_face_index(0,xx,yy,zz,n,h)];
    };
    const auto v=[&](int xx,int yy,int zz){
        xx=max(0,min(n-1,xx));yy=max(0,min(h,yy));zz=max(0,min(n-1,zz));
        return grid.face[1][smoke_grid_face_index(1,xx,yy,zz,n,h)];
    };
    const auto w=[&](int xx,int yy,int zz){
        xx=max(0,min(n-1,xx));yy=max(0,min(h-1,yy));zz=max(0,min(n,zz));
        return grid.face[2][smoke_grid_face_index(2,xx,yy,zz,n,h)];
    };
    const float u0=u(x,y,z),u1=u(x+1,y,z);
    const float v0=v(x,y,z),v1=v(x,y+1,z);
    const float w0=w(x,y,z),w1=w(x,y,z+1);
    const float inverse=1.0F/grid.spacing;
    const float transverse=0.25F*inverse;
    const float dux=(u1-u0)*inverse;
    const float duy=(u(x,y+1,z)+u(x+1,y+1,z)-
                     u(x,y-1,z)-u(x+1,y-1,z))*transverse;
    const float duz=(u(x,y,z+1)+u(x+1,y,z+1)-
                     u(x,y,z-1)-u(x+1,y,z-1))*transverse;
    const float dvx=(v(x+1,y,z)+v(x+1,y+1,z)-
                     v(x-1,y,z)-v(x-1,y+1,z))*transverse;
    const float dvy=(v1-v0)*inverse;
    const float dvz=(v(x,y,z+1)+v(x,y+1,z+1)-
                     v(x,y,z-1)-v(x,y+1,z-1))*transverse;
    const float dwx=(w(x+1,y,z)+w(x+1,y,z+1)-
                     w(x-1,y,z)-w(x-1,y,z+1))*transverse;
    const float dwy=(w(x,y+1,z)+w(x,y+1,z+1)-
                     w(x,y-1,z)-w(x,y-1,z+1))*transverse;
    const float dwz=(w1-w0)*inverse;
    if(cell_velocity)cell_velocity[cell]={0.5F*(u0+u1),0.5F*(v0+v1),
                                          0.5F*(w0+w1)};
    vorticity[cell]={dwy-dvz,duz-dwx,dvx-duy};
    const float sxy=0.5F*(duy+dvx),sxz=0.5F*(duz+dwx);
    const float syz=0.5F*(dvz+dwy);
    strain[cell]=sqrtf(2.0F*(dux*dux+dvy*dvy+dwz*dwz+
        2.0F*(sxy*sxy+sxz*sxz+syz*syz)));
    if(divergence)divergence[cell]=dux+dvy+dwz;
}

__global__ void smoke_grid_subgrid_force(SmokeGridField grid,
    const Vec3 *vorticity,Vec3 *force,float strength) {
    const auto cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=grid.n*grid.height*grid.n)return;
    const int n=int(grid.n),h=int(grid.height);
    const int x=int(cell%grid.n),y=int(cell/grid.n%grid.height);
    const int z=int(cell/(grid.n*grid.height));
    const auto magnitude=[&](int xx,int yy,int zz){
        xx=max(0,min(n-1,xx));yy=max(0,min(h-1,yy));zz=max(0,min(n-1,zz));
        return vector_length(vorticity[smoke_grid_index(xx,yy,zz,n,h)]);
    };
    Vec3 gradient{magnitude(x+1,y,z)-magnitude(x-1,y,z),
                  magnitude(x,y+1,z)-magnitude(x,y-1,z),
                  magnitude(x,y,z+1)-magnitude(x,y,z-1)};
    gradient=multiply(gradient,0.5F/grid.spacing);
    if(length_squared(gradient)<1.0e-12F){force[cell]={};return;}
    force[cell]=multiply(cross(normalized_or(gradient,{1,0,0}),vorticity[cell]),
                         strength*grid.spacing);
}

__global__ void smoke_grid_apply_face_forces(int axis,SmokeGridField grid,
    const float *source,float *destination,const float *open,const float *wall,
    const float *strain,const Vec3 *subgrid,const float *temperature,
    SmokeOptions options,float dt,Vec3 gravity) {
    const auto face=blockIdx.x*blockDim.x+threadIdx.x;
    const auto count=smoke_grid_face_count(axis,int(grid.n),int(grid.height));
    if(face>=count)return;
    if(open[face]<=1.0e-4F){destination[face]=wall[face];return;}
    int x{},y{},z{};
    smoke_grid_face_coordinates(axis,face,int(grid.n),int(grid.height),x,y,z);
    const Vec3 point=smoke_grid_face_position(axis,x,y,z,grid);
    const float center=source[face];
    const int sx=int(grid.n)+(axis==0),sy=int(grid.height)+(axis==1);
    const int sz=int(grid.n)+(axis==2);
    const auto sample=[&](int xx,int yy,int zz){
        xx=max(0,min(sx-1,xx));yy=max(0,min(sy-1,yy));zz=max(0,min(sz-1,zz));
        return source[smoke_grid_face_index(axis,xx,yy,zz,int(grid.n),
                                            int(grid.height))];
    };
    const float laplacian=(sample(x-1,y,z)+sample(x+1,y,z)+sample(x,y-1,z)+
        sample(x,y+1,z)+sample(x,y,z-1)+sample(x,y,z+1)-6.0F*center)/
        (grid.spacing*grid.spacing);
    const float local_strain=smoke_grid_sample_scalar(strain,point,grid);
    const float viscosity=options.grid_kinematic_viscosity+
        options.grid_les_coefficient*options.grid_les_coefficient*
        grid.spacing*grid.spacing*local_strain;
    const Vec3 curl_force=smoke_grid_sample_vector(subgrid,point,grid);
    const float density=smoke_grid_sample_scalar(grid.density,point,grid);
    const float thermal_loading=smoke_grid_sample_scalar(temperature,point,grid);
    // The splat stores density-weighted temperature.  Divide by loading to
    // recover the local mean thermal acceleration instead of weakening an
    // isolated hot parcel by the interpolation kernel's fractional weight.
    const float heat=density>1.0e-5F?clamp_scalar(
        thermal_loading/density,0.0F,options.maximum_speed):0.0F;
    const Vec3 up=normalized_or(multiply(gravity,-1.0F),{0,1,0});
    const Vec3 buoyancy=multiply(up,options.buoyancy*density*
        options.rest_number_density+2.0F*heat);
    const float body=axis==0?curl_force.x+buoyancy.x:
                     axis==1?curl_force.y+buoyancy.y:curl_force.z+buoyancy.z;
    const float value=center+dt*(viscosity*laplacian+body);
    destination[face]=clamp_scalar(value,-options.maximum_speed,
                                   options.maximum_speed);
}

__global__ void smoke_grid_apply_face_boundaries(int axis,SmokeGridField grid,
    float *values,const float *open,const float *wall,SmokeOptions options) {
    const auto face=blockIdx.x*blockDim.x+threadIdx.x;
    const auto count=smoke_grid_face_count(axis,int(grid.n),int(grid.height));
    if(face>=count)return;
    int x{},y{},z{};
    smoke_grid_face_coordinates(axis,face,int(grid.n),int(grid.height),x,y,z);
    const int extent=axis==0?int(grid.n):axis==1?int(grid.height):int(grid.n);
    const int coordinate=axis==0?x:axis==1?y:z;
    if(coordinate==0||coordinate==extent){
        if(open[face]<=1.0e-4F)values[face]=wall[face];
        else {
            const int nx=axis==0?(coordinate==0?1:extent-1):x;
            const int ny=axis==1?(coordinate==0?1:extent-1):y;
            const int nz=axis==2?(coordinate==0?1:extent-1):z;
            values[face]=values[smoke_grid_face_index(axis,nx,ny,nz,
                int(grid.n),int(grid.height))];
        }
    }
    const Vec3 point=smoke_grid_face_position(axis,x,y,z,grid);
    if(fabsf(point.x-options.emitter_center.x)<1.5F*grid.spacing&&
       fabsf(point.y-options.emitter_center.y)<=options.emitter_half_extents.x+
            0.5F*grid.spacing&&
       fabsf(point.z-options.emitter_center.z)<=options.emitter_half_extents.y+
            0.5F*grid.spacing)
        values[face]=axis==0?options.initial_velocity.x:
                     axis==1?options.initial_velocity.y:options.initial_velocity.z;
    // Store the cut face's volume-averaged velocity. Divergence consumes this
    // flux directly and projection applies an aperture-scaled pressure
    // gradient, so the aperture enters the discrete operator exactly once.
    values[face]=open[face]*values[face]+(1.0F-open[face])*wall[face];
}

__global__ void smoke_grid_divergence(SmokeGridField grid,
    float *divergence,float *rhs_max,float dt) {
    const auto cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=grid.n*grid.height*grid.n)return;
    const int n=int(grid.n),h=int(grid.height);
    const int x=int(cell%grid.n),y=int(cell/grid.n%grid.height);
    const int z=int(cell/(grid.n*grid.height));
    const auto flux=[&](int axis,int xx,int yy,int zz){
        return grid.face[axis][smoke_grid_face_index(axis,xx,yy,zz,n,h)];
    };
    const float value=(flux(0,x+1,y,z)-flux(0,x,y,z)+
        flux(1,x,y+1,z)-flux(1,x,y,z)+
        flux(2,x,y,z+1)-flux(2,x,y,z))/grid.spacing;
    divergence[cell]=value/dt;
    if(rhs_max)atomicMax(reinterpret_cast<unsigned int *>(rhs_max),
                         __float_as_uint(fabsf(value/dt)));
}

__device__ float smoke_pressure_open(const float *const open[3],int direction,
    int x,int y,int z,int n,int h) {
    if(direction==0)return open[0][smoke_grid_face_index(0,x,y,z,n,h)];
    if(direction==1)return open[0][smoke_grid_face_index(0,x+1,y,z,n,h)];
    if(direction==2)return open[1][smoke_grid_face_index(1,x,y,z,n,h)];
    if(direction==3)return open[1][smoke_grid_face_index(1,x,y+1,z,n,h)];
    if(direction==4)return open[2][smoke_grid_face_index(2,x,y,z,n,h)];
    return open[2][smoke_grid_face_index(2,x,y,z+1,n,h)];
}

__global__ void smoke_grid_pressure_smooth(int n,int h,float spacing,
    const float *rhs,const float *pressure,float *next,
    const float *open_x,const float *open_y,const float *open_z,
    const unsigned int *converged) {
    const auto cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=std::uint32_t(n*h*n)||(converged&&*converged))return;
    const int x=int(cell%std::uint32_t(n));
    const int y=int(cell/std::uint32_t(n)%std::uint32_t(h));
    const int z=int(cell/std::uint32_t(n*h));
    const float *open[3]={open_x,open_y,open_z};
    const int dx[6]={-1,1,0,0,0,0},dy[6]={0,0,-1,1,0,0};
    const int dz[6]={0,0,0,0,-1,1};
    float sum{},diagonal{};
    for(int direction=0;direction<6;++direction){
        const float coefficient=smoke_pressure_open(open,direction,x,y,z,n,h);
        if(coefficient<=1.0e-5F)continue;
        const int xx=x+dx[direction],yy=y+dy[direction],zz=z+dz[direction];
        if(xx>=0&&yy>=0&&zz>=0&&xx<n&&yy<h&&zz<n)
            sum+=coefficient*pressure[smoke_grid_index(xx,yy,zz,n,h)];
        diagonal+=coefficient;
    }
    // Undamped Jacobi preserves the highest-frequency checkerboard mode of
    // the 3-D Poisson operator.  That mode was being prolonged between
    // levels until the projection became unstable.  Weighted Jacobi is a
    // proper multigrid smoother: it damps those modes while the coarse grids
    // remove the long-wavelength error.
    const float jacobi=diagonal>1.0e-6F?
        (sum-spacing*spacing*rhs[cell])/diagonal:0.0F;
    constexpr float omega=2.0F/3.0F;
    next[cell]=pressure[cell]+omega*(jacobi-pressure[cell]);
}

__global__ void smoke_grid_pressure_residual(int n,int h,float spacing,
    const float *rhs,const float *pressure,float *residual,
    const float *open_x,const float *open_y,const float *open_z,
    float *maximum,const unsigned int *converged) {
    const auto cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=std::uint32_t(n*h*n)||(converged&&*converged))return;
    const int x=int(cell%std::uint32_t(n));
    const int y=int(cell/std::uint32_t(n)%std::uint32_t(h));
    const int z=int(cell/std::uint32_t(n*h));
    const float *open[3]={open_x,open_y,open_z};
    const int dx[6]={-1,1,0,0,0,0},dy[6]={0,0,-1,1,0,0};
    const int dz[6]={0,0,0,0,-1,1};
    float sum{},diagonal{};
    for(int direction=0;direction<6;++direction){
        const float coefficient=smoke_pressure_open(open,direction,x,y,z,n,h);
        if(coefficient<=1.0e-5F)continue;
        const int xx=x+dx[direction],yy=y+dy[direction],zz=z+dz[direction];
        if(xx>=0&&yy>=0&&zz>=0&&xx<n&&yy<h&&zz<n)
            sum+=coefficient*pressure[smoke_grid_index(xx,yy,zz,n,h)];
        diagonal+=coefficient;
    }
    const float laplacian=(sum-diagonal*pressure[cell])/(spacing*spacing);
    residual[cell]=rhs[cell]-laplacian;
    if(maximum)atomicMax(reinterpret_cast<unsigned int *>(maximum),
                         __float_as_uint(fabsf(residual[cell])));
}

__global__ void smoke_grid_restrict_residual(const float *fine,int fine_n,
    int fine_h,float *coarse,int coarse_n,int coarse_h) {
    const auto cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=std::uint32_t(coarse_n*coarse_h*coarse_n))return;
    const int x=int(cell%std::uint32_t(coarse_n));
    const int y=int(cell/std::uint32_t(coarse_n)%std::uint32_t(coarse_h));
    const int z=int(cell/std::uint32_t(coarse_n*coarse_h));
    float sum{};int samples{};
    for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
        const int xx=min(fine_n-1,2*x+dx),yy=min(fine_h-1,2*y+dy);
        const int zz=min(fine_n-1,2*z+dz);
        sum+=fine[smoke_grid_index(xx,yy,zz,fine_n,fine_h)];++samples;
    }
    coarse[cell]=sum/float(samples);
}

__global__ void smoke_grid_clear_scalar(float *values,std::uint32_t count) {
    const auto index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<count)values[index]=0.0F;
}

__global__ void smoke_grid_restrict_open(int axis,const float *fine,
    int fine_n,int fine_h,float *coarse,int coarse_n,int coarse_h) {
    const auto face=blockIdx.x*blockDim.x+threadIdx.x;
    const auto count=smoke_grid_face_count(axis,coarse_n,coarse_h);
    if(face>=count)return;
    int x{},y{},z{};
    smoke_grid_face_coordinates(axis,face,coarse_n,coarse_h,x,y,z);
    float sum{};int samples{};
    for(int b=0;b<2;++b)for(int a=0;a<2;++a){
        int xx=2*x,yy=2*y,zz=2*z;
        if(axis==0){yy+=a;zz+=b;}
        if(axis==1){xx+=a;zz+=b;}
        if(axis==2){xx+=a;yy+=b;}
        const int sx=fine_n+(axis==0),sy=fine_h+(axis==1),sz=fine_n+(axis==2);
        xx=min(sx-1,xx);yy=min(sy-1,yy);zz=min(sz-1,zz);
        sum+=fine[smoke_grid_face_index(axis,xx,yy,zz,fine_n,fine_h)];
        ++samples;
    }
    coarse[face]=sum/float(samples);
}

__global__ void smoke_grid_mark_domain_boundaries(int axis,
    SmokeGridField grid,float *open,float *wall,SmokeOptions options) {
    const auto face=blockIdx.x*blockDim.x+threadIdx.x;
    const auto count=smoke_grid_face_count(axis,int(grid.n),int(grid.height));
    if(face>=count)return;
    int x{},y{},z{};
    smoke_grid_face_coordinates(axis,face,int(grid.n),int(grid.height),x,y,z);
    const int extent=axis==0?int(grid.n):axis==1?int(grid.height):int(grid.n);
    const int coordinate=axis==0?x:axis==1?y:z;
    if(coordinate!=0&&coordinate!=extent)return;
    const float wind=axis==0?options.wind.x:axis==1?options.wind.y:options.wind.z;
    const float outward=coordinate==0?-1.0F:1.0F;
    if(wind*outward<0.0F){
        open[face]=0.0F;
        wall[face]=wind;
    }
}

__global__ void smoke_grid_prolong_add(const float *coarse,int coarse_n,
    int coarse_h,float *fine,int fine_n,int fine_h) {
    const auto cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=std::uint32_t(fine_n*fine_h*fine_n))return;
    const int x=int(cell%std::uint32_t(fine_n));
    const int y=int(cell/std::uint32_t(fine_n)%std::uint32_t(fine_h));
    const int z=int(cell/std::uint32_t(fine_n*fine_h));
    const float gx=0.5F*float(x)-0.25F;
    const float gy=0.5F*float(y)-0.25F;
    const float gz=0.5F*float(z)-0.25F;
    const int x0=int(floorf(gx)),y0=int(floorf(gy)),z0=int(floorf(gz));
    const float fx=gx-float(x0),fy=gy-float(y0),fz=gz-float(z0);
    float correction{};
    for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)
        for(int dx=0;dx<2;++dx){
            const int cx=max(0,min(coarse_n-1,x0+dx));
            const int cy=max(0,min(coarse_h-1,y0+dy));
            const int cz=max(0,min(coarse_n-1,z0+dz));
            const float weight=(dx?fx:1.0F-fx)*(dy?fy:1.0F-fy)*
                               (dz?fz:1.0F-fz);
            correction+=weight*coarse[smoke_grid_index(
                cx,cy,cz,coarse_n,coarse_h)];
        }
    // Averaged cut-face apertures form a cheap non-Galerkin coarse operator.
    // Damping its correction keeps the V-cycle contractive near thin meshes.
    fine[cell]+=0.5F*correction;
}

__global__ void smoke_grid_compare_residual(const float *rhs_max,
    const float *residual_max,float tolerance,unsigned int *converged) {
    if(blockIdx.x||threadIdx.x)return;
    *converged=*residual_max<=tolerance*fmaxf(*rhs_max,1.0e-5F);
}

__global__ void smoke_grid_reset_residual(float *residual_max,
    const unsigned int *converged) {
    if(blockIdx.x==0U&&threadIdx.x==0U&&!*converged)*residual_max=0.0F;
}

__global__ void smoke_grid_project_face(int axis,SmokeGridField grid,
    const float *pressure,float *velocity,const float *open,const float *wall,
    float dt) {
    const auto face=blockIdx.x*blockDim.x+threadIdx.x;
    const auto count=smoke_grid_face_count(axis,int(grid.n),int(grid.height));
    if(face>=count)return;
    int x{},y{},z{};
    smoke_grid_face_coordinates(axis,face,int(grid.n),int(grid.height),x,y,z);
    const int extent=axis==0?int(grid.n):axis==1?int(grid.height):int(grid.n);
    const int coordinate=axis==0?x:axis==1?y:z;
    if(coordinate==0||coordinate==extent){
        if(open[face]<=1.0e-4F){velocity[face]=wall[face];return;}
        int cx=x,cy=y,cz=z;
        if(axis==0)cx=coordinate==0?0:int(grid.n)-1;
        if(axis==1)cy=coordinate==0?0:int(grid.height)-1;
        if(axis==2)cz=coordinate==0?0:int(grid.n)-1;
        const float inside=pressure[smoke_grid_index(cx,cy,cz,
            int(grid.n),int(grid.height))];
        const float gradient=(coordinate==0?inside:-inside)/grid.spacing;
        velocity[face]-=dt*open[face]*gradient;
        return;
    }
    int lx=x,ly=y,lz=z,rx=x,ry=y,rz=z;
    if(axis==0){lx=x-1;rx=x;}
    if(axis==1){ly=y-1;ry=y;}
    if(axis==2){lz=z-1;rz=z;}
    const float gradient=(pressure[smoke_grid_index(rx,ry,rz,int(grid.n),
        int(grid.height))]-pressure[smoke_grid_index(lx,ly,lz,int(grid.n),
        int(grid.height))])/grid.spacing;
    velocity[face]=open[face]<=1.0e-4F?wall[face]:
        velocity[face]-dt*open[face]*gradient;
}

__global__ void smoke_grid_trace(Vec3 *positions,Vec3 *previous_positions,
    Vec3 *velocities,float *ages,float *thermal_lift,float *number_densities,
    float *pressures,Vec3 *particle_vorticities,std::uint32_t count,
    SmokeOptions options,SmokeGridField grid,float dt,Vec3 gravity) {
    const auto particle=blockIdx.x*blockDim.x+threadIdx.x;
    if(particle>=count||ages[particle]>=options.lifetime)return;
    const Vec3 point=positions[particle];
    previous_positions[particle]=point;
    Vec3 velocity=velocities[particle];
    if(smoke_grid_contains(point,grid)&&ages[particle]>0.0F){
        const Vec3 v0=smoke_grid_sample_velocity(point,grid);
        const Vec3 midpoint=add(point,multiply(v0,0.5F*dt));
        velocity=smoke_grid_sample_velocity(midpoint,grid);
        number_densities[particle]=smoke_grid_sample_scalar(
            grid.density,point,grid)*options.rest_number_density;
        pressures[particle]=smoke_grid_sample_pressure(point,grid);
        particle_vorticities[particle]=smoke_grid_sample_vector(
            grid.vorticity,point,grid);
    }
    velocity=clamp_length(velocity,options.maximum_speed);
    positions[particle]=add(point,multiply(velocity,dt));
    velocities[particle]=velocity;ages[particle]+=dt;
    thermal_lift[particle]*=expf(-dt/3.0F);
}

__global__ void smoke_grid_soft_body_force(const Vec3 *surface,
    const std::uint32_t *indices,std::uint32_t index_count,
    const SoftBodySurfaceBinding *bindings,const Vec3 *node_velocity_snapshot,
    Vec3 *node_velocities,const float *inverse_masses,SmokeGridField grid,
    float drag_coefficient,float density_scale,float kinematic_viscosity,
    float les_coefficient,
    float maximum_acceleration,float dt) {
    const auto triangle=blockIdx.x*blockDim.x+threadIdx.x;
    if(triangle>=index_count/3U)return;
    const auto base=triangle*3U,ia=indices[base],ib=indices[base+1U],ic=indices[base+2U];
    const Vec3 a=surface[ia],b=surface[ib],c=surface[ic];
    const Vec3 twice_area=cross(subtract(b,a),subtract(c,a));
    const float area=0.5F*vector_length(twice_area);
    if(area<=1.0e-9F)return;
    const Vec3 normal=normalized_or(twice_area,{0,1,0});
    const Vec3 center=multiply(add(add(a,b),c),1.0F/3.0F);
    const float offset=1.5F*grid.spacing;
    const Vec3 plus=add(center,multiply(normal,offset));
    const Vec3 minus=subtract(center,multiply(normal,offset));
    if(!smoke_grid_contains(plus,grid)||!smoke_grid_contains(minus,grid))return;
    Vec3 plus_air{},minus_air{};float plus_density{},minus_density{};
    smoke_grid_sample(plus,grid,plus_air,plus_density);
    smoke_grid_sample(minus,grid,minus_air,minus_density);
    plus_density*=density_scale;minus_density*=density_scale;
    if(plus_density+minus_density<1.0e-4F)return;
    const Vec3 body=multiply(add(add(
        smoke_soft_surface_velocity(bindings[ia],node_velocity_snapshot),
        smoke_soft_surface_velocity(bindings[ib],node_velocity_snapshot)),
        smoke_soft_surface_velocity(bindings[ic],node_velocity_snapshot)),
        1.0F/3.0F);
    const Vec3 plus_relative=subtract(plus_air,body);
    const Vec3 minus_relative=subtract(minus_air,body);
    const Vec3 plus_tangent=subtract(plus_relative,
        multiply(normal,dot(plus_relative,normal)));
    const Vec3 minus_tangent=subtract(minus_relative,
        multiply(normal,dot(minus_relative,normal)));
    const float pressure_difference=minus_density*
        smoke_grid_sample_pressure(minus,grid)-plus_density*
        smoke_grid_sample_pressure(plus,grid);
    const float strain=smoke_grid_sample_scalar(grid.strain,center,grid);
    const float viscosity=kinematic_viscosity+les_coefficient*les_coefficient*
        grid.spacing*grid.spacing*strain;
    const Vec3 force=multiply(add(multiply(normal,pressure_difference),
        multiply(add(multiply(plus_tangent,plus_density),
                     multiply(minus_tangent,minus_density)),
                 drag_coefficient*viscosity/offset)),area);
    const std::uint32_t corners[3]={ia,ib,ic};
    for(int corner=0;corner<3;++corner){
        const auto binding=bindings[corners[corner]];
        for(int j=0;j<4;++j){
            const auto node=binding.nodes[j];const float weight=binding.weights[j];
            if(weight<=0.0F||inverse_masses[node]==0.0F)continue;
            const Vec3 dv=clamp_length(multiply(force,dt*inverse_masses[node]*
                weight/3.0F),maximum_acceleration*dt*weight/3.0F);
            atomicAdd(&node_velocities[node].x,dv.x);
            atomicAdd(&node_velocities[node].y,dv.y);
            atomicAdd(&node_velocities[node].z,dv.z);
        }
    }
}

__global__ void smoke_grid_cloth_force(const Vec3 *positions,
    const Vec3 *velocities,const float *inverse_masses,
    const std::uint32_t *indices,std::uint32_t index_count,SmokeGridField grid,
    float drag_coefficient,float density_scale,float kinematic_viscosity,
    float les_coefficient,Vec3 *forces) {
    const auto triangle=blockIdx.x*blockDim.x+threadIdx.x;
    if(triangle>=index_count/3U)return;
    const auto base=triangle*3U;
    const auto ia=indices[base],ib=indices[base+1U],ic=indices[base+2U];
    if(inverse_masses[ia]==0.0F&&inverse_masses[ib]==0.0F&&
       inverse_masses[ic]==0.0F)return;
    const Vec3 a=positions[ia],b=positions[ib],c=positions[ic];
    const Vec3 twice_area=cross(subtract(b,a),subtract(c,a));
    const float area=0.5F*vector_length(twice_area);
    if(area<=1.0e-9F)return;
    const Vec3 normal=normalized_or(twice_area,{0,1,0});
    const Vec3 center=multiply(add(add(a,b),c),1.0F/3.0F);
    const Vec3 plus=add(center,multiply(normal,1.5F*grid.spacing));
    const Vec3 minus=subtract(center,multiply(normal,1.5F*grid.spacing));
    if(!smoke_grid_contains(plus,grid)||!smoke_grid_contains(minus,grid))return;
    Vec3 plus_air{},minus_air{};float plus_density{},minus_density{};
    smoke_grid_sample(plus,grid,plus_air,plus_density);
    smoke_grid_sample(minus,grid,minus_air,minus_density);
    plus_density*=density_scale;minus_density*=density_scale;
    if(plus_density+minus_density<1.0e-4F)return;
    const Vec3 cloth_velocity=multiply(add(add(
        velocities[ia],velocities[ib]),velocities[ic]),1.0F/3.0F);
    const Vec3 plus_relative=subtract(plus_air,cloth_velocity);
    const Vec3 minus_relative=subtract(minus_air,cloth_velocity);
    const Vec3 plus_tangent=subtract(plus_relative,
        multiply(normal,dot(plus_relative,normal)));
    const Vec3 minus_tangent=subtract(minus_relative,
        multiply(normal,dot(minus_relative,normal)));
    const float pressure_difference=minus_density*
        smoke_grid_sample_pressure(minus,grid)-plus_density*
        smoke_grid_sample_pressure(plus,grid);
    const float strain=smoke_grid_sample_scalar(grid.strain,center,grid);
    const float viscosity=kinematic_viscosity+les_coefficient*les_coefficient*
        grid.spacing*grid.spacing*strain;
    const Vec3 corner_force=multiply(add(multiply(normal,pressure_difference),
        multiply(add(multiply(plus_tangent,plus_density),
                     multiply(minus_tangent,minus_density)),
                 drag_coefficient*viscosity/(1.5F*grid.spacing))),area/3.0F);
    const std::uint32_t corners[3]={ia,ib,ic};
    for(int corner=0;corner<3;++corner){
        const auto vertex=corners[corner];
        atomicAdd(&forces[vertex].x,corner_force.x);
        atomicAdd(&forces[vertex].y,corner_force.y);
        atomicAdd(&forces[vertex].z,corner_force.z);
    }
}

__global__ void smoke_grid_apply_cloth_force(Vec3 *velocities,
    const float *inverse_masses,const Vec3 *forces,std::uint32_t count,
    float maximum_acceleration,float dt) {
    const auto vertex=blockIdx.x*blockDim.x+threadIdx.x;
    if(vertex>=count||inverse_masses[vertex]==0.0F)return;
    const Vec3 delta=clamp_length(multiply(forces[vertex],
        inverse_masses[vertex]*dt),maximum_acceleration*dt);
    velocities[vertex]=add(velocities[vertex],delta);
}

__global__ void smoke_grid_rigid_force(TriangleMeshResource mesh,
    std::uint32_t body,const BodyParameters *parameters,RigidBodyState *states,
    SmokeGridField grid,float air_density,float drag_coefficient,float density_scale,
    float kinematic_viscosity,float les_coefficient,float dt) {
    __shared__ Vec3 linear[128],angular[128];
    Vec3 local_linear{},local_angular{};
    const RigidBodyState state=states[body];
    for(std::uint32_t triangle=threadIdx.x;triangle<mesh.index_count/3U;
        triangle+=blockDim.x){
        const auto base=triangle*3U;
        const Vec3 a=transform_point(state,mesh.vertices[mesh.indices[base]]);
        const Vec3 b=transform_point(state,mesh.vertices[mesh.indices[base+1U]]);
        const Vec3 c=transform_point(state,mesh.vertices[mesh.indices[base+2U]]);
        const Vec3 twice_area=cross(subtract(b,a),subtract(c,a));
        const float area=0.5F*vector_length(twice_area);
        if(area<=1.0e-9F)continue;
        const Vec3 normal=normalized_or(twice_area,{1,0,0});
        const Vec3 center=multiply(add(add(a,b),c),1.0F/3.0F);
        const Vec3 plus=add(center,multiply(normal,1.5F*grid.spacing));
        const Vec3 minus=subtract(center,multiply(normal,1.5F*grid.spacing));
        if(!smoke_grid_contains(plus,grid)||!smoke_grid_contains(minus,grid))continue;
        Vec3 plus_air{},minus_air{};float plus_density{},minus_density{};
        smoke_grid_sample(plus,grid,plus_air,plus_density);
        smoke_grid_sample(minus,grid,minus_air,minus_density);
        plus_density*=density_scale;minus_density*=density_scale;
        if(plus_density+minus_density<1.0e-4F)continue;
        const Vec3 arm=subtract(center,state.position);
        const Vec3 wall=add(state.linear_velocity,cross(state.angular_velocity,arm));
        const Vec3 plus_relative=subtract(plus_air,wall);
        const Vec3 minus_relative=subtract(minus_air,wall);
        const Vec3 plus_tangent=subtract(plus_relative,
            multiply(normal,dot(plus_relative,normal)));
        const Vec3 minus_tangent=subtract(minus_relative,
            multiply(normal,dot(minus_relative,normal)));
        const float pressure_difference=minus_density*
            smoke_grid_sample_pressure(minus,grid)-plus_density*
            smoke_grid_sample_pressure(plus,grid);
        const float strain=smoke_grid_sample_scalar(grid.strain,center,grid);
        const float viscosity=kinematic_viscosity+
            les_coefficient*les_coefficient*grid.spacing*grid.spacing*strain;
        const Vec3 force=multiply(add(multiply(normal,pressure_difference),
            multiply(add(multiply(plus_tangent,plus_density),
                         multiply(minus_tangent,minus_density)),
                     drag_coefficient*viscosity/(1.5F*grid.spacing))),
            air_density*area);
        local_linear=add(local_linear,force);
        local_angular=add(local_angular,cross(arm,force));
    }
    linear[threadIdx.x]=local_linear;angular[threadIdx.x]=local_angular;
    __syncthreads();
    for(std::uint32_t stride=blockDim.x/2U;stride;stride/=2U){
        if(threadIdx.x<stride){linear[threadIdx.x]=add(linear[threadIdx.x],
            linear[threadIdx.x+stride]);angular[threadIdx.x]=add(
            angular[threadIdx.x],angular[threadIdx.x+stride]);}
        __syncthreads();
    }
    if(threadIdx.x||parameters[body].motion!=MotionType::dynamic)return;
    RigidBodyState next=state;const BodyParameters settings=parameters[body];
    next.linear_velocity=clamp_length(add(next.linear_velocity,multiply(
        linear[0],dt*settings.inverse_mass)),settings.maximum_linear_speed);
    next.angular_velocity=clamp_length(add(next.angular_velocity,
        inverse_inertia_world(settings,next,multiply(angular[0],dt))),
        settings.maximum_angular_speed);
    states[body]=next;
}

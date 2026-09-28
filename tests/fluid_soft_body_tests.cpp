// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>
#include <cuda_runtime_api.h>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace parallel_mater;
void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void check(Status s, const char *message) {
    if (!s) throw std::runtime_error(std::string(message) + ": " + s.message);
}
Vec3 add(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
Vec3 mul(Vec3 a, float s) { return {a.x*s,a.y*s,a.z*s}; }
float length(Vec3 v) { return std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z); }
template<class T> std::vector<T> read(DeviceSpan<const T> span) {
    std::vector<T> output(span.size);
    check(cudaMemcpy(output.data(), span.data, span.size*sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess, "read");
    return output;
}

void impact(bool pinned, unsigned particle_count, bool reverse_winding = false) {
    constexpr float dt = 1.0F/60;
    const std::array<Vec3,8> nodes{{{-1,-0.05F,-1},{1,-0.05F,-1},{1,-0.05F,1},{-1,-0.05F,1},
                                   {-1,0.05F,-1},{1,0.05F,-1},{1,0.05F,1},{-1,0.05F,1}}};
    std::array<std::uint32_t,36> indices{0,1,2,0,2,3,4,6,5,4,7,6,0,4,5,0,5,1,
                                       1,5,6,1,6,2,2,6,7,2,7,3,3,7,4,3,4,0};
    if (reverse_winding) for (unsigned i=0;i<indices.size();i+=3) std::swap(indices[i+1],indices[i+2]);
    std::vector<SoftBodyBond> bonds;
    for (unsigned a=0;a<8;++a) for(unsigned b=a+1;b<8;++b)
        bonds.push_back({a,b,length(add(nodes[a],mul(nodes[b],-1)))});
    std::array<float,8> masses{};
    std::array<SoftBodySurfaceBinding,8> bindings{};
    for (unsigned i=0;i<8;++i) {
        masses[i] = pinned && i>=4 ? 0 : 100;
        // Duplicate influences must coalesce before inverse mass/counting.
        bindings[i].nodes[0] = bindings[i].nodes[1] = i;
        bindings[i].weights[0] = 0.4F; bindings[i].weights[1] = 0.6F;
    }
    World world;
    check(World::create({.soft_body_capacity=2,.fluid_soft_body_coupling_capacity=1,
                        .physics_debug={.frame_capacity=2}},world),"create");
    SoftBodyId soft;
    const SoftBodyOptions body_options{
        .nodes={nodes.data(),nodes.size()}, .bonds={bonds.data(),bonds.size()},
        .inverse_masses={masses.data(),masses.size()}, .surface_vertices={nodes.data(),nodes.size()},
        .surface_triangle_indices={indices.data(),indices.size()}, .surface_bindings={bindings.data(),bindings.size()},
        .velocity_damping=0,.spring_damping=0,.maximum_speed=2,.solver_iterations=1};
    check(world.add_soft_body(body_options,soft),"soft body");
    std::vector<FluidParticle> particles(particle_count, {{0,0.16F,0},{0,-12,0}});
    FluidParticle *device=nullptr;
    check(cudaMalloc(reinterpret_cast<void **>(&device),particles.size()*sizeof(FluidParticle))==cudaSuccess,"upload allocation");
    check(cudaMemcpy(device,particles.data(),particles.size()*sizeof(FluidParticle),cudaMemcpyHostToDevice)==cudaSuccess,"upload");
    FluidId fluid;
    constexpr float mass=0.01F;
    auto added=world.add_fluid({.capacity=particle_count,.particle_radius=0.02F,.support_radius=0.08F,
        .solver_iterations=1,.maximum_neighbors=particle_count+1,.repulsion=0,.viscosity=0,.velocity_damping=0,
        .maximum_speed=12,.normal_damping=0,.rest_particle_volume=mass/1000},
        {device,particles.size()},fluid);
    cudaFree(device); check(added,"fluid");
    FluidSoftBodyCouplingId coupling;
    check(world.add_fluid_soft_body_coupling({.fluid=fluid,.soft_body=soft},coupling),"couple");
    check(world.step({.timestep=dt,.substeps=1,.gravity={}}),"step");
    FluidDeviceView water;
    SoftBodyDeviceView body;
    check(world.fluid_view(fluid,water),"water view");
    check(world.soft_body_view(soft,body),"soft view");
    Vec3 fluid_impulse{}, soft_impulse{};
    for (Vec3 v:read(water.velocities)) fluid_impulse=add(fluid_impulse,mul(add(v,{0,12,0}),mass));
    for (Vec3 f:read(body.fluid_contact_forces)) soft_impulse=add(soft_impulse,mul(f,dt));
    const float imbalance=length(add(fluid_impulse,soft_impulse));
    check(imbalance < 2.0e-4F,"fluid and soft reaction impulses are unbalanced");
    float max_speed=0;
    for (Vec3 v:read(body.velocities)) max_speed=std::max(max_speed,length(v));
    check(max_speed<=2.0001F,"dense water overshot soft speed ceiling");
    if (!pinned) {
        const auto positions = read(body.positions);
        float displacement = 0;
        for (unsigned i = 0; i < nodes.size(); ++i)
            displacement = std::max(displacement, length(add(positions[i],mul(nodes[i],-1))));
        check(displacement > 1.0e-6F && max_speed > 0,
              "water failed to deflect the soft body in zero gravity");
    }
    for (Vec3 p:read(water.positions)) check(p.y>0.0F,"particle tunneled through thin soft body");
    if (pinned) {
        auto positions=read(body.positions);
        for(unsigned i=4;i<8;++i) check(length(add(positions[i],mul(nodes[i],-1)))<1.0e-7F,"pin moved");
    }
    PhysicsDebugFrameView capture;
    check(world.physics_debug_frame(capture),"debug capture");
    float capture_force=0;
    for(std::size_t i=0;i<capture.soft_body_nodes.size;++i)
        capture_force+=length(capture.soft_body_nodes.data[i].fluid_contact_force);
    check(capture_force>0,"capture omitted fluid reactions");
    check(world.remove_fluid_soft_body_coupling(coupling),"remove");
    auto open=body_options;
    open.surface_triangle_indices.size-=3;
    SoftBodyId open_body;
    check(world.add_soft_body(open,open_body),"open soft body");
    check(!world.add_fluid_soft_body_coupling({.fluid=fluid,.soft_body=open_body},coupling),"open surface accepted");
    std::cout << "particles=" << particle_count << " pinned=" << pinned << " reverse=" << reverse_winding
              << " impulse_error=" << imbalance << " speed=" << max_speed << '\n';
}

int main() {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
    try { impact(true,1); impact(false,1); impact(false,64); impact(true,1,true); }
    catch(const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}

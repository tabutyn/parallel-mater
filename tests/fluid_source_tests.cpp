// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace parallel_mater;
void check(bool value, const char *message) { if(!value) throw std::runtime_error(message); }
void check(Status value, const char *message) { if(!value) throw std::runtime_error(std::string(message)+": "+value.message); }
double distance2(Vec3 a,Vec3 b) { return (a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z); }
int main() {
    try {
        // Tilted open triangle, repeated face: no rectangular bounding-box fill,
        // no duplicate seam sites, and every site remains on the authored face.
        const std::array<Vec3,3> vertices{{{0,0,0},{1,1,0},{0,0,1}}};
        const std::array<std::uint32_t,6> indices{{0,1,2,0,1,2}};
        ParticleSourceMesh mesh{{vertices.data(),3},{indices.data(),6},0.2F};
        std::vector<Vec3> points,again;
        check(sample_fluid_source(mesh,points),"sample triangle");
        check(points.size()>10,"subdivides coarse triangles");
        check(sample_fluid_source(mesh,again),"repeat sample");
        check(again.size()==points.size(),"deterministic site count");
        for(std::size_t i=0;i<points.size();++i) {
            const auto p=points[i];
            check(std::abs(p.x-p.y)<1e-6 && p.x>=0 && p.z>=0 && p.x+p.z<=1.00001F,"sites lie on triangle");
            check(distance2(p,again[i])==0,"deterministic sample order");
            for(std::size_t j=0;j<i;++j) check(distance2(p,points[j])>=0.039999,"minimum source spacing");
        }
        auto bad=mesh; bad.spacing=0;
        check(!sample_fluid_source(bad,again),"reject zero explicit sampling spacing");
        check(again.size()==points.size(),"failure leaves output unchanged");
        const std::array<std::uint32_t,3> invalid{{0,1,99}};
        bad=mesh; bad.triangle_indices={invalid.data(),3};
        check(!sample_fluid_source(bad,again),"reject invalid mesh index");
        int devices=0;
        if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
        auto run=[&](float speed, unsigned capacity, bool duplicate) {
            World world;
            check(World::create({.rigid_body_capacity=1,.triangle_mesh_capacity=1},world),"create source world");
            FluidId fluid;
            check(world.add_fluid({.capacity=capacity,.particle_radius=0.04F,.support_radius=0.2F,
                .solver_iterations=1,.repulsion=0,.viscosity=0,.velocity_damping=0}, {},fluid),"add fluid");
            ParticleSourceId source,second;
            ParticleSourceOptions options{.fluid=fluid,.initial_velocity={0,-speed,0}};
            check(world.add_particle_source(mesh,options,source),"add source");
            if(duplicate) check(world.add_particle_source(mesh,options,second),"add overlapping source");
            WorldStatistics stats,first;
            for(int i=0;i<120;++i) {
                check(world.step({.timestep=1.0F/120,.substeps=1,.gravity={}}),"source step");
                if(i==0) check(world.collect_statistics(first),"first statistics");
            }
            check(world.collect_statistics(stats),"source statistics");
            check(stats.allocated_bytes==first.allocated_bytes,"allocation-free emission");
            check(stats.particle_count<=capacity,"bounded source capacity");
            if(speed==0) check(stats.emitted_particle_count==std::min<std::size_t>(capacity,points.size()),"occupied sites never reemit; overlapping sources share occupancy");
            if(capacity==1) check(stats.spawn_capacity_miss_count>0,"report full source capacity");
            FluidDeviceView view;
            check(world.fluid_view(fluid,view),"source view");
            std::vector<std::uint32_t> ids(view.particle_count);
            check(cudaMemcpy(ids.data(),view.stable_particle_ids.data,ids.size()*sizeof(ids[0]),cudaMemcpyDeviceToHost)==cudaSuccess,"read source IDs");
            for(std::size_t i=1;i<ids.size();++i) check(ids[i]>ids[i-1],"unique monotonic source IDs");
            options.enabled=false;
            check(world.update_particle_source(source,options),"disable source");
            if(duplicate) check(world.update_particle_source(second,options),"disable duplicate");
            check(world.step({.timestep=0.1F,.substeps=1,.gravity={}}),"disabled step");
            WorldStatistics disabled;
            check(world.collect_statistics(disabled),"disabled stats");
            check(disabled.emitted_particle_count==stats.emitted_particle_count,"disabled sources do not emit");
            check(world.remove_particle_source(source),"remove source");
            check(!world.update_particle_source(source,options),"reject stale source handle");
            check(world.remove_fluid(fluid),"remove destination invalidates remaining sources");
            if(duplicate) check(!world.remove_particle_source(second),"removed-fluid source stale");
            return stats.emitted_particle_count;
        };
        const auto stationary=run(0,1000,true), slow=run(1,1000,false),fast=run(2,1000,false);
        run(0,1,true);
        check(fast>slow*1.6 && fast<slow*2.4,"velocity controls flow without a rate setting");
        std::cout << "sites=" << points.size() << " stationary=" << stationary << " speed1=" << slow << " speed2=" << fast << '\n';
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}

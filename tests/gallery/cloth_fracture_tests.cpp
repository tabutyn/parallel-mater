// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>
#include "../../examples/support/vector_math.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

using namespace parallel_mater;
namespace m = parallel_mater::gallery::math;
void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void check(Status status, const char *message) { check(bool(status), message); }
template<class T> std::vector<T> read(DeviceSpan<const T> span) {
    std::vector<T> out(span.size);
    check(cudaMemcpy(out.data(), span.data, out.size()*sizeof(T), cudaMemcpyDeviceToHost)
          == cudaSuccess, "readback failed");
    return out;
}
// Test-only fault injection: impose a known fracture/impulse so isolation is
// tested independently of impact thresholds. Applications must not mutate views.
template<class T> void inject(DeviceSpan<const T> span, const std::vector<T> &data) {
    check(span.size == data.size(), "injection size mismatch");
    check(cudaMemcpy(const_cast<T*>(span.data), data.data(), data.size()*sizeof(T),
                     cudaMemcpyHostToDevice) == cudaSuccess, "injection failed");
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        const std::vector<Vec3> rest{{0,0,0},{1,0,0},{0,1,0},{1,1,0}};
        const std::vector<unsigned> faces{0,1,2,1,3,2};
        World world; check(World::create({.cloth_capacity=1}, world), "create world");
        ClothId id;
        check(world.add_cloth({.vertices={rest.data(),rest.size()},
            .triangle_indices={faces.data(),faces.size()}, .vertex_mass=1,
            .velocity_damping=0, .break_strain=0.1F}, id), "add cloth");
        // Allocate coupling before the tear, then exercise the appended cloth
        // nodes later. This catches undersized contact buffers and reactions
        // incorrectly routed back through the original, now remote vertices.
        const std::vector<Vec3> soft_nodes{{.6F,.6F,.04F},{.8F,.6F,.04F},
                                         {.6F,.8F,.04F},{.6F,.6F,.24F}};
        const std::vector<unsigned> soft_faces{0,2,1,0,1,3,0,3,2,1,2,3};
        std::vector<SoftBodyBond> soft_bonds;
        std::vector<SoftBodySurfaceBinding> bindings(4);
        for(unsigned i=0;i<4;++i) {
            bindings[i].nodes[0]=i; bindings[i].weights[0]=1;
            for(unsigned j=i+1;j<4;++j)
                soft_bonds.push_back({i,j,m::length(m::subtract(soft_nodes[i],soft_nodes[j]))});
        }
        SoftBodyId soft;
        check(world.add_soft_body({.nodes={soft_nodes.data(),soft_nodes.size()},
            .bonds={soft_bonds.data(),soft_bonds.size()},
            .surface_vertices={soft_nodes.data(),soft_nodes.size()},
            .surface_triangle_indices={soft_faces.data(),soft_faces.size()},
            .surface_bindings={bindings.data(),bindings.size()}},soft),"add soft test body");
        SoftBodyClothCouplingId coupling;
        check(world.add_soft_body_cloth_coupling(
            {.soft_body=soft,.cloth=id,.enabled=false},coupling),"allocate coupling before tear");
        ClothDeviceView view; check(world.cloth_view(id,view), "view cloth");
        auto bonds=read(view.bonds); auto active=read(view.active_bonds);
        unsigned seam=~0U;
        for(unsigned i=0;i<bonds.size();++i)
            if(std::min(bonds[i].first,bonds[i].second)==1 &&
               std::max(bonds[i].first,bonds[i].second)==2) seam=i;
        check(seam!=~0U, "missing shared edge");
        const Vec3 drift{0.2F,0.1F,0.3F};
        inject(view.velocities,std::vector<Vec3>(view.vertex_count,drift));
        active[seam]=0; inject(view.active_bonds,active);
        check(world.step({.gravity={}}), "split cloth");
        check(world.cloth_view(id,view), "view split cloth");
        check(view.vertex_count==6, "tear did not split shared vertices");
        Vec2 *uvs{};
        check(cudaMallocManaged(reinterpret_cast<void**>(&uvs),4*sizeof(Vec2))==cudaSuccess, "allocate UVs");
        uvs[0]={0,0}; uvs[1]={1,0}; uvs[2]={0,1}; uvs[3]={1,1};
        PaintFieldId paint;
        const auto paint_status=world.add_paint_field(
            {.cloth=id,.vertex_uvs={uvs,4},.width=8,.height=8},paint);
        cudaFree(uvs);
        check(paint_status,"paint field must still accept authored UVs after tear");
        auto indices=read(view.triangle_indices); auto sources=read(view.vertex_source_indices);
        auto masses=read(view.inverse_masses); auto positions=read(view.positions);
        auto velocities=read(view.velocities); active=read(view.active_bonds);
        for (Vec3 velocity:velocities)
            check(m::length(m::subtract(velocity,drift)) < 1.e-4F,
                  "split nodes did not inherit velocity");
        float mass=0;
        for(float inverse:masses) mass+=1/inverse;
        check(std::abs(mass-4)<1.e-6F, "tear changed total mass");
        for(unsigned i=0;i<indices.size();++i)
            check(sources[indices[i]]==faces[i], "tear lost authored vertex mapping");
        for(unsigned i=0;i<bonds.size();++i)
            check(!bonds[i].bending || !active[i], "bend still crosses broken seam");
        std::set<unsigned> fragment(indices.begin(),indices.begin()+3);
        for(unsigned i=3;i<6;++i) check(!fragment.contains(indices[i]), "fragments share a vertex");
        const Vec3 kick{0.7F,0.3F,0.4F};
        std::fill(velocities.begin(),velocities.end(),Vec3{});
        for(unsigned node:fragment) velocities[node]=kick;
        inject(view.velocities,velocities);
        float maximum_error=0;
        for(unsigned frame=0;frame<120;++frame) {
            check(world.step({.gravity={}}), "advance free fragments");
            check(world.cloth_view(id,view), "refresh free fragments");
            auto current=read(view.positions); auto surface=read(view.surface_positions);
            for(unsigned node=0;node<current.size();++node) {
                const Vec3 expected=m::add(positions[node], fragment.contains(node)
                    ? m::multiply(kick, (frame+1)/60.F) : Vec3{});
                maximum_error=std::max(maximum_error,m::length(m::subtract(current[node],expected)));
            }
            for(unsigned c=0;c<indices.size();++c)
                check(m::length(m::subtract(surface[c],current[indices[c]]))<1.e-7F,
                      "surface is not its physical triangle");
        }
        std::cout<<"fragment_isolation max_ballistic_error="<<maximum_error<<" mass="<<mass<<'\n';
        check(maximum_error<2.e-4F, "force or motion leaked across the tear");
        check(world.update_soft_body_cloth_coupling(coupling,{.soft_body=soft,.cloth=id}),
              "enable contact with separated triangle");
        SoftBodyDeviceView soft_view;
        check(world.soft_body_view(soft,soft_view),"view soft test body");
        inject(soft_view.velocities,std::vector<Vec3>(soft_view.node_count,{0,0,-1}));
        float split_node_force=0, remote_force=0;
        for(unsigned frame=0;frame<20;++frame) {
            check(world.step({.timestep=1.F/240,.substeps=1,.gravity={}}),"contact split triangle");
            check(world.cloth_view(id,view),"refresh contact forces");
            const auto forces=read(view.soft_body_contact_forces);
            for(unsigned node=0;node<forces.size();++node) {
                if(node>=rest.size()) split_node_force=std::max(split_node_force,m::length(forces[node]));
                if(fragment.contains(node)) remote_force=std::max(remote_force,m::length(forces[node]));
            }
        }
        std::cout<<"split_contact_force="<<split_node_force<<" remote_fragment_force="<<remote_force<<'\n';
        check(split_node_force>1.e-5F && remote_force==0,
              "contact did not stay on its actual fragment vertices");
        return 0;
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}

// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

using namespace parallel_mater;
void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
float distance(Vec3 a, Vec3 b) {
    return std::sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)+(a.z-b.z)*(a.z-b.z));
}
int main() {
    try {
        // Duplicate render corners deliberately: the API must weld physical
        // seams while retaining source interpolation for UVs and hard normals.
        const Vec3 corners[]{{0,0,0},{1,0,0},{1,1,0},{0,1,0},
                             {0,0,1},{1,0,1},{1,1,1},{0,1,1}};
        const unsigned faces[]{0,2,1,0,3,2,4,5,6,4,6,7,
                               0,1,5,0,5,4,3,7,6,3,6,2,
                               0,4,7,0,7,3,1,2,6,1,6,5};
        std::vector<Vec3> vertices;
        std::vector<std::uint32_t> indices;
        std::vector<float> pins;
        for (auto corner : faces) {
            indices.push_back(vertices.size());
            vertices.push_back(corners[corner]);
            pins.push_back(corners[corner].x == 1 ? 1 : 0);
        }
        SoftBodyGeometrySource source{{vertices.data(),vertices.size()},
            {indices.data(),indices.size()},{pins.data(),pins.size()},0.2F,3.0F};
        SoftBodyGeometry mesh;
        auto status = build_soft_body_geometry(source, mesh);
        check(status.ok(), status.message);
        check(mesh.surface_vertices.size() > vertices.size(), "surface was not subdivided");
        check(mesh.surface_sources.size() == mesh.surface_vertices.size(), "missing provenance");
        check(mesh.surface_bindings.size() == mesh.surface_vertices.size(), "missing skin bindings");
        check(std::abs(mesh.node_mass*mesh.nodes.size()-3) < 1e-5F, "mass changed");
        std::vector<unsigned> degrees(mesh.nodes.size());
        for (const auto &bond : mesh.bonds) { ++degrees[bond.first]; ++degrees[bond.second]; }
        std::map<std::pair<unsigned,unsigned>,std::pair<unsigned,int>> edges;
        std::vector<bool> on_surface(mesh.nodes.size());
        for (std::size_t i=0;i<mesh.surface_vertices.size();++i) {
            const auto &binding=mesh.surface_bindings[i];
            const auto p=mesh.surface_vertices[i];
            check(binding.weights[0]==1 && distance(p,mesh.nodes[binding.nodes[0]])==0,
                  "surface is not directly simulated");
            on_surface[binding.nodes[0]]=true;
            Vec3 reconstructed{}; float sum=0;
            const auto &origin=mesh.surface_sources[i];
            for (unsigned j=0;j<3;++j) {
                const auto old=vertices[origin.vertices[j]];
                reconstructed.x+=old.x*origin.weights[j];
                reconstructed.y+=old.y*origin.weights[j];
                reconstructed.z+=old.z*origin.weights[j];
                sum+=origin.weights[j];
            }
            check(std::abs(sum-1)<1e-6F && distance(p,reconstructed)<1e-6F, "bad attribute interpolation");
            check((mesh.inverse_masses[binding.nodes[0]]==0)==(p.x==1), "pin face not inherited");
        }
        double volume=0;
        for (std::size_t i=0;i<mesh.surface_triangle_indices.size();i+=3) {
            Vec3 p[3];
            for (unsigned j=0;j<3;++j) {
                const auto a=mesh.surface_triangle_indices[i+j], b=mesh.surface_triangle_indices[i+(j+1)%3];
                p[j]=mesh.surface_vertices[a];
                check(distance(p[j],mesh.surface_vertices[b])<=source.spacing*1.50001F, "surface edge exceeds lattice resolution");
                const auto na=mesh.surface_bindings[a].nodes[0], nb=mesh.surface_bindings[b].nodes[0];
                auto &edge=edges[std::minmax(na,nb)];
                ++edge.first; edge.second+=na<nb?1:-1;
            }
            volume+=(p[0].x*(p[1].y*p[2].z-p[1].z*p[2].y)+
                     p[0].y*(p[1].z*p[2].x-p[1].x*p[2].z)+
                     p[0].z*(p[1].x*p[2].y-p[1].y*p[2].x))/6.0;
        }
        for (const auto &[key,edge] : edges)
            check(edge.first==2 && edge.second==0, "crack or winding error in refined surface");
        check(std::abs(volume-1)<1e-5, "refinement changed enclosed shape");
        unsigned supported=0;
        for (const auto &bond : mesh.bonds)
            supported+=on_surface[bond.first]!=on_surface[bond.second];
        check(supported>100, "surface lacks interior supports");
        check(*std::min_element(degrees.begin(),degrees.end())>0, "isolated node");
        check(std::count(mesh.inverse_masses.begin(),mesh.inverse_masses.end(),0)>4, "only original corners pinned");
        auto resolved_source=source;
        resolved_source.spacing=1.0F;
        SoftBodyGeometry resolved;
        check(build_soft_body_geometry(resolved_source,resolved).ok(), "resolved surface rejected");
        check(resolved.surface_vertices.size()==vertices.size() &&
              resolved.surface_triangle_indices==indices, "already-resolved surface was refined unnecessarily");
        const auto original_count=mesh.nodes.size();
        source.maximum_nodes=4;
        check(build_soft_body_geometry(source,mesh).code==StatusCode::capacity_exceeded, "node capacity ignored");
        check(mesh.nodes.size()==original_count, "failure modified output");
        source.maximum_nodes=100000; source.maximum_surface_triangles=12;
        check(build_soft_body_geometry(source,mesh).code==StatusCode::capacity_exceeded, "surface capacity ignored");
        source.maximum_surface_triangles=200000; source.triangle_indices.size-=3;
        check(build_soft_body_geometry(source,mesh).code==StatusCode::invalid_argument, "open surface accepted");
        std::cout << "nodes=" << mesh.nodes.size() << " triangles=" << mesh.surface_triangle_indices.size()/3
                  << " supports=" << supported << '\n';
        return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}

// SPDX-License-Identifier: MIT
#if defined(PARALLEL_MATER_D3D12_GEOMETRY)
#include <parallel_mater/d3d12.hpp>
#define PM_GEOMETRY_NAMESPACE parallel_mater::d3d12
#define PM_GEOMETRY_SUCCESS 0
#elif defined(PARALLEL_MATER_METAL_GEOMETRY)
#include <parallel_mater/metal.hpp>
#define PM_GEOMETRY_NAMESPACE parallel_mater::metal
#define PM_GEOMETRY_SUCCESS 0
#else
#include <parallel_mater/parallel_mater.hpp>
#define PM_GEOMETRY_NAMESPACE parallel_mater
#define PM_GEOMETRY_SUCCESS cudaSuccess
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <vector>

namespace PM_GEOMETRY_NAMESPACE {
Status sample_fluid_source(ParticleSourceMesh mesh, std::vector<Vec3> &output) noexcept {
    const auto fail = [](const char *message) {
        return Status{StatusCode::invalid_argument, PM_GEOMETRY_SUCCESS, message};
    };
    if (!mesh.vertices.data || mesh.vertices.size < 3 ||
        !mesh.triangle_indices.data || mesh.triangle_indices.size < 3 ||
        mesh.triangle_indices.size % 3 || !std::isfinite(mesh.spacing) || mesh.spacing <= 0)
        return fail("invalid fluid source triangle mesh or spacing");
    try {
        using Cell = std::array<int, 3>;
        std::map<Cell, std::vector<Vec3>> cells;
        std::vector<Vec3> result;
        const double inverse = 1.0 / mesh.spacing;
        auto key = [inverse](Vec3 p) -> Cell {
            return {int(std::floor(p.x*inverse)), int(std::floor(p.y*inverse)),
                    int(std::floor(p.z*inverse))};
        };
        auto squared = [](Vec3 a, Vec3 b) -> double {
            const double x=double(a.x)-b.x, y=double(a.y)-b.y, z=double(a.z)-b.z;
            return x*x+y*y+z*z;
        };
        for (std::size_t i=0; i<mesh.vertices.size; ++i) {
            const auto p=mesh.vertices.data[i];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                std::max({std::abs(double(p.x)), std::abs(double(p.y)), std::abs(double(p.z))})*inverse > 1e8)
                return fail("fluid source vertex is nonfinite or outside sampling range");
        }
        std::size_t candidates=0;
        for (std::size_t t=0; t<mesh.triangle_indices.size; t+=3) {
            const auto ia=mesh.triangle_indices.data[t], ib=mesh.triangle_indices.data[t+1], ic=mesh.triangle_indices.data[t+2];
            if (ia>=mesh.vertices.size || ib>=mesh.vertices.size || ic>=mesh.vertices.size)
                return fail("fluid source triangle index is out of range");
            const auto a=mesh.vertices.data[ia], b=mesh.vertices.data[ib], c=mesh.vertices.data[ic];
            const double ux=double(b.x)-a.x, uy=double(b.y)-a.y, uz=double(b.z)-a.z;
            const double vx=double(c.x)-a.x, vy=double(c.y)-a.y, vz=double(c.z)-a.z;
            const double nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx;
            if (nx*nx+ny*ny+nz*nz < 1e-24) continue;
            const double divisions=std::ceil(std::sqrt(std::max({squared(a,b),squared(a,c),squared(b,c)}))*inverse*1.5);
            if (divisions>2000)
                return {StatusCode::capacity_exceeded,PM_GEOMETRY_SUCCESS,"fluid source subdivision budget exceeded"};
            const int n=std::max(1,int(divisions));
            candidates += std::size_t(n+1)*(n+2)/2;
            if (candidates>2'000'000)
                return {StatusCode::capacity_exceeded,PM_GEOMETRY_SUCCESS,"fluid source sample budget exceeded"};
            for (int i=0;i<=n;++i) for(int j=0;j<=n-i;++j) {
                const float u=float(i)/n, v=float(j)/n;
                const Vec3 p{a.x+u*(b.x-a.x)+v*(c.x-a.x),a.y+u*(b.y-a.y)+v*(c.y-a.y),a.z+u*(b.z-a.z)+v*(c.z-a.z)};
                const auto cell=key(p);
                bool occupied=false;
                for(int z=-1;z<=1 && !occupied;++z) for(int y=-1;y<=1 && !occupied;++y) for(int x=-1;x<=1 && !occupied;++x) {
                    auto it=cells.find({cell[0]+x,cell[1]+y,cell[2]+z});
                    if(it==cells.end()) continue;
                    for(auto q:it->second) if(squared(p,q)<double(mesh.spacing)*mesh.spacing) {occupied=true;break;}
                }
                if (occupied) continue;
                if (result.size()==65'536)
                    return {StatusCode::capacity_exceeded,PM_GEOMETRY_SUCCESS,"too many fluid source sites"};
                cells[cell].push_back(p);
                result.push_back(p);
            }
        }
        if(result.empty()) return fail("fluid source has no nondegenerate triangles");
        output.swap(result);
        return {};
    } catch (...) {
        return {StatusCode::out_of_memory,PM_GEOMETRY_SUCCESS,"fluid source sampling allocation failed"};
    }
}
} // namespace parallel_mater
#undef PM_GEOMETRY_SUCCESS
#undef PM_GEOMETRY_NAMESPACE

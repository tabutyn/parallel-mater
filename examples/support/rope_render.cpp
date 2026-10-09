// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include "vector_math.hpp"
#include <cmath>
#if defined(PARALLEL_MATER_GALLERY_METAL)
namespace parallel_mater::metal::gallery {
#elif defined(PARALLEL_MATER_GALLERY_D3D12)
namespace parallel_mater::d3d12::gallery {
#else
namespace parallel_mater::gallery {
#endif
void update_rope_render_mesh(const std::vector<Vec3> &nodes,float radius,TriangleMesh &mesh) {
    if(nodes.size()<2)return;
    constexpr unsigned sides=8;
    mesh.vertices.resize(nodes.size()*sides);
    auto unit=[](Vec3 v) {const float len=std::sqrt(math::dot(v,v));return len>1e-8F?math::multiply(v,1/len):Vec3{1,0,0};};
    Vec3 previous_axis{};
    for(unsigned i=0;i<nodes.size();++i) {
        const Vec3 tangent=unit(math::subtract(nodes[std::min<std::size_t>(i+1,nodes.size()-1)],nodes[i?i-1:0]));
        Vec3 axis=i?math::subtract(previous_axis,math::multiply(tangent,math::dot(previous_axis,tangent))):Vec3{};
        if(math::dot(axis,axis)<1e-6F)axis=math::cross(tangent,std::abs(tangent.y)<0.9F?Vec3{0,1,0}:Vec3{1,0,0});
        axis=unit(axis);previous_axis=axis;
        const auto other=math::cross(tangent,axis);
        for(unsigned j=0;j<sides;++j) {
            const float angle=6.28318530718F*j/sides;
            const Vec3 normal=math::add(math::multiply(axis,std::cos(angle)),math::multiply(other,std::sin(angle)));
            mesh.vertices[i*sides+j]={math::add(nodes[i],math::multiply(normal,radius)),normal,{float(i)/(nodes.size()-1),float(j)/sides}};
        }
    }
    if(!mesh.indices.empty())return;
    for(unsigned i=0;i+1<nodes.size();++i)for(unsigned j=0;j<sides;++j) {
        const unsigned a=i*sides+j,b=i*sides+(j+1)%sides,c=b+sides,d=a+sides;
        mesh.indices.insert(mesh.indices.end(),{a,b,c,a,c,d});
    }
    for(unsigned j=1;j+1<sides;++j) {
        const unsigned last=(nodes.size()-1)*sides;
        mesh.indices.insert(mesh.indices.end(),{0,j+1,j,last,last+j,last+j+1});
    }
}
#if defined(PARALLEL_MATER_GALLERY_METAL)
} // namespace parallel_mater::metal::gallery
#elif defined(PARALLEL_MATER_GALLERY_D3D12)
} // namespace parallel_mater::d3d12::gallery
#else
} // namespace parallel_mater::gallery
#endif

// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
namespace parallel_mater {
Status sample_rope_centerline(HostSpan<const Vec3> line, float spacing,
                             std::vector<Vec3> &output) noexcept {
    if (!line.data || line.size < 2 || !std::isfinite(spacing) || spacing <= 0)
        return {StatusCode::invalid_argument,cudaSuccess,"invalid rope centerline or spacing"};
    try {
        std::vector<double> arc(line.size,0);
        for(std::size_t i=0;i<line.size;++i) {
            const Vec3 p=line.data[i];
            if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))
                return {StatusCode::invalid_argument,cudaSuccess,"nonfinite rope centerline"};
            if(i) {
                const auto q=line.data[i-1];
                const double x=double(p.x)-q.x,y=double(p.y)-q.y,z=double(p.z)-q.z;
                arc[i]=arc[i-1]+std::sqrt(x*x+y*y+z*z);
            }
        }
        const double length=arc.back(), steps=std::ceil(length/spacing);
        if(length<1e-6 || steps>1023)
            return {StatusCode::invalid_argument,cudaSuccess,"rope needs positive length and at most 1024 nodes"};
        const auto n=std::max(1U,static_cast<unsigned>(steps));
        std::vector<Vec3> nodes;
        nodes.reserve(n+1);
        std::size_t edge=1;
        for(unsigned i=0;i<=n;++i) {
            const double s=length*i/n;
            while(edge+1<arc.size() && (arc[edge]<s || arc[edge]==arc[edge-1])) ++edge;
            const auto a=line.data[edge-1],b=line.data[edge];
            const float t=arc[edge]>arc[edge-1]?float((s-arc[edge-1])/(arc[edge]-arc[edge-1])):1;
            nodes.push_back({a.x+t*(b.x-a.x),a.y+t*(b.y-a.y),a.z+t*(b.z-a.z)});
        }
        nodes.front()=line.data[0];nodes.back()=line.data[line.size-1];
        output.swap(nodes);
        return {};
    } catch(...) {return {StatusCode::out_of_memory,cudaSuccess,"rope sampling allocation failed"};}
}
} // namespace parallel_mater

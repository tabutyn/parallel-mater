// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace parallel_mater {
namespace {
Vec3 add(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
Vec3 sub(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
Vec3 mul(Vec3 a, float s) { return {a.x*s, a.y*s, a.z*s}; }
float squared(Vec3 p) { return p.x*p.x+p.y*p.y+p.z*p.z; }
bool finite(Vec3 p) { return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z); }
std::uint64_t edge_key(std::uint32_t a, std::uint32_t b) {
    return (std::uint64_t(std::min(a,b))<<32U)|std::max(a,b);
}
SoftBodySurfaceSource midpoint_source(const SoftBodySurfaceSource &a,
                                      const SoftBodySurfaceSource &b) {
    SoftBodySurfaceSource result{};
    std::uint32_t count = 0;
    for (const auto *source : {&a,&b}) for (unsigned i=0;i<3;++i) {
        if (source->weights[i] == 0) continue;
        unsigned slot=0;
        while(slot<count && result.vertices[slot]!=source->vertices[i]) ++slot;
        if(slot==count) {
            if(count==3) throw std::logic_error("inconsistent surface provenance");
            result.vertices[count++]=source->vertices[i];
        }
        result.weights[slot]+=0.5F*source->weights[i];
    }
    return result;
}
} // namespace

Status build_soft_body_geometry(SoftBodyGeometrySource source,
                                SoftBodyGeometry &output) noexcept {
    const auto invalid = [](const char *message) {
        return Status{StatusCode::invalid_argument,cudaSuccess,message};
    };
    const auto capacity = [] {
        return Status{StatusCode::capacity_exceeded,cudaSuccess,
                      "soft-body geometry exceeds configured capacity"};
    };
    if(!source.vertices.data || source.vertices.size<4 || source.vertices.size>UINT32_MAX ||
       !source.triangle_indices.data || source.triangle_indices.size<12 ||
       source.triangle_indices.size%3 || source.triangle_indices.size>UINT32_MAX ||
       !std::isfinite(source.spacing) || source.spacing<=0 ||
       !std::isfinite(source.total_mass) || source.total_mass<=0 ||
       source.maximum_nodes<4 || source.maximum_surface_triangles<4 ||
       (source.pin_weights.size && (!source.pin_weights.data ||
                                    source.pin_weights.size!=source.vertices.size)))
        return invalid("invalid soft-body geometry source");
    try {
        SoftBodyGeometry result;
        result.surface_vertices.assign(source.vertices.data,source.vertices.data+source.vertices.size);
        result.surface_triangle_indices.assign(source.triangle_indices.data,
                                                source.triangle_indices.data+source.triangle_indices.size);
        result.surface_sources.resize(source.vertices.size);
        std::vector<float> pins(source.vertices.size,0);
        for(std::size_t i=0;i<source.vertices.size;++i) {
            if(!finite(source.vertices.data[i])) return invalid("nonfinite soft-body surface vertex");
            if(source.pin_weights.size) pins[i]=source.pin_weights.data[i];
            if(!std::isfinite(pins[i]) || pins[i]<0 || pins[i]>1)
                return invalid("invalid soft-body pin weight");
            result.surface_sources[i].vertices[0]=static_cast<std::uint32_t>(i);
            result.surface_sources[i].weights[0]=1;
        }
        for(auto index:result.surface_triangle_indices)
            if(index>=source.vertices.size) return invalid("soft-body triangle index outside vertex buffer");
        if(result.surface_triangle_indices.size()/3>source.maximum_surface_triangles) return capacity();

        // Check topology after welding attribute seams, not rendering indices.
        std::map<std::array<float,3>,std::uint32_t> source_nodes;
        std::vector<std::uint32_t> canonical(source.vertices.size);
        for(std::size_t i=0;i<source.vertices.size;++i) {
            const auto p=source.vertices.data[i];
            canonical[i]=source_nodes.emplace(std::array{p.x,p.y,p.z},source_nodes.size()).first->second;
        }
        std::unordered_map<std::uint64_t,std::pair<unsigned,int>> incidences;
        for(std::size_t i=0;i<source.triangle_indices.size;i+=3) {
            for(unsigned j=0;j<3;++j) {
                const auto a=canonical[source.triangle_indices.data[i+j]];
                const auto b=canonical[source.triangle_indices.data[i+(j+1)%3]];
                if(a==b) return invalid("soft-body surface contains a degenerate triangle");
                auto &edge=incidences[edge_key(a,b)];
                ++edge.first; edge.second+=a<b?1:-1;
            }
        }
        for(const auto &[key,edge]:incidences) {
            (void)key;
            if(edge.first!=2 || edge.second!=0)
                return invalid("soft-body surface must be closed and consistently wound");
        }

        // Validate the sampling domain before allocating a refined surface.
        std::vector<FluidParticle> samples;
        Status sampled=sample_fluid_geometry({source.vertices,source.triangle_indices,{}, {},source.spacing},samples);
        if(!sampled) return sampled;

        // Split every long edge on both incident faces in the same round.
        // Separate render indices at seams get bit-identical midpoint positions,
        // while keeping their own attribute provenance. No T-junctions.
        // Triangle diagonals can be longer than point spacing (sqrt(2) on
        // a square grid). Avoid refining an already lattice-resolution skin
        // into a denser, heavier spring graph than its interior.
        const float limit=source.spacing*source.spacing*2.25F*(1.0F+1.0e-6F);
        for(unsigned round=0;round<32;++round) {
            std::unordered_map<std::uint64_t,std::uint32_t> midpoints;
            const auto midpoint = [&](std::uint32_t a,std::uint32_t b) {
                if(squared(sub(result.surface_vertices[a],result.surface_vertices[b]))<=limit)
                    return UINT32_MAX;
                const auto key=edge_key(a,b);
                const auto found=midpoints.find(key);
                if(found!=midpoints.end()) return found->second;
                if(result.surface_vertices.size()>=UINT32_MAX) throw std::length_error("surface capacity");
                const auto vertex=static_cast<std::uint32_t>(result.surface_vertices.size());
                result.surface_vertices.push_back(mul(add(result.surface_vertices[a],result.surface_vertices[b]),0.5F));
                result.surface_sources.push_back(midpoint_source(result.surface_sources[a],result.surface_sources[b]));
                pins.push_back(0.5F*(pins[a]+pins[b]));
                midpoints.emplace(key,vertex);
                return vertex;
            };
            std::vector<std::uint32_t> triangles;
            triangles.reserve(result.surface_triangle_indices.size());
            const auto emit=[&](std::uint32_t a,std::uint32_t b,std::uint32_t c) {
                if(triangles.size()/3>=source.maximum_surface_triangles) throw std::length_error("surface capacity");
                triangles.insert(triangles.end(),{a,b,c});
            };
            for(std::size_t i=0;i<result.surface_triangle_indices.size();i+=3) {
                std::uint32_t v[3],m[3]; unsigned count=0;
                for(unsigned j=0;j<3;++j) v[j]=result.surface_triangle_indices[i+j];
                for(unsigned j=0;j<3;++j) { m[j]=midpoint(v[j],v[(j+1)%3]); count+=m[j]!=UINT32_MAX; }
                if(count==0) emit(v[0],v[1],v[2]);
                else if(count==3) {
                    emit(v[0],m[0],m[2]); emit(m[0],v[1],m[1]);
                    emit(m[2],m[1],v[2]); emit(m[0],m[1],m[2]);
                } else {
                    unsigned j=0;
                    if(count==1) {
                        while(m[j]==UINT32_MAX) ++j;
                        emit(v[j],m[j],v[(j+2)%3]); emit(m[j],v[(j+1)%3],v[(j+2)%3]);
                    } else {
                        while(m[j]==UINT32_MAX || m[(j+1)%3]==UINT32_MAX) ++j;
                        const auto a=v[j],b=v[(j+1)%3],c=v[(j+2)%3],ab=m[j],bc=m[(j+1)%3];
                        emit(b,bc,ab); emit(a,ab,c); emit(ab,bc,c);
                    }
                }
            }
            result.surface_triangle_indices.swap(triangles);
            if(midpoints.empty()) break;
            if(round==31) return capacity();
        }

        // Weld rendering seams into physical nodes. Every refined face vertex
        // is simulated, rather than being a decorative interpolation between
        // the original distant corners.
        std::map<std::array<float,3>,std::uint32_t> welded;
        std::vector<float> node_pins;
        result.surface_bindings.resize(result.surface_vertices.size());
        for(std::size_t i=0;i<result.surface_vertices.size();++i) {
            const auto p=result.surface_vertices[i];
            const auto [entry,inserted]=welded.emplace(std::array{p.x,p.y,p.z},result.nodes.size());
            if(inserted) { result.nodes.push_back(p); node_pins.push_back(pins[i]); }
            else node_pins[entry->second]=std::max(node_pins[entry->second],pins[i]);
            result.surface_bindings[i].nodes[0]=entry->second;
            result.surface_bindings[i].weights[0]=1;
        }
        if(result.nodes.size()>source.maximum_nodes) return capacity();
        const float duplicate_squared=source.spacing*source.spacing*0.04F;
        for(const auto &sample:samples) {
            if(std::any_of(result.nodes.begin(),result.nodes.end(),[&](Vec3 p) {
                return squared(sub(p,sample.position))<=duplicate_squared;
            })) continue;
            if(result.nodes.size()>=source.maximum_nodes) return capacity();
            result.nodes.push_back(sample.position); node_pins.push_back(0);
        }
        std::unordered_set<std::uint64_t> edges;
        std::vector<std::uint32_t> degrees(result.nodes.size(),0);
        const auto bond=[&](std::uint32_t a,std::uint32_t b) {
            if(a>b) std::swap(a,b);
            if(a==b || !edges.insert(edge_key(a,b)).second) return;
            const float rest=std::sqrt(squared(sub(result.nodes[a],result.nodes[b])));
            if(rest<=1.0e-6F) return;
            result.bonds.push_back({a,b,rest}); ++degrees[a]; ++degrees[b];
        };
        for(std::size_t i=0;i<result.surface_triangle_indices.size();i+=3)
            for(unsigned j=0;j<3;++j) bond(
                result.surface_bindings[result.surface_triangle_indices[i+j]].nodes[0],
                result.surface_bindings[result.surface_triangle_indices[i+(j+1)%3]].nodes[0]);
        const float reach=source.spacing*1.8F;
        using Cell=std::array<int,3>;
        const Vec3 origin=result.nodes.front();
        const auto cell=[&](Vec3 p) {
            p=sub(p,origin);
            return Cell{int(std::floor(p.x/reach)),int(std::floor(p.y/reach)),int(std::floor(p.z/reach))};
        };
        std::map<Cell,std::vector<std::uint32_t>> cells;
        std::vector<std::uint64_t> nearby;
        for(std::uint32_t i=0;i<result.nodes.size();++i) {
            const auto key=cell(result.nodes[i]);
            for(int x=-1;x<=1;++x) for(int y=-1;y<=1;++y) for(int z=-1;z<=1;++z) {
                const auto found=cells.find({key[0]+x,key[1]+y,key[2]+z});
                if(found==cells.end()) continue;
                for(auto neighbor:found->second)
                    if(squared(sub(result.nodes[i],result.nodes[neighbor]))<=reach*reach)
                        nearby.push_back(edge_key(neighbor,i));
            }
            cells[key].push_back(i);
        }
        // Preserve lexicographic spring order, independently of cell traversal.
        // Jacobi sums then remain reproducible for already-resolved surfaces.
        std::sort(nearby.begin(),nearby.end());
        for(const auto edge:nearby) bond(static_cast<std::uint32_t>(edge>>32U),static_cast<std::uint32_t>(edge));
        if(std::any_of(degrees.begin(),degrees.end(),[](auto degree){return degree==0;}))
            return invalid("soft-body geometry contains an unsupported node; reduce spacing");
        result.node_mass=source.total_mass/static_cast<float>(result.nodes.size());
        if (!std::isfinite(1.0F/result.node_mass))
            return invalid("soft-body mass per node is too small");
        result.inverse_masses.reserve(result.nodes.size());
        for(float pin:node_pins) result.inverse_masses.push_back(pin>=1.0F-1.0e-6F?0.0F:1.0F/result.node_mass);
        output=std::move(result);
        return {};
    } catch(const std::length_error &) { return capacity(); }
      catch(const std::logic_error &) { return {StatusCode::internal_error,cudaSuccess,"soft-body refinement provenance failed"}; }
      catch(...) { return {StatusCode::out_of_memory,cudaSuccess,"soft-body geometry allocation failed"}; }
}
} // namespace parallel_mater

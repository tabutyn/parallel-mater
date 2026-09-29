// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include "../../examples/support/vector_math.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace parallel_mater;
using namespace parallel_mater::gallery;
namespace math=parallel_mater::gallery::math;
static void check(bool condition,const char *message) {
    if(!condition)throw std::runtime_error(message);
}
static void check(Status status,const char *message) {
    if(!status)throw std::runtime_error(std::string(message)+": "+status.message);
}
template<class T>static std::vector<T> read(DeviceSpan<const T> span) {
    std::vector<T> output(span.size);
    check(cudaMemcpy(output.data(),span.data,span.size*sizeof(T),
        cudaMemcpyDeviceToHost)==cudaSuccess,"device read failed");
    return output;
}
static Vec3 closest_triangle(Vec3 point,Vec3 a,Vec3 b,Vec3 c) {
    const auto ab=math::subtract(b,a),ac=math::subtract(c,a);
    const auto ap=math::subtract(point,a);
    const float d1=math::dot(ab,ap),d2=math::dot(ac,ap);
    if(d1<=0 && d2<=0)return a;
    const auto bp=math::subtract(point,b);
    const float d3=math::dot(ab,bp),d4=math::dot(ac,bp);
    if(d3>=0 && d4<=d3)return b;
    const float vc=d1*d4-d3*d2;
    if(vc<=0 && d1>=0 && d3<=0)
        return math::add(a,math::multiply(ab,d1/(d1-d3)));
    const auto cp=math::subtract(point,c);
    const float d5=math::dot(ab,cp),d6=math::dot(ac,cp);
    if(d6>=0 && d5<=d6)return c;
    const float vb=d5*d2-d1*d6;
    if(vb<=0 && d2>=0 && d6<=0)
        return math::add(a,math::multiply(ac,d2/(d2-d6)));
    const float va=d3*d6-d5*d4;
    if(va<=0 && d4-d3>=0 && d5-d6>=0)
        return math::add(b,math::multiply(math::subtract(c,b),
            (d4-d3)/((d4-d3)+(d5-d6))));
    const float denominator=1.0F/(va+vb+vc);
    return math::add(a,math::add(math::multiply(ab,vb*denominator),
        math::multiply(ac,vc*denominator)));
}
int main(int argc,char **argv) {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0)return 77;
    try {
        SceneDefinition scene;std::string error;
        check(load_glb_scene(PARALLEL_MATER_ROPE_SOFT_BODY_SCENE_PATH,scene,error),
            error.c_str());
        check(scene.ropes.size()==1 && scene.soft_bodies.size()==1 &&
            scene.rigid_bodies.size()==3,"expected rope, soft post, rigid ball, and arena");
        const auto &rope=scene.ropes.front();
        check(rope.first_body>=0 && rope.last_soft_body>=0 &&
            rope.first_soft_body<0 && rope.last_body<0,
            "rigid ball and soft post Hooks must be imported");
        check(std::abs(scene.rigid_bodies[rope.first_body].options.mass-1.0F)<0.001F &&
            scene.soft_bodies[rope.last_soft_body].solver_iterations==8U &&
            std::abs(scene.soft_bodies[rope.last_soft_body].shape_matching_stiffness-0.9F)<0.001F &&
            rope.options.solver_iterations==4U,
            "authored dry-rope and soft-post settings were not exported");
        if(argc>3)scene.soft_bodies[rope.last_soft_body].solver_iterations=
            static_cast<std::uint32_t>(std::stoul(argv[3]));
        if(argc>4)scene.ropes[0].options.solver_iterations=
            static_cast<std::uint32_t>(std::stoul(argv[4]));
        if(argc>5)scene.soft_bodies[rope.last_soft_body].shape_matching_stiffness=
            std::stof(argv[5]);
        {
            const auto &mesh=scene.meshes[scene.soft_bodies[rope.last_soft_body].mesh_index];
            const auto origin=mesh.vertices[0].position;
            double volume=0;
            for(unsigned i=0;i<mesh.indices.size();i+=3) {
                const auto a=math::subtract(mesh.vertices[mesh.indices[i]].position,origin),
                           b=math::subtract(mesh.vertices[mesh.indices[i+1]].position,origin),
                           c=math::subtract(mesh.vertices[mesh.indices[i+2]].position,origin);
                volume+=math::dot(a,math::cross(b,c))/6.0;
            }
            std::cout<<"post_signed_volume="<<volume<<'\n';
        }
        check(std::count(scene.soft_bodies[rope.last_soft_body].inverse_masses.begin(),
            scene.soft_bodies[rope.last_soft_body].inverse_masses.end(),0.0F)>0,
            "post base must be pinned");
        std::cout<<"post_nodes="<<scene.soft_bodies[rope.last_soft_body].nodes.size()
            <<" post_triangles="<<scene.meshes[scene.soft_bodies[rope.last_soft_body].mesh_index].indices.size()/3U
            <<" fixed_nodes="<<std::count(
                scene.soft_bodies[rope.last_soft_body].inverse_masses.begin(),
                scene.soft_bodies[rope.last_soft_body].inverse_masses.end(),0.0F)<<'\n';
        World world;SceneInstance instance;
        check(create_scene_world(scene,world,instance),"instantiate rope soft-body scene");
        check(instance.rope_soft_body_couplings.size()==1,"post coupling registered");
        check(!world.remove_rope(instance.ropes[0]),"coupled rope removed");
        check(!world.remove_soft_body(instance.soft_bodies[rope.last_soft_body]),
            "coupled soft body removed");
        RopeSoftBodyCouplingId duplicate{};
        check(!world.add_rope_soft_body_coupling({.rope=instance.ropes[0],
            .soft_body=instance.soft_bodies[rope.last_soft_body]},duplicate),
            "duplicate coupling accepted");
        auto valid=RopeSoftBodyCouplingOptions{.rope=instance.ropes[0],
            .soft_body=instance.soft_bodies[rope.last_soft_body],.attach_last=true};
        auto invalid=valid;invalid.maximum_soft_body_acceleration=0;
        check(!world.update_rope_soft_body_coupling(
            instance.rope_soft_body_couplings[0],invalid),
            "invalid acceleration accepted");
        check(world.update_rope_soft_body_coupling(
            instance.rope_soft_body_couplings[0],valid),
            "update coupling");
        const int frames=argc>1?std::stoi(argv[1]):120;
        const bool wind=argc>2 && std::string(argv[2])=="--wind";
        float maximum_strain=0,maximum_speed=0,maximum_anchor_distance=0;
        unsigned peak_edge=0;Vec3 peak_a{},peak_b{};
        float peak_winding=0,last_winding=0;
        float ball_turns=0;
        Vec3 previous_ball_offset{};
        float peak_soft_reaction=0;
        double gpu_ms=0,rope_ms=0,soft_ms=0;
        unsigned total_contacts=0;
        const auto center_of=[&](unsigned index) {
            SoftBodyDeviceView view{};
            check(world.soft_body_view(instance.soft_bodies[index],view),"soft center view");
            const auto skin=read(view.surface_positions);
            Vec3 center{};
            for(const auto point:skin)center=math::add(center,point);
            return math::multiply(center,1.0F/skin.size());
        };
        const auto &post_mesh=scene.meshes[scene.soft_bodies[rope.last_soft_body].mesh_index];
        float rest_top=-1.0e9F,rest_bottom=1.0e9F;
        for(const auto &vertex:post_mesh.vertices) {
            rest_top=std::max(rest_top,vertex.position.y);
            rest_bottom=std::min(rest_bottom,vertex.position.y);
        }
        std::vector<unsigned> top_vertices,bottom_vertices;
        for(unsigned i=0;i<post_mesh.vertices.size();++i) {
            if(post_mesh.vertices[i].position.y>rest_top-0.001F)
                top_vertices.push_back(i);
            if(post_mesh.vertices[i].position.y<rest_bottom+0.001F)
                bottom_vertices.push_back(i);
        }
        check(!top_vertices.empty() && !bottom_vertices.empty(),"post caps missing");
        Vec3 rest_base{};
        for(unsigned index:bottom_vertices)
            rest_base=math::add(rest_base,post_mesh.vertices[index].position);
        rest_base=math::multiply(rest_base,1.0F/bottom_vertices.size());
        float post_radius=0;
        for(unsigned index:bottom_vertices)
            post_radius=std::max(post_radius,std::hypot(
                post_mesh.vertices[index].position.x-rest_base.x,
                post_mesh.vertices[index].position.z-rest_base.z));
        float minimum_clearance=100;
        float clearance_to_skin=100;
        int minimum_clearance_frame=-1;
        float maximum_post_bend=0,maximum_base_drift=0,minimum_post_height=100;
        unsigned minimum_segment=0,minimum_sample=0;
        Vec3 minimum_point{};
        for(int frame=0;frame<frames;++frame) {
            const float angle=std::max(0,frame-40)*0.025F;
            Vec3 gravity=(wind || frame<40)?Vec3{0,-9.81F,0}:
                Vec3{6.0F*std::cos(angle),-7.8F,6.0F*std::sin(angle)};
            if(wind && frame>=120) {
                RigidBodyState ball_state{};
                check(world.read_rigid_body_state(instance.rigid_bodies[rope.first_body],
                    ball_state),"read rigid ball");
                const auto ball=ball_state.position,
                           post=center_of(rope.last_soft_body);
                const auto delta=math::subtract(ball,post);
                const float radius=std::max(0.01F,std::hypot(delta.x,delta.z));
                gravity={(-6.0F*delta.z-3.0F*delta.x)/radius,-6.9367F,
                    (6.0F*delta.x-3.0F*delta.z)/radius};
            }
            check(world.step({.timestep=1.0F/60.0F,.substeps=4,
                .gravity=gravity,.collect_kernel_timings=true}),"step rope soft-body scene");
            RopeDeviceView view{};
            check(world.rope_view(instance.ropes[0],view),"rope view");
            const auto positions=read(view.positions), velocities=read(view.velocities);
            const auto rest=read(view.rest_lengths);
            if(frame==0) {
                float length=0;for(float edge:rest)length+=edge;
                std::cout<<"rest_length="<<length<<" nodes="<<positions.size()<<'\n';
            }
            for(unsigned i=0;i<positions.size();++i) {
                check(std::isfinite(math::length(positions[i])) &&
                    math::length(positions[i])<20.0F,"rope became nonfinite or escaped");
                maximum_speed=std::max(maximum_speed,math::length(velocities[i]));
                if(i) {
                    const float strain=std::abs(math::length(
                        math::subtract(positions[i],positions[i-1]))/rest[i-1]-1);
                    if(strain>maximum_strain) {
                        maximum_strain=strain;peak_edge=i-1;
                        peak_a=positions[i-1];peak_b=positions[i];
                    }
                }
            }
            SoftBodyDeviceView soft{};
            check(world.soft_body_view(instance.soft_bodies[rope.last_soft_body],soft),
                "soft view");
            const auto skin=read(soft.surface_positions);
            const auto centroid=[&](const std::vector<unsigned> &vertices) {
                Vec3 result{};
                for(unsigned index:vertices)result=math::add(result,skin[index]);
                return math::multiply(result,1.0F/vertices.size());
            };
            const auto top=centroid(top_vertices),bottom=centroid(bottom_vertices);
            const float post_bend=std::hypot(top.x-bottom.x,top.z-bottom.z);
            maximum_post_bend=std::max(maximum_post_bend,post_bend);
            maximum_base_drift=std::max(maximum_base_drift,
                math::length(math::subtract(bottom,rest_base)));
            minimum_post_height=std::min(minimum_post_height,top.y-bottom.y);
            for(unsigned i=0;i+1<positions.size();++i)
                for(unsigned sample=0;sample<=8;++sample) {
                    if(i+3>=positions.size())continue;
                    const float fraction=float(sample)/8;
                    const auto point=math::add(positions[i],math::multiply(
                        math::subtract(positions[i+1],positions[i]),fraction));
                    if(point.y<bottom.y+view.radius ||
                       point.y>top.y-view.radius)continue;
                    const float height_fraction=(point.y-bottom.y)/
                        std::max(top.y-bottom.y,1.0e-5F);
                    const auto axis=math::add(bottom,math::multiply(
                        math::subtract(top,bottom),height_fraction));
                    const float clearance=std::hypot(
                        point.x-axis.x,point.z-axis.z)-post_radius;
                    if(clearance<minimum_clearance) {
                        minimum_clearance=clearance;
                        minimum_segment=i;minimum_sample=sample;
                        minimum_point=point;
                        minimum_clearance_frame=frame;
                    }
                }
            if(minimum_clearance_frame==frame) {
                float nearest=100;
                for(unsigned i=0;i<post_mesh.indices.size();i+=3) {
                    const auto a=skin[post_mesh.indices[i]],
                               b=skin[post_mesh.indices[i+1]],
                               c=skin[post_mesh.indices[i+2]];
                    const auto point=closest_triangle(minimum_point,a,b,c);
                    const auto delta=math::subtract(minimum_point,point);
                    const float distance=math::length(delta);
                    if(distance>=nearest)continue;
                    nearest=distance;
                    const auto normal=math::normalize_or(math::cross(
                        math::subtract(b,a),math::subtract(c,a)),{});
                    clearance_to_skin=(math::dot(delta,normal)<0?-distance:distance)-
                        view.radius;
                }
            }
            RigidBodyState ball_state{};
            check(world.read_rigid_body_state(instance.rigid_bodies[rope.first_body],
                ball_state),"read ball position");
            const Vec3 ball_offset=math::subtract(ball_state.position,
                center_of(rope.last_soft_body));
            if(frame)ball_turns+=std::atan2(
                previous_ball_offset.x*ball_offset.z-
                    previous_ball_offset.z*ball_offset.x,
                previous_ball_offset.x*ball_offset.x+
                    previous_ball_offset.z*ball_offset.z)/6.28318530718F;
            previous_ball_offset=ball_offset;
            for(const auto force:read(soft.rope_contact_forces))
                peak_soft_reaction=std::max(peak_soft_reaction,math::length(force));
            float minimum=100;
            for(unsigned i=0;i<post_mesh.indices.size();i+=3)
                minimum=std::min(minimum,math::length(
                    math::subtract(positions.back(),closest_triangle(
                        positions.back(),skin[post_mesh.indices[i]],
                        skin[post_mesh.indices[i+1]],skin[post_mesh.indices[i+2]]))));
            maximum_anchor_distance=std::max(maximum_anchor_distance,minimum);
            WorldStatistics stats{};
            check(world.collect_statistics(stats),"rope soft statistics");
            total_contacts+=stats.rope_soft_body_contact_count;
            check(std::isfinite(stats.maximum_rope_soft_body_penetration),
                "nonfinite rope soft-body penetration");
            WorldStepTimings timing{};
            check(world.collect_step_timings(timing),"rope soft timing");
            gpu_ms+=timing.total_gpu_milliseconds;
            rope_ms+=timing.rope_solve.total_milliseconds;
            soft_ms+=timing.rope_soft_body_contacts.total_milliseconds;
            const auto post=center_of(rope.last_soft_body);
            float winding=0;
            for(unsigned i=1;i<positions.size();++i) {
                const auto a=math::subtract(positions[i-1],post),
                           b=math::subtract(positions[i],post);
                winding+=std::atan2(a.x*b.z-a.z*b.x,a.x*b.x+a.z*b.z);
            }
            last_winding=std::abs(winding)/6.28318530718F;
            peak_winding=std::max(peak_winding,last_winding);
            if(frame%30==29)std::cout<<"frame="<<frame+1<<" strain="<<maximum_strain
                <<" anchor_distance="<<maximum_anchor_distance
                <<" contacts="<<total_contacts<<" turns="<<last_winding
                <<" gpu_ms="<<gpu_ms/(frame+1)<<" peak_edge="<<peak_edge
                <<" a="<<peak_a.x<<','<<peak_a.y<<','<<peak_a.z
                <<" b="<<peak_b.x<<','<<peak_b.y<<','<<peak_b.z<<'\n';
            if(frame%30==29)std::cout<<"post_bend="<<post_bend
                <<" post_height="<<top.y-bottom.y
                <<" ball="<<ball_state.position.x<<','<<ball_state.position.y
                <<','<<ball_state.position.z<<" ball_turns="<<ball_turns
                <<" clearance="<<minimum_clearance
                <<" segment="<<minimum_segment<<" sample="<<minimum_sample
                <<" point="<<minimum_point.x<<','<<minimum_point.y<<','
                <<minimum_point.z<<'\n';
        }
        std::cout<<"maximum_strain="<<maximum_strain<<" maximum_speed="
            <<maximum_speed<<" maximum_anchor_distance="<<maximum_anchor_distance
            <<" contacts="<<total_contacts<<" peak_turns="<<peak_winding
            <<" last_turns="<<last_winding<<" gpu_ms="<<gpu_ms/frames
            <<" rope_ms="<<rope_ms/frames<<" soft_ms="<<soft_ms/frames
            <<" soft_reaction="<<peak_soft_reaction<<'\n';
        std::cout<<"maximum_post_bend="<<maximum_post_bend
            <<" maximum_base_drift="<<maximum_base_drift
            <<" minimum_post_height="<<minimum_post_height
            <<" minimum_clearance="<<minimum_clearance
            <<" minimum_clearance_frame="<<minimum_clearance_frame
            <<" clearance_to_skin="<<clearance_to_skin<<'\n';
        check(maximum_speed<15.0F,"rope became unstable");
        check(maximum_strain<0.02F,"rope stretched excessively");
        check(maximum_anchor_distance<0.02F,"soft Hook drifted from its surface");
        check(maximum_post_bend<0.25F && maximum_base_drift<0.03F &&
            minimum_post_height>0.50F,
            "soft post collapsed or uprooted");
        if(wind) {
            check(peak_winding>2.8F,"rope failed to wind three times around the soft post");
            check(clearance_to_skin>-0.003F,"rope crossed the soft post");
        }
        check(total_contacts>0 && peak_soft_reaction>0,
            "rope did not transfer contact force to the soft post");
        for(auto coupling:instance.rope_soft_body_couplings)
            check(world.remove_rope_soft_body_coupling(coupling),"remove coupling");
        check(world.remove_rope(instance.ropes[0]),"remove uncoupled rope");
        return 0;
    }catch(const std::exception &exception){
        std::cerr<<exception.what()<<'\n';return 1;
    }
}

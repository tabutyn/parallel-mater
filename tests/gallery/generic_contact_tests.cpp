// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include "rigid_mesh_checks.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace parallel_mater;
using namespace parallel_mater::gallery;
using namespace parallel_mater::gallery::math;

void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void require(Status status) {
    check(bool(status), status.message ? status.message : "physics operation failed");
}

Vec3 inertia_of(const SceneDefinition &scene, const RigidBodyDefinition &body) {
    Vec3 low{1.e30F,1.e30F,1.e30F}, high{-1.e30F,-1.e30F,-1.e30F};
    for (auto index : body.mesh_indices) for (const auto &vertex : scene.meshes[index].vertices) {
        const auto v = vertex.position;
        low = {std::min(low.x,v.x),std::min(low.y,v.y),std::min(low.z,v.z)};
        high = {std::max(high.x,v.x),std::max(high.y,v.y),std::max(high.z,v.z)};
    }
    const auto size = subtract(high,low);
    const float scale = body.options.mass / 12.0F;
    return {scale*(size.y*size.y+size.z*size.z),scale*(size.x*size.x+size.z*size.z),
            scale*(size.x*size.x+size.y*size.y)};
}

void run_drop(unsigned substeps, bool sphere_first) {
    SceneDefinition scene;
    std::string error;
    check(load_glb_scene(PARALLEL_MATER_GENERIC_TRAY_SCENE_PATH,scene,error),error.c_str());
    const auto index_of = [&](const char *name) {
        auto it=std::find_if(scene.rigid_bodies.begin(),scene.rigid_bodies.end(),
            [&](const auto &body){return body.source_name==name;});
        check(it!=scene.rigid_bodies.end(),"Generic scene body missing");
        return std::size_t(it-scene.rigid_bodies.begin());
    };
    auto sphere=index_of("GenericSphere"), tray=index_of("GenericBlockB");
    if(sphere_first) {
        std::swap(scene.rigid_bodies[sphere],scene.rigid_bodies[tray]);
        for(auto &joint:scene.rigid_constraints) {
            const auto remap=[&](std::size_t index){return index==sphere?tray:index==tray?sphere:index;};
            joint.body_a=remap(joint.body_a);joint.body_b=remap(joint.body_b);
        }
        std::swap(sphere,tray);
    }
    World world;
    SceneInstance instance;
    require(create_scene_world(scene,world,instance));
    const rigid_mesh_checks::MeshSamples sphere_mesh(scene,sphere),tray_mesh(scene,tray);
    std::vector<Vec3> inertia;
    for(const auto &body:scene.rigid_bodies) inertia.push_back(inertia_of(scene,body));
    float maximum_penetration=0;
    double energy_at_30=0,maximum_later_gain=0;
    for(unsigned frame=1;frame<=300;++frame) {
        require(world.step({.timestep=1.0F/60,.substeps=substeps,.gravity={0,-9.81F,0},
                            .collect_rigid_contacts=true}));
        std::vector<RigidBodyState> states(instance.rigid_bodies.size());
        double energy=0;
        for(std::size_t i=0;i<states.size();++i) {
            require(world.read_rigid_body_state(instance.rigid_bodies[i],states[i]));
            const auto &s=states[i];const auto &body=scene.rigid_bodies[i];
            if(body.options.motion!=MotionType::dynamic)continue;
            const auto q=s.orientation;
            const auto w=rigid_mesh_checks::rotate({-q.x,-q.y,-q.z,q.w},s.angular_velocity);
            energy+=body.options.mass*(9.81*s.position.y+0.5*dot(s.linear_velocity,s.linear_velocity))+
                0.5*(inertia[i].x*w.x*w.x+inertia[i].y*w.y*w.y+inertia[i].z*w.z*w.z);
        }
        check(std::isfinite(energy),"Generic scene must remain finite");
        const float penetration=std::max(
            sphere_mesh.penetration(states[sphere],tray_mesh,states[tray]),
            tray_mesh.penetration(states[tray],sphere_mesh,states[sphere]));
        maximum_penetration=std::max(maximum_penetration,penetration);
        if(penetration>0.002F) {
            std::cerr<<"Generic penetration "<<penetration<<" m at frame "<<frame
                     <<", substeps "<<substeps<<", sphere first "<<sphere_first<<'\n';
            check(false,"Sphere must not cross GenericBlockB's actual collision surface");
        }
        if(frame==30)energy_at_30=energy;
        if(frame>30)maximum_later_gain=std::max(maximum_later_gain,energy-energy_at_30);
    }
    std::cout<<"Generic substeps="<<substeps<<" sphere_first="<<sphere_first
             <<" penetration="<<maximum_penetration<<" energy_gain="<<maximum_later_gain<<'\n';
    check(maximum_later_gain<0.10,"Contact recovery must not propel sphere uphill by adding energy");
}
} // namespace

int main() try {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices==0)return 77;
    for(unsigned substeps:{4U,8U,16U})for(bool reversed:{false,true})run_drop(substeps,reversed);
    return 0;
} catch(const std::exception &error) {
    std::cerr<<"FAIL: "<<error.what()<<'\n';return 1;
}

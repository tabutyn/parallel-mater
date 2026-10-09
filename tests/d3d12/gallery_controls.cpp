// SPDX-License-Identifier: MIT
#include "../../examples/d3d12_gallery/ui.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
    using namespace parallel_mater::d3d12::viewer;
    InputState input;
    auto require=[](bool condition,const char *message) {
        if(!condition) { std::cerr<<message<<'\n';std::exit(1); }
    };
    input.key(Key::fps);
    require(input.show_fps,"F must show FPS");
    input.key(Key::fps);
    require(!input.show_fps,"F must hide FPS");
    input.key(Key::catalog);
    require(input.catalog_visible,"Tab must open catalog");
    input.key(Key::next); input.key(Key::enter);
    require(input.requested_scene==1U && !input.catalog_visible,"Enter must load selected scene");
    input.requested_scene.reset();
    input.key(Key::catalog); input.selected=rigid_scene_count;
    input.key(Key::enter);
    require(!input.requested_scene && input.catalog_visible,"Unavailable scene must not load");
    input.key(Key::escape);
    require(!input.catalog_visible && !input.close_requested,"Escape must close catalog first");
    input.current=3; input.key(Key::reset);
    require(input.requested_scene==3U,"R must reset current scene");
    input.key(Key::pause); require(input.paused,"P must pause");
    input.arrow(ArrowKey::right,true);
    input.arrow(ArrowKey::up,true);
    require(input.right_input()==1.0F && input.up_input()==1.0F,
            "Held arrow state must drive physics");
    input.arrow(ArrowKey::right,false);
    require(input.right_input()==0.0F && input.up_input()==1.0F,
            "Released arrow state must stop that axis");
    input.key(Key::action);
    require(input.scene_action,"Space must request the current scene action");
    const auto camera=input.camera.camera();
    const auto tilted=control_gravity(
        ::parallel_mater::gallery::GalleryControlPolicy::rigid_gravity,
        false,camera,1.0F,0.0F,1.0F);
    require(std::fabs(tilted.x)>0.1F || std::fabs(tilted.z)>0.1F,
            "Rigid arrow input must tilt gravity");
    const auto forced=control_gravity(
        ::parallel_mater::gallery::GalleryControlPolicy::rigid_gravity,
        true,camera,1.0F,0.0F,1.0F);
    require(forced.x==0.0F && forced.y==-9.81F && forced.z==0.0F,
            "Authored force scenes must keep vertical gravity");
    input.key(Key::escape); require(input.close_requested,"Escape must close viewer");
    std::cout<<"Gallery controls passed\n";
}

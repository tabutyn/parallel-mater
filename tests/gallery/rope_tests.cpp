// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/scene.hpp>
#include "../../examples/support/vector_math.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace parallel_mater;
using namespace parallel_mater::gallery;
namespace math=parallel_mater::gallery::math;
void check(bool b,const char *m){if(!b)throw std::runtime_error(m);}
void check(Status s,const char *m){if(!s)throw std::runtime_error(std::string(m)+": "+s.message);}
Vec3 rotate(Quaternion q,Vec3 v){const Vec3 a{q.x,q.y,q.z};auto t=math::multiply(math::cross(a,v),2);return math::add(v,math::add(math::multiply(t,q.w),math::cross(a,t)));}
template<class T> std::vector<T> read(DeviceSpan<const T> s){std::vector<T> v(s.size);check(cudaMemcpy(v.data(),s.data,s.size*sizeof(T),cudaMemcpyDeviceToHost)==cudaSuccess,"read");return v;}
int main(int argc,char **argv){
 int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
 try{
  SceneDefinition scene;std::string error;
  check(load_glb_scene(PARALLEL_MATER_ROPE_SCENE_PATH,scene,error),error.c_str());
  check(scene.ropes.size()==1 && scene.rigid_bodies.size()==4,"authored rope systems");
  auto &rope=scene.ropes[0];
  if(argc>3)rope.options.node_spacing=std::stof(argv[3]);
  if(argc>4)rope.options.solver_iterations=std::stoul(argv[4]);
  if(argc>5)rope.options.velocity_damping=std::stof(argv[5]);
  if(argc>2 && std::string(argv[2])=="--no-self")rope.options.self_collision=false;
  const bool wrapped=argc>2 && std::string(argv[2])=="--wrapped";
  const bool release=argc>2 && std::string(argv[2])=="--settle-after-motion";
  const bool settle=release || (argc>2 && std::string(argv[2])=="--settle");
  if(argc>2 && std::string(argv[2])=="--no-enclosure") {
   for(const auto &body:scene.rigid_bodies)if(body.source_name=="Plane")for(auto index:body.mesh_indices)
    for(auto &v:scene.meshes[index].vertices)v.position.y+=1000;
  }
  if(argc>2 && std::string(argv[2])=="--no-contacts") {
   rope.options.self_collision=false;
   for(const auto &body:scene.rigid_bodies)for(auto index:body.mesh_indices)
    for(auto &v:scene.meshes[index].vertices)v.position.y+=1000;
  }
  if(wrapped) {
   const auto post=scene.rigid_bodies[rope.last_body].options.initial_state.position;
   auto &ball=scene.rigid_bodies[rope.first_body].options.initial_state;
   ball.position={post.x+0.45F,0.20F,post.z};ball.orientation={};
   rope.options.first.local_anchor={-0.17F,0,0};
   rope.centerline.clear();rope.centerline.push_back({post.x+0.28F,0.20F,post.z});
   for(unsigned i=0;i<=360;++i){const float t=float(i)/360,a=18.84955592F*t;
    rope.centerline.push_back({post.x+0.078F*std::cos(a),0.20F+0.30F*t,post.z+0.078F*std::sin(a)});}
   rope.options.last.local_anchor=math::subtract(rope.centerline.back(),post);
  }
  check(rope.options.first.enabled&&rope.options.last.enabled,"both Hooks imported");
  World world;SceneInstance instance;check(create_scene_world(scene,world,instance),"create rope scene");
  check(!world.remove_rigid_body(instance.rigid_bodies[rope.first_body]),"attached body cannot be removed");
  RopeDeviceView view;check(world.rope_view(instance.ropes[0],view),"view rope");
  const auto rest=read(view.rest_lengths);
  struct Plane {Vec3 normal;float offset;};
  std::vector<Plane> ball_planes;
  const auto &ball_mesh=scene.meshes[scene.rigid_bodies[rope.first_body].mesh_indices[0]];
  for(unsigned i=0;i<ball_mesh.indices.size();i+=3) {
   const auto a=ball_mesh.vertices[ball_mesh.indices[i]].position,
              b=ball_mesh.vertices[ball_mesh.indices[i+1]].position,
              c=ball_mesh.vertices[ball_mesh.indices[i+2]].position;
   auto normal=math::normalize_or(math::cross(math::subtract(b,a),math::subtract(c,a)),{});
   if(math::dot(normal,a)<0)normal=math::multiply(normal,-1);
   ball_planes.push_back({normal,math::dot(normal,a)});
  }
  float max_ball_penetration=0;
  float max_strain=0,anchor_error=0,maximum_speed=0,min_clearance=100,min_winding=100;
  double gpu=0,rope_gpu=0;
  int post=-1;for(unsigned i=0;i<scene.rigid_bodies.size();++i)if(scene.rigid_bodies[i].source_name=="Cylinder")post=i;
  check(post>=0,"post imported");
  const auto post_state=scene.rigid_bodies[post].options.initial_state;
  const auto &mesh=scene.meshes[scene.rigid_bodies[post].mesh_indices[0]];
  float radius=0,lo=1e9,hi=-1e9;
  for(auto v:mesh.vertices){radius=std::max(radius,std::hypot(v.position.x,v.position.z));lo=std::min(lo,v.position.y+post_state.position.y);hi=std::max(hi,v.position.y+post_state.position.y);}
  const int frames=argc>1?std::stoi(argv[1]):600;
  float late_max_speed=0,late_rms_speed=0,late_displacement=0;
  unsigned late_count=0,ground_nodes=0;
  unsigned late_peak_node=0;Vec3 late_peak_position{};
  std::vector<Vec3> settling_start;
  for(int frame=0;frame<frames;++frame){
   Vec3 gravity{0,-9.81F,0};
   if(!settle && frame>=120){const float a=(frame-120)/60.0F;gravity={6.9367F*std::sin(a),-6.9367F,6.9367F*std::cos(a)};}
   if(wrapped)gravity={frame>=120?6.9367F:0,-9.81F,0};
   if(release && frame>=120 && frame<240)gravity={0,-6.9367F,6.9367F};
   check(world.step({.timestep=1.0F/60,.substeps=argc>6?unsigned(std::stoul(argv[6])):4U,.gravity=gravity,.collect_kernel_timings=true}),"step rope");
   check(world.rope_view(instance.ropes[0],view),"rope view");auto p=read(view.positions),v=read(view.velocities);
   if(settle && frame>=frames-120) {
    if(settling_start.empty())settling_start=p;
    ground_nodes=0;
    for(unsigned i=1;i+1<p.size();++i) {
     if(math::length(v[i])>late_max_speed){late_max_speed=math::length(v[i]);late_peak_node=i;late_peak_position=p[i];}
     late_rms_speed+=math::dot(v[i],v[i]);++late_count;
     late_displacement=std::max(late_displacement,math::length(math::subtract(p[i],settling_start[i])));
     ground_nodes+=p[i].y<0.025F;
    }
   }
   float winding=0;
   for(unsigned i=0;i<p.size();++i){if(!(std::isfinite(math::length(p[i]))&&math::length(p[i])<20))std::cout<<"invalid frame="<<frame<<" node="<<i<<" p="<<p[i].x<<','<<p[i].y<<','<<p[i].z<<" v="<<v[i].x<<','<<v[i].y<<','<<v[i].z<<std::endl;check(std::isfinite(math::length(p[i]))&&math::length(p[i])<20,"rope finite and bounded");maximum_speed=std::max(maximum_speed,math::length(v[i]));
    if(i){const float strain=std::abs(math::length(math::subtract(p[i],p[i-1]))/rest[i-1]-1);
      if(strain>max_strain){max_strain=strain;if(strain>0.03F)std::cout<<"new_peak frame="<<frame<<" edge="<<i-1<<" strain="<<strain<<" p="<<p[i-1].x<<','<<p[i-1].y<<','<<p[i-1].z<<" q="<<p[i].x<<','<<p[i].y<<','<<p[i].z<<std::endl;}}
    if(i>2&&i+3<p.size()&&p[i].y>lo+view.radius&&p[i].y<hi-view.radius)
      min_clearance=std::min(min_clearance,std::hypot(p[i].x-post_state.position.x,p[i].z-post_state.position.z)-radius);
    if(i){const auto a=math::subtract(p[i-1],post_state.position),b=math::subtract(p[i],post_state.position);
     winding+=std::atan2(a.x*b.z-a.z*b.x,a.x*b.x+a.z*b.z);}
   }
   min_winding=std::min(min_winding,std::abs(winding)/6.28318530718F);
   for(int end=0;end<2;++end){const auto anchor=end?rope.options.last:rope.options.first;RigidBodyState state;
    check(world.read_rigid_body_state(instance.rigid_bodies[end?rope.last_body:rope.first_body],state),"anchor state");
    const auto target=math::add(state.position,rotate(state.orientation,anchor.local_anchor));
    anchor_error=std::max(anchor_error,math::length(math::subtract(target,end?p.back():p.front())));
    if(!end)for(unsigned i=3;i+1<p.size();++i) {
     const auto q=state.orientation;
     const auto local=rotate({-q.x,-q.y,-q.z,q.w},math::subtract(p[i],state.position));
     float side=-1e9;
     for(const auto &plane:ball_planes)side=std::max(side,math::dot(plane.normal,local)-plane.offset);
     if(-side>max_ball_penetration){max_ball_penetration=-side;
      if(max_ball_penetration>0.003F)std::cout<<"ball_penetration frame="<<frame<<" node="<<i<<" depth="<<max_ball_penetration<<" ball="<<state.position.x<<','<<state.position.y<<','<<state.position.z<<std::endl;}
    }
   }
   // Check each capsule centerline, not only its endpoints: a chord can cut
   // into the post even when both sampled nodes are outside it.
   for(unsigned i=3;i+4<p.size();++i)for(unsigned sample=1;sample<8;++sample) {
    const auto point=math::add(p[i],math::multiply(math::subtract(p[i+1],p[i]),float(sample)/8));
    if(point.y>lo+view.radius && point.y<hi-view.radius)
      min_clearance=std::min(min_clearance,std::hypot(point.x-post_state.position.x,point.z-post_state.position.z)-radius);
   }
   WorldStepTimings timing;check(world.collect_step_timings(timing),"rope timings");gpu+=timing.total_gpu_milliseconds;rope_gpu+=timing.rope_solve.total_milliseconds;
   if(frame%60==59)std::cout<<"frame="<<frame+1<<" strain="<<max_strain<<" anchor="<<anchor_error<<" clearance="<<min_clearance<<" gpu_ms="<<gpu/(frame+1)<<std::endl;
  }
  std::cout<<"nodes="<<view.positions.size<<" strain="<<max_strain<<" anchor_error="<<anchor_error<<" min_post_clearance="<<min_clearance<<" winding="<<min_winding<<" speed="<<maximum_speed<<" gpu_ms="<<gpu/frames<<" rope_ms="<<rope_gpu/frames<<std::endl;
  std::cout<<"max_ball_penetration="<<max_ball_penetration<<std::endl;
  if(settle)std::cout<<"settle_max_speed="<<late_max_speed<<" settle_rms_speed="<<std::sqrt(late_rms_speed/std::max(1U,late_count))<<" settle_drift="<<late_displacement<<" ground_nodes="<<ground_nodes<<std::endl;
  if(settle)std::cout<<"settle_peak_node="<<late_peak_node<<" position="<<late_peak_position.x<<','<<late_peak_position.y<<','<<late_peak_position.z<<std::endl;
  check(max_strain<0.03F,"rope segment stretch/compression exceeds 3 percent");
  check(anchor_error<1e-4F,"hook attachment drift");
  check(max_ball_penetration<0.003F,"rope clips through active sphere");
  check(min_clearance>=view.radius-0.003F,"rope clips through post");
  if(wrapped)check(min_winding>2.8F,"wrapped rope slipped through post");
  if(settle) {
   check(late_max_speed<0.08F,"rope still jitters after settling");
   check(std::sqrt(late_rms_speed/std::max(1U,late_count))<0.004F,"rope retains too much motion");
   check(late_displacement<0.008F,"settled rope drifts across ground");
   check(ground_nodes*5>view.positions.size*2,"loose rope does not settle on ground");
  }
  check(world.remove_rope(instance.ropes[0]),"remove rope");check(!world.rope_view(instance.ropes[0],view),"stale rope handle");
  return 0;
 }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}

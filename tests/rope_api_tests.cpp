// SPDX-License-Identifier: MIT
#include <parallel_mater/parallel_mater.hpp>
#include <cuda_runtime_api.h>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace parallel_mater;
void check(bool b,const char *m){if(!b)throw std::runtime_error(m);}
void check(Status s,const char *m){if(!s)throw std::runtime_error(std::string(m)+": "+s.message);}
int main(){try{
 const std::array<Vec3,3> line{{{0,1,0},{0.1F,1.2F,0},{0.3F,1,0}}};
 std::vector<Vec3> nodes;
 check(sample_rope_centerline({line.data(),line.size()},0.02F,nodes),"sample polyline");
 check(nodes.size()>20 && nodes.front().y==1 && nodes.back().x==0.3F,"sample endpoints and resolution");
 const auto size=nodes.size();
 check(!sample_rope_centerline({line.data(),line.size()},0,nodes),"invalid spacing");
 check(nodes.size()==size,"transactional failed sampling");
 int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
 World world;check(World::create({.rigid_body_capacity=1,.triangle_mesh_capacity=1,.rope_capacity=1,
     .physics_debug={.frame_capacity=3}},world),"create rope world");
 RopeOptions options{.centerline={line.data(),line.size()},.node_spacing=0.02F,.self_collision=false};
 RopeId rope,other;
 auto invalid=options;invalid.radius=-1;
 check(!world.add_rope(invalid,rope),"reject negative rope radius");
 invalid=options;invalid.maximum_substep_timestep=0;
 check(!world.add_rope(invalid,rope),"reject zero rope timestep limit");
 invalid=options;invalid.first.enabled=true;
 check(!world.add_rope(invalid,rope),"reject invalid Hook body");
 check(world.add_rope(options,rope),"add free rope");
 check(!world.step({.timestep=3.0F}),"reject excessive rope substep budget before advancing");
 check(!world.add_rope(options,other),"bounded rope capacity");
 RopeDeviceView view;check(world.rope_view(rope,view),"rope view");
 WorldStatistics first,last;
 for(int i=0;i<120;++i) {
  check(world.step({.timestep=1.0F/120,.substeps=2,.gravity={0,-1,0},.collect_kernel_timings=i==0}),"free rope step");
  if(i==0) {
   check(world.collect_statistics(first),"first statistics");
   WorldStepTimings timing;check(world.collect_step_timings(timing),"rope substep timing");
   check(timing.rope_solve.launch_count==4,"API enforces rope timestep limit");
  }
 }
 check(world.collect_statistics(last),"last statistics");
 // The rolling capture is bounded too; compare only the fixed physics buffers.
 check(last.rope_count==1 && last.rope_node_count==size,"rope statistics");
 PhysicsDebugCapture capture;check(world.copy_physics_debug_capture(capture),"rope capture");
 check(capture.frames.size()==3 && capture.frames.back().rope_nodes.size()==size,"opt-in rope history");
 check(world.rope_view(rope,view),"advanced rope view");
 std::vector<Vec3> p(view.positions.size);check(cudaMemcpy(p.data(),view.positions.data,p.size()*sizeof(Vec3),cudaMemcpyDeviceToHost)==cudaSuccess,"read positions");
 for(unsigned i=0;i<p.size();++i)check(std::isfinite(p[i].y)&&p[i].y<nodes[i].y-0.4F,"free rope obeys gravity");
 std::vector<Vec3> velocity(view.velocities.size);
 check(cudaMemcpy(velocity.data(),view.velocities.data,velocity.size()*sizeof(Vec3),cudaMemcpyDeviceToHost)==cudaSuccess,"read free rope velocity");
 float reference=0;
 for(unsigned substep=0;substep<240;++substep)reference=(reference-1.0F/240)*std::exp(-options.velocity_damping/240);
 for(const auto v:velocity)
  check(std::abs(v.y-reference)<0.002F && std::abs(v.x)<0.002F && std::abs(v.z)<0.002F,
        "rope constraints preserve uniform free-fall velocity");
 check(world.remove_rope(rope),"remove rope");check(!world.rope_view(rope,view),"stale rope view");
 check(world.add_rope(options,other),"reuse rope slot");check(other.generation!=rope.generation,"generation changes");
 check(world.remove_rope(other),"remove reused rope");
 const std::array<Vec3,4> floor{{{-2,0,-2},{2,0,-2},{2,0,2},{-2,0,2}}};
 const std::array<std::uint32_t,6> indices{{0,2,1,0,3,2}};
 Vec3 *device_vertices=nullptr;std::uint32_t *device_indices=nullptr;
 check(cudaMalloc(reinterpret_cast<void **>(&device_vertices),sizeof(floor))==cudaSuccess,"allocate floor vertices");
 check(cudaMalloc(reinterpret_cast<void **>(&device_indices),sizeof(indices))==cudaSuccess,"allocate floor indices");
 check(cudaMemcpy(device_vertices,floor.data(),sizeof(floor),cudaMemcpyHostToDevice)==cudaSuccess,"upload floor vertices");
 check(cudaMemcpy(device_indices,indices.data(),sizeof(indices),cudaMemcpyHostToDevice)==cudaSuccess,"upload floor indices");
 TriangleMeshId mesh;
 check(world.add_triangle_mesh({device_vertices,floor.size()},{device_indices,indices.size()},mesh),"add rope floor");
 cudaFree(device_vertices);cudaFree(device_indices);
 RigidBodyId body;check(world.add_rigid_body({.motion=MotionType::static_body,.mesh=mesh},body),"add floor body");
 const std::array<Vec3,2> crossing{{{0,-0.2F,0},{0,0.2F,0}}};
 options.centerline={crossing.data(),crossing.size()};
 const auto rejected=world.add_rope(options,other);
 check(!rejected && rejected.code==StatusCode::invalid_argument,"reject rope threaded through floor");
 check(world.collect_statistics(last),"statistics after rejection");
 check(last.rope_count==0,"invalid rest pose leaves no rope allocation");
 options.centerline={line.data(),line.size()};
 check(world.add_rope(options,other),"valid rest pose after rejection");
 // An unanchored rope must land, slide under tilted gravity, then stop via
 // contact friction. No scene-specific resting force or sleep flag is used.
 float slide_start=0,slide_end=0;
 for(unsigned frame=0;frame<480;++frame) {
  const bool tilted=frame>=180 && frame<240;
  check(world.step({.timestep=1.0F/60,.substeps=4,
      .gravity={tilted?4.0F:0.0F,-9.81F,0}}),"rope floor friction step");
  if(frame==179 || frame==239) {
   check(world.rope_view(other,view),"sliding rope view");
   check(cudaMemcpy(p.data(),view.positions.data,p.size()*sizeof(Vec3),cudaMemcpyDeviceToHost)==cudaSuccess,"read sliding rope");
   if(frame==179)slide_start=p[p.size()/2].x;else slide_end=p[p.size()/2].x;
  }
 }
 check(slide_end>slide_start+0.005F,"floor contact does not glue rope in place");
 check(world.rope_view(other,view),"resting rope view");
 check(cudaMemcpy(p.data(),view.positions.data,p.size()*sizeof(Vec3),cudaMemcpyDeviceToHost)==cudaSuccess,"read resting rope");
 check(cudaMemcpy(velocity.data(),view.velocities.data,velocity.size()*sizeof(Vec3),cudaMemcpyDeviceToHost)==cudaSuccess,"read resting velocity");
 for(unsigned i=0;i<p.size();++i) {
  check(p[i].y>=options.radius-0.001F && p[i].y<options.radius+0.02F,"rope settles above floor");
  check(std::sqrt(velocity[i].x*velocity[i].x+velocity[i].y*velocity[i].y+velocity[i].z*velocity[i].z)<0.005F,
        "floor friction stops resting rope");
 }
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}

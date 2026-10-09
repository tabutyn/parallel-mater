// SPDX-License-Identifier: MIT
#include <parallel_mater/d3d12.hpp>

// The public surface is intentionally complete from the first release.  Later
// system gates can replace these transactional stubs without an API break.
namespace parallel_mater::d3d12 {
namespace {
[[nodiscard]] constexpr Status unavailable() noexcept {
    return {StatusCode::not_supported, 0,
            "This system is not implemented by the D3D12 rigid-body gate"};
}
[[nodiscard]] constexpr Status no_world() noexcept {
    return {StatusCode::invalid_argument, 0, "World is not initialized"};
}
} // namespace

#define PM_ADD_STUB(name, Options, Id)                                      \
    Status World::name(Options, Id &output) noexcept {                       \
        output = {};                                                         \
        if (!impl_) return no_world();                                       \
        return unavailable();                                                \
    }
#define PM_REMOVE_STUB(name, Id)                                             \
    Status World::name(Id) noexcept {                                        \
        if (!impl_) return no_world();                                       \
        return unavailable();                                                \
    }
#define PM_UPDATE_STUB(name, Id, Options)                                    \
    Status World::name(Id, Options) noexcept {                               \
        if (!impl_) return no_world();                                       \
        return unavailable();                                                \
    }

Status World::add_fluid(FluidOptions, BufferSpan<const FluidParticle>,
                        FluidId &output) noexcept {
    output = {};
    return impl_ ? unavailable() : no_world();
}
Status World::add_fluid(FluidOptions, HostSpan<const FluidParticle>,
                        FluidId &output) noexcept {
    output = {};
    return impl_ ? unavailable() : no_world();
}
Status World::add_fluid_geometry(FluidOptions, FluidGeometrySource,
                                 FluidId &output) noexcept {
    output = {};
    return impl_ ? unavailable() : no_world();
}
PM_REMOVE_STUB(remove_fluid, FluidId)
Status World::fluid_view(FluidId, FluidDeviceView &output) const noexcept {
    output = {};
    return impl_ ? unavailable() : no_world();
}

PM_ADD_STUB(add_smoke, SmokeOptions, SmokeId)
PM_REMOVE_STUB(remove_smoke, SmokeId)
Status World::smoke_view(SmokeId, SmokeDeviceView &output) const noexcept {
    output = {};
    return impl_ ? unavailable() : no_world();
}
PM_ADD_STUB(add_fluid_smoke_coupling, FluidSmokeCouplingOptions,
            FluidSmokeCouplingId)
PM_REMOVE_STUB(remove_fluid_smoke_coupling, FluidSmokeCouplingId)
PM_ADD_STUB(add_smoke_soft_body_coupling, SmokeSoftBodyCouplingOptions,
            SmokeSoftBodyCouplingId)
PM_REMOVE_STUB(remove_smoke_soft_body_coupling, SmokeSoftBodyCouplingId)
PM_ADD_STUB(add_smoke_cloth_coupling, SmokeClothCouplingOptions,
            SmokeClothCouplingId)
PM_REMOVE_STUB(remove_smoke_cloth_coupling, SmokeClothCouplingId)
PM_ADD_STUB(add_smoke_rope_coupling, SmokeRopeCouplingOptions,
            SmokeRopeCouplingId)
PM_REMOVE_STUB(remove_smoke_rope_coupling, SmokeRopeCouplingId)
PM_ADD_STUB(add_smoke_rigid_coupling, SmokeRigidCouplingOptions,
            SmokeRigidCouplingId)
PM_REMOVE_STUB(remove_smoke_rigid_coupling, SmokeRigidCouplingId)

PM_ADD_STUB(add_cloth, ClothOptions, ClothId)
PM_REMOVE_STUB(remove_cloth, ClothId)
Status World::cloth_view(ClothId, ClothDeviceView &output) const noexcept {
    output = {};
    return impl_ ? unavailable() : no_world();
}
PM_ADD_STUB(add_soft_body, SoftBodyOptions, SoftBodyId)
PM_REMOVE_STUB(remove_soft_body, SoftBodyId)
Status World::soft_body_view(SoftBodyId,
                            SoftBodyDeviceView &output) const noexcept {
    output = {};
    return impl_ ? unavailable() : no_world();
}
PM_ADD_STUB(add_rope, RopeOptions, RopeId)
PM_REMOVE_STUB(remove_rope, RopeId)
Status World::rope_view(RopeId, RopeDeviceView &output) const noexcept {
    output = {};
    return impl_ ? unavailable() : no_world();
}

PM_ADD_STUB(add_fluid_rope_coupling, FluidRopeCouplingOptions,
            FluidRopeCouplingId)
PM_UPDATE_STUB(update_fluid_rope_coupling, FluidRopeCouplingId,
               FluidRopeCouplingOptions)
PM_REMOVE_STUB(remove_fluid_rope_coupling, FluidRopeCouplingId)
PM_ADD_STUB(add_rope_soft_body_coupling, RopeSoftBodyCouplingOptions,
            RopeSoftBodyCouplingId)
PM_UPDATE_STUB(update_rope_soft_body_coupling, RopeSoftBodyCouplingId,
               RopeSoftBodyCouplingOptions)
PM_REMOVE_STUB(remove_rope_soft_body_coupling, RopeSoftBodyCouplingId)
PM_ADD_STUB(add_rope_cloth_coupling, RopeClothCouplingOptions,
            RopeClothCouplingId)
PM_UPDATE_STUB(update_rope_cloth_coupling, RopeClothCouplingId,
               RopeClothCouplingOptions)
PM_REMOVE_STUB(remove_rope_cloth_coupling, RopeClothCouplingId)
PM_ADD_STUB(add_fluid_cloth_coupling, FluidClothCouplingOptions,
            FluidClothCouplingId)
PM_UPDATE_STUB(update_fluid_cloth_coupling, FluidClothCouplingId,
               FluidClothCouplingOptions)
PM_REMOVE_STUB(remove_fluid_cloth_coupling, FluidClothCouplingId)
PM_ADD_STUB(add_soft_body_cloth_coupling, SoftBodyClothCouplingOptions,
            SoftBodyClothCouplingId)
PM_UPDATE_STUB(update_soft_body_cloth_coupling, SoftBodyClothCouplingId,
               SoftBodyClothCouplingOptions)
PM_REMOVE_STUB(remove_soft_body_cloth_coupling, SoftBodyClothCouplingId)
PM_ADD_STUB(add_fluid_soft_body_coupling, FluidSoftBodyCouplingOptions,
            FluidSoftBodyCouplingId)
PM_UPDATE_STUB(update_fluid_soft_body_coupling, FluidSoftBodyCouplingId,
               FluidSoftBodyCouplingOptions)
PM_REMOVE_STUB(remove_fluid_soft_body_coupling, FluidSoftBodyCouplingId)

Status World::add_particle_source(ParticleSourceMesh,
    ParticleSourceOptions, ParticleSourceId &output) noexcept {
    output = {};
    return impl_ ? unavailable() : no_world();
}
PM_UPDATE_STUB(update_particle_source, ParticleSourceId, ParticleSourceOptions)
PM_REMOVE_STUB(remove_particle_source, ParticleSourceId)
PM_ADD_STUB(add_particle_destroy_plane, ParticleDestroyPlaneOptions,
            ParticleDestroyPlaneId)
PM_UPDATE_STUB(update_particle_destroy_plane, ParticleDestroyPlaneId,
               ParticleDestroyPlaneOptions)
PM_REMOVE_STUB(remove_particle_destroy_plane, ParticleDestroyPlaneId)

Status World::add_paint_field(PaintFieldOptions,
                              PaintFieldId &output) noexcept {
    output = {};
    return impl_ ? unavailable() : no_world();
}
Status World::add_paint_field(PaintFieldHostOptions,
                              PaintFieldId &output) noexcept {
    output = {};
    return impl_ ? unavailable() : no_world();
}
PM_REMOVE_STUB(remove_paint_field, PaintFieldId)
PM_REMOVE_STUB(clear_paint_field, PaintFieldId)
Status World::paint_field_view(PaintFieldId,
                              PaintFieldDeviceView &output) const noexcept {
    output = {};
    return impl_ ? unavailable() : no_world();
}
PM_ADD_STUB(add_paint_rule, PaintRuleOptions, PaintRuleId)
PM_REMOVE_STUB(remove_paint_rule, PaintRuleId)

Status World::physics_debug_frame(PhysicsDebugFrameView &output) const noexcept {
    output = {};
    return impl_ ? unavailable() : no_world();
}
Status World::copy_physics_debug_capture(PhysicsDebugCapture &output) const noexcept {
    output = {};
    return impl_ ? unavailable() : no_world();
}

Status World::systems_reserve_debug_samples(std::uint64_t, std::uint64_t,
    std::uint64_t, std::uint64_t) noexcept {
    return impl_ ? unavailable() : no_world();
}
Status World::systems_validate_rope_rest(HostSpan<const Vec3>, RopeAttachment,
    RopeAttachment, std::uint32_t &, std::uint32_t &) const noexcept {
    return impl_ ? unavailable() : no_world();
}

#undef PM_ADD_STUB
#undef PM_REMOVE_STUB
#undef PM_UPDATE_STUB
} // namespace parallel_mater::d3d12

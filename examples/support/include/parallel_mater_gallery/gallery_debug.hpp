// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater_gallery/gallery_context.hpp>
#include <parallel_mater_gallery/renderer.hpp>

namespace parallel_mater::gallery {

// One debug-mode state shared by every gallery entry. Scene capabilities only
// decide what the primary V mode means; force-vector toggles remain available
// uniformly across rigid, fluid, and cloth combinations.
struct GalleryDebugState {
    bool particle_view{};
    bool normals{};
    bool rigid_forces{};
    bool fluid_forces{};
    bool cloth_bonds{};
    bool velocities{};
    bool structure{};
    bool rigid_contacts{};
    SmokeDebugMode smoke_mode{SmokeDebugMode::none};

    void reset(bool initial_particle_view = false) noexcept {
        *this = {};
        particle_view = initial_particle_view;
    }

    void toggle_smoke(SmokeDebugMode mode) noexcept {
        smoke_mode = smoke_mode == mode ? SmokeDebugMode::none : mode;
    }

    void toggle_primary(GalleryContext context) noexcept {
        const bool deformable = is_cloth_context(context) ||
            is_soft_body_context(context) || is_rope_context(context);
        if (deformable) structure = !structure;
        if (is_fluid_context(context)) particle_view = !particle_view;
        if (!deformable && !is_fluid_context(context))
            rigid_contacts = !rigid_contacts;
    }

    [[nodiscard]] bool vectors_visible() const noexcept {
        return normals || rigid_forces || fluid_forces || velocities;
    }

    [[nodiscard]] FluidRenderMode fluid_render_mode(
        GalleryContext context) const noexcept {
        if (particle_view) return FluidRenderMode::particles;
        if (context == GalleryContext::water_cloth && structure)
            return FluidRenderMode::wireframe;
        return FluidRenderMode::surface;
    }
};

} // namespace parallel_mater::gallery

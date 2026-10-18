// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/gallery_input.hpp>
#include <parallel_mater_gallery/control_text.hpp>
#include <parallel_mater_gallery/runtime_platform.hpp>

#include <cstdlib>
#include <iostream>

namespace gallery = parallel_mater::gallery;

namespace {
void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}
} // namespace

int main() {
    static_assert(gallery::gallery_entries.size() == 28U);
    require(gallery::gallery_backend_from_name("cuda") ==
                PM_NATIVE_BACKEND_CUDA,
            "CUDA backend option must parse");
    require(gallery::gallery_backend_from_name("d3d12") ==
                PM_NATIVE_BACKEND_D3D12,
            "D3D12 backend option must parse");
    require(gallery::gallery_backend_from_name("dx12") ==
                PM_NATIVE_BACKEND_D3D12,
            "DX12 backend alias must parse");
    require(!gallery::gallery_backend_from_name("directcompute"),
            "retired DirectCompute backend must not parse");

    const auto force = gallery::gallery_directional_control_text(
        gallery::GalleryControlPolicy::authored_force);
    require(force.label == "Push objects" && force.left == "Push left" &&
                force.forward == "Push forward" && force.backward == "Push back" &&
                force.right == "Push right",
            "Authored-force controls must expose common user-facing semantics");
    const auto motor = gallery::gallery_directional_control_text(
        gallery::GalleryControlPolicy::motor_drive);
    require(motor.label == "Drive and steer" && motor.left == "Steer left" &&
                motor.forward == "Drive forward" && motor.backward == "Drive reverse" &&
                motor.right == "Steer right",
            "Motor controls must expose steering and drive semantics");
    for (const auto policy : {gallery::GalleryControlPolicy::rigid_gravity,
             gallery::GalleryControlPolicy::collector_gravity,
             gallery::GalleryControlPolicy::peg_gravity,
             gallery::GalleryControlPolicy::cloth_gravity}) {
        const auto gravity = gallery::gallery_directional_control_text(policy);
        require(gravity.label == "Tilt gravity" &&
                    gravity.left == "Tilt gravity left" &&
                    gravity.forward == "Tilt gravity forward" &&
                    gravity.backward == "Tilt gravity back" &&
                    gravity.right == "Tilt gravity right",
                "Gravity controls must expose common directional semantics");
    }
    const auto camera = gallery::gallery_directional_control_text(
        gallery::GalleryControlPolicy::none);
    require(camera.label == "Camera only" && camera.left.empty() &&
                camera.forward.empty() && camera.backward.empty() &&
                camera.right.empty(),
            "Scenes without directional actions must not expose action buttons");
    require(gallery::gallery_primary_action_label(
                gallery::GalleryPrimaryAction::none, false, 0.0F).empty(),
            "Scenes without a primary action must not expose one");
    require(gallery::gallery_primary_action_label(
                gallery::GalleryPrimaryAction::release_restore_constraints,
                false, 0.0F) == "Release",
            "Constraint action must initially offer release");
    require(gallery::gallery_primary_action_label(
                gallery::GalleryPrimaryAction::release_restore_constraints,
                true, 0.0F) == "Restore",
            "Released constraints must offer restore");
    require(gallery::gallery_primary_action_label(
                gallery::GalleryPrimaryAction::toggle_dump_bed,
                false, 0.0F) == "Raise bed",
            "Lowered dump bed must offer raise");
    require(gallery::gallery_primary_action_label(
                gallery::GalleryPrimaryAction::toggle_dump_bed,
                false, 0.1F) == "Lower bed",
            "Raised dump bed must offer lower");

    require(gallery::gallery_backend_from_name("vulkan") == PM_NATIVE_BACKEND_VULKAN,
            "Vulkan backend option must parse");
    PmPhysicalDeviceFeatures preview{};
    preview.rigidBodies = preview.rigidConstraints = PM_TRUE;
    preview.particleLiquids = preview.liquidRigidCoupling = PM_TRUE;
    require(gallery::gallery_backend_supports(gallery::GalleryContext::fluid_rigid, preview),
            "Liquid-rigid preview must remain selectable");
    for (auto context : {gallery::GalleryContext::cloth, gallery::GalleryContext::rope,
                         gallery::GalleryContext::soft_body, gallery::GalleryContext::smoke,
                         gallery::GalleryContext::peg_paint, gallery::GalleryContext::water_cloth})
        require(!gallery::gallery_backend_supports(context, preview),
                "Unavailable preview domains must remain disabled");
    preview.liquidRigidCoupling = PM_FALSE;
    require(!gallery::gallery_backend_supports(gallery::GalleryContext::fluid_rigid, preview),
            "Liquid-rigid scene requires its coupling feature");

    for (std::size_t current = 0; current < gallery::gallery_entries.size();
         ++current) {
        gallery::GalleryInputState input;
        input.current = current;
        input.selected = current;
        input.arrow(gallery::GalleryArrowKey::right, true);
        input.key(gallery::GalleryKey::fps);
        require(input.show_fps && input.right_input() == 1.0F && !input.scene_action,
                "F must only toggle the shared overlay in every scene");
        input.key(gallery::GalleryKey::fps);
        require(!input.show_fps, "F must hide the shared overlay");

        if (gallery::gallery_entries[current].count_kind !=
            gallery::GalleryCountKind::none) {
            input.key(gallery::GalleryKey::parameters);
            require(input.parameters_visible,
                    "P must open the current scene parameter editor");
        }

        input.key(gallery::GalleryKey::catalog);
        require(input.catalog_visible,
                "Tab must open the catalog from every scene and modal state");
        require(!input.parameters_visible,
                "Opening the catalog must dismiss parameter editing");
        require(input.selected == current,
                "Opening the catalog must select the current scene");
        require(input.right_input() == 0.0F,
                "Opening a modal must clear held movement");
        input.key(gallery::GalleryKey::fps);
        require(input.show_fps && input.catalog_visible,
                "F must preserve the open modal");

        while (input.selected + 1U < gallery::gallery_entries.size())
            input.key(gallery::GalleryKey::next);
        require(input.selected == gallery::gallery_entries.size() - 1U,
                "The last canonical catalog entry must be reachable");
        input.key(gallery::GalleryKey::enter);
        require(input.requested_scene == gallery::gallery_entries.size() - 1U,
                "Every canonical catalog entry must be selectable");
    }

    std::cout << "Gallery input contract passed\n";
}

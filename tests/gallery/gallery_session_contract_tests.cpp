// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/mobile_qualification.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace gallery = parallel_mater::gallery;

namespace {
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
} // namespace

int main() {
    try {
        require(gallery::gallery_backend_label("vulkan") == "Vulkan" &&
                    gallery::gallery_backend_label("cuda") == "CUDA" &&
                    gallery::gallery_backend_label("d3d12") == "Direct3D 12" &&
                    gallery::gallery_backend_label("metal") == "Metal",
                "Backend presentation labels changed");
        const std::filesystem::path root = PM_GALLERY_SESSION_TEST_ROOT;
        require(gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Runtime count was not initially zero");
        {
            gallery::GallerySessionConfig config{
                .assets = root / "examples/assets",
                .profiles = root /
                    "tests/gallery/assets/session_contract_profiles.json",
                .platform = "ContractTest",
                .model = "test-host",
                .backend = PM_GALLERY_SESSION_TEST_BACKEND,
            };
            gallery::GallerySession session(std::move(config));
            session.resize(96U, 72U);
            require(gallery::GalleryRuntime::live_runtime_count() == 1U,
                    "Session did not own exactly one live runtime");

            require(session.catalog().size() == gallery::gallery_entries.size(),
                    "Shared catalog changed");
            require(session.ready() && session.selection() == 0U,
                    "Session did not select its initial scene");
            const auto startup = session.presentation();
            require(startup.platform == "ContractTest" &&
                        startup.backend == PM_GALLERY_SESSION_TEST_BACKEND &&
                        !startup.device.empty() && startup.ready &&
                        startup.scene_id == "rigid-body" &&
                        !startup.scene_summary.empty() &&
                        startup.renderer ==
                            gallery::GalleryRendererKind::software_image &&
                        startup.readback ==
                            gallery::GalleryReadbackMode::host_frame,
                    "Session did not expose structured presentation state");

            session.frame();
            require(session.frames == 1U && session.timing.latest() &&
                        session.timing.latest()->physics_steps == 1U,
                    "Running frame did not advance exactly once");

            session.set_paused(true);
            session.frame();
            require(session.frames == 1U && session.timing.latest() &&
                        session.timing.latest()->physics_steps == 0U,
                    "Paused frame advanced physics");

            session.steer(5.0F, -5.0F);
            session.set_active(false);
            session.frame();
            require(session.frames == 1U && !session.timing.latest(),
                    "Inactive session advanced or retained timing");
            session.set_active(true);
            session.set_paused(false);
            session.frame();
            require(session.frames == 2U && session.timing.latest() &&
                        session.timing.latest()->physics_steps == 1U,
                    "Foreground session did not resume with one normal step");

            session.select(2U); // Constraint: Point in bounded catalog.
            require(session.selection() == 2U && session.frames == 0U &&
                        !session.timing.latest(),
                    "Scene selection retained stale frame state");
            require(gallery::GalleryRuntime::live_runtime_count() == 1U,
                    "Scene replacement retained more than one runtime");
            session.frame();
            require(session.primary_action() ==
                        gallery::GalleryActionOutcome::applied &&
                        session.controller_state().constraints_released,
                    "Mobile semantic action did not release point constraints");
            session.frame();
            require(session.frames == 2U,
                    "Point release did not preserve the live scene");
            require(session.primary_action() ==
                        gallery::GalleryActionOutcome::reset_requested,
                    "Mobile semantic action did not request point restore");
            require(session.selection() == 2U && session.frames == 0U &&
                        !session.timing.latest() &&
                        !session.controller_state().constraints_released,
                    "Point restore did not reset the selected scene");

            const auto selected = session.selection();
            bool rejected = false;
            try {
                session.select(999U);
            } catch (const std::runtime_error &) {
                rejected = true;
            }
            require(rejected && session.ready() &&
                        session.selection() == selected &&
                        gallery::GalleryRuntime::live_runtime_count() == 1U,
                    "Rejected selection replaced the live scene");

            session.frame();
            session.reset();
            require(session.selection() == selected && session.frames == 0U &&
                        !session.timing.latest(),
                    "Reset changed selection or retained frame state");

            // Real platform adapters pass monotonic timestamps. The first
            // timestamp establishes a baseline; only elapsed fixed intervals
            // become submissions, and pause time is never caught up.
            using FrameClock = gallery::GallerySession::FrameClock;
            auto timestamp = FrameClock::time_point{};
            const auto interval = std::chrono::duration_cast<FrameClock::duration>(
                std::chrono::duration<double>(
                    1.0 / static_cast<double>(session.limits.target_fps)));
            session.frame_at(timestamp);
            require(session.frames == 0U && session.timing.latest() &&
                        session.timing.latest()->physics_steps == 0U,
                    "Timestamp baseline advanced physics");
            timestamp += interval;
            session.frame_at(timestamp);
            require(session.frames == 1U &&
                        session.timing.latest()->physics_steps == 1U,
                    "Timestamp interval did not advance physics");
            session.set_paused(true);
            timestamp += std::chrono::seconds(10);
            session.frame_at(timestamp);
            require(session.frames == 1U &&
                        session.timing.latest()->physics_steps == 0U,
                    "Timestamp trace caught up paused time");
            session.set_paused(false);
            timestamp += std::chrono::seconds(10);
            session.frame_at(timestamp);
            require(session.frames == 1U,
                    "Resume timestamp caught up paused time");
            timestamp += interval;
            session.frame_at(timestamp);
            require(session.frames == 2U,
                    "Resumed timestamp trace did not advance");

            // Rich desktop rendering supplies callbacks, but cadence,
            // stepping, timing, and presentation ordering remain session-owned.
            session.reset();
            timestamp = FrameClock::time_point{};
            unsigned renders = 0U;
            unsigned presents = 0U;
            float interpolation = 0.0F;
            gallery::GalleryFramePlan desktop_frame;
            desktop_frame.controls.camera = session.camera().camera();
            const auto render = [&](gallery::GallerySession &owner,
                                    float alpha) {
                require(&owner == &session,
                        "Desktop renderer received a foreign session");
                ++renders;
                interpolation = alpha;
            };
            const auto present = [&](const gallery::GallerySession &owner) {
                require(&owner == &session && renders == presents + 1U,
                        "Desktop presentation ran out of order");
                ++presents;
            };
            session.frame_at(timestamp, desktop_frame, render, present);
            require(session.frames == 0U && renders == 1U && presents == 1U &&
                        session.frame_timing().latest() &&
                        session.frame_timing().latest()->physics_steps == 0U,
                    "Desktop callback baseline bypassed common scheduling");
            timestamp += interval;
            session.frame_at(timestamp, desktop_frame, render, present);
            require(session.frames == 1U && renders == 2U && presents == 2U &&
                        session.frame_timing().latest()->physics_steps == 1U &&
                        interpolation >= 0.0F && interpolation <= 1.0F,
                    "Desktop callbacks did not use common frame policy");
            desktop_frame.simulation_enabled = false;
            timestamp += std::chrono::seconds(10);
            session.frame_at(timestamp, desktop_frame, render, present);
            desktop_frame.simulation_enabled = true;
            timestamp += interval;
            session.frame_at(timestamp, desktop_frame, render, present);
            require(session.frames == 2U && renders == 4U && presents == 4U,
                    "Desktop modal frame accumulated catch-up work");

            if (session.backend_name() == "vulkan") {
                for (const gallery::GalleryContext context : {
                         gallery::GalleryContext::cloth,
                         gallery::GalleryContext::cloth_tear,
                         gallery::GalleryContext::cloth_paint,
                         gallery::GalleryContext::water_cloth,
                         gallery::GalleryContext::soft_body,
                         gallery::GalleryContext::soft_body_rigid,
                         gallery::GalleryContext::soft_body_fluid,
                         gallery::GalleryContext::soft_body_cloth,
                         gallery::GalleryContext::rope,
                         gallery::GalleryContext::rope_fluid,
                         gallery::GalleryContext::rope_soft_body,
                         gallery::GalleryContext::rope_cloth,
                         gallery::GalleryContext::smoke,
                         gallery::GalleryContext::smoke_water,
                         gallery::GalleryContext::smoke_soft_body,
                         gallery::GalleryContext::smoke_cloth,
                         gallery::GalleryContext::smoke_rope}) {
                    const auto item = std::find_if(
                        session.catalog().begin(), session.catalog().end(),
                        [context](const auto &candidate) {
                            return candidate.scene.context == context;
                        });
                    require(item != session.catalog().end() && item->available,
                            "Qualified Vulkan scene was not advertised");
                    session.select(static_cast<unsigned>(
                        item - session.catalog().begin()));
                    (void)session.qualify(3U);
                    require(session.ready() && session.selection() ==
                                static_cast<unsigned>(
                                    item - session.catalog().begin()),
                            "Vulkan scene qualification lost its scene");
                }
            }

            std::cout << "Gallery session contract passed on "
                      << session.backend_name() << '\n';
        }
        require(gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Session teardown did not release its runtime exactly once");

        const auto mobile_config = [&] {
            return gallery::GallerySessionConfig{
                .assets = root / "examples/assets",
                .profiles = root /
                    "tests/gallery/assets/session_contract_profiles.json",
                .platform = "ContractTest",
                .model = "test-host",
                .backend = PM_GALLERY_SESSION_TEST_BACKEND,
            };
        };
        for (unsigned recreation = 0U; recreation < 2U; ++recreation) {
            gallery::MobileGallerySession mobile(mobile_config());
            require(!mobile.state().presentation.active &&
                        !mobile.state().surface_ready &&
                        !mobile.should_frame(),
                    "Mobile session ran before a surface was attached");
            require(mobile.attach_surface(96U, 72U) &&
                        mobile.state().surface_generation == 1U &&
                        mobile.should_frame(),
                    "Mobile surface attachment did not admit frames");
            require(mobile.resize_surface(88U, 66U) &&
                        mobile.gallery().width == 88U &&
                        mobile.gallery().height == 66U &&
                        mobile.resize_surface(96U, 72U) &&
                        mobile.gallery().width == 96U &&
                        mobile.gallery().height == 72U,
                    "Shared mobile resize did not update and restore the surface");
            require(mobile.frame() && mobile.gallery().frames == 1U,
                    "Mobile surface did not produce a frame");

            require(mobile.execute(
                        gallery::MobileGalleryCommand::toggle_paused) &&
                        mobile.state().presentation.paused && mobile.frame() &&
                        mobile.gallery().frames == 1U,
                    "Shared mobile pause advanced physics");
            require(mobile.execute(
                        gallery::MobileGalleryCommand::toggle_paused) &&
                        !mobile.state().presentation.paused,
                    "Shared mobile resume did not clear pause");

            mobile.set_foreground(false);
            require(!mobile.should_frame() && !mobile.frame() &&
                        !mobile.state().presentation.active,
                    "Background mobile session admitted a frame");
            mobile.set_foreground(true);
            require(mobile.should_frame() && mobile.frame(),
                    "Foreground mobile session did not recover");

            const auto old_selection = mobile.gallery().selection();
            require(!mobile.execute(
                        gallery::MobileGalleryCommand::select_scene, 999.0F) &&
                        mobile.state().failed && !mobile.should_frame() &&
                        mobile.gallery().selection() == old_selection,
                    "Mobile failure did not latch without replacing the scene");
            require(mobile.execute(
                        gallery::MobileGalleryCommand::reset_scene) &&
                        !mobile.state().failed && mobile.should_frame(),
                    "Mobile reset did not recover a failed shell");

            require(!mobile.frame([](const gallery::GallerySession &) {
                        throw std::runtime_error("forced presentation failure");
                    }) &&
                        mobile.state().failed && !mobile.should_frame(),
                    "Presentation failure did not stop mobile scheduling");
            mobile.detach_surface();
            require(!mobile.should_frame() && !mobile.frame() &&
                        mobile.state().surface_generation == 2U,
                    "Detached mobile surface admitted work");
            require(mobile.attach_surface(80U, 60U) &&
                        !mobile.state().failed &&
                        mobile.state().surface_generation == 3U &&
                        mobile.gallery().width == 80U &&
                        mobile.gallery().height == 60U && mobile.frame(),
                    "Replacement mobile surface did not recover presentation failure");

            require(mobile.execute(
                        gallery::MobileGalleryCommand::select_scene, 2.0F) &&
                        mobile.state().presentation.controls.primary ==
                            gallery::GalleryPrimaryAction::release_restore_constraints &&
                        !mobile.state().presentation.controls.constraints_released &&
                        mobile.execute(
                            gallery::MobileGalleryCommand::primary_action) &&
                        mobile.state().presentation.controls.constraints_released,
                    "Mobile shell did not derive Point controls from shared state");
        }
        require(gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Mobile shell recreation leaked a runtime");

        gallery::MobileGalleryQualificationReport mobile_report{
            .revision = "contract-revision",
            .platform = "ContractTest",
            .backend = PM_GALLERY_SESSION_TEST_BACKEND,
            .device = "contract-device",
            .renderer = "software-image",
            .readback = "host-frame",
            .nonce = "quote-\"-and-newline-\n",
            .frames_per_scene = 3U,
            .completed = true,
            .passed = true,
            .lifecycle = {
                {.id = "attach", .passed = true},
                {.id = "resize", .passed = true},
                {.id = "background", .passed = true},
                {.id = "failure_recovery", .passed = true},
            },
            .controls = {
                {.id = "scene_selection", .passed = true},
                {.id = "pause_resume", .passed = true},
            },
        };
        const std::string mobile_json = mobile_report.json();
        require(mobile_json.find(std::string(gallery::mobile_gallery_qualification_schema)) !=
                        std::string::npos &&
                    mobile_json.find("\\\"") != std::string::npos &&
                    mobile_json.find("\\n") != std::string::npos &&
                    mobile_json.find("\"renderer\":{\"kind\":\"software-image\"") !=
                        std::string::npos &&
                    mobile_json.find("\"failures\":[]") != std::string::npos,
                "Unified mobile qualification report schema changed");

        // Desktop supplies resolved limits and exact scene parameters instead
        // of making the common session select a mobile device profile.
        bool validated_scene = false;
        gallery::GallerySessionConfig desktop_config{
            .assets = root / "examples/assets",
            .profiles = root / "missing-device-profiles.json",
            .platform = "Desktop",
            .model = "test-host",
            .backend = PM_GALLERY_SESSION_TEST_BACKEND,
#ifdef PM_GALLERY_SESSION_TEST_NATIVE_BACKEND
            .native_backend = PM_GALLERY_SESSION_TEST_NATIVE_BACKEND,
#endif
            .workload_limits = gallery::GalleryWorkloadLimits{
                .name = "desktop-contract",
                .platform = "Desktop",
                .machine_model = "*",
                .backend = PM_GALLERY_SESSION_TEST_BACKEND,
                .brick_count = 4U,
                .fluid_particles = 64U,
                .coupled_bodies = std::numeric_limits<std::uint32_t>::max(),
                .smoke_particles = std::numeric_limits<std::uint32_t>::max(),
                .dump_spheres = 13U,
                .render_width = 96U,
                .render_height = 72U,
                .target_fps = 60U,
            },
            .bricks = gallery::BrickSceneConfig{4U, 1.5F, 2U},
            .dump_spheres = 13U,
            .fluid_particles = 64U,
            .initial_scene = static_cast<unsigned>(gallery::gallery_context_index(
                gallery::GalleryContext::fluid)),
            .replacement = gallery::GalleryReplacementPolicy::preserve_until_ready,
            .create_host_renderer = false,
            .prime_scene = false,
        };
        desktop_config.validate_scene =
            [&](gallery::GalleryContext, const gallery::SceneDefinition &,
                std::string &) {
                validated_scene = true;
                return true;
            };
        {
            gallery::GallerySession desktop(std::move(desktop_config));
            require(validated_scene && desktop.ready() &&
                        desktop.context() == gallery::GalleryContext::fluid &&
                        desktop.scene().fluid_options.capacity == 64U &&
                        desktop.revision() == 1U,
                    "Explicit desktop session configuration was not applied");
            auto parameters = desktop.scene_parameters();
            parameters.fluid_particles = 96U;
            desktop.set_scene_parameters(parameters);
            desktop.select(static_cast<unsigned>(gallery::gallery_context_index(
                gallery::GalleryContext::fluid)));
            require(desktop.scene().fluid_options.capacity == 96U &&
                        desktop.revision() == 2U,
                    "Desktop fluid parameter did not survive common selection");
            desktop.select(static_cast<unsigned>(gallery::gallery_context_index(
                gallery::GalleryContext::dump)));
            require(gallery::dump_payload_count(desktop.scene()) == 13U &&
                        desktop.revision() == 3U,
                    "Desktop dump parameter did not survive common selection");
            desktop.reset();
            require(gallery::dump_payload_count(desktop.scene()) == 13U &&
                        desktop.revision() == 4U,
                    "Common reset lost desktop scene parameters");
        }
        require(gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Explicit desktop session leaked its runtime");

        gallery::GalleryRuntimeCreateInfo valid;
        valid.asset = root / "examples/assets/RigidBody.glb";
        valid.width = 96U;
        valid.height = 72U;
#ifdef PM_GALLERY_SESSION_TEST_NATIVE_BACKEND
        valid.backend = PM_GALLERY_SESSION_TEST_NATIVE_BACKEND;
#endif
        gallery::GalleryRuntime runtime;
        std::string failure;
        require(runtime.replace(valid, failure),
                "Direct runtime creation failed");
        require(runtime.ready() &&
                    gallery::GalleryRuntime::live_runtime_count() == 1U,
                "Direct runtime did not own one scene");
        const PmDevice preserved_device = runtime.simulation.device();

        auto missing = valid;
        missing.asset = root / "examples/assets/Missing.glb";
        require(!runtime.replace(missing, failure) && runtime.ready() &&
                    runtime.simulation.device() == preserved_device &&
                    gallery::GalleryRuntime::live_runtime_count() == 1U,
                "Preflight failure did not preserve the live scene");

        auto rejected_before_release = valid;
        rejected_before_release.replacement =
            gallery::GalleryReplacementPolicy::release_before_device_create;
        rejected_before_release.configure_scene =
            [](gallery::SceneDefinition &, std::string &error) {
                error = "forced preflight failure";
                return false;
            };
        require(!runtime.replace(rejected_before_release, failure) &&
                    runtime.ready() &&
                    runtime.simulation.device() == preserved_device &&
                    gallery::GalleryRuntime::live_runtime_count() == 1U,
                "Release-first policy destroyed the scene during preflight");

        auto failed_after_release = valid;
        failed_after_release.replacement =
            gallery::GalleryReplacementPolicy::release_before_device_create;
        failed_after_release.prepare_runtime =
            [](gallery::GalleryRuntime &, std::string &error) {
                error = "forced post-create failure";
                return false;
            };
        require(!runtime.replace(failed_after_release, failure) &&
                    !runtime.ready() &&
                    gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Post-create failure retained stale runtime resources");
        require(!runtime.reset(failure) && !runtime.ready() &&
                    gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Retry failure leaked a runtime");

        require(runtime.replace(valid, failure) &&
                    gallery::GalleryRuntime::live_runtime_count() == 1U,
                "Runtime did not recover after replacement failure");
        runtime.clear();
        runtime.clear();
        require(gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Repeated teardown released the runtime more than once");

        gallery::GalleryRuntimeCreateInfo kinematic = valid;
        kinematic.asset = root / "examples/assets/PassiveActive.glb";
        kinematic.create_host_renderer = false;
        gallery::GalleryRuntime kinematic_runtime;
        require(kinematic_runtime.replace(kinematic, failure) &&
                    kinematic_runtime.controller_state().moves_kinematic &&
                    kinematic_runtime.kinematic_index <
                        kinematic_runtime.instance.rigid_bodies.size(),
                "Rigid semantic action did not bind an authored kinematic body");
        const auto initial_kinematic =
            kinematic_runtime.kinematic_target.position;
        require(kinematic_runtime.controlled_step({
                    .right = 1.0F,
                    .forward = 1.0F,
                    .camera = kinematic_runtime.camera.camera(),
                }, {}, failure) &&
                    kinematic_runtime.kinematic_target.position.x >
                        initial_kinematic.x &&
                    kinematic_runtime.kinematic_target.position.z <
                        initial_kinematic.z,
                "Shared directional action did not move the kinematic target");
        kinematic_runtime.clear();

        gallery::GalleryRuntimeCreateInfo desktop_point = valid;
        desktop_point.context = gallery::GalleryContext::constraint_point;
        desktop_point.asset = root / "examples/assets/Celestial.glb";
        desktop_point.create_host_renderer = false;
        gallery::GalleryRuntime desktop_runtime;
        require(desktop_runtime.replace(desktop_point, failure),
                "Desktop adapter point runtime creation failed");
        require(desktop_runtime.controller_state().primary ==
                    gallery::GalleryPrimaryAction::release_restore_constraints &&
                    desktop_runtime.primary_action(failure) ==
                    gallery::GalleryActionOutcome::applied &&
                    desktop_runtime.controller_state().constraints_released,
                "Desktop semantic action did not match mobile release");
        require(desktop_runtime.primary_action(failure) ==
                    gallery::GalleryActionOutcome::reset_requested &&
                    desktop_runtime.reset(failure) &&
                    !desktop_runtime.controller_state().constraints_released,
                "Desktop semantic action did not match mobile reset");
        desktop_runtime.clear();
        require(gallery::GalleryRuntime::live_runtime_count() == 0U,
                "Desktop adapter controller test leaked its runtime");
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        if (std::string_view(error.what()) == "no Parallel Mater physical device") {
            std::cout << "SKIP: no selected Gallery physics device\n";
            return 77;
        }
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

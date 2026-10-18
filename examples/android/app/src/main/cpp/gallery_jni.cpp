// SPDX-License-Identifier: MIT
// Android owns JNI, surfaces and lifecycle. Shared C++ owns the Gallery session.
#if PM_ANDROID_VULKAN
#ifndef VK_USE_PLATFORM_ANDROID_KHR
#define VK_USE_PLATFORM_ANDROID_KHR 1
#endif
#include <vulkan/vulkan.h>
#endif
#include <parallel_mater_gallery/mobile_qualification.hpp>
#include <android/native_window_jni.h>
#include <android/log.h>
#include <jni.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#if !PM_ANDROID_VULKAN
#include <cstring>
#endif
#include <cmath>
#include <memory>
#include <locale>
#include <stdexcept>
#include <sstream>
#include <utility>

namespace {
using namespace parallel_mater::gallery;
constexpr const char *backend_name = PM_ANDROID_VULKAN ? "vulkan" : "slang-cpu";

std::string directional_label(GalleryControlPolicy policy) {
    switch (policy) {
    case GalleryControlPolicy::authored_force: return "Push objects";
    case GalleryControlPolicy::motor_drive: return "Drive and steer";
    case GalleryControlPolicy::none: return "Camera only";
    case GalleryControlPolicy::rigid_gravity:
    case GalleryControlPolicy::collector_gravity:
    case GalleryControlPolicy::peg_gravity:
    case GalleryControlPolicy::cloth_gravity: return "Tilt gravity";
    }
    return "Camera only";
}

std::string primary_action_label(const GalleryControllerState &controls) {
    switch (controls.primary) {
    case GalleryPrimaryAction::release_restore_constraints:
        return controls.constraints_released ? "Restore" : "Release";
    case GalleryPrimaryAction::toggle_dump_bed:
        return controls.dump_bed_angle > 0.05F ? "Lower bed" : "Raise bed";
    case GalleryPrimaryAction::none: return {};
    }
    return {};
}

std::string renderer_label(const GalleryPresentationState &state) {
    return state.renderer == GalleryRendererKind::native_gpu
        ? std::string(gallery_backend_label(state.backend)) + " GPU renderer"
        : "Software image renderer";
}

std::string readback_label(GalleryReadbackMode mode) {
    return mode == GalleryReadbackMode::diagnostic_only
        ? "Diagnostic captures only" : "Host image each frame";
}

std::string status_text(const MobileGalleryState &state) {
    if (state.failed) return state.failure;
    const auto &view = state.presentation;
    if (!view.ready) return "Gallery unavailable";
    std::ostringstream text;
    text << (view.paused ? "Paused · " : "")
         << gallery_backend_label(view.backend) << " · " << view.steps
         << " steps · "
         << (view.renderer == GalleryRendererKind::native_gpu
                 ? "GPU graphics" : "software graphics");
    if (view.particle_count != 0U)
        text << " · " << view.particle_count << " particles";
    return text.str();
}

std::string details_text(const GalleryPresentationState &state) {
    std::ostringstream text;
    text.imbue(std::locale::classic());
    text << state.device << "\n" << state.platform << " · "
         << gallery_backend_label(state.backend)
         << "\nScene: " << state.scene_name << " (" << state.scene_id << ")"
         << "\n" << state.scene_summary
         << "\nRequires: " << state.required_features
         << "\nRenderer: " << renderer_label(state)
         << "\nReadback: " << readback_label(state.readback)
         << "\n\nProfile: " << state.workload_profile
         << "\n" << state.fluid_particle_limit << " liquid particles · "
         << state.coupled_body_limit << " coupled bodies"
         << "\n" << state.width << " × " << state.height << " · target "
         << state.target_updates_per_second << " updates/s"
         << "\nHost work: " << state.host_work_frames_per_second << " frames/s";
    if (state.latest_timing) {
        const auto &sample = *state.latest_timing;
        text << "\nFrame work: " << sample.total_ms << " ms"
             << "\nPhysics: " << sample.physics_ms << " ms · render: "
             << sample.render_ms << " ms";
        if (sample.present_ms)
            text << " · present callback: " << *sample.present_ms << " ms";
    }
    text << "\n\nDrag to orbit. Pinch to zoom.";
    if (state.controls.directional != GalleryControlPolicy::none)
        text << " Direction buttons: "
             << directional_label(state.controls.directional) << '.';
    const std::string primary = primary_action_label(state.controls);
    if (!primary.empty()) text << " Use " << primary << " for the scene action.";
    text << "\n\nBuild " << state.build_revision;
    return text.str();
}

struct Session {
    std::unique_ptr<MobileGallerySession> mobile;
    ANativeWindow *window{};
#if PM_ANDROID_VULKAN
    VkInstance vulkan_instance{};
    VkSurfaceKHR vulkan_surface{};
#endif
    unsigned surface_width{};
    unsigned surface_height{};
    unsigned logged_frames{};
    std::uint64_t software_presentations{};
    ~Session() { release_surface(); }
    GallerySession &gallery() { return mobile->gallery(); }
    const GallerySession &gallery() const { return mobile->gallery(); }
    void detach_render_target() noexcept {
        if (window && mobile) mobile->detach_surface();
#if PM_ANDROID_VULKAN
        if (vulkan_surface && vulkan_instance)
            vkDestroySurfaceKHR(vulkan_instance, vulkan_surface, nullptr);
        vulkan_surface = {};
        vulkan_instance = {};
#endif
    }
    void release_surface() noexcept {
        detach_render_target();
        if (window) ANativeWindow_release(window);
        window = nullptr;
        surface_width = 0U;
        surface_height = 0U;
    }
#if PM_ANDROID_VULKAN
    GpuRenderTarget create_render_target(unsigned width, unsigned height) {
        const PmNativeDeviceInfo native = gallery().native_render_device();
        if (native.backend != PM_NATIVE_BACKEND_VULKAN ||
            native.instance == 0U || native.device == 0U)
            throw std::runtime_error(
                "Android Gallery did not receive its Vulkan physics device");
        vulkan_instance = reinterpret_cast<VkInstance>(
            static_cast<std::uintptr_t>(native.instance));
        VkAndroidSurfaceCreateInfoKHR info{};
        info.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
        info.window = window;
        const VkResult result = vkCreateAndroidSurfaceKHR(
            vulkan_instance, &info, nullptr, &vulkan_surface);
        if (result != VK_SUCCESS) {
            vulkan_instance = {};
            throw std::runtime_error(
                "Could not create Android Vulkan surface (VkResult " +
                std::to_string(static_cast<int>(result)) + ")");
        }
        return {
            PM_NATIVE_BACKEND_VULKAN,
            native.device,
            static_cast<PmNativeHandle>(reinterpret_cast<std::uintptr_t>(
                vulkan_surface)),
            width,
            height,
        };
    }
#endif
    void attach_render_target() {
        if (!window) return;
        GpuRenderTarget target{};
#if PM_ANDROID_VULKAN
        target = create_render_target(surface_width, surface_height);
#endif
        if (!mobile->attach_surface(surface_width, surface_height, target))
            throw std::runtime_error(mobile->state().failure);
#if !PM_ANDROID_VULKAN
        if (ANativeWindow_setBuffersGeometry(window, gallery().width,
                gallery().height, WINDOW_FORMAT_RGBA_8888) != 0) {
            mobile->report_failure("Cannot configure display surface");
            throw std::runtime_error("Cannot configure display surface");
        }
#endif
    }
    bool execute(MobileGalleryCommand command, float x = 0.0F,
                 float y = 0.0F, bool may_replace_runtime = false) {
#if PM_ANDROID_VULKAN
        // VkSurfaceKHR belongs to its creating VkInstance. Gallery runtime
        // replacement creates a fresh instance/device, so retire presentation
        // before the old runtime and bind it again only after replacement.
        if (may_replace_runtime && window) detach_render_target();
#endif
        const bool ok = mobile->execute(command, x, y);
#if PM_ANDROID_VULKAN
        if (may_replace_runtime && ok && window) attach_render_target();
#endif
        return ok;
    }
#if !PM_ANDROID_VULKAN
    void present(const GallerySession &frame) {
        if (!window) return;
        ANativeWindow_Buffer buffer{};
        // A retired surface can race its queued destruction callback.
        if (ANativeWindow_lock(window, &buffer, nullptr) != 0) return;
        if (buffer.format == WINDOW_FORMAT_RGBA_8888 &&
            buffer.width == static_cast<int>(frame.width) &&
            buffer.height == static_cast<int>(frame.height)) {
            for (unsigned y = 0; y < frame.height; ++y)
                std::memcpy(static_cast<std::uint32_t *>(buffer.bits) + y * buffer.stride,
                    frame.pixels.data() + (frame.height - 1 - y) * frame.width,
                    frame.width * sizeof(std::uint32_t));
        }
        (void)ANativeWindow_unlockAndPost(window);
        ++software_presentations;
    }
#endif
    void log_scene() {
        logged_frames = 0;
        const auto &g = gallery();
        const auto view = g.presentation();
        __android_log_print(ANDROID_LOG_INFO, "ParallelMater",
            "Physics buffers: %llu bytes (excludes driver-private allocations)",
            static_cast<unsigned long long>(g.allocated_bytes()));
        __android_log_print(ANDROID_LOG_INFO, "ParallelMater",
            "Scene ready: %s; backend=%.*s; device=%.*s; bodies=%zu; particles=%zu",
            g.scene_file().c_str(), static_cast<int>(view.backend.size()),
            view.backend.data(), static_cast<int>(view.device.size()),
            view.device.data(),
            view.moving_body_count, view.particle_count);
    }
};
Session &session(jlong handle) {
    if (!handle) throw std::runtime_error("Gallery not initialized");
    return *reinterpret_cast<Session *>(handle);
}
std::string string(JNIEnv *env, jstring text) {
    if (!text) throw std::runtime_error("Missing app path");
    const char *chars = env->GetStringUTFChars(text, nullptr);
    if (!chars) throw std::bad_alloc();
    std::string result(chars); env->ReleaseStringUTFChars(text, chars); return result;
}
void exception(JNIEnv *env, const char *message) {
    __android_log_print(ANDROID_LOG_ERROR, "ParallelMater", "%s", message);
    if (!env->ExceptionCheck()) env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), message);
}
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeSceneCatalog(
    JNIEnv *env, jclass, jlong handle) {
    try {
        const auto &catalog = session(handle).gallery().catalog();
        constexpr std::size_t fields = 6U;
        auto result = env->NewObjectArray(
            static_cast<jsize>(catalog.size() * fields),
            env->FindClass("java/lang/String"), nullptr);
        if (!result) return nullptr;
        for (std::size_t i = 0; i < catalog.size(); ++i) {
            const auto &item = catalog[i];
            const std::array<std::string, fields> values{
                std::string(item.id), std::string(item.scene.name),
                std::string(item.scene.summary), item.required_feature_names,
                item.available ? "1" : "0", item.unavailable_reason};
            for (std::size_t field = 0; field < fields; ++field) {
                auto value = env->NewStringUTF(values[field].c_str());
                if (!value) return nullptr;
                env->SetObjectArrayElement(
                    result, static_cast<jsize>(i * fields + field), value);
                env->DeleteLocalRef(value);
                if (env->ExceptionCheck()) return nullptr;
            }
        }
        return result;
    } catch (const std::exception &error) { exception(env, error.what()); return nullptr; }
}
extern "C" JNIEXPORT jstring JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeQualify(
    JNIEnv *env, jclass, jlong handle, jint count, jstring nonce) {
    try {
        auto &s = session(handle);
        auto &g = s.gallery();
        if (count < 1 || count > 100000)
            throw std::runtime_error("Invalid qualification arguments");
        if (!s.window)
            throw std::runtime_error("Qualification requires an attached surface");
#if PM_ANDROID_VULKAN
        // The common qualification sweep replaces physics runtimes. A
        // VkSurfaceKHR belongs to the VkInstance that created it, so qualify
        // against the public offscreen target and bind a fresh Android surface
        // to the final runtime afterward.
        s.detach_render_target();
#endif
        MobileGalleryQualificationConfig config{
            .frames_per_scene = static_cast<unsigned>(count),
            .width = std::max(g.width, 1U),
            .height = std::max(g.height, 1U),
            .nonce = string(env, nonce),
#if !PM_ANDROID_VULKAN
            .configure_surface = [&s](unsigned width, unsigned height,
                                      std::string &error) {
                if (!s.window) {
                    error = "Android qualification surface was detached";
                    return false;
                }
                if (ANativeWindow_setBuffersGeometry(
                        s.window, static_cast<int>(width),
                        static_cast<int>(height), WINDOW_FORMAT_RGBA_8888) != 0) {
                    error = "Cannot configure Android qualification surface";
                    return false;
                }
                return true;
            },
#else
            .configure_surface = {},
#endif
            .memory_footprint = [&g] { return g.allocated_bytes(); },
            .checkpoint = {},
        };
        MobileGalleryQualificationReport report;
        try {
            report = qualify_mobile_gallery(*s.mobile, std::move(config));
#if PM_ANDROID_VULKAN
            s.mobile->detach_surface();
            s.attach_render_target();
#endif
        } catch (...) {
#if PM_ANDROID_VULKAN
            try {
                s.mobile->detach_surface();
                if (s.window) s.attach_render_target();
            } catch (...) {
            }
#endif
            throw;
        }
        const std::string serialized = report.json();
        return env->NewStringUTF(serialized.c_str());
    } catch (const std::exception &error) { exception(env, error.what()); return nullptr; }
}
extern "C" JNIEXPORT jstring JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeRuntimeInfo(JNIEnv *env, jclass, jlong handle) {
    try {
        const auto &s = session(handle);
        const auto &g = s.gallery();
        const auto state = s.mobile->state();
        const auto &view = state.presentation;
        std::ostringstream info;
        info.imbue(std::locale::classic());
        info << "{\"revision\":\"" << view.build_revision
             << "\",\"backend\":\"" << view.backend
             << "\",\"frames\":" << view.steps
             << ",\"active\":" << (view.active ? "true" : "false")
             << ",\"paused\":" << (view.paused ? "true" : "false")
             << ",\"surface_ready\":" << (state.surface_ready ? "true" : "false")
             << ",\"surface_generation\":" << state.surface_generation
             << ",\"render_width\":" << view.width
             << ",\"render_height\":" << view.height
             << ",\"presentation\":\""
             << (view.renderer == GalleryRendererKind::native_gpu
                     ? "vulkan-direct" : "software-copy")
             << "\",\"ordinary_readback_pixels\":" << g.pixels.size()
             << ",\"software_presentations\":" << s.software_presentations
             << "}";
        return env->NewStringUTF(info.str().c_str());
    } catch (const std::exception &error) { exception(env, error.what()); return nullptr; }
}
extern "C" JNIEXPORT jlong JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeCreate(JNIEnv *env, jclass,
    jstring assets, jstring manifest, jstring profiles, jstring model, jstring cache) {
    try {
        const auto manifest_path = string(env, manifest);
        if (setenv("PM_DRIVER_FILES", manifest_path.c_str(), 1) != 0 ||
            setenv("PM_LAYER_PATH", std::filesystem::path(manifest_path).parent_path().c_str(), 1) != 0)
            throw std::runtime_error("Could not configure bundled driver/layer");
        if (PM_ANDROID_VULKAN && setenv("PM_VULKAN_CACHE_DIR", string(env, cache).c_str(), 1) != 0)
            throw std::runtime_error("Could not configure Vulkan pipeline cache");
        auto result = std::make_unique<Session>();
        GallerySessionConfig config;
        config.assets = string(env, assets);
        config.profiles = string(env, profiles);
        config.platform = "Android";
        config.model = string(env, model);
        config.backend = backend_name;
#if PM_ANDROID_VULKAN
        config.native_backend = PM_NATIVE_BACKEND_VULKAN;
        config.gpu_rendering = true;
#endif
        result->mobile = std::make_unique<MobileGallerySession>(std::move(config));
        const auto &limits = result->gallery().limits;
        __android_log_print(ANDROID_LOG_INFO, "ParallelMater",
            "Profile %s: %u bricks, %u fluid particles, %u coupled bodies, %ux%u, target %u FPS (unmeasured)",
            limits.name.c_str(), limits.brick_count, limits.fluid_particles,
            limits.coupled_bodies, limits.render_width, limits.render_height, limits.target_fps);
        result->log_scene();
        return reinterpret_cast<jlong>(result.release());
    } catch (const std::exception &error) { exception(env, error.what()); }
    catch (...) { exception(env, "Gallery initialization failed"); }
    return 0;
}
extern "C" JNIEXPORT void JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeDestroy(JNIEnv *, jclass, jlong handle) {
    delete reinterpret_cast<Session *>(handle);
    __android_log_print(
        ANDROID_LOG_INFO, "ParallelMater", "Gallery session destroyed");
}
extern "C" JNIEXPORT void JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeSurface(JNIEnv *env, jclass,
    jlong handle, jobject surface, jint w, jint h) {
    try {
        auto &s = session(handle);
        s.release_surface();
        s.window = surface ? ANativeWindow_fromSurface(env, surface) : nullptr;
        s.logged_frames = 0;
        if (!s.window) return;
        s.surface_width = static_cast<unsigned>(std::max(w, 1));
        s.surface_height = static_cast<unsigned>(std::max(h, 1));
        try {
            s.attach_render_target();
        } catch (...) {
            s.release_surface();
            throw;
        }
    } catch (const std::exception &error) { exception(env, error.what()); }
    catch (...) { exception(env, "Display initialization failed"); }
}
extern "C" JNIEXPORT jstring JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeFrame(JNIEnv *env, jclass, jlong handle) {
    try {
        auto &s = session(handle); auto &g = s.gallery();
#if PM_ANDROID_VULKAN
        const bool framed =
            s.mobile->frame_at(GallerySession::FrameClock::now());
#else
        const bool framed = s.mobile->frame_at(GallerySession::FrameClock::now(),
            [&](const GallerySession &frame) { s.present(frame); });
#endif
        const auto state = s.mobile->state();
        if (!framed && state.failed) throw std::runtime_error(state.failure);
        if (framed && !g.is_paused() && g.active() && ++s.logged_frames == 60) {
            const auto stats = g.timing.statistics();
            __android_log_print(ANDROID_LOG_INFO, "ParallelMater",
                "FrameWindow scene=%s backend=%s frames=%llu count=%zu mean_ms=%.3f median_ms=%.3f p95_ms=%.3f physics_ms=%.3f render_ms=%.3f present_ms=%.3f fps=%.2f particles=%zu bodies=%zu",
                g.scene_file().c_str(), backend_name, static_cast<unsigned long long>(g.frames), stats.count,
                stats.mean_ms, stats.median_ms, stats.p95_ms, stats.physics_ms, stats.render_ms,
                stats.present_ms, stats.cadence_fps, g.particle_count(), g.moving_bodies());
            s.logged_frames = 0;
        }
        const std::string status = status_text(state) + "\n" +
            directional_label(state.presentation.controls.directional);
        return env->NewStringUTF(status.c_str());
    } catch (const std::exception &error) { exception(env, error.what()); }
    catch (...) { exception(env, "Gallery frame failed"); }
    return nullptr;
}
extern "C" JNIEXPORT jstring JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeDetails(JNIEnv *env, jclass, jlong handle) {
    try {
        const std::string details = details_text(
            session(handle).gallery().presentation());
        return env->NewStringUTF(details.c_str());
    }
    catch (const std::exception &error) { exception(env, error.what()); }
    return nullptr;
}
extern "C" JNIEXPORT void JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeCommand(JNIEnv *env, jclass,
    jlong handle, jint command, jfloat x, jfloat y) {
    try {
        auto &s = session(handle);
        bool ok = false;
        switch (command) {
        case 0: ok = s.execute(MobileGalleryCommand::select_scene, x, y, true); break;
        case 1: ok = s.execute(MobileGalleryCommand::reset_scene, x, y, true); break;
        case 2: ok = s.mobile->execute(MobileGalleryCommand::set_paused, x); break;
        case 3: ok = s.mobile->execute(MobileGalleryCommand::steer, x, y); break;
        case 4: ok = s.mobile->execute(MobileGalleryCommand::orbit, x, y); break;
        case 5: ok = s.mobile->execute(MobileGalleryCommand::zoom, x); break;
        case 6: ok = s.execute(MobileGalleryCommand::primary_action, x, y, true); break;
        case 7: ok = s.execute(MobileGalleryCommand::set_count, x, y, true); break;
        case 8: ok = s.mobile->execute(MobileGalleryCommand::focus); break;
        case 9: s.mobile->set_foreground(x != 0); ok = true; break;
        case 10: ok = s.mobile->execute(MobileGalleryCommand::toggle_paused); break;
        default: throw std::runtime_error("Unknown Gallery command");
        }
        if (!ok) throw std::runtime_error(s.mobile->state().failure);
        if (command == 0 || command == 1) s.log_scene();
        if (command == 2 || command == 7 || command == 9) s.logged_frames = 0;
    } catch (const std::exception &error) { exception(env, error.what()); }
    catch (...) { exception(env, "Gallery command failed"); }
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeControlState(
    JNIEnv *env, jclass, jlong handle) {
    try {
        const auto state = session(handle).mobile->state();
        const auto &view = state.presentation;
        const std::string primary = primary_action_label(view.controls);
        const std::array<std::string, 7U> values{
            std::to_string(view.selection), view.paused ? "1" : "0",
            view.controls.primary != GalleryPrimaryAction::none ? "1" : "0",
            primary, directional_label(view.controls.directional),
            state.failed ? "1" : "0", state.failure};
        auto result = env->NewObjectArray(static_cast<jsize>(values.size()),
            env->FindClass("java/lang/String"), nullptr);
        if (!result) return nullptr;
        for (std::size_t i = 0; i < values.size(); ++i) {
            auto value = env->NewStringUTF(values[i].c_str());
            if (!value) return nullptr;
            env->SetObjectArrayElement(result, static_cast<jsize>(i), value);
            env->DeleteLocalRef(value);
            if (env->ExceptionCheck()) return nullptr;
        }
        return result;
    } catch (const std::exception &error) {
        exception(env, error.what());
        return nullptr;
    }
}

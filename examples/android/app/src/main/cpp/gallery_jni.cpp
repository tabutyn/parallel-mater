// SPDX-License-Identifier: MIT
// Android owns JNI, surfaces and lifecycle. Shared C++ owns the Gallery session.
#if PM_ANDROID_VULKAN
#ifndef VK_USE_PLATFORM_ANDROID_KHR
#define VK_USE_PLATFORM_ANDROID_KHR 1
#endif
#include <vulkan/vulkan.h>
#endif
#include <parallel_mater_gallery/mobile_session.hpp>
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
#include <stdexcept>
#include <sstream>

namespace {
using namespace parallel_mater::gallery;
constexpr const char *backend_name = PM_ANDROID_VULKAN ? "vulkan" : "slang-cpu";
struct Session {
    std::unique_ptr<MobileGallerySession> mobile;
    ANativeWindow *window{};
#if PM_ANDROID_VULKAN
    VkInstance vulkan_instance{};
    VkSurfaceKHR vulkan_surface{};
#endif
    unsigned logged_frames{};
    std::uint64_t software_presentations{};
    ~Session() { release_surface(); }
    GallerySession &gallery() { return mobile->gallery(); }
    const GallerySession &gallery() const { return mobile->gallery(); }
    void release_surface() noexcept {
        if (window && mobile) mobile->detach_surface();
#if PM_ANDROID_VULKAN
        if (vulkan_surface && vulkan_instance)
            vkDestroySurfaceKHR(vulkan_instance, vulkan_surface, nullptr);
        vulkan_surface = {};
        vulkan_instance = {};
#endif
        if (window) ANativeWindow_release(window);
        window = nullptr;
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
        __android_log_print(ANDROID_LOG_INFO, "ParallelMater",
            "Physics buffers: %llu bytes (excludes driver-private allocations)",
            static_cast<unsigned long long>(g.allocated_bytes()));
        __android_log_print(ANDROID_LOG_INFO, "ParallelMater",
            "Scene ready: %s; backend=%s; device=%s; bodies=%zu; particles=%zu",
            g.scene_file().c_str(), backend_name, g.device_name().c_str(), g.moving_bodies(), g.particle_count());
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
                std::string(item.scene.help), item.required_feature_names,
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
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeQualifyScene(JNIEnv *env, jclass, jlong handle, jint index, jint count) {
    try {
        auto &s = session(handle);
        auto &g = s.gallery();
        if (index < 0 || count < 1 || count > 100000) throw std::runtime_error("Invalid qualification arguments");
        if (!s.mobile->execute(MobileGalleryCommand::select_scene,
                               static_cast<float>(index)))
            throw std::runtime_error(s.mobile->state().failure);
        const auto stats = g.qualify(static_cast<unsigned>(count));
        const jdouble values[]{stats.mean_ms, stats.p95_ms, static_cast<double>(stats.count)};
        auto result = env->NewDoubleArray(3);
        if (result) env->SetDoubleArrayRegion(result, 0, 3, values);
        return result;
    } catch (const std::exception &error) { exception(env, error.what()); return nullptr; }
}
extern "C" JNIEXPORT jstring JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeRuntimeInfo(JNIEnv *env, jclass, jlong handle) {
    try {
        const auto &s = session(handle);
        const auto &g = s.gallery();
        const auto state = s.mobile->state();
        std::ostringstream info;
        info.imbue(std::locale::classic());
        info << "{\"revision\":\"" << GallerySession::build_revision() << "\",\"backend\":\"" << backend_name
             << "\",\"frames\":" << g.frames << ",\"active\":" << (g.active() ? "true" : "false")
             << ",\"paused\":" << (g.is_paused() ? "true" : "false")
             << ",\"surface_ready\":" << (state.surface_ready ? "true" : "false")
             << ",\"surface_generation\":" << state.surface_generation
             << ",\"render_width\":" << g.width
             << ",\"render_height\":" << g.height
             << ",\"presentation\":\""
             << (PM_ANDROID_VULKAN ? "vulkan-direct" : "software-copy")
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
        const unsigned width = static_cast<unsigned>(std::max(w, 1));
        const unsigned height = static_cast<unsigned>(std::max(h, 1));
        try {
            GpuRenderTarget target{};
#if PM_ANDROID_VULKAN
            target = s.create_render_target(width, height);
#endif
            if (!s.mobile->attach_surface(width, height, target))
                throw std::runtime_error(s.mobile->state().failure);
#if !PM_ANDROID_VULKAN
            if (ANativeWindow_setBuffersGeometry(s.window, s.gallery().width,
                    s.gallery().height, WINDOW_FORMAT_RGBA_8888) != 0) {
                s.mobile->report_failure("Cannot configure display surface");
                throw std::runtime_error("Cannot configure display surface");
            }
#endif
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
        const std::string status = state.status + "\n" + state.directional_label;
        return env->NewStringUTF(status.c_str());
    } catch (const std::exception &error) { exception(env, error.what()); }
    catch (...) { exception(env, "Gallery frame failed"); }
    return nullptr;
}
extern "C" JNIEXPORT jstring JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeDetails(JNIEnv *env, jclass, jlong handle) {
    try { return env->NewStringUTF(session(handle).gallery().details().c_str()); }
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
        case 0: ok = s.mobile->execute(MobileGalleryCommand::select_scene, x); break;
        case 1: ok = s.mobile->execute(MobileGalleryCommand::reset_scene); break;
        case 2: ok = s.mobile->execute(MobileGalleryCommand::set_paused, x); break;
        case 3: ok = s.mobile->execute(MobileGalleryCommand::steer, x, y); break;
        case 4: ok = s.mobile->execute(MobileGalleryCommand::orbit, x, y); break;
        case 5: ok = s.mobile->execute(MobileGalleryCommand::zoom, x); break;
        case 6: ok = s.mobile->execute(MobileGalleryCommand::primary_action); break;
        case 7: ok = s.mobile->execute(MobileGalleryCommand::set_count, x); break;
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
        const std::array<std::string, 7U> values{
            std::to_string(state.selection), state.paused ? "1" : "0",
            state.primary_action_available ? "1" : "0",
            state.primary_action_label, state.directional_label,
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

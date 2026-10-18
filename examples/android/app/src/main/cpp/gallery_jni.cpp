// SPDX-License-Identifier: MIT
// Android owns JNI, surfaces and lifecycle. Shared C++ owns the Gallery session.
#include <parallel_mater_gallery/session.hpp>
#include <android/native_window_jni.h>
#include <android/log.h>
#include <jni.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <sstream>

namespace {
using namespace parallel_mater::gallery;
constexpr const char *backend_name = PM_ANDROID_VULKAN ? "vulkan" : "slang-cpu";
struct Session {
    std::unique_ptr<GallerySession> gallery;
    ANativeWindow *window{};
    unsigned logged_frames{};
    ~Session() { if (window) ANativeWindow_release(window); }
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
    }
    void log_scene() {
        logged_frames = 0;
        const auto &g = *gallery;
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
        const auto &catalog = session(handle).gallery->catalog();
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
        auto &g = *session(handle).gallery;
        if (index < 0 || count < 1 || count > 100000) throw std::runtime_error("Invalid qualification arguments");
        g.select(static_cast<unsigned>(index));
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
        const auto &g = *session(handle).gallery;
        std::ostringstream info;
        info.imbue(std::locale::classic());
        info << "{\"revision\":\"" << GallerySession::build_revision() << "\",\"backend\":\"" << backend_name
             << "\",\"frames\":" << g.frames << ",\"active\":" << (g.active() ? "true" : "false")
             << ",\"paused\":" << (g.is_paused() ? "true" : "false") << "}";
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
        GallerySessionConfig config{.assets = string(env, assets), .profiles = string(env, profiles),
            .platform = "Android", .model = string(env, model), .backend = backend_name,
            .native_backend = std::nullopt};
        if (PM_ANDROID_VULKAN) config.native_backend = PM_NATIVE_BACKEND_VULKAN;
        result->gallery = std::make_unique<GallerySession>(std::move(config));
        const auto &limits = result->gallery->limits;
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
}
extern "C" JNIEXPORT void JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeSurface(JNIEnv *env, jclass,
    jlong handle, jobject surface, jint w, jint h) {
    try {
        auto &s = session(handle);
        if (s.window) ANativeWindow_release(s.window);
        s.window = surface ? ANativeWindow_fromSurface(env, surface) : nullptr;
        s.gallery->set_active(s.window != nullptr); s.logged_frames = 0;
        if (!s.window) return;
        s.gallery->resize(std::max(w, 1), std::max(h, 1));
        if (ANativeWindow_setBuffersGeometry(s.window, s.gallery->width, s.gallery->height, WINDOW_FORMAT_RGBA_8888) != 0)
            throw std::runtime_error("Cannot configure display surface");
    } catch (const std::exception &error) { exception(env, error.what()); }
    catch (...) { exception(env, "Display initialization failed"); }
}
extern "C" JNIEXPORT jstring JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeFrame(JNIEnv *env, jclass, jlong handle) {
    try {
        auto &s = session(handle); auto &g = *s.gallery;
        g.frame_at(GallerySession::FrameClock::now(),
            [&](const GallerySession &frame) { s.present(frame); });
        if (!g.is_paused() && g.active() && ++s.logged_frames == 60) {
            const auto stats = g.timing.statistics();
            __android_log_print(ANDROID_LOG_INFO, "ParallelMater",
                "FrameWindow scene=%s backend=%s frames=%llu count=%zu mean_ms=%.3f median_ms=%.3f p95_ms=%.3f physics_ms=%.3f render_ms=%.3f present_ms=%.3f fps=%.2f particles=%zu bodies=%zu",
                g.scene_file().c_str(), backend_name, static_cast<unsigned long long>(g.frames), stats.count,
                stats.mean_ms, stats.median_ms, stats.p95_ms, stats.physics_ms, stats.render_ms,
                stats.present_ms, stats.cadence_fps, g.particle_count(), g.moving_bodies());
            s.logged_frames = 0;
        }
        return env->NewStringUTF(g.status().c_str());
    } catch (const std::exception &error) { exception(env, error.what()); }
    catch (...) { exception(env, "Gallery frame failed"); }
    return nullptr;
}
extern "C" JNIEXPORT jstring JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeDetails(JNIEnv *env, jclass, jlong handle) {
    try { return env->NewStringUTF(session(handle).gallery->details().c_str()); }
    catch (const std::exception &error) { exception(env, error.what()); }
    return nullptr;
}
extern "C" JNIEXPORT void JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeCommand(JNIEnv *env, jclass,
    jlong handle, jint command, jfloat x, jfloat y) {
    try {
        auto &s = session(handle); auto &g = *s.gallery;
        if (!std::isfinite(x) || !std::isfinite(y)) throw std::runtime_error("Non-finite Gallery input");
        switch (command) {
        case 0:
            if (x < 0 || x >= g.catalog().size())
                throw std::runtime_error("Unknown Gallery scene");
            g.select(static_cast<unsigned>(x)); s.log_scene(); break;
        case 1: g.reset(); s.log_scene(); break;
        case 2: g.set_paused(x != 0); s.logged_frames = 0; break;
        case 3: g.steer(x, y); break;
        case 4: g.orbit(x, y); break;
        case 5: g.zoom(x); break;
        case 6: (void)g.primary_action(); break;
        case 7: g.set_bricks(static_cast<unsigned>(std::max(1.0F, x))); s.logged_frames = 0; break;
        case 8: g.focus(); break;
        case 9: g.set_active(x != 0); s.logged_frames = 0; break;
        default: throw std::runtime_error("Unknown Gallery command");
        }
    } catch (const std::exception &error) { exception(env, error.what()); }
    catch (...) { exception(env, "Gallery command failed"); }
}
extern "C" JNIEXPORT jint JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeTargetFps(JNIEnv *, jclass, jlong handle) {
    return handle ? static_cast<jint>(reinterpret_cast<Session *>(handle)->gallery->limits.target_fps) : 30;
}

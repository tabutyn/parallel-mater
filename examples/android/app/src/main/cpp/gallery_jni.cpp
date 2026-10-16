// SPDX-License-Identifier: MIT
#include "scene_builder.hpp"
#include "render/render_scene.hpp"
#include <parallel_mater_gallery/renderer.hpp>
#include <parallel_mater_gallery/gallery_context.hpp>
#include <parallel_mater_gallery/arrow_forces.hpp>
#include <parallel_mater_gallery/fixed_collector.hpp>
#include <android/native_window_jni.h>
#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace {
using namespace parallel_mater;
using namespace parallel_mater::gallery;

constexpr const char *scene_files[]{"RigidBody.glb", "ConstraintFixed.glb",
    "Celestial.glb", "ConstraintHinge.glb", "ConstraintPiston.glb",
    "ConstraintGeneric.glb", "Fluid.glb", "FluidRigid.glb"};
constexpr std::size_t scene_count = std::size(scene_files);
constexpr GalleryContext contexts[]{GalleryContext::rigid_body,
    GalleryContext::constraint_fixed, GalleryContext::constraint_point,
    GalleryContext::constraint_hinge, GalleryContext::constraint_piston,
    GalleryContext::constraint_generic, GalleryContext::fluid,
    GalleryContext::fluid_rigid};

struct Scene {
    SceneDefinition definition;
    CommandScene simulation;
    CommandSceneInstance instance;
    CommandSceneFrame output;
    RenderSceneFrame frame;
    RendererHandle renderer;
    ArrowForces arrows;
    FixedContactCollector collector;
};

// All calls run on the Activity's one worker thread, including destruction.
struct Session {
    std::filesystem::path assets;
    GalleryWorkloadLimits limits;
    std::unique_ptr<Scene> scene;
    CameraController camera;
    ANativeWindow *window{};
    std::uint32_t width{640U}, height{360U};
    int selection{};
    int bricks{8};
    bool paused{}, released{};
    float right{}, up{};
    std::uint64_t frames{};
    double work_sum{}, work_mean{};
    unsigned work_count{};
    std::vector<std::uint32_t> pixels;
    std::string error;

    ~Session() { if (window) ANativeWindow_release(window); }

    std::size_t moving_bodies() const {
        return static_cast<std::size_t>(std::count_if(
            scene->definition.rigid_bodies.begin(), scene->definition.rigid_bodies.end(),
            [](const auto &body) { return body.options.motion == MotionType::dynamic; }));
    }

    void focus() {
        camera = CameraController{};
        if (selection < 6) {
            camera.set_preset(selection == 0
                ? brick_camera_preset({static_cast<std::uint32_t>(bricks), 1.0F, 1U})
                : gallery_entry(contexts[selection]).camera);
            return;
        }
        // Frame the active particles and moving bodies, not the large static
        // course. Focus can be tapped again after the action moves downstream.
        Vec3 low{INFINITY, INFINITY, INFINITY}, high{-INFINITY, -INFINITY, -INFINITY};
        const auto include = [&](float x, float y, float z) {
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) return;
            low = {std::min(low.x, x), std::min(low.y, y), std::min(low.z, z)};
            high = {std::max(high.x, x), std::max(high.y, y), std::max(high.z, z)};
        };
        for (const auto &p : scene->output.fluid_particles)
            include(p.position[0], p.position[1], p.position[2]);
        for (std::size_t i = 0; i < scene->output.rigid_states.size(); ++i) {
            if (scene->definition.rigid_bodies[i].options.motion != MotionType::dynamic) continue;
            const auto &p = scene->output.rigid_states[i].position;
            include(p[0] - 0.35F, p[1] - 0.35F, p[2] - 0.35F);
            include(p[0] + 0.35F, p[1] + 0.35F, p[2] + 0.35F);
        }
        if (!std::isfinite(low.x)) {
            camera.set_preset(gallery_entry(contexts[selection]).camera);
            return;
        }
        const Vec3 extent{high.x-low.x, high.y-low.y, high.z-low.z};
        const float radius = 0.5F * std::sqrt(extent.x*extent.x +
            extent.y*extent.y + extent.z*extent.z);
        camera.set_preset({.target = {(low.x+high.x)*0.5F,
            (low.y+high.y)*0.5F, (low.z+high.z)*0.5F},
            .distance_scale = std::max(0.28F, (radius + 0.25F) * 2.8F / 11.0F),
            .pitch = 0.8F});
    }

    bool select(int index) {
        if (index < 0 || index >= static_cast<int>(scene_count)) {
            error = "Scene unavailable in this preview";
            return false;
        }
        auto next = std::make_unique<Scene>();
        SceneDefinition authored;
        if (!load_glb_scene(assets / scene_files[index], authored, error))
            return false;
        const BrickSceneConfig config{static_cast<std::uint32_t>(bricks), 1.0F, 1U};
        if (index == 0) {
            if (!make_brick_scene(authored, config, next->definition, error))
                return false;
        } else next->definition = std::move(authored);
        if (index >= 6) {
            if (!apply_liquid_workload_limits(next->definition, limits, error)) return false;
        }
        if (!CommandScene::create(next->definition, 0U, next->simulation,
                                  next->instance, error) ||
            !next->arrows.initialize(next->definition, next->instance, error))
            return false;
        if (index == 1 && !next->collector.initialize(
                next->definition, next->instance, error)) return false;
        if (!create_renderer(next->definition, next->instance, width, height,
                             next->renderer, error) ||
            !next->simulation.step(1.0F / 60.0F, 4U,
                {0, -9.81F * next->definition.gravity_scale, 0},
                next->output, error) ||
            !make_render_scene_frame(next->definition, next->output,
                                      next->frame, error)) return false;
        scene = std::move(next);
        selection = index;
        right = up = 0;
        released = false;
        frames = 0;
        work_sum = work_mean = 0;
        work_count = 0;
        focus();
        error.clear();
        __android_log_print(ANDROID_LOG_INFO, "ParallelMater",
            "Scene ready: %s; bodies=%zu", scene_files[index],
            scene->definition.rigid_bodies.size());
        return true;
    }

    bool advance() {
        const float magnitude = 9.81F * scene->definition.gravity_scale;
        const float steering_right = scene->arrows.active() ? 0 : right;
        const float steering_up = scene->arrows.active() ? 0 : up;
        const Vec3 gravity = screen_space_gravity(camera.camera(),
            steering_right, steering_up, magnitude, 30.0F);
        if (scene->collector.active() &&
            !scene->collector.apply_loose_gravity(scene->simulation,
                scene->definition, scene->instance, {0, -magnitude, 0},
                gravity, error)) return false;
        if (!scene->arrows.apply(scene->simulation, camera.camera(), right,
                                 up, error) ||
            !scene->simulation.step(1.0F / limits.target_fps,
                (240U + limits.target_fps - 1U) / limits.target_fps, gravity, scene->output,
                error, scene->collector.active())) return false;
        if (scene->collector.active() && !scene->collector.collect(
                scene->simulation, scene->definition, scene->instance,
                scene->output, error)) return false;
        ++frames;
        return make_render_scene_frame(scene->definition, scene->output,
                                        scene->frame, error);
    }

    bool present() {
        if (!window || !scene) return true;
        if (!scene->renderer.render(scene->frame, camera.camera(), pixels,
                                     error, nullptr, FluidRenderMode::particles))
            return false;
        ANativeWindow_Buffer buffer{};
        if (ANativeWindow_lock(window, &buffer, nullptr) != 0) {
            // Android may retire the surface while its destruction callback
            // waits in the worker queue. Skip presentation during that handoff.
            return true;
        }
        if (buffer.format == WINDOW_FORMAT_RGBA_8888 &&
            buffer.width == static_cast<int>(width) &&
            buffer.height == static_cast<int>(height)) {
            for (std::uint32_t y = 0; y < height; ++y)
                std::memcpy(static_cast<std::uint32_t *>(buffer.bits) +
                                y * buffer.stride,
                            pixels.data() + (height - 1U - y) * width,
                            width * sizeof(std::uint32_t));
        }
        (void)ANativeWindow_unlockAndPost(window);
        return true;
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
    std::string result(chars);
    env->ReleaseStringUTFChars(text, chars);
    return result;
}

void exception(JNIEnv *env, const char *text) {
    __android_log_print(ANDROID_LOG_ERROR, "ParallelMater", "%s", text);
    if (!env->ExceptionCheck())
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), text);
}
} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeCreate(
    JNIEnv *env, jclass, jstring assets, jstring manifest, jstring catalog_path,
    jstring model) {
    try {
        if (setenv("PM_DRIVER_FILES", string(env, manifest).c_str(), 1) != 0)
            throw std::runtime_error("Could not configure bundled driver");
        if (setenv("PM_LAYER_PATH", std::filesystem::path(string(env, manifest))
                .parent_path().c_str(), 1) != 0)
            throw std::runtime_error("Could not configure bundled validation layer");
        auto result = std::make_unique<Session>();
        result->assets = string(env, assets);
        DeviceProfileCatalog catalog;
        if (!load_device_profiles(string(env, catalog_path), catalog, result->error))
            throw std::runtime_error(result->error);
        const auto *limits = select_workload_limits(
            catalog, "Android", string(env, model), "slang-cpu");
        if (!limits) throw std::runtime_error("Android workload profile missing");
        result->limits = *limits;
        result->bricks = static_cast<int>(limits->brick_count);
        result->width = limits->render_width;
        result->height = limits->render_height;
        __android_log_print(ANDROID_LOG_INFO, "ParallelMater",
            "Profile %s: %u bricks, %u fluid particles, %u coupled bodies, %ux%u, target %u FPS (unmeasured)",
            limits->name.c_str(), limits->brick_count, limits->fluid_particles,
            limits->coupled_bodies,
            limits->render_width, limits->render_height, limits->target_fps);
        if (!result->select(0)) throw std::runtime_error(result->error);
        return reinterpret_cast<jlong>(result.release());
    } catch (const std::exception &error) { exception(env, error.what()); }
    catch (...) { exception(env, "Gallery initialization failed"); }
    return 0;
}

extern "C" JNIEXPORT void JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeDestroy(
    JNIEnv *, jclass, jlong handle) { delete reinterpret_cast<Session *>(handle); }

extern "C" JNIEXPORT void JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeSurface(
    JNIEnv *env, jclass, jlong handle, jobject surface, jint w, jint h) {
    try {
        auto &s = session(handle);
        if (s.window) ANativeWindow_release(s.window);
        s.window = surface ? ANativeWindow_fromSurface(env, surface) : nullptr;
        if (!s.window) return;
        const float scale = std::min({1.0F,
            static_cast<float>(s.limits.render_width) / std::max(w, 1),
            static_cast<float>(s.limits.render_height) / std::max(h, 1)});
        s.width = std::max(1U, static_cast<std::uint32_t>(std::max(w, 1) * scale));
        s.height = std::max(1U, static_cast<std::uint32_t>(std::max(h, 1) * scale));
        if (ANativeWindow_setBuffersGeometry(s.window, s.width, s.height,
                                             WINDOW_FORMAT_RGBA_8888) != 0)
            throw std::runtime_error("Cannot configure display surface");
        if (!create_renderer(s.scene->definition, s.scene->instance,
                              s.width, s.height, s.scene->renderer, s.error))
            throw std::runtime_error(s.error);
    } catch (const std::exception &error) { exception(env, error.what()); }
    catch (...) { exception(env, "Display initialization failed"); }
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeFrame(
    JNIEnv *env, jclass, jlong handle) {
    try {
        auto &s = session(handle);
        const auto start = std::chrono::steady_clock::now();
        if (!s.paused && !s.advance()) throw std::runtime_error(s.error);
        if (!s.present()) throw std::runtime_error(s.error);
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (!s.paused) {
            s.work_sum += milliseconds;
            if (++s.work_count == 60) {
                s.work_mean = s.work_sum / s.work_count;
                s.work_count = 0;
                s.work_sum = 0;
            }
        }
        char text[256];
        const auto bodies = s.moving_bodies();
        std::string population = s.selection >= 6
            ? std::to_string(s.scene->output.fluid_particles.size()) + " particles" : "";
        if (bodies > 0) {
            if (!population.empty()) population += " · ";
            population += std::to_string(bodies) + (bodies == 1 ? " moving body" : " moving bodies");
        }
        std::snprintf(text, sizeof(text), "%s%s\n%s",
            s.paused ? "Paused · " : "", population.c_str(),
            s.scene->arrows.active() ? "Push objects" : "Tilt gravity");
        return env->NewStringUTF(text);
    } catch (const std::exception &error) { exception(env, error.what()); }
    catch (...) { exception(env, "Physics frame failed"); }
    return nullptr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeDetails(JNIEnv *env, jclass, jlong handle) {
    try {
        const auto &s = session(handle);
        const double mean = s.work_mean > 0 ? s.work_mean
            : (s.work_count ? s.work_sum / s.work_count : 0);
        char text[768];
        std::snprintf(text, sizeof(text),
            "CPU preview · Slang physics\n\n"
            "Profile: %s\nLimits: %u particles · %u coupled bodies\n"
            "Render: %u × %u · target %u FPS\n"
            "Average frame work: %.1f ms\n(last window, up to 60 running frames)\n"
            "Frames simulated: %llu\n\n"
            "Workload limits are provisional. Slow frames slow simulation time.",
            s.limits.name.c_str(), s.limits.fluid_particles, s.limits.coupled_bodies,
            s.width, s.height, s.limits.target_fps, mean,
            static_cast<unsigned long long>(s.frames));
        return env->NewStringUTF(text);
    } catch (const std::exception &error) { exception(env, error.what()); }
    return nullptr;
}

extern "C" JNIEXPORT void JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeCommand(
    JNIEnv *env, jclass, jlong handle, jint command, jfloat x, jfloat y) {
    try {
        auto &s = session(handle);
        bool ok = true;
        switch (command) {
        case 0: ok = s.select(static_cast<int>(x)); break;
        case 1: ok = s.select(s.selection); break;
        case 2: s.paused = x != 0; break;
        case 3: s.right = std::clamp(x, -1.0F, 1.0F);
                s.up = std::clamp(y, -1.0F, 1.0F); break;
        case 4: s.camera.begin_drag(CameraDragMode::orbit, 0, 0);
                s.camera.move_cursor(x, y, s.height); s.camera.end_drag(); break;
        case 5: s.camera.zoom(x); break;
        case 6:
            if (s.selection != 2) break;
            if (s.released) { ok = s.select(s.selection); break; }
            for (auto slot : s.scene->instance.rigid_constraints)
                if (!s.scene->simulation.set_constraint_active(slot, false,
                                                               s.error)) {
                    ok = false; break;
                }
            if (ok) s.released = true;
            break;
        case 7: s.bricks = std::clamp(static_cast<int>(x), 1,
                                     static_cast<int>(s.limits.brick_count));
                if (s.selection == 0) ok = s.select(0); break;
        case 8: s.focus(); break;
        default: throw std::runtime_error("Unknown gallery command");
        }
        if (!ok) throw std::runtime_error(s.error);
    } catch (const std::exception &error) { exception(env, error.what()); }
    catch (...) { exception(env, "Gallery command failed"); }
}

extern "C" JNIEXPORT jint JNICALL
Java_org_parallelmater_gallery_GalleryActivity_nativeTargetFps(
    JNIEnv *, jclass, jlong handle) {
    return handle ? static_cast<jint>(reinterpret_cast<Session *>(handle)->limits.target_fps) : 30;
}

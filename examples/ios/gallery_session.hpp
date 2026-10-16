// SPDX-License-Identifier: MIT
#pragma once
#include <parallel_mater_gallery/device_profiles.hpp>
#include <parallel_mater_gallery/frame_timing.hpp>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace parallel_mater::gallery {
// Worker-owned adapter to shared Gallery services; no platform physics or timer.
class IPadGallerySession {
public:
    IPadGallerySession(const std::filesystem::path &resources, const std::string &model);
    ~IPadGallerySession();
    void select(unsigned index);
    void reset();
    void resize(unsigned width, unsigned height);
    void orbit(float x, float y);
    void zoom(float amount);
    void focus();
    void release();
    void frame();
    void validate_snapshot() const;
    std::string details() const;
    std::string status() const;
    static std::vector<std::string> scene_names();
    unsigned selection() const { return selected_; }
    GalleryWorkloadLimits limits;
    bool paused{};
    float right{}, up{};
    unsigned width{}, height{};
    std::uint64_t frames{};
    std::vector<std::uint32_t> pixels;
    FrameTimingCollector timing;
private:
    struct Scene;
    std::unique_ptr<Scene> scene_;
    std::filesystem::path resources_;
    unsigned selected_{};
    bool released_{};
    std::string device_;
};
}

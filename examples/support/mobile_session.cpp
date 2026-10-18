// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/mobile_session.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <utility>

namespace parallel_mater::gallery {
namespace {

std::string directional_label(GalleryControlPolicy policy) {
    switch (policy) {
    case GalleryControlPolicy::authored_force: return "Push objects";
    case GalleryControlPolicy::motor_drive: return "Drive motor";
    case GalleryControlPolicy::none: return "No directional control";
    case GalleryControlPolicy::rigid_gravity:
    case GalleryControlPolicy::collector_gravity:
    case GalleryControlPolicy::peg_gravity:
    case GalleryControlPolicy::cloth_gravity: return "Tilt gravity";
    }
    return "No directional control";
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

} // namespace

MobileGallerySession::MobileGallerySession(GallerySessionConfig config)
    : gallery_(std::move(config)) {
    gallery_.set_active(false);
}

void MobileGallerySession::synchronize_activity() noexcept {
    gallery_.set_active(should_frame());
}

void MobileGallerySession::clear_failure() noexcept {
    failed_ = false;
    failure_.clear();
    synchronize_activity();
}

void MobileGallerySession::report_failure(std::string failure) noexcept {
    failed_ = true;
    failure_ = failure.empty() ? "Mobile Gallery operation failed"
                               : std::move(failure);
    synchronize_activity();
}

bool MobileGallerySession::attach_surface(unsigned width, unsigned height,
                                          GpuRenderTarget target) noexcept {
    ++surface_generation_;
    surface_ready_ = true;
    try {
        gallery_.set_render_target(target);
        gallery_.resize(std::max(1U, width), std::max(1U, height));
        synchronize_activity();
        return true;
    } catch (const std::exception &error) {
        surface_ready_ = false;
        gallery_.set_render_target({});
        report_failure(error.what());
    } catch (...) {
        surface_ready_ = false;
        gallery_.set_render_target({});
        report_failure("Display surface attachment failed");
    }
    return false;
}

void MobileGallerySession::detach_surface() noexcept {
    ++surface_generation_;
    surface_ready_ = false;
    gallery_.set_render_target({});
    synchronize_activity();
}

bool MobileGallerySession::resize_surface(unsigned width,
                                          unsigned height) noexcept {
    if (!surface_ready_) return false;
    try {
        gallery_.resize(std::max(1U, width), std::max(1U, height));
        return true;
    } catch (const std::exception &error) {
        report_failure(error.what());
    } catch (...) {
        report_failure("Display surface resize failed");
    }
    return false;
}

void MobileGallerySession::set_foreground(bool foreground) noexcept {
    foreground_ = foreground;
    synchronize_activity();
}

bool MobileGallerySession::execute(MobileGalleryCommand command, float x,
                                   float y) noexcept {
    const bool recovery = command == MobileGalleryCommand::select_scene ||
                          command == MobileGalleryCommand::reset_scene;
    if (failed_ && !recovery) return false;
    try {
        if (!std::isfinite(x) || !std::isfinite(y))
            throw std::runtime_error("Non-finite Gallery input");
        switch (command) {
        case MobileGalleryCommand::select_scene:
            if (std::trunc(x) != x || x < 0.0F ||
                x >= static_cast<float>(gallery_.catalog().size()))
                throw std::runtime_error("Unknown Gallery scene");
            gallery_.select(static_cast<unsigned>(x));
            break;
        case MobileGalleryCommand::reset_scene: gallery_.reset(); break;
        case MobileGalleryCommand::toggle_paused:
            gallery_.set_paused(!gallery_.is_paused());
            break;
        case MobileGalleryCommand::set_paused: gallery_.set_paused(x != 0.0F); break;
        case MobileGalleryCommand::steer: gallery_.steer(x, y); break;
        case MobileGalleryCommand::orbit: gallery_.orbit(x, y); break;
        case MobileGalleryCommand::zoom: gallery_.zoom(x); break;
        case MobileGalleryCommand::primary_action:
            (void)gallery_.primary_action();
            break;
        case MobileGalleryCommand::set_count:
            gallery_.set_bricks(static_cast<unsigned>(std::clamp(x, 1.0F,
                static_cast<float>(gallery_.limits.brick_count))));
            break;
        case MobileGalleryCommand::focus: gallery_.focus(); break;
        }
        if (recovery) clear_failure();
        return true;
    } catch (const std::exception &error) {
        report_failure(error.what());
    } catch (...) {
        report_failure("Gallery command failed");
    }
    return false;
}

bool MobileGallerySession::frame(const Presenter &present) noexcept {
    if (!should_frame()) return false;
    try {
        gallery_.frame(present);
        return true;
    } catch (const std::exception &error) {
        report_failure(error.what());
    } catch (...) {
        report_failure("Gallery frame failed");
    }
    return false;
}

bool MobileGallerySession::frame_at(FrameClock::time_point timestamp,
                                    const Presenter &present) noexcept {
    if (!should_frame()) return false;
    try {
        gallery_.frame_at(timestamp, present);
        return true;
    } catch (const std::exception &error) {
        report_failure(error.what());
    } catch (...) {
        report_failure("Gallery frame failed");
    }
    return false;
}

MobileGalleryState MobileGallerySession::state() const {
    const GalleryControllerState controls = gallery_.controller_state();
    MobileGalleryState result{
        .selection = gallery_.selection(),
        .paused = gallery_.is_paused(),
        .active = gallery_.active(),
        .foreground = foreground_,
        .surface_ready = surface_ready_,
        .failed = failed_,
        .primary_action_available =
            controls.primary != GalleryPrimaryAction::none,
        .primary_action_active = controls.constraints_released ||
            (controls.primary == GalleryPrimaryAction::toggle_dump_bed &&
             controls.dump_bed_angle > 0.05F),
        .surface_generation = surface_generation_,
        .primary_action_label = primary_action_label(controls),
        .directional_label = directional_label(controls.directional),
        .status = failed_ ? failure_ :
            (gallery_.ready() ? gallery_.status() : "Gallery unavailable"),
        .failure = failure_,
    };
    return result;
}

} // namespace parallel_mater::gallery

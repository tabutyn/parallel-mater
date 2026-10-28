// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/mobile_qualification.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace parallel_mater::gallery {
namespace {
std::string json_string(std::string_view value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20U) {
                output << "\\u" << std::hex << std::setw(4)
                       << std::setfill('0') << static_cast<unsigned>(character)
                       << std::dec << std::setfill(' ');
            } else {
                output << static_cast<char>(character);
            }
        }
    }
    output << '"';
    return output.str();
}

const char *json_bool(bool value) noexcept { return value ? "true" : "false"; }

void write_check(std::ostream &output, const MobileQualificationCheck &check) {
    output << "{\"id\":" << json_string(check.id)
           << ",\"applicable\":" << json_bool(check.applicable)
           << ",\"passed\":" << json_bool(check.passed);
    if (!check.error.empty())
        output << ",\"error\":" << json_string(check.error);
    output << '}';
}

template <class Value, class Writer>
void write_array(std::ostream &output, const std::vector<Value> &values,
                 Writer writer) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0U) output << ',';
        writer(output, values[index]);
    }
    output << ']';
}

std::string failure_message(const MobileGallerySession &session,
                            std::string fallback) {
    const auto state = session.state();
    return state.failure.empty() ? std::move(fallback) : state.failure;
}
} // namespace

std::string MobileGalleryQualificationReport::json() const {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(17)
           << "{\"schema\":" << json_string(mobile_gallery_qualification_schema)
           << ",\"scope\":\"scene output, shared controls, host frame timing, public realtime work and platform thermal state; native traces and full physics qualification remain separate artifacts\""
           << ",\"revision\":" << json_string(revision)
           << ",\"platform\":" << json_string(platform)
           << ",\"backend\":" << json_string(backend)
           << ",\"execution_contract\":" << json_string(execution_contract)
           << ",\"device\":" << json_string(device)
           << ",\"renderer\":{\"kind\":" << json_string(renderer)
           << ",\"readback\":" << json_string(readback) << '}'
           << ",\"nonce\":" << json_string(nonce)
           << ",\"frames_per_scene\":" << frames_per_scene
           << ",\"completed\":" << json_bool(completed)
           << ",\"passed\":" << json_bool(passed)
           << ",\"lifecycle\":";
    write_array(output, lifecycle,
                [](std::ostream &stream, const MobileQualificationCheck &check) {
                    write_check(stream, check);
                });
    output << ",\"controls\":";
    write_array(output, controls,
                [](std::ostream &stream, const MobileQualificationCheck &check) {
                    write_check(stream, check);
                });
    output << ",\"mobile_session_contract\":{";
    for (std::size_t index = 0; index < lifecycle.size(); ++index) {
        if (index != 0U) output << ',';
        output << json_string(lifecycle[index].id) << ':'
               << json_bool(lifecycle[index].passed);
    }
    for (const auto &check : controls)
        output << ',' << json_string(check.id) << ':'
               << json_bool(!check.applicable || check.passed);
    output << "},\"failures\":";
    write_array(output, failures,
        [](std::ostream &stream, const MobileQualificationFailure &failure) {
            stream << "{\"stage\":" << json_string(failure.stage)
                   << ",\"message\":" << json_string(failure.message) << '}';
        });
    output << ",\"scenes\":";
    write_array(output, scenes,
        [](std::ostream &stream, const MobileQualificationSceneResult &scene) {
            stream << "{\"id\":" << json_string(scene.id)
                   << ",\"scene\":" << json_string(scene.scene)
                   << ",\"scene_asset\":" << json_string(scene.scene_asset)
                   << ",\"requirements\":" << json_string(scene.requirements)
                   << ",\"workload_profile\":"
                   << json_string(scene.workload_profile)
                   << ",\"available\":" << json_bool(scene.available)
                   << ",\"skipped\":" << json_bool(scene.skipped)
                   << ",\"passed\":" << json_bool(scene.passed)
                   << ",\"steps\":" << scene.steps
                   << ",\"mean_ms\":" << scene.mean_ms
                   << ",\"p50_ms\":" << scene.p50_ms
                   << ",\"p95_ms\":" << scene.p95_ms
                   << ",\"p99_ms\":" << scene.p99_ms
                   << ",\"timing_ms\":{\"frame\":{\"mean\":"
                   << scene.mean_ms << ",\"p50\":" << scene.p50_ms
                   << ",\"p95\":" << scene.p95_ms << ",\"p99\":"
                   << scene.p99_ms << "},\"physics\":{\"mean\":"
                   << scene.physics_mean_ms << ",\"p50\":"
                   << scene.physics_p50_ms << ",\"p95\":"
                   << scene.physics_p95_ms << ",\"p99\":"
                   << scene.physics_p99_ms << "},\"render\":{\"mean\":"
                   << scene.render_mean_ms << ",\"p50\":"
                   << scene.render_p50_ms << ",\"p95\":"
                   << scene.render_p95_ms << ",\"p99\":"
                   << scene.render_p99_ms << "},\"present\":{\"mean\":"
                   << scene.present_mean_ms << ",\"p50\":"
                   << scene.present_p50_ms << ",\"p95\":"
                   << scene.present_p95_ms << ",\"p99\":"
                   << scene.present_p99_ms << "}}"
                   << ",\"simulation_to_wall_clock_rate\":"
                   << scene.simulation_to_wall_clock_rate
                   << ",\"timing_window_frames\":"
                   << scene.timing_window_frames
                   << ",\"realtime_work\":{\"dispatches\":"
                   << scene.realtime_statistics.total.dispatches
                   << ",\"workgroups\":"
                   << scene.realtime_statistics.total.workgroups
                   << ",\"barriers\":"
                   << scene.realtime_statistics.total.barriers
                   << ",\"submissions\":"
                   << scene.realtime_statistics.total.queueSubmissions
                   << ",\"bytes_moved\":"
                   << scene.realtime_statistics.total.bytesMoved
                   << ",\"time_advances\":"
                   << scene.realtime_statistics.timeAdvances
                   << ",\"spatial_builds\":"
                   << scene.realtime_statistics.total.spatialIndexBuilds
                   << ",\"spatial_refits\":"
                   << scene.realtime_statistics.total.spatialIndexRefits
                   << ",\"spatial_publications\":"
                   << scene.realtime_statistics.total.spatialIndexPublications
                   << ",\"numerical_iterations\":"
                   << scene.realtime_statistics.total.numericalIterations
                   << ",\"intra_tick_host_readbacks\":"
                   << scene.realtime_statistics.total.intraTickHostReadbacks
                   << ",\"intra_tick_host_waits\":"
                   << scene.realtime_statistics.total.intraTickHostWaits
                   << ",\"domain_iterations\":{\"rigid\":"
                   << scene.realtime_statistics.domainIterations[0]
                   << ",\"fluid\":"
                   << scene.realtime_statistics.domainIterations[1]
                   << ",\"cloth\":"
                   << scene.realtime_statistics.domainIterations[2]
                   << ",\"soft_body\":"
                   << scene.realtime_statistics.domainIterations[3]
                   << ",\"rope\":"
                   << scene.realtime_statistics.domainIterations[4]
                   << ",\"smoke\":"
                   << scene.realtime_statistics.domainIterations[5] << '}'
                   << ",\"coupling_edges\":"
                   << scene.realtime_statistics.couplingEdgeCount
                   << ",\"coupling_sweeps\":"
                   << scene.realtime_statistics.couplingSweeps
                   << '}'
                   << ",\"thermal_state_before\":"
                   << json_string(scene.thermal_state_before)
                   << ",\"thermal_state_after\":"
                   << json_string(scene.thermal_state_after)
                   << ",\"session_controls\":true"
                   << ",\"memory_footprint_bytes\":"
                   << scene.memory_footprint_bytes
                   << ",\"workload\":{\"target_updates_per_second\":"
                   << scene.target_updates_per_second
                   << ",\"rigid_body_limit\":"
                   << scene.rigid_body_limit
                   << ",\"fluid_particle_limit\":"
                   << scene.fluid_particle_limit
                   << ",\"coupled_body_limit\":"
                   << scene.coupled_body_limit
                   << ",\"smoke_particle_limit\":"
                   << scene.smoke_particle_limit
                   << ",\"render_width\":" << scene.render_width
                   << ",\"render_height\":" << scene.render_height
                   << ",\"peak_particle_count\":"
                   << scene.peak_particle_count
                   << ",\"moving_body_count\":"
                   << scene.moving_body_count << '}';
            if (!scene.reason.empty())
                stream << ",\"reason\":" << json_string(scene.reason);
            if (!scene.error.empty())
                stream << ",\"error\":" << json_string(scene.error);
            stream << '}';
        });
    output << '}';
    return output.str();
}

MobileGalleryQualificationReport qualify_mobile_gallery(
    MobileGallerySession &session, MobileGalleryQualificationConfig config) {
    if (config.frames_per_scene == 0U || config.frames_per_scene > 100000U)
        throw std::invalid_argument(
            "Mobile Gallery qualification frames must be 1..100000");
    if (config.width == 0U || config.height == 0U)
        throw std::invalid_argument(
            "Mobile Gallery qualification surface must be non-zero");

    const auto identity = session.state().presentation;
    MobileGalleryQualificationReport report{
        .revision = std::string(identity.build_revision),
        .platform = std::string(identity.platform),
        .backend = std::string(identity.backend),
        .execution_contract = std::string(identity.execution_contract),
        .device = std::string(identity.device),
        .renderer = identity.renderer == GalleryRendererKind::native_gpu
            ? "native-gpu" : "software-image",
        .readback = identity.readback == GalleryReadbackMode::diagnostic_only
            ? "diagnostic-only" : "host-frame",
        .nonce = std::move(config.nonce),
        .frames_per_scene = config.frames_per_scene,
        .completed = false,
        .passed = false,
        .lifecycle = {},
        .controls = {},
        .scenes = {},
        .failures = {},
    };

    const auto checkpoint = [&] {
        if (config.checkpoint) {
            const std::string serialized = report.json();
            config.checkpoint(serialized);
        }
    };
    const auto failure = [&](std::string stage, std::string message) {
        report.failures.push_back(
            {.stage = std::move(stage), .message = std::move(message)});
    };
    const auto check = [&](std::vector<MobileQualificationCheck> &destination,
                           std::string id, bool applicable,
                           const std::function<void()> &operation) {
        MobileQualificationCheck result{
            .id = std::move(id),
            .applicable = applicable,
            .passed = !applicable,
            .error = {},
        };
        if (applicable) {
            try {
                operation();
                result.passed = true;
            } catch (const std::exception &error) {
                result.error = error.what();
            } catch (...) {
                result.error = "Unknown mobile qualification failure";
            }
        }
        if (result.applicable && !result.passed)
            failure(result.id, result.error);
        destination.push_back(std::move(result));
        checkpoint();
    };
    const auto require = [](bool condition, std::string message) {
        if (!condition) throw std::runtime_error(std::move(message));
    };
    const auto configure_surface = [&](unsigned width, unsigned height) {
        if (!config.configure_surface) return;
        std::string error;
        require(config.configure_surface(width, height, error),
                error.empty() ? "Native surface configuration failed" : error);
    };

    checkpoint();
    check(report.lifecycle, "attach", true, [&] {
        const auto generation = session.state().surface_generation;
        session.detach_surface();
        require(!session.state().surface_ready && !session.frame(),
                "Detached mobile surface admitted a frame");
        configure_surface(config.width, config.height);
        require(session.attach_surface(config.width, config.height,
                                       config.render_target),
                failure_message(session, "Surface attachment failed"));
        const auto state = session.state();
        require(state.surface_ready && state.surface_generation >= generation + 2U &&
                    session.gallery().width == config.width &&
                    session.gallery().height == config.height,
                "Replacement mobile surface did not become current");
    });

    check(report.lifecycle, "resize", session.state().surface_ready, [&] {
        const unsigned changed_width = config.width > 1U
            ? config.width - 1U : config.width + 1U;
        const unsigned changed_height = config.height > 1U
            ? config.height - 1U : config.height + 1U;
        configure_surface(changed_width, changed_height);
        require(session.resize_surface(changed_width, changed_height) &&
                    session.gallery().width == changed_width &&
                    session.gallery().height == changed_height,
                failure_message(session, "Surface resize failed"));
        configure_surface(config.width, config.height);
        require(session.resize_surface(config.width, config.height) &&
                    session.gallery().width == config.width &&
                    session.gallery().height == config.height,
                failure_message(session, "Surface resize restoration failed"));
    });

    check(report.lifecycle, "background", true, [&] {
        const auto frames = session.gallery().frames;
        session.set_foreground(false);
        require(!session.should_frame() && !session.frame() &&
                    !session.state().presentation.active &&
                    session.gallery().frames == frames,
                "Background mobile session admitted work");
    });
    check(report.lifecycle, "foreground", true, [&] {
        session.set_foreground(true);
        require(session.should_frame() && session.state().presentation.active,
                "Foreground mobile session did not recover");
    });
    check(report.lifecycle, "failure_recovery", session.should_frame(), [&] {
        require(!session.frame([](const GallerySession &) {
                    throw std::runtime_error(
                        "qualification presentation failure");
                }) && session.state().failed && !session.should_frame(),
                "Presentation failure did not stop mobile scheduling");
        require(session.execute(MobileGalleryCommand::reset_scene) &&
                    !session.state().failed && session.should_frame(),
                "Reset did not recover the mobile session");
    });

    const auto &catalog = session.gallery().catalog();
    const auto first_available = std::find_if(catalog.begin(), catalog.end(),
        [](const GalleryCatalogItem &item) { return item.available; });
    const bool has_available = first_available != catalog.end();
    const unsigned first_index = has_available
        ? static_cast<unsigned>(first_available - catalog.begin()) : 0U;
    check(report.controls, "scene_selection", has_available, [&] {
        require(session.execute(MobileGalleryCommand::select_scene,
                                static_cast<float>(first_index)) &&
                    session.gallery().selection() == first_index,
                failure_message(session, "Scene selection failed"));
    });
    check(report.controls, "pause_resume", has_available, [&] {
        require(session.execute(MobileGalleryCommand::toggle_paused) &&
                    session.state().presentation.paused &&
                    session.execute(MobileGalleryCommand::toggle_paused) &&
                    !session.state().presentation.paused,
                failure_message(session, "Pause/resume failed"));
    });
    check(report.controls, "camera", has_available, [&] {
        require(session.execute(MobileGalleryCommand::orbit, 1.0F, -1.0F) &&
                    session.execute(MobileGalleryCommand::zoom, 1.0F) &&
                    session.execute(MobileGalleryCommand::focus),
                failure_message(session, "Camera controls failed"));
    });
    check(report.controls, "directional", has_available, [&] {
        require(session.execute(MobileGalleryCommand::steer, 0.25F, -0.25F),
                failure_message(session, "Directional control failed"));
    });

    const auto count_scene = std::find_if(catalog.begin(), catalog.end(),
        [](const GalleryCatalogItem &item) {
            return item.available &&
                item.scene.count_kind != GalleryCountKind::none;
        });
    check(report.controls, "count", count_scene != catalog.end(), [&] {
        const unsigned index = static_cast<unsigned>(count_scene - catalog.begin());
        require(session.execute(MobileGalleryCommand::select_scene,
                                static_cast<float>(index)) &&
                    session.execute(MobileGalleryCommand::set_count,
                                    static_cast<float>(
                                        std::max(1U, count_scene->scene.minimum_count))),
                failure_message(session, "Count control failed"));
    });

    const auto primary_scene = std::find_if(catalog.begin(), catalog.end(),
        [](const GalleryCatalogItem &item) {
            return item.available &&
                item.scene.primary_action != GalleryPrimaryAction::none;
        });
    check(report.controls, "primary_action", primary_scene != catalog.end(), [&] {
        const unsigned index =
            static_cast<unsigned>(primary_scene - catalog.begin());
        require(session.execute(MobileGalleryCommand::select_scene,
                                static_cast<float>(index)) &&
                    session.execute(MobileGalleryCommand::primary_action),
                failure_message(session, "Primary action failed"));
        require(session.execute(MobileGalleryCommand::reset_scene),
                failure_message(session, "Primary action reset failed"));
    });

    check(report.lifecycle, "invalid_scene_recovery", has_available, [&] {
        const unsigned selection = session.gallery().selection();
        require(!session.execute(MobileGalleryCommand::select_scene,
                                 static_cast<float>(catalog.size())) &&
                    session.state().failed &&
                    session.gallery().selection() == selection,
                "Invalid scene did not latch a recoverable failure");
        require(session.execute(MobileGalleryCommand::reset_scene) &&
                    !session.state().failed && session.should_frame(),
                "Invalid-scene recovery failed");
    });

    for (unsigned index = 0U; index < catalog.size(); ++index) {
        const auto &item = catalog[index];
        MobileQualificationSceneResult result{
            .id = std::string(item.id),
            .scene = std::string(item.scene.name),
            .scene_asset = std::string(item.asset_file),
            .requirements = item.required_feature_names,
            .workload_profile = {},
            .available = item.available,
            .skipped = !item.available,
            .passed = false,
            .steps = config.frames_per_scene,
            .mean_ms = 0.0,
            .p50_ms = 0.0,
            .p95_ms = 0.0,
            .p99_ms = 0.0,
            .physics_mean_ms = 0.0,
            .physics_p50_ms = 0.0,
            .physics_p95_ms = 0.0,
            .physics_p99_ms = 0.0,
            .render_mean_ms = 0.0,
            .render_p50_ms = 0.0,
            .render_p95_ms = 0.0,
            .render_p99_ms = 0.0,
            .present_mean_ms = 0.0,
            .present_p50_ms = 0.0,
            .present_p95_ms = 0.0,
            .present_p99_ms = 0.0,
            .simulation_to_wall_clock_rate = 0.0,
            .timing_window_frames = 0U,
            .realtime_statistics = {},
            .thermal_state_before = "unavailable",
            .thermal_state_after = "unavailable",
            .memory_footprint_bytes = 0U,
            .target_updates_per_second = 0U,
            .rigid_body_limit = 0U,
            .fluid_particle_limit = 0U,
            .coupled_body_limit = 0U,
            .smoke_particle_limit = 0U,
            .render_width = 0U,
            .render_height = 0U,
            .peak_particle_count = 0U,
            .moving_body_count = 0U,
            .reason = item.available ? std::string{} : item.unavailable_reason,
            .error = {},
        };
        if (item.available) {
            try {
                if (config.thermal_state)
                    result.thermal_state_before = config.thermal_state();
                require(session.execute(MobileGalleryCommand::select_scene,
                                        static_cast<float>(index)),
                        failure_message(session, "Scene selection failed"));
                const auto workload = session.gallery().presentation();
                result.workload_profile = workload.workload_profile;
                result.target_updates_per_second =
                    workload.target_updates_per_second;
                result.rigid_body_limit = workload.rigid_body_limit;
                result.fluid_particle_limit = workload.fluid_particle_limit;
                result.coupled_body_limit = workload.coupled_body_limit;
                result.smoke_particle_limit = workload.smoke_particle_limit;
                result.render_width = workload.width;
                result.render_height = workload.height;
                result.peak_particle_count = workload.particle_count;
                result.moving_body_count = workload.moving_body_count;
                const auto statistics = session.gallery().qualify(
                    config.frames_per_scene,
                    [&](const FrameTimingSample &,
                        const PmRealtimeStepStatistics &realtime) {
                        result.realtime_statistics = realtime;
                        const auto observed = session.gallery().presentation();
                        result.peak_particle_count = std::max(
                            result.peak_particle_count,
                            observed.particle_count);
                        result.moving_body_count = std::max(
                            result.moving_body_count,
                            observed.moving_body_count);
                    });
                result.mean_ms = statistics.mean_ms;
                result.p50_ms = statistics.median_ms;
                result.p95_ms = statistics.p95_ms;
                result.p99_ms = statistics.p99_ms;
                result.physics_mean_ms = statistics.physics_ms;
                result.physics_p50_ms = statistics.physics_median_ms;
                result.physics_p95_ms = statistics.physics_p95_ms;
                result.physics_p99_ms = statistics.physics_p99_ms;
                result.render_mean_ms = statistics.render_ms;
                result.render_p50_ms = statistics.render_median_ms;
                result.render_p95_ms = statistics.render_p95_ms;
                result.render_p99_ms = statistics.render_p99_ms;
                result.present_mean_ms = statistics.present_ms;
                result.present_p50_ms = statistics.present_median_ms;
                result.present_p95_ms = statistics.present_p95_ms;
                result.present_p99_ms = statistics.present_p99_ms;
                result.simulation_to_wall_clock_rate =
                    statistics.mean_ms > 0.0
                    ? 1000.0 * session.gallery().frame_timestep() /
                        statistics.mean_ms
                    : 0.0;
                result.timing_window_frames = statistics.count;
                result.memory_footprint_bytes = config.memory_footprint
                    ? config.memory_footprint() : session.gallery().allocated_bytes();
                if (config.thermal_state)
                    result.thermal_state_after = config.thermal_state();
                result.passed = true;
            } catch (const std::exception &error) {
                result.error = error.what();
                failure("scene:" + result.id, result.error);
            } catch (...) {
                result.error = "Unknown scene qualification failure";
                failure("scene:" + result.id, result.error);
            }
        }
        report.scenes.push_back(std::move(result));
        checkpoint();
    }

    if (has_available) {
        if (!session.execute(MobileGalleryCommand::select_scene,
                             static_cast<float>(first_index)))
            failure("restore", failure_message(session,
                        "Could not restore the initial mobile scene"));
    }
    session.set_foreground(true);
    report.completed = report.scenes.size() == catalog.size();
    const auto checks_passed = [](const auto &checks) {
        return std::all_of(checks.begin(), checks.end(), [](const auto &item) {
            return !item.applicable || item.passed;
        });
    };
    report.passed = report.completed && report.failures.empty() &&
        checks_passed(report.lifecycle) && checks_passed(report.controls) &&
        std::all_of(report.scenes.begin(), report.scenes.end(),
            [](const auto &scene) { return !scene.available || scene.passed; });
    checkpoint();
    return report;
}

} // namespace parallel_mater::gallery

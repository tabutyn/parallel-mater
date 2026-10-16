// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/frame_timing.hpp>

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <locale>
#include <sstream>
#include <cmath>
#include <vector>

namespace parallel_mater::gallery {
namespace {
double milliseconds(FrameTimingCollector::TimePoint from,
                    FrameTimingCollector::TimePoint to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

void csv_text(std::ostream &stream, std::string_view value) {
    stream << '"';
    for (char c : value) {
        if (c == '"') stream << '"';
        stream << c;
    }
    stream << '"';
}
} // namespace

void FrameTimingCollector::begin_frame(TimePoint now) {
    cancel_frame();
    active_ = true;
    frame_started_ = now;
}

void FrameTimingCollector::begin_phase(FramePhase phase, TimePoint now) {
    assert(active_ && !phase_.has_value());
    phase_ = phase;
    phase_started_ = now;
}

double FrameTimingCollector::end_phase(TimePoint now) {
    assert(active_ && phase_.has_value());
    const double elapsed = milliseconds(phase_started_, now);
    switch (*phase_) {
    case FramePhase::physics:
        pending_.physics_ms += elapsed;
        ++pending_.physics_steps;
        break;
    case FramePhase::render: pending_.render_ms += elapsed; break;
    case FramePhase::present:
        pending_.present_ms = pending_.present_ms.value_or(0.0) + elapsed;
        break;
    }
    phase_.reset();
    return elapsed;
}

bool FrameTimingCollector::finish_frame(TimePoint now) {
    if (!active_ || phase_.has_value()) return false;
    pending_.frame = ++completed_;
    pending_.total_ms = milliseconds(frame_started_, now);
    pending_.other_ms = std::max(0.0, pending_.total_ms - pending_.physics_ms -
        pending_.render_ms - pending_.present_ms.value_or(0.0));
    latest_ = pending_;
    recent_.push_back({pending_, frame_started_, now});
    if (recent_.size() > 60) recent_.pop_front();
    history_.push_back(pending_.total_ms);
    history_ms_ += pending_.total_ms;
    // Keep roughly one second, including at least the most recent sample.
    // Bound storage even for zero-duration synthetic frames.
    while (history_.size() > 1U &&
           (history_ms_ - history_.front() >= 1000.0 || history_.size() > 4096U)) {
        history_ms_ -= history_.front();
        history_.pop_front();
    }
    cancel_frame();
    return true;
}

void FrameTimingCollector::cancel_frame() {
    active_ = false;
    phase_.reset();
    pending_ = {};
}

void FrameTimingCollector::clear_history() {
    latest_.reset();
    history_.clear();
    history_ms_ = 0.0;
    recent_.clear();
}

FrameTimingStatistics FrameTimingCollector::statistics() const {
    FrameTimingStatistics result;
    result.count = recent_.size();
    if (!result.count) return result;
    std::vector<double> sorted;
    sorted.reserve(result.count);
    for (const auto &entry : recent_) {
        sorted.push_back(entry.sample.total_ms);
        result.mean_ms += entry.sample.total_ms;
        result.physics_ms += entry.sample.physics_ms;
        result.render_ms += entry.sample.render_ms;
        result.present_ms += entry.sample.present_ms.value_or(0);
    }
    const double count = static_cast<double>(result.count);
    result.mean_ms /= count; result.physics_ms /= count;
    result.render_ms /= count; result.present_ms /= count;
    std::sort(sorted.begin(), sorted.end());
    result.median_ms = (sorted[(result.count - 1) / 2] + sorted[result.count / 2]) * .5;
    result.p95_ms = sorted[static_cast<std::size_t>(std::ceil(count * .95)) - 1];
    const double elapsed = milliseconds(recent_.front().start, recent_.back().end);
    result.cadence_fps = elapsed > 0 ? 1000 * count / elapsed : 0;
    return result;
}

double FrameTimingCollector::fps() const {
    return history_ms_ > 0.0 ? 1000.0 * static_cast<double>(history_.size()) / history_ms_ : 0.0;
}

bool FrameTimingCsv::write_header(std::ostream &stream, std::string &error) {
    stream << "frame,scene,mode,physics_steps,total_ms,physics_ms,render_ms,present_ms,other_ms\n";
    if (stream) return true;
    error = "Frame timing CSV header write failed";
    return false;
}

bool FrameTimingCsv::write_sample(std::ostream &stream, const FrameTimingSample &sample,
                                std::string_view scene, std::string_view mode,
                                std::string &error) {
    std::ostringstream row;
    row.imbue(std::locale::classic());
    row << std::fixed << std::setprecision(6) << sample.frame << ',';
    csv_text(row, scene);
    row << ',';
    csv_text(row, mode);
    row << ',' << sample.physics_steps << ',' << sample.total_ms << ','
        << sample.physics_ms << ',' << sample.render_ms << ',';
    if (sample.present_ms) row << *sample.present_ms;
    row << ',' << sample.other_ms << '\n';
    stream << row.str();
    if (stream) return true;
    error = "Frame timing CSV write failed";
    return false;
}

bool FrameTimingCsv::flush(std::ostream &stream, std::string &error) {
    stream.flush();
    if (stream) return true;
    error = "Frame timing CSV flush failed";
    return false;
}

bool FrameTimingCsv::open(const std::filesystem::path &path, std::string &error) {
    path_ = path;
    stream_.open(path);
    if (!stream_) {
        error = "Frame timing CSV open failed: " + path.string();
        return false;
    }
    if (write_header(stream_, error)) return true;
    error += ": " + path_.string();
    return false;
}

bool FrameTimingCsv::write(const FrameTimingSample &sample, std::string_view scene,
                          std::string_view mode, std::string &error) {
    if (!stream_.is_open() || write_sample(stream_, sample, scene, mode, error)) return true;
    error += ": " + path_.string();
    return false;
}

bool FrameTimingCsv::close(std::string &error) {
    if (!stream_.is_open()) return true;
    const bool flushed = flush(stream_, error);
    stream_.close();
    if (flushed && stream_) return true;
    if (flushed) error = "Frame timing CSV close failed";
    error += ": " + path_.string();
    return false;
}

} // namespace parallel_mater::gallery

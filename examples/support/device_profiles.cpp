// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/device_profiles.hpp>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
#include <system_error>
#include <variant>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace parallel_mater::gallery {
namespace {

struct Json {
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json>;
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value{};
};

class JsonParser {
  public:
    explicit JsonParser(std::string_view source) : source_(source) {}

    bool parse(Json &output, std::string &error) {
        skip_space();
        if (!parse_value(output, error)) return false;
        skip_space();
        if (position_ != source_.size()) return fail(error, "trailing JSON data");
        return true;
    }

  private:
    std::string_view source_;
    std::size_t position_{};

    bool fail(std::string &error, std::string_view message) const {
        error = std::string(message) + " at byte " + std::to_string(position_);
        return false;
    }

    void skip_space() {
        while (position_ < source_.size() &&
               (source_[position_] == ' ' || source_[position_] == '\n' ||
                source_[position_] == '\r' || source_[position_] == '\t'))
            ++position_;
    }

    bool consume(char expected) {
        skip_space();
        if (position_ >= source_.size() || source_[position_] != expected)
            return false;
        ++position_;
        return true;
    }

    bool parse_value(Json &output, std::string &error) {
        skip_space();
        if (position_ >= source_.size()) return fail(error, "missing JSON value");
        const char first = source_[position_];
        if (first == '{') return parse_object(output, error);
        if (first == '[') return parse_array(output, error);
        if (first == '"') {
            std::string text;
            if (!parse_string(text, error)) return false;
            output.value = std::move(text);
            return true;
        }
        if (source_.substr(position_, 4) == "true") {
            position_ += 4U; output.value = true; return true;
        }
        if (source_.substr(position_, 5) == "false") {
            position_ += 5U; output.value = false; return true;
        }
        if (source_.substr(position_, 4) == "null") {
            position_ += 4U; output.value = nullptr; return true;
        }
        return parse_number(output, error);
    }

    bool parse_string(std::string &output, std::string &error) {
        if (!consume('"')) return fail(error, "expected string");
        output.clear();
        while (position_ < source_.size()) {
            const char character = source_[position_++];
            if (character == '"') return true;
            if (static_cast<unsigned char>(character) < 0x20U)
                return fail(error, "control character in string");
            if (character != '\\') { output.push_back(character); continue; }
            if (position_ >= source_.size()) return fail(error, "unfinished escape");
            const char escaped = source_[position_++];
            switch (escaped) {
            case '"': case '\\': case '/': output.push_back(escaped); break;
            case 'b': output.push_back('\b'); break;
            case 'f': output.push_back('\f'); break;
            case 'n': output.push_back('\n'); break;
            case 'r': output.push_back('\r'); break;
            case 't': output.push_back('\t'); break;
            default: return fail(error, "unsupported JSON escape");
            }
        }
        return fail(error, "unfinished string");
    }

    bool parse_number(Json &output, std::string &error) {
        const std::size_t begin = position_;
        while (position_ < source_.size()) {
            const char c = source_[position_];
            if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' ||
                c == 'e' || c == 'E') ++position_;
            else break;
        }
        if (begin == position_) return fail(error, "expected JSON value");
        double number = 0.0;
        const auto result = std::from_chars(source_.data() + begin,
            source_.data() + position_, number);
        if (result.ec != std::errc{} || result.ptr != source_.data() + position_ ||
            !std::isfinite(number)) return fail(error, "invalid number");
        output.value = number;
        return true;
    }

    bool parse_array(Json &output, std::string &error) {
        consume('[');
        Json::Array values;
        skip_space();
        if (consume(']')) { output.value = std::move(values); return true; }
        for (;;) {
            Json item;
            if (!parse_value(item, error)) return false;
            values.push_back(std::move(item));
            if (consume(']')) break;
            if (!consume(',')) return fail(error, "expected comma in array");
        }
        output.value = std::move(values);
        return true;
    }

    bool parse_object(Json &output, std::string &error) {
        consume('{');
        Json::Object values;
        skip_space();
        if (consume('}')) { output.value = std::move(values); return true; }
        for (;;) {
            std::string key;
            if (!parse_string(key, error)) return false;
            if (!consume(':')) return fail(error, "expected colon in object");
            Json item;
            if (!parse_value(item, error)) return false;
            if (!values.emplace(std::move(key), std::move(item)).second)
                return fail(error, "duplicate object member");
            if (consume('}')) break;
            if (!consume(',')) return fail(error, "expected comma in object");
        }
        output.value = std::move(values);
        return true;
    }
};

const Json::Object *object(const Json &value) {
    return std::get_if<Json::Object>(&value.value);
}
const Json::Array *array(const Json &value) {
    return std::get_if<Json::Array>(&value.value);
}
const Json *member(const Json::Object &value, std::string_view name) {
    const auto found = value.find(std::string(name));
    return found == value.end() ? nullptr : &found->second;
}
std::string string(const Json::Object &value, std::string_view name) {
    const Json *item = member(value, name);
    if (item == nullptr) return {};
    const auto *result = std::get_if<std::string>(&item->value);
    return result != nullptr ? *result : std::string{};
}
double number(const Json::Object &value, std::string_view name) {
    const Json *item = member(value, name);
    if (item == nullptr) return 0.0;
    const auto *result = std::get_if<double>(&item->value);
    return result != nullptr ? *result : 0.0;
}
bool boolean(const Json::Object &value, std::string_view name) {
    const Json *item = member(value, name);
    if (item == nullptr) return false;
    const auto *result = std::get_if<bool>(&item->value);
    return result != nullptr && *result;
}

void quote(std::ostream &output, std::string_view value) {
    output << '"';
    for (char character : value) {
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default: output << character; break;
        }
    }
    output << '"';
}

void write_hardware(std::ostream &out, const HardwareRecord &value) {
    out << "    {\"vendor\":"; quote(out, value.vendor);
    out << ",\"model\":"; quote(out, value.model);
    out << ",\"variant\":"; quote(out, value.variant);
    if (value.compute_units != 0U)
        out << ",\"compute_units\":" << value.compute_units;
    if (value.memory_bandwidth_gigabytes_per_second > 0.0)
        out << ",\"memory_bandwidth_gigabytes_per_second\":"
            << value.memory_bandwidth_gigabytes_per_second;
    if (value.memory_bytes != 0U)
        out << ",\"memory_bytes\":" << value.memory_bytes;
    if (value.power_watts > 0.0)
        out << ",\"power_watts\":" << value.power_watts;
    out << ",\"source_url\":";
    quote(out, value.source_url); out << '}';
}

void write_identity(std::ostream &out, const HardwareIdentity &value) {
    out << "{\"machine_model\":"; quote(out, value.machine_model);
    out << ",\"cpu_model\":"; quote(out, value.cpu_model);
    out << ",\"gpu_model\":"; quote(out, value.gpu_model);
    out << ",\"gpu_variant\":"; quote(out, value.gpu_variant);
    out << ",\"memory_bytes\":" << value.memory_bytes
        << ",\"backend\":"; quote(out, value.backend);
    out << ",\"operating_system\":"; quote(out, value.operating_system);
    out << ",\"driver\":"; quote(out, value.driver);
    out << ",\"power_mode\":"; quote(out, value.power_mode); out << '}';
}

void write_profile(std::ostream &out, const VerifiedBrickProfile &value) {
    out << "    {\"hardware\":"; write_identity(out, value.hardware);
    out << ",\"scene\":{\"brick_count\":" << value.scene.brick_count
        << ",\"brick_scale\":" << value.scene.brick_scale
        << ",\"wall_planes\":" << value.scene.wall_planes << "}"
        << ",\"width\":" << value.width << ",\"height\":" << value.height
        << ",\"scene_version\":" << value.scene_version
        << ",\"solver_version\":"; quote(out, value.solver_version);
    out << ",\"build_revision\":"; quote(out, value.build_revision);
    out << ",\"metrics\":{\"duration_seconds\":" << value.metrics.duration_seconds
        << ",\"quiet_p95_milliseconds\":" << value.metrics.quiet_p95_milliseconds
        << ",\"quiet_maximum_milliseconds\":" << value.metrics.quiet_maximum_milliseconds
        << ",\"collision_p95_milliseconds\":" << value.metrics.collision_p95_milliseconds
        << ",\"collision_maximum_milliseconds\":" << value.metrics.collision_maximum_milliseconds
        << ",\"simulation_progress_ratio\":" << value.metrics.simulation_progress_ratio
        << ",\"stable\":" << (value.metrics.stable ? "true" : "false")
        << ",\"no_dropped_steps\":" << (value.metrics.no_dropped_steps ? "true" : "false")
        << "},\"verified_at\":"; quote(out, value.verified_at);
    out << ",\"verifier\":"; quote(out, value.verifier); out << '}';
}

void write_measurement(std::ostream &out, const BrickMeasurement &value) {
    out << "    {\"hardware\":"; write_identity(out, value.hardware);
    out << ",\"scene\":{\"brick_count\":" << value.scene.brick_count
        << ",\"brick_scale\":" << value.scene.brick_scale
        << ",\"wall_planes\":" << value.scene.wall_planes << "}"
        << ",\"width\":" << value.width << ",\"height\":" << value.height
        << ",\"scene_version\":" << value.scene_version
        << ",\"solver_version\":"; quote(out, value.solver_version);
    out << ",\"build_revision\":"; quote(out, value.build_revision);
    out << ",\"measured_at\":"; quote(out, value.measured_at);
    out << ",\"source\":"; quote(out, value.source);
    out << ",\"quiet_median_milliseconds\":" << value.quiet_median_milliseconds
        << ",\"collision_p95_milliseconds\":" << value.collision_p95_milliseconds
        << ",\"stable\":" << (value.stable ? "true" : "false") << '}';
}

bool parse_identity(const Json &json, HardwareIdentity &output) {
    const auto *value = object(json); if (value == nullptr) return false;
    output.machine_model = string(*value, "machine_model");
    output.cpu_model = string(*value, "cpu_model");
    output.gpu_model = string(*value, "gpu_model");
    output.gpu_variant = string(*value, "gpu_variant");
    output.memory_bytes = static_cast<std::uint64_t>(number(*value, "memory_bytes"));
    output.backend = string(*value, "backend");
    output.operating_system = string(*value, "operating_system");
    output.driver = string(*value, "driver");
    output.power_mode = string(*value, "power_mode");
    return !output.gpu_model.empty() && !output.backend.empty();
}

bool parse_catalog(const Json &json, DeviceProfileCatalog &output,
                   std::string &error) {
    const auto *root = object(json);
    if (root == nullptr) { error = "profile catalog root must be an object"; return false; }
    output = {};
    output.schema_version = static_cast<std::uint32_t>(number(*root, "schema_version"));
    if (output.schema_version != brick_profile_schema_version) {
        error = "unsupported device profile schema version"; return false;
    }
    if (const Json *items = member(*root, "hardware")) {
        const auto *values = array(*items);
        if (values == nullptr) { error = "hardware must be an array"; return false; }
        for (const Json &item : *values) {
            const auto *value = object(item);
            if (value == nullptr) { error = "hardware entry must be an object"; return false; }
            output.hardware.push_back({
                string(*value, "vendor"), string(*value, "model"),
                string(*value, "variant"),
                static_cast<std::uint32_t>(number(*value, "compute_units")),
                number(*value, "memory_bandwidth_gigabytes_per_second"),
                static_cast<std::uint64_t>(number(*value, "memory_bytes")),
                number(*value, "power_watts"), string(*value, "source_url")});
        }
    }
    if (const Json *items = member(*root, "verified_profiles")) {
        const auto *values = array(*items);
        if (values == nullptr) { error = "verified_profiles must be an array"; return false; }
        for (const Json &item : *values) {
            const auto *value = object(item);
            if (value == nullptr) { error = "verified profile must be an object"; return false; }
            VerifiedBrickProfile profile;
            const Json *hardware = member(*value, "hardware");
            const Json *scene = member(*value, "scene");
            const Json *metrics = member(*value, "metrics");
            if (hardware == nullptr || !parse_identity(*hardware, profile.hardware) ||
                scene == nullptr || object(*scene) == nullptr || metrics == nullptr ||
                object(*metrics) == nullptr) {
                error = "verified profile is incomplete"; return false;
            }
            const auto &s = *object(*scene); const auto &m = *object(*metrics);
            profile.scene = {static_cast<std::uint32_t>(number(s, "brick_count")),
                             static_cast<float>(number(s, "brick_scale")),
                             static_cast<std::uint32_t>(number(s, "wall_planes"))};
            if (!validate_brick_config(profile.scene, error)) return false;
            profile.width = static_cast<std::uint32_t>(number(*value, "width"));
            profile.height = static_cast<std::uint32_t>(number(*value, "height"));
            profile.scene_version = static_cast<std::uint32_t>(number(*value, "scene_version"));
            profile.solver_version = string(*value, "solver_version");
            profile.build_revision = string(*value, "build_revision");
            profile.metrics = {number(m, "duration_seconds"),
                number(m, "quiet_p95_milliseconds"), number(m, "quiet_maximum_milliseconds"),
                number(m, "collision_p95_milliseconds"), number(m, "collision_maximum_milliseconds"),
                number(m, "simulation_progress_ratio"), boolean(m, "stable"),
                boolean(m, "no_dropped_steps")};
            profile.verified_at = string(*value, "verified_at");
            profile.verifier = string(*value, "verifier");
            output.verified_profiles.push_back(std::move(profile));
        }
    }
    if (const Json *items = member(*root, "measurements")) {
        const auto *values = array(*items);
        if (values == nullptr) { error = "measurements must be an array"; return false; }
        for (const Json &item : *values) {
            const auto *value = object(item);
            if (value == nullptr) { error = "measurement must be an object"; return false; }
            BrickMeasurement measurement;
            const Json *hardware = member(*value, "hardware");
            const Json *scene = member(*value, "scene");
            if (hardware == nullptr || !parse_identity(*hardware, measurement.hardware) ||
                scene == nullptr || object(*scene) == nullptr) {
                error = "measurement is incomplete"; return false;
            }
            const auto &s = *object(*scene);
            measurement.scene = {static_cast<std::uint32_t>(number(s, "brick_count")),
                static_cast<float>(number(s, "brick_scale")),
                static_cast<std::uint32_t>(number(s, "wall_planes"))};
            if (!validate_brick_config(measurement.scene, error)) return false;
            measurement.width = static_cast<std::uint32_t>(number(*value, "width"));
            measurement.height = static_cast<std::uint32_t>(number(*value, "height"));
            measurement.scene_version = static_cast<std::uint32_t>(number(*value, "scene_version"));
            measurement.solver_version = string(*value, "solver_version");
            measurement.build_revision = string(*value, "build_revision");
            measurement.measured_at = string(*value, "measured_at");
            measurement.source = string(*value, "source");
            measurement.quiet_median_milliseconds = number(*value, "quiet_median_milliseconds");
            measurement.collision_p95_milliseconds = number(*value, "collision_p95_milliseconds");
            measurement.stable = boolean(*value, "stable");
            output.measurements.push_back(std::move(measurement));
        }
    }
    return true;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t index = std::min(values.size() - 1U,
        static_cast<std::size_t>(std::ceil(fraction * values.size())) - 1U);
    return values[index];
}

bool same_hardware(const HardwareIdentity &left, const HardwareIdentity &right) {
    return left.machine_model == right.machine_model &&
           left.cpu_model == right.cpu_model && left.gpu_model == right.gpu_model &&
           left.gpu_variant == right.gpu_variant && left.memory_bytes == right.memory_bytes &&
           left.backend == right.backend &&
           left.operating_system == right.operating_system &&
           left.driver == right.driver && left.power_mode == right.power_mode;
}

class CatalogSaveLock {
  public:
    bool acquire(const std::filesystem::path &catalog, std::string &error) {
        std::error_code filesystem_error;
        const auto canonical = std::filesystem::absolute(catalog, filesystem_error);
        const std::string key = (filesystem_error ? catalog : canonical).string();
        std::uint64_t hash = 14'695'981'039'346'656'037ULL;
        for (const unsigned char byte : key) {
            hash ^= byte;
            hash *= 1'099'511'628'211ULL;
        }
        path_ = std::filesystem::temp_directory_path(filesystem_error) /
            ("parallel-mater-profile-" +
             std::to_string(hash) + ".lock");
        if (filesystem_error ||
            !std::filesystem::create_directory(path_, filesystem_error)) {
            error = filesystem_error
                ? "could not create device profile save lock: " +
                      filesystem_error.message()
                : "another process is updating the device profile file";
            path_.clear();
            return false;
        }
        return true;
    }

    ~CatalogSaveLock() {
        if (path_.empty()) return;
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    CatalogSaveLock(const CatalogSaveLock &) = delete;
    CatalogSaveLock &operator=(const CatalogSaveLock &) = delete;
    CatalogSaveLock() = default;

  private:
    std::filesystem::path path_{};
};

} // namespace

bool validate_brick_config(const BrickSceneConfig &config,
                           std::string &error) noexcept {
    if (config.brick_count < brick_minimum_count ||
        config.brick_count > brick_maximum_count) {
        error = "brick count must be between 1 and 4096"; return false;
    }
    if (!std::isfinite(config.brick_scale) || config.brick_scale < brick_minimum_scale ||
        config.brick_scale > brick_maximum_scale) {
        error = "brick scale must be between 0.5 and 2.0"; return false;
    }
    if (config.wall_planes < 1U || config.wall_planes > brick_maximum_planes ||
        config.wall_planes > config.brick_count) {
        error = "wall planes must be between 1 and min(brick count, 16)"; return false;
    }
    error.clear(); return true;
}

std::uint64_t estimated_metal_contact_bytes(std::uint32_t rigid_body_count) noexcept {
    const std::uint64_t count = rigid_body_count;
    const std::uint64_t pairs = count > 1U ? count * (count - 1U) / 2U : 1U;
    // Conservative sum for manifold/cache/island/active-pair and dense flag buffers.
    return pairs * 1'024U + count * count * 4U + count * 512U;
}

const std::array<BrickSceneConfig, 12> &brick_calibration_presets() noexcept {
    static constexpr std::array values{
        BrickSceneConfig{1U, 2.0F, 1U}, BrickSceneConfig{8U, 2.0F, 1U},
        BrickSceneConfig{24U, 2.0F, 1U}, BrickSceneConfig{48U, 2.0F, 1U},
        BrickSceneConfig{80U, 1.5F, 1U}, BrickSceneConfig{192U, 1.0F, 1U},
        BrickSceneConfig{384U, 1.0F, 2U}, BrickSceneConfig{640U, 0.75F, 2U},
        BrickSceneConfig{960U, 0.75F, 3U}, BrickSceneConfig{1'280U, 0.75F, 4U},
        BrickSceneConfig{3'072U, 0.5F, 4U}, BrickSceneConfig{3'840U, 0.5F, 5U}};
    return values;
}

CalibrationMetrics summarize_calibration(const std::vector<CalibrationSample> &samples,
                                         double duration_seconds, bool stable,
                                         bool dropped_steps) noexcept {
    std::vector<double> quiet, collision;
    std::size_t advanced = 0U;
    for (const auto &sample : samples) {
        (sample.collision ? collision : quiet).push_back(sample.milliseconds);
        advanced += sample.simulation_advanced;
    }
    const auto maximum = [](const std::vector<double> &values) {
        return values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
    };
    return {duration_seconds, percentile(quiet, 0.95), maximum(quiet),
            percentile(collision, 0.95), maximum(collision),
            samples.empty() ? 0.0 : static_cast<double>(advanced) / samples.size(),
            stable, !dropped_steps};
}

bool calibration_passes(const CalibrationMetrics &metrics) noexcept {
    return metrics.duration_seconds > 0.0 && metrics.quiet_p95_milliseconds > 0.0 &&
           metrics.collision_p95_milliseconds > 0.0 &&
           metrics.quiet_p95_milliseconds <= 15.0 &&
           metrics.collision_p95_milliseconds <= 15.0 &&
           metrics.quiet_maximum_milliseconds <= 30.0 &&
           metrics.collision_maximum_milliseconds <= 30.0 &&
           metrics.simulation_progress_ratio >= 0.995 && metrics.stable &&
           metrics.no_dropped_steps;
}

bool load_device_profiles(const std::filesystem::path &path,
                          DeviceProfileCatalog &output, std::string &error) {
    std::ifstream input(path);
    if (!input) { error = "could not open device profiles: " + path.string(); return false; }
    std::ostringstream contents; contents << input.rdbuf();
    Json root;
    if (!JsonParser(contents.str()).parse(root, error)) return false;
    return parse_catalog(root, output, error);
}

bool save_verified_profile(const std::filesystem::path &path,
                           const VerifiedBrickProfile &profile, std::string &error) {
    if (!calibration_passes(profile.metrics)) {
        error = "only a passing calibration can be verified"; return false;
    }
    if (profile.metrics.duration_seconds < 180.0) {
        error = "verified calibration must include three sustained minutes";
        return false;
    }
    CatalogSaveLock save_lock;
    if (!save_lock.acquire(path, error)) return false;
    std::string validation;
    if (!validate_brick_config(profile.scene, validation)) { error = validation; return false; }
    if (profile.width != brick_render_width || profile.height != brick_render_height ||
        profile.scene_version != brick_scene_version || profile.solver_version.empty() ||
        profile.build_revision.empty() || profile.verified_at.empty() ||
        profile.verifier.empty() || profile.hardware.gpu_model.empty() ||
        profile.hardware.backend.empty()) {
        error = "verified profile metadata is incomplete or incompatible";
        return false;
    }
    DeviceProfileCatalog catalog;
    std::error_code existence_error;
    const bool original_exists = std::filesystem::exists(path, existence_error);
    std::string original_contents;
    if (original_exists) {
        std::ifstream original(path, std::ios::binary);
        std::ostringstream snapshot;
        snapshot << original.rdbuf();
        if (!original) { error = "could not read device profile snapshot"; return false; }
        original_contents = snapshot.str();
        if (!load_device_profiles(path, catalog, error)) return false;
    }
    auto existing = std::find_if(catalog.verified_profiles.begin(), catalog.verified_profiles.end(),
        [&](const auto &item) { return same_hardware(item.hardware, profile.hardware) &&
            item.width == profile.width && item.height == profile.height &&
            item.scene_version == profile.scene_version; });
    if (existing == catalog.verified_profiles.end()) catalog.verified_profiles.push_back(profile);
    else *existing = profile;
    std::sort(catalog.verified_profiles.begin(), catalog.verified_profiles.end(),
        [](const auto &a, const auto &b) { return a.hardware.gpu_model < b.hardware.gpu_model; });

    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    static std::atomic<std::uint64_t> temporary_serial{};
    const auto temporary = path.string() + ".tmp." + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()) + "." +
        std::to_string(temporary_serial.fetch_add(1U));
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) { error = "could not create temporary profile file"; return false; }
    output << "{\n  \"schema_version\": " << catalog.schema_version << ",\n  \"hardware\": [\n";
    for (std::size_t i = 0; i < catalog.hardware.size(); ++i) {
        write_hardware(output, catalog.hardware[i]);
        output << (i + 1U == catalog.hardware.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"verified_profiles\": [\n";
    for (std::size_t i = 0; i < catalog.verified_profiles.size(); ++i) {
        write_profile(output, catalog.verified_profiles[i]);
        output << (i + 1U == catalog.verified_profiles.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"measurements\": [\n";
    for (std::size_t i = 0; i < catalog.measurements.size(); ++i) {
        write_measurement(output, catalog.measurements[i]);
        output << (i + 1U == catalog.measurements.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    output.close();
    if (!output) { error = "could not write temporary profile file"; return false; }
    std::error_code current_existence_error;
    const bool current_exists = std::filesystem::exists(
        path, current_existence_error);
    bool changed = current_existence_error || current_exists != original_exists;
    if (!changed && current_exists) {
        std::ifstream current(path, std::ios::binary);
        std::ostringstream snapshot;
        snapshot << current.rdbuf();
        changed = !current || snapshot.str() != original_contents;
    }
    if (changed) {
        std::filesystem::remove(temporary);
        error = "device profile file changed during save";
        return false;
    }
    std::error_code rename_error;
#if defined(_WIN32)
    if (!MoveFileExW(std::filesystem::path(temporary).c_str(), path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        rename_error = std::error_code(
            static_cast<int>(GetLastError()), std::system_category());
#else
    std::filesystem::rename(temporary, path, rename_error);
#endif
    if (rename_error) { std::filesystem::remove(temporary); error = "could not replace profile file: " + rename_error.message(); return false; }
    error.clear(); return true;
}

const VerifiedBrickProfile *find_matching_profile(const DeviceProfileCatalog &catalog,
    const HardwareIdentity &hardware, std::uint32_t width, std::uint32_t height,
    std::string_view solver_version) noexcept {
    for (const auto &profile : catalog.verified_profiles)
        if (same_hardware(profile.hardware, hardware) && profile.width == width &&
            profile.height == height && profile.scene_version == brick_scene_version &&
            profile.metrics.duration_seconds >= 180.0 &&
            calibration_passes(profile.metrics) &&
            (solver_version.empty() || profile.solver_version == solver_version))
            return &profile;
    return nullptr;
}

BrickStartupSelection select_startup_bricks(const DeviceProfileCatalog &catalog,
    const HardwareIdentity &hardware, std::string_view solver_version,
    std::uint32_t width, std::uint32_t height) {
    if (const auto *verified = find_matching_profile(
            catalog, hardware, width, height, solver_version))
        return {verified->scene, BrickSelectionSource::verified};

    const auto os_family = [](std::string_view os) {
        return os.substr(0U, os.find(' '));
    };
    const auto match_score = [&](const BrickMeasurement &sample) {
        const auto &recorded = sample.hardware;
        const auto memory_difference = std::max(recorded.memory_bytes, hardware.memory_bytes) -
            std::min(recorded.memory_bytes, hardware.memory_bytes);
        // Usable VRAM can change slightly with driver reservations. Never
        // extrapolate counts between different GPU models, capacities or backends.
        if (hardware.gpu_model.empty() || hardware.gpu_variant.empty() ||
            hardware.memory_bytes == 0U || recorded.memory_bytes == 0U ||
            recorded.gpu_model != hardware.gpu_model ||
            recorded.gpu_variant != hardware.gpu_variant ||
            recorded.backend != hardware.backend ||
            recorded.power_mode != hardware.power_mode ||
            os_family(recorded.operating_system).empty() ||
            os_family(recorded.operating_system) != os_family(hardware.operating_system) ||
            memory_difference > 256ULL * 1024U * 1024U ||
            sample.width != width || sample.height != height ||
            sample.scene_version != brick_scene_version ||
            solver_version.empty() || sample.solver_version != solver_version ||
            sample.build_revision.empty() || sample.measured_at.empty() || sample.source.empty())
            return -1;
        return (recorded.machine_model == hardware.machine_model ? 4 : 0) +
            (recorded.cpu_model == hardware.cpu_model ? 2 : 0) +
            (recorded.driver == hardware.driver ? 1 : 0);
    };
    // Prefer this machine's data, even if a different machine with the same
    // GPU measured faster. An OS patch/driver update need not erase estimates.
    int best_match = -1;
    for (const auto &sample : catalog.measurements)
        best_match = std::max(best_match, match_score(sample));
    BrickStartupSelection selection;
    if (best_match < 0) return selection;
    for (const auto &sample : catalog.measurements) {
        std::string error;
        if (match_score(sample) != best_match || !sample.stable ||
            !validate_brick_config(sample.scene, error) ||
            !std::isfinite(sample.quiet_median_milliseconds) ||
            !std::isfinite(sample.collision_p95_milliseconds) ||
            sample.quiet_median_milliseconds <= 0.0 ||
            sample.collision_p95_milliseconds <= 0.0 ||
            sample.quiet_median_milliseconds > 1000.0 / 60.0 ||
            sample.collision_p95_milliseconds > 1000.0 / 30.0) continue;
        if (selection.source == BrickSelectionSource::fallback ||
            sample.scene.brick_count > selection.scene.brick_count)
            selection = {sample.scene, BrickSelectionSource::measured};
    }
    return selection;
}

const char *brick_selection_label(BrickSelectionSource source) noexcept {
    switch (source) {
    case BrickSelectionSource::verified: return "verified profile";
    case BrickSelectionSource::measured: return "measured recommendation";
    case BrickSelectionSource::fallback: return "fallback; no suitable measurements";
    }
    return "fallback";
}

std::filesystem::path default_local_profiles_path() {
#if defined(__APPLE__)
    if (const char *home = std::getenv("HOME"))
        return std::filesystem::path(home) / "Library/Application Support/ParallelMater/device-profiles.json";
#elif defined(_WIN32)
    if (const char *appdata = std::getenv("LOCALAPPDATA"))
        return std::filesystem::path(appdata) / "ParallelMater/device-profiles.json";
#else
    if (const char *state = std::getenv("XDG_STATE_HOME"))
        return std::filesystem::path(state) / "parallel-mater/device-profiles.json";
    if (const char *home = std::getenv("HOME"))
        return std::filesystem::path(home) / ".local/state/parallel-mater/device-profiles.json";
#endif
    return "device-profiles.json";
}

} // namespace parallel_mater::gallery

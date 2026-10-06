// SPDX-License-Identifier: MIT
#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace parallel_mater::test::parity {

inline constexpr std::string_view format_magic =
    "parallel-mater-parity-v1";

enum class ValueKind { unsigned_integer, floating_point };

struct Record {
    ValueKind kind{};
    std::string path;
    std::uint64_t unsigned_value{};
    double floating_value{};
};

struct Capture {
    std::string scenario;
    std::string backend;
    std::vector<Record> records;
};

struct ComparisonOptions {
    double absolute_tolerance{1.0e-5};
    double relative_tolerance{1.0e-3};
};

struct ComparisonResult {
    bool matches{};
    std::string message;
};

class Writer {
  public:
    Writer(std::ostream &output, std::string_view scenario,
           std::string_view backend)
        : output_(output) {
        output_ << format_magic << '\n'
                << "scenario " << scenario << '\n'
                << "backend " << backend << '\n';
    }

    void unsigned_value(std::string_view path, std::uint64_t value) {
        output_ << "u " << path << ' ' << value << '\n';
    }

    void floating_value(std::string_view path, double value) {
        output_ << "f " << path << ' ';
        if (std::isfinite(value)) {
            output_ << std::setprecision(std::numeric_limits<double>::max_digits10)
                    << value;
        } else if (std::isnan(value)) {
            output_ << "nan";
        } else {
            output_ << (value < 0.0 ? "-inf" : "inf");
        }
        output_ << '\n';
    }

  private:
    std::ostream &output_;
};

inline bool read_header_line(std::istream &input, std::string_view key,
                             std::string &value, std::string &error) {
    std::string line;
    if (!std::getline(input, line)) {
        error = "missing " + std::string(key) + " header";
        return false;
    }
    const std::string prefix = std::string(key) + ' ';
    if (!line.starts_with(prefix) || line.size() == prefix.size()) {
        error = "invalid " + std::string(key) + " header";
        return false;
    }
    value = line.substr(prefix.size());
    if (value.find_first_of(" \t\r\n") != std::string::npos) {
        error = std::string(key) + " must be one token";
        return false;
    }
    return true;
}

inline bool parse(std::istream &input, Capture &capture, std::string &error) {
    capture = {};
    std::string line;
    if (!std::getline(input, line) || line != format_magic) {
        error = "unsupported or missing parity capture version";
        return false;
    }
    if (!read_header_line(input, "scenario", capture.scenario, error) ||
        !read_header_line(input, "backend", capture.backend, error)) {
        return false;
    }

    std::size_t line_number = 3U;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) {
            error = "empty record at line " + std::to_string(line_number);
            return false;
        }
        std::istringstream record_stream(line);
        char kind = '\0';
        std::string path;
        std::string value;
        std::string trailing;
        if (!(record_stream >> kind >> path >> value) ||
            (record_stream >> trailing)) {
            error = "invalid record at line " + std::to_string(line_number);
            return false;
        }

        Record record{};
        record.path = std::move(path);
        if (kind == 'u') {
            record.kind = ValueKind::unsigned_integer;
            const char *begin = value.data();
            const char *end = begin + value.size();
            const auto parsed =
                std::from_chars(begin, end, record.unsigned_value, 10);
            if (parsed.ec != std::errc{} || parsed.ptr != end) {
                error = "invalid unsigned value at line " +
                        std::to_string(line_number);
                return false;
            }
        } else if (kind == 'f') {
            record.kind = ValueKind::floating_point;
            const char *begin = value.data();
            const char *end = begin + value.size();
            const auto parsed = std::from_chars(
                begin, end, record.floating_value, std::chars_format::general);
            if (parsed.ec != std::errc{} || parsed.ptr != end ||
                !std::isfinite(record.floating_value)) {
                error = "invalid or non-finite float at line " +
                        std::to_string(line_number);
                return false;
            }
        } else {
            error = "unknown record kind at line " +
                    std::to_string(line_number);
            return false;
        }
        capture.records.push_back(std::move(record));
    }
    if (!input.eof()) {
        error = "failed while reading capture";
        return false;
    }
    return true;
}

inline ComparisonResult compare(const Capture &reference,
                                const Capture &candidate,
                                ComparisonOptions options = {}) {
    if (reference.scenario != candidate.scenario) {
        return {false, "scenario mismatch: " + reference.scenario + " != " +
                           candidate.scenario};
    }
    if (options.absolute_tolerance < 0.0 ||
        options.relative_tolerance < 0.0 ||
        !std::isfinite(options.absolute_tolerance) ||
        !std::isfinite(options.relative_tolerance)) {
        return {false, "comparison tolerances must be finite and non-negative"};
    }
    if (reference.records.size() != candidate.records.size()) {
        return {false, "record count mismatch: " +
                           std::to_string(reference.records.size()) + " != " +
                           std::to_string(candidate.records.size())};
    }

    for (std::size_t index = 0; index < reference.records.size(); ++index) {
        const Record &expected = reference.records[index];
        const Record &actual = candidate.records[index];
        if (expected.kind != actual.kind || expected.path != actual.path) {
            return {false, "schema/order mismatch at record " +
                               std::to_string(index) + ": " + expected.path +
                               " != " + actual.path};
        }
        if (expected.kind == ValueKind::unsigned_integer) {
            if (expected.unsigned_value != actual.unsigned_value) {
                return {false, expected.path + ": exact mismatch " +
                                   std::to_string(expected.unsigned_value) +
                                   " != " +
                                   std::to_string(actual.unsigned_value)};
            }
            continue;
        }

        const double difference =
            std::abs(expected.floating_value - actual.floating_value);
        const double scale = std::max(std::abs(expected.floating_value),
                                      std::abs(actual.floating_value));
        const double allowed =
            options.absolute_tolerance + options.relative_tolerance * scale;
        if (difference > allowed) {
            std::ostringstream message;
            message << std::setprecision(17) << expected.path
                    << ": float mismatch " << expected.floating_value << " != "
                    << actual.floating_value << " (difference " << difference
                    << ", allowed " << allowed << ')';
            return {false, message.str()};
        }
    }
    return {true, "captures match (reference backend " + reference.backend +
                      ", candidate backend " + candidate.backend + ')'};
}

} // namespace parallel_mater::test::parity

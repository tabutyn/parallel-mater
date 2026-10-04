// SPDX-License-Identifier: MIT
#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace parallel_mater::conformance {

class Json {
  public:
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json, std::less<>>;

    Json() = default;
    Json(std::nullptr_t) : value_(nullptr) {}
    Json(bool value) : value_(value) {}
    Json(std::int64_t value) : value_(value) {}
    Json(std::uint64_t value) : value_(value) {}
    Json(double value) : value_(value) {}
    Json(const char *value) : value_(std::string(value)) {}
    Json(std::string value) : value_(std::move(value)) {}
    Json(Array value) : value_(std::move(value)) {}
    Json(Object value) : value_(std::move(value)) {}

    static Json array() { return Array{}; }
    static Json object() { return Object{}; }

    Json &operator[](std::string key) {
        if (!std::holds_alternative<Object>(value_)) value_ = Object{};
        return std::get<Object>(value_)[std::move(key)];
    }

    void push_back(Json value) {
        if (!std::holds_alternative<Array>(value_)) value_ = Array{};
        std::get<Array>(value_).push_back(std::move(value));
    }

    [[nodiscard]] std::string serialize() const {
        std::string result;
        append(result);
        result.push_back('\n');
        return result;
    }

  private:
    using Value = std::variant<std::nullptr_t, bool, std::int64_t,
                               std::uint64_t, double, std::string,
                               Array, Object>;
    Value value_{nullptr};

    static void append_quoted(std::string &output, std::string_view value) {
        output.push_back('\"');
        constexpr char hex[] = "0123456789abcdef";
        for (const unsigned char byte : value) {
            switch (byte) {
            case '\"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (byte < 0x20U) {
                    output += "\\u00";
                    output.push_back(hex[byte >> 4U]);
                    output.push_back(hex[byte & 0x0fU]);
                } else {
                    output.push_back(static_cast<char>(byte));
                }
            }
        }
        output.push_back('\"');
    }

    static void append_double(std::string &output, double value) {
        if (!std::isfinite(value))
            throw std::runtime_error("non-finite conformance JSON value");
        char buffer[64]{};
        const auto [end, error] = std::to_chars(
            buffer, buffer + sizeof(buffer), value,
            std::chars_format::general,
            std::numeric_limits<double>::max_digits10);
        if (error != std::errc{})
            throw std::runtime_error("conformance JSON float serialization failed");
        output.append(buffer, end);
    }

    void append(std::string &output) const {
        if (std::holds_alternative<std::nullptr_t>(value_)) {
            output += "null";
        } else if (const auto *value = std::get_if<bool>(&value_)) {
            output += *value ? "true" : "false";
        } else if (const auto *value = std::get_if<std::int64_t>(&value_)) {
            output += std::to_string(*value);
        } else if (const auto *value = std::get_if<std::uint64_t>(&value_)) {
            output += std::to_string(*value);
        } else if (const auto *value = std::get_if<double>(&value_)) {
            append_double(output, *value);
        } else if (const auto *value = std::get_if<std::string>(&value_)) {
            append_quoted(output, *value);
        } else if (const auto *value = std::get_if<Array>(&value_)) {
            output.push_back('[');
            for (std::size_t index = 0U; index < value->size(); ++index) {
                if (index != 0U) output.push_back(',');
                (*value)[index].append(output);
            }
            output.push_back(']');
        } else {
            const Object &members = std::get<Object>(value_);
            output.push_back('{');
            bool first = true;
            for (const auto &[key, member] : members) {
                if (!first) output.push_back(',');
                first = false;
                append_quoted(output, key);
                output.push_back(':');
                member.append(output);
            }
            output.push_back('}');
        }
    }
};

} // namespace parallel_mater::conformance

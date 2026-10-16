// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>

namespace parallel_mater {

enum class StatusCode : std::uint8_t {
    success,
    invalid_argument,
    not_supported,
    invalid_handle,
    capacity_exceeded,
    busy,
    out_of_memory,
    backend_failure,
    internal_error,
};

struct Status {
    StatusCode code{StatusCode::success};
    std::int32_t backend_error{};
    const char *message{};

    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == StatusCode::success;
    }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok(); }
};

} // namespace parallel_mater

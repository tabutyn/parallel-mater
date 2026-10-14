// SPDX-License-Identifier: MIT
#pragma once

#include "device.hpp"

#include <cuda_runtime_api.h>

namespace parallel_mater::device {

[[nodiscard]] inline PmResult from_cuda(cudaError_t error) noexcept {
    return from_driver_error(static_cast<std::int32_t>(error));
}

[[nodiscard]] inline cudaStream_t cuda_stream(Services &services) noexcept {
    return static_cast<cudaStream_t>(services.native_queue_handle());
}

} // namespace parallel_mater::device

// SPDX-License-Identifier: MIT
#pragma once

#include <parallel_mater/parallel_mater.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace parallel_mater::device {

[[nodiscard]] PmResult from_cuda(cudaError_t error) noexcept;

struct Description {
    int ordinal{};
    std::uint32_t driver_version{};
    std::uint32_t vendor_id{0x10deU};
    std::uint32_t device_id{};
    std::uint8_t uuid[16]{};
    std::string name;
    PmDeviceSize total_memory{};
};

class Services {
  public:
    virtual ~Services() = default;

    [[nodiscard]] virtual int ordinal() const noexcept = 0;
    [[nodiscard]] virtual cudaStream_t stream() const noexcept = 0;
    [[nodiscard]] virtual cudaEvent_t epoch() const noexcept = 0;
    [[nodiscard]] virtual PmResult allocate(PmDeviceSize size,
                                            bool host_visible,
                                            void **output) noexcept = 0;
    virtual void free(void *allocation) noexcept = 0;
    [[nodiscard]] virtual PmResult wait_idle() noexcept = 0;
    [[nodiscard]] virtual PmResult memory_budget(PmDeviceSize &available,
                                                 PmDeviceSize &total) noexcept = 0;
};

[[nodiscard]] PmResult enumerate(std::uint32_t *count,
                                 Description *descriptions) noexcept;
[[nodiscard]] std::unique_ptr<Services> create_cuda(
    int ordinal, PmResult &result) noexcept;

} // namespace parallel_mater::device

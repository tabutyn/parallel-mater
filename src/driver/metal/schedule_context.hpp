// SPDX-License-Identifier: MIT
#pragma once
#include "device/compute.hpp"
#include "device/device.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>

namespace parallel_mater::metal {
namespace compute = device::compute;

struct HostMember { const char *path; std::size_t offset, size; };
struct RelocatedRecord {
    std::span<const std::byte> bytes;
    std::span<const HostMember> members;
};

// Transitional pointer-oriented resource adapter for the shared scheduler.
// Host addresses identify allocations; only gpuAddress values reach shaders.
class ScheduleContext {
public:
    struct ProfileState;
    explicit ScheduleContext(std::unique_ptr<compute::Device>);
    ~ScheduleContext();
    ScheduleContext(const ScheduleContext &) = delete;
    ScheduleContext &operator=(const ScheduleContext &) = delete;
    void activate() noexcept;
    void *allocate(std::size_t);
    void free(void *);
    void *native_resource(const void *) const;
    compute::Device &device() { return *device_; }
    compute::KernelPtr kernel(const std::string &entry);
    void bind(compute::Arguments &, const std::string &, const void *);
    void record(compute::Arguments &, const std::string &, const RelocatedRecord &);
    void meshes(const void *, const void *parameters, std::uint32_t body_count);
    void paint_fields(const void *table, std::uint32_t count, const void *meshes);
    void dispatch(const compute::Arguments &, dim3);
    void copy(void *, const void *, std::size_t);
    void clear(void *, int, std::size_t);
    compute::SubmissionPtr submit();
    void wait();
    [[nodiscard]] bool schedule_profile_enabled() const noexcept;
    cudaError_t begin_schedule_profile() noexcept;
    cudaError_t record_schedule_profile_stage(
        std::string_view name, std::uint32_t launches) noexcept;
    cudaError_t end_schedule_profile() noexcept;
    std::recursive_mutex mutex;
    std::uint64_t dispatch_count{};

    // Public service errors are returned directly, not injected into a later
    // scheduler launch's sticky compatibility error state.
    template<class F> cudaError_t checked(F &&call, bool retain_error = true) noexcept {
        std::lock_guard lock(mutex);
        const auto report = [&](cudaError_t error) {
            // Common encoding/submit failures cancel the private list. Never
            // keep that terminal list as the scheduler's next recording batch.
            // Ordinary validation errors leave it recording and must preserve
            // previously encoded valid work.
            if (command_ && !command_->is_recording()) command_.reset();
            return retain_error ? remember(error) : error;
        };
        try { call(); return cudaSuccess; }
        catch (const std::bad_alloc &) { return report(cudaErrorMemoryAllocation); }
        catch (const std::invalid_argument &error) {
            std::fprintf(stderr, "Metal preview: %s\n", error.what());
            return report(cudaErrorInvalidValue);
        }
        catch (const std::exception &error) {
            std::fprintf(stderr, "Metal preview: %s\n", error.what());
            return report(cudaErrorLaunchFailure);
        }
    }
    cudaError_t error(bool take) noexcept;
    template<class F> PmResult checked_result(F &&call) noexcept {
        return device::from_driver_error(checked(std::forward<F>(call)));
    }

    template<class... T> cudaError_t launch(const char *entry, dim3 grid, const T &...values) noexcept {
        return checked([&] {
            if (!grid.x || !grid.y || !grid.z) return;
            const auto program = kernel(entry);
            compute::Arguments arguments(program);
            std::vector<std::string> names;
            for (const auto &[name, field] : program->parameters) {
                (void)field; names.push_back(name);
            }
            std::sort(names.begin(), names.end(), [&](const auto &a, const auto &b) {
                return program->parameter(a).offset < program->parameter(b).offset;
            });
            if (names.size() != sizeof...(T)) throw std::invalid_argument("Metal argument count mismatch");
            std::size_t index = 0;
            const auto set = [&](const auto &value) {
                const auto &name = names.at(index++);
                using Value = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Value, RelocatedRecord>) record(arguments, name, value);
                else if constexpr (std::is_pointer_v<Value>) bind(arguments, name, value);
                else arguments.set(name, value);
            };
            (set(values), ...);
            dispatch(arguments, grid);
        });
    }
private:
    compute::Record relocate_record(compute::Field, const RelocatedRecord &);
    [[nodiscard]] compute::CommandStatistics command_statistics() const noexcept;
    void collect_profiles(bool wait);
    cudaError_t remember(cudaError_t) noexcept;
    std::unique_ptr<compute::Device> device_;
    std::map<std::string, compute::KernelPtr> kernels_;
    std::map<std::string, std::string> paths_;
    std::map<const void *, std::size_t> table_counts;
    std::unique_ptr<compute::CommandList> command_;
    std::vector<compute::SubmissionPtr> pending_;
    compute::CommandStatistics submitted_statistics_{};
    std::unique_ptr<ProfileState> profile_;
    cudaError_t error_{cudaSuccess};
};
ScheduleContext *context(cudaStream_t stream = nullptr) noexcept;
template<class... T>
cudaError_t launch(cudaStream_t stream, const char *entry, dim3 grid, const T &...values) noexcept {
    auto *execution = context(stream);
    return execution ? execution->launch(entry, grid, values...) : cudaErrorInvalidResourceHandle;
}
inline ScheduleContext *context(device::ExecutionQueue queue) noexcept {
    return context(queue.native_handle);
}
template<class... T>
PmResult launch(device::ExecutionQueue queue, const char *entry, device::DispatchSize grid, const T &...values) noexcept {
    auto *execution = context(queue);
    return execution ? device::from_driver_error(execution->launch(entry, dim3{grid.x, grid.y, grid.z}, values...))
                     : PM_ERROR_INVALID_ARGUMENT;
}
} // namespace parallel_mater::metal

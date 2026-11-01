// SPDX-License-Identifier: MIT
#pragma once
#include "device/compute.hpp"
#include "device/device.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

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
    void bind(compute::Arguments &, const std::string &, const compute::Field &,
              const void *);
    void record(compute::Arguments &, const std::string &, const compute::Field &,
                const RelocatedRecord &);
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
            if (command_ && !command_->is_recording()) {
                command_.reset();
                launch_cursor_ = 0U;
            }
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
            const auto &parameters = parameters_.at(program.get());
            if (parameters.size() != sizeof...(T)) throw std::invalid_argument("Metal argument count mismatch");
            auto &signature = signature_scratch_;
            signature.clear();
            if (signature.capacity() == 0U)
                signature.reserve(sizeof...(T) * sizeof(std::uint64_t));
            std::size_t index = 0;
            const auto append = [&](const auto &value) {
                append_launch_value(signature, parameters.at(index++), value);
            };
            (append(values), ...);
            auto &resource_signature = resource_signature_scratch_;
            resource_signature.clear();
            index = 0;
            const auto append_resources = [&](const auto &value) {
                append_launch_resources(resource_signature,
                    parameters.at(index++), value);
            };
            (append_resources(values), ...);
            if (launch_cursor_ == launch_cache_.size())
                launch_cache_.emplace_back();
            auto &variants = launch_cache_.at(launch_cursor_++);
            for (auto &cached : variants) {
                if (cached.entry == entry && cached.grid.x == grid.x &&
                    cached.grid.y == grid.y && cached.grid.z == grid.z &&
                    cached.signature == signature) {
                    dispatch(*cached.arguments, grid);
                    return;
                }
            }
            for (auto &cached : variants) {
                if (cached.entry != entry || cached.grid.x != grid.x ||
                    cached.grid.y != grid.y || cached.grid.z != grid.z ||
                    cached.resource_signature != resource_signature)
                    continue;
                index = 0;
                const auto update = [&](const auto &value) {
                    const auto &parameter = parameters.at(index++);
                    using Value = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<Value, RelocatedRecord>)
                        cached.arguments->update_record_positional(
                            *parameter.field, value.bytes);
                    else if constexpr (!std::is_pointer_v<Value>)
                        cached.arguments->update_positional(
                            *parameter.field, value);
                };
                (update(values), ...);
                cached.signature.assign(signature.begin(), signature.end());
                dispatch(*cached.arguments, grid);
                return;
            }
            compute::Arguments arguments(program);
            index = 0;
            const auto set = [&](const auto &value) {
                const auto &parameter = parameters.at(index++);
                using Value = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Value, RelocatedRecord>)
                    record(arguments, parameter.name, *parameter.field, value);
                else if constexpr (std::is_pointer_v<Value>)
                    bind(arguments, parameter.name, *parameter.field, value);
                else arguments.set_positional(*parameter.field, value);
            };
            (set(values), ...);
            auto cached_arguments =
                std::make_shared<compute::Arguments>(std::move(arguments));
            if (variants.size() == 4U) variants.erase(variants.begin());
            variants.push_back({entry, grid, signature, resource_signature,
                                cached_arguments});
            try {
                dispatch(*cached_arguments, grid);
            } catch (...) {
                variants.pop_back();
                throw;
            }
        });
    }
private:
    struct Parameter {
        std::string name;
        const compute::Field *field{};
    };
    struct CachedLaunch {
        std::string entry;
        dim3 grid{};
        std::vector<std::byte> signature;
        std::vector<std::byte> resource_signature;
        std::shared_ptr<compute::Arguments> arguments;
    };
    void append_record_resources(std::vector<std::byte> &output,
                                 const compute::Field &field,
                                 std::span<const std::byte> bytes,
                                 std::size_t base) const {
        const auto append = [&](const void *data, std::size_t size) {
            const auto *first = static_cast<const std::byte *>(data);
            output.insert(output.end(), first, first + size);
        };
        if (field.pointer) {
            if (base > bytes.size() || sizeof(void *) > bytes.size() - base)
                throw std::invalid_argument(
                    "Metal record pointer outside signature storage");
            append(bytes.data() + base, sizeof(void *));
        } else if (field.element) {
            for (std::size_t index = 0; index < field.element_count; ++index)
                append_record_resources(output, *field.element, bytes,
                    base + index * field.element_stride);
        } else {
            for (const auto &[name, child] : field.fields) {
                (void)name;
                append_record_resources(output, child, bytes,
                                        base + child.offset);
            }
        }
    }
    template<class T> void append_launch_value(
        std::vector<std::byte> &output, const Parameter &parameter,
        const T &value) const {
        using Value = std::decay_t<T>;
        const auto append = [&](const void *data, std::size_t size) {
            const auto *first = static_cast<const std::byte *>(data);
            output.insert(output.end(), first, first + size);
        };
        if constexpr (std::is_same_v<Value, RelocatedRecord>) {
            append(value.bytes.data(), value.bytes.size());
        } else {
            static_assert(std::is_trivially_copyable_v<Value>);
            append(&value, sizeof(value));
            if constexpr (std::is_pointer_v<Value>) {
                if (value != nullptr &&
                    (parameter.name == "meshes" ||
                     parameter.name == "paint_fields")) {
                    const auto found = table_counts.find(value);
                    if (found == table_counts.end())
                        throw std::invalid_argument(
                            "Metal resource table extent missing: " +
                            parameter.name);
                    append(&found->second, sizeof(found->second));
                    const auto stride = table_strides.find(value);
                    if (stride == table_strides.end())
                        throw std::invalid_argument(
                            "Metal resource table stride missing: " +
                            parameter.name);
                    append(value, found->second * stride->second);
                }
            }
        }
    }
    template<class T> void append_launch_resources(
        std::vector<std::byte> &output, const Parameter &parameter,
        const T &value) const {
        using Value = std::decay_t<T>;
        const auto append = [&](const void *data, std::size_t size) {
            const auto *first = static_cast<const std::byte *>(data);
            output.insert(output.end(), first, first + size);
        };
        if constexpr (std::is_same_v<Value, RelocatedRecord>) {
            for (const auto &[name, child] : parameter.field->fields) {
                (void)name;
                append_record_resources(output, child, value.bytes,
                                        child.offset);
            }
        } else if constexpr (std::is_pointer_v<Value>) {
            append(&value, sizeof(value));
            if (value != nullptr &&
                (parameter.name == "meshes" ||
                 parameter.name == "paint_fields")) {
                const auto found = table_counts.find(value);
                if (found == table_counts.end())
                    throw std::invalid_argument(
                        "Metal resource table extent missing: " +
                        parameter.name);
                append(&found->second, sizeof(found->second));
                const auto stride = table_strides.find(value);
                if (stride == table_strides.end())
                    throw std::invalid_argument(
                        "Metal resource table stride missing: " +
                        parameter.name);
                append(value, found->second * stride->second);
            }
        }
    }
    struct CachedSlice {
        compute::BufferPtr buffer;
        std::size_t offset{};
    };
    [[nodiscard]] CachedSlice slice(const void *, std::size_t bytes = 1U);
    compute::Record relocate_record(compute::Field, const RelocatedRecord &);
    [[nodiscard]] compute::CommandStatistics command_statistics() const noexcept;
    void collect_profiles(bool wait);
    cudaError_t remember(cudaError_t) noexcept;
    std::unique_ptr<compute::Device> device_;
    std::map<std::string, compute::KernelPtr> kernels_;
    std::map<const compute::Kernel *, std::vector<Parameter>> parameters_;
    std::map<std::string, std::string> paths_;
    std::map<const void *, std::size_t> table_counts;
    std::map<const void *, std::size_t> table_strides;
    std::unordered_map<const void *, CachedSlice> slices_;
    std::vector<std::vector<CachedLaunch>> launch_cache_;
    std::size_t launch_cursor_{};
    std::vector<std::byte> signature_scratch_;
    std::vector<std::byte> resource_signature_scratch_;
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

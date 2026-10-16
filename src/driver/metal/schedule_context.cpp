// SPDX-License-Identifier: MIT
#include "schedule_context.hpp"
#include "simulation/geometry_storage.hpp"
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <cstring>

namespace parallel_mater::metal {
namespace {
thread_local ScheduleContext *active{};
struct Allocation { compute::BufferPtr buffer; ScheduleContext *owner{}; };
std::mutex allocation_mutex;
std::map<std::uintptr_t, Allocation> allocations;
struct Slice { compute::BufferPtr buffer; std::size_t offset{}; ScheduleContext *owner{}; };
Slice find(const void *pointer, std::size_t bytes = 1) {
    if (!pointer) return {};
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    std::lock_guard lock(allocation_mutex);
    auto it = allocations.upper_bound(address);
    if (it == allocations.begin()) return {};
    --it;
    const auto offset = address - it->first;
    if (offset >= it->second.buffer->size()) return {};
    if (bytes > it->second.buffer->size() - offset)
        throw std::invalid_argument("Metal allocation range exceeded");
    return {it->second.buffer, offset, it->second.owner};
}
void library_anchor() {}
}
ScheduleContext *context(cudaStream_t stream) noexcept {
    return stream ? static_cast<ScheduleContext *>(stream) : active;
}
ScheduleContext::ScheduleContext(std::unique_ptr<compute::Device> device) : device_(std::move(device)) {
    if (!device_) throw std::runtime_error("Apple GPU unavailable (no CPU fallback)");
    Dl_info library{};
    if (!dladdr(reinterpret_cast<const void *>(&library_anchor), &library) || !library.dli_fname)
        throw std::runtime_error("cannot locate Metal driver library");
    const auto root = std::filesystem::path(library.dli_fname).parent_path() / "slang-metal";
    std::ifstream manifest(root / "manifest.txt");
    if (!manifest) throw std::runtime_error("bundled Metal shader manifest missing: " + root.string());
    std::string line;
    while (std::getline(manifest, line)) {
        auto first = line.find('|'), second = line.find('|', first + 1);
        if (first == std::string::npos || second == std::string::npos)
            throw std::runtime_error("invalid bundled Metal manifest");
        paths_.emplace(line.substr(first + 1, second - first - 1), (root / line.substr(second + 1)).string());
    }
}
ScheduleContext::~ScheduleContext() {
    try { wait(); } catch (...) {}
    std::lock_guard lock(allocation_mutex);
    std::erase_if(allocations, [this](const auto &entry) { return entry.second.owner == this; });
    if (active == this) active = nullptr;
}
void ScheduleContext::activate() noexcept { active = this; }
void *ScheduleContext::allocate(std::size_t bytes) {
    if (!bytes) throw std::invalid_argument("empty Metal allocation");
    auto buffer = device_->allocate(bytes, true);
    auto *pointer = buffer->host_address();
    if (!pointer) throw std::bad_alloc();
    std::lock_guard lock(allocation_mutex);
    allocations.emplace(reinterpret_cast<std::uintptr_t>(pointer), Allocation{buffer, this});
    return pointer;
}
void ScheduleContext::free(void *pointer) {
    if (!pointer) return;
    wait();
    std::lock_guard lock(allocation_mutex);
    const auto it = allocations.find(reinterpret_cast<std::uintptr_t>(pointer));
    if (it == allocations.end() || it->second.owner != this)
        throw std::invalid_argument("foreign Metal allocation");
    allocations.erase(it);
}
void *ScheduleContext::native_resource(const void *pointer) const {
    const auto slice = find(pointer);
    return slice.owner == this && slice.offset == 0 ? slice.buffer->native_handle() : nullptr;
}
compute::KernelPtr ScheduleContext::kernel(const std::string &entry) {
    if (auto found = kernels_.find(entry); found != kernels_.end()) return found->second;
    const auto &path = paths_.at(entry);
    auto result = device_->load_kernel(path + ".metallib", path + ".json", entry);
    kernels_.emplace(entry, result);
    return result;
}
void ScheduleContext::bind(compute::Arguments &arguments, const std::string &name, const void *pointer) {
    if (!pointer) { arguments.bind(name, {}); return; }
    const auto &field = arguments.kernel()->parameter(name);
    auto slice = find(pointer, field.pointee_size);
    if (!slice.buffer || slice.owner != this) throw std::invalid_argument("unregistered Metal shader pointer: " + name);
    // Mesh descriptors alone contain pointers in the rigid preview. Validate
    // every host/native offset before relocation, never mutate the host table.
    if (name == "meshes") {
        using simulation::TriangleMeshResource;
        if (field.pointee_size != sizeof(TriangleMeshResource))
            throw std::invalid_argument("Metal mesh record size mismatch");
        const std::pair<const char *, std::size_t> members[]{
#define PM_MESH_FIELD(member) {#member, offsetof(TriangleMeshResource, member)}
            PM_MESH_FIELD(vertices), PM_MESH_FIELD(indices), PM_MESH_FIELD(vertex_count),
            PM_MESH_FIELD(index_count), PM_MESH_FIELD(generation), PM_MESH_FIELD(alive),
            PM_MESH_FIELD(minimum), PM_MESH_FIELD(maximum), PM_MESH_FIELD(bounding_center),
            PM_MESH_FIELD(bounding_radius), PM_MESH_FIELD(unit_inertia), PM_MESH_FIELD(bvh_nodes),
            PM_MESH_FIELD(bvh_node_count), PM_MESH_FIELD(bvh_leaves), PM_MESH_FIELD(bvh_leaf_count),
            PM_MESH_FIELD(solid_planes), PM_MESH_FIELD(solid_unique_plane_count), PM_MESH_FIELD(shell_normals)
#undef PM_MESH_FIELD
        };
        for (auto [member, offset] : members)
            if (field.fields.at(member).offset != offset) throw std::invalid_argument("Metal mesh field offset mismatch");
        // Mesh arrays may live inside an arena. The allocation tail is not its
        // extent: the resource manager registers the array length below.
        const auto count_field = mesh_counts.find(pointer);
        if (count_field == mesh_counts.end()) throw std::invalid_argument("Metal mesh table extent missing");
        const auto count = count_field->second;
        if (count > (slice.buffer->size() - slice.offset) / sizeof(TriangleMeshResource))
            throw std::invalid_argument("Metal mesh table outside allocation");
        auto translated = device_->allocate(count * sizeof(TriangleMeshResource), true);
        auto *output = static_cast<std::byte *>(translated->host_address());
        auto *input = static_cast<const TriangleMeshResource *>(pointer);
        for (std::size_t i = 0; i < count; ++i) {
            compute::Record record(field, input[i]);
            for (const auto &[member, layout] : field.fields) {
                if (!layout.pointer) continue;
                const void *value{};
                std::memcpy(&value, reinterpret_cast<const std::byte *>(&input[i]) + layout.offset, sizeof(value));
                const auto indirect = find(value, layout.pointee_size);
                if (value && (!indirect.buffer || indirect.owner != this)) throw std::invalid_argument("unregistered nested Metal pointer");
                record.bind(member, indirect.buffer, indirect.offset);
            }
            const auto bytes = record.bytes();
            std::memcpy(output + i * sizeof(TriangleMeshResource), bytes.data(), bytes.size());
            for (const auto &resource : record.resources()) arguments.reference(resource);
        }
        arguments.bind(name, translated);
    } else arguments.bind(name, slice.buffer, slice.offset);
}
void ScheduleContext::meshes(const void *table, const void *parameters, std::uint32_t count) {
    std::lock_guard lock(mutex);
    const auto *bodies = static_cast<const simulation::BodyParameters *>(parameters);
    std::size_t size = 1;
    for (std::uint32_t i = 0; i < count; ++i)
        size = std::max(size, std::size_t(bodies[i].mesh.index) + 1);
    mesh_counts[table] = size;
}
void ScheduleContext::dispatch(const compute::Arguments &arguments, dim3 groups) {
    if (!command_) command_ = device_->commands();
    command_->dispatch(arguments, {groups.x, groups.y, groups.z});
    ++dispatch_count;
}
compute::SubmissionPtr ScheduleContext::submit() {
    std::erase_if(pending_, [](const auto &completion) { return completion->ready(); });
    if (!command_) command_ = device_->commands();
    auto completion = command_->submit();
    command_.reset();
    pending_.push_back(completion);
    return completion;
}
void ScheduleContext::wait() {
    std::lock_guard lock(mutex);
    if (command_) (void)submit();
    for (const auto &completion : pending_) completion->wait();
    pending_.clear();
}
void ScheduleContext::copy(void *destination, const void *source, std::size_t bytes) {
    if (!bytes) return;
    if (!source || !destination) throw std::invalid_argument("null Metal copy pointer");
    const auto dst = find(destination, bytes), src = find(source, bytes);
    if ((dst.buffer && dst.owner != this) || (src.buffer && src.owner != this))
        throw std::invalid_argument("cross-device Metal copy");
    if (dst.buffer && src.buffer && dst.buffer != src.buffer) {
        if (!command_) command_ = device_->commands();
        command_->copy(dst.buffer, dst.offset, src.buffer, src.offset, bytes);
    } else if (dst.buffer && !src.buffer) {
        if (!command_) command_ = device_->commands();
        command_->upload(dst.buffer, dst.offset, {static_cast<const std::byte *>(source), bytes});
    } else {
        // Preview readback boundary, including overlapping shared-arena copies.
        wait();
        std::memmove(destination, source, bytes);
    }
}
void ScheduleContext::clear(void *destination, int value, std::size_t bytes) {
    if (!bytes) return;
    auto dst = find(destination, bytes);
    if (!dst.buffer || dst.owner != this) throw std::invalid_argument("unregistered Metal clear pointer");
    if (!command_) command_ = device_->commands();
    command_->fill(dst.buffer, dst.offset, bytes, static_cast<std::uint8_t>(value));
}
cudaError_t ScheduleContext::remember(cudaError_t value) noexcept { error_ = value; return value; }
cudaError_t ScheduleContext::error(bool take) noexcept {
    std::lock_guard lock(mutex);
    const auto result = error_;
    if (take) error_ = cudaSuccess;
    return result;
}
} // namespace parallel_mater::metal

using parallel_mater::metal::context;
cudaError_t pmMetalAllocate(void **output, std::size_t size) noexcept {
    if (!output || !size || !context()) return cudaErrorInvalidValue;
    *output = nullptr;
    return context()->checked([&] { *output = context()->allocate(size); });
}
cudaError_t pmMetalFree(void *pointer) noexcept {
    if (!pointer) return cudaSuccess;
    try {
        auto slice = parallel_mater::metal::find(pointer);
        if (!slice.owner || slice.offset) return cudaErrorInvalidValue;
        return slice.owner->checked([&] { slice.owner->free(pointer); });
    } catch (...) { return cudaErrorInvalidValue; }
}
cudaError_t pmMetalMemcpyAsync(void *dst, const void *src, std::size_t size, cudaMemcpyKind, cudaStream_t stream) noexcept {
    auto *execution = context(stream);
    if (!execution) return cudaErrorInvalidResourceHandle;
    return execution->checked([&] { execution->copy(dst, src, size); if (!stream) execution->wait(); });
}
cudaError_t pmMetalMemsetAsync(void *dst, int value, std::size_t size, cudaStream_t stream) noexcept {
    auto *execution = context(stream);
    if (!execution) return cudaErrorInvalidResourceHandle;
    return execution->checked([&] { execution->clear(dst, value, size); if (!stream) execution->wait(); });
}
cudaError_t pmMetalStreamSynchronize(cudaStream_t stream) noexcept {
    auto *execution = context(stream);
    return execution ? execution->checked([&] { execution->wait(); }) : cudaErrorInvalidResourceHandle;
}
cudaError_t pmMetalRecordEvent(cudaEvent_t event, cudaStream_t stream) noexcept {
    auto *execution = context(stream);
    if (!event || !execution) return cudaErrorInvalidResourceHandle;
    return execution->checked([&] {
        event->submission = execution->submit(); event->stream = execution;
        event->recorded = std::chrono::steady_clock::now();
    });
}
cudaError_t pmMetalWaitEvent(cudaEvent_t event) noexcept {
    if (!event || !event->submission) return cudaErrorInvalidResourceHandle;
    try { event->submission->wait(); return cudaSuccess; } catch (...) { return cudaErrorLaunchFailure; }
}
cudaError_t pmMetalQueryEvent(cudaEvent_t event) noexcept {
    if (!event || !event->submission) return cudaErrorInvalidResourceHandle;
    try { return event->submission->ready() ? cudaSuccess : cudaErrorNotReady; } catch (...) { return cudaErrorLaunchFailure; }
}
cudaError_t pmMetalPeekError(cudaStream_t stream, bool take) noexcept {
    auto *execution = context(stream);
    return execution ? execution->error(take) : cudaErrorInvalidResourceHandle;
}

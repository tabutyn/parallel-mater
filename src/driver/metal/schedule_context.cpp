// SPDX-License-Identifier: MIT
#include "schedule_context.hpp"
#include "resource_records.hpp"
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <functional>

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
    if (name == "meshes" || name == "paint_fields") {
        const auto extent = table_counts.find(pointer);
        if (extent == table_counts.end())
            throw std::invalid_argument("Metal resource table extent missing: " + name);
        const auto count = extent->second;
        if (count == 0) { arguments.bind(name, {}); return; }
        const auto stride = name == "meshes" ? sizeof(simulation::TriangleMeshResource)
                                            : sizeof(simulation::PaintFieldResource);
        if (field.pointee_size != stride || count > (slice.buffer->size() - slice.offset) / stride)
            throw std::invalid_argument("Metal resource table layout/range mismatch: " + name);
        auto translated = device_->allocate(count * stride, true);
        auto *output = static_cast<std::byte *>(translated->host_address());
        for (std::size_t i = 0; i < count; ++i) {
            const auto host = name == "meshes"
                ? mesh_record(static_cast<const simulation::TriangleMeshResource *>(pointer)[i])
                : paint_record(static_cast<const simulation::PaintFieldResource *>(pointer)[i]);
            const auto record = relocate_record(field, host);
            const auto bytes = record.bytes();
            std::memcpy(output + i * stride, bytes.data(), bytes.size());
            for (const auto &binding : record.bindings())
                arguments.reference(name + "[]." + binding.name, binding.range.buffer,
                    binding.range.offset, binding.range.size, binding.range.access);
        }
        arguments.bind(name, translated);
    } else {
        // Unknown pointer-bearing arrays need an explicit extent and layout.
        for (const auto &[member, child] : field.fields)
            if (compute::contains_pointers(child))
                throw std::invalid_argument("unrelocated Metal resource table: " + name + "." + member);
        arguments.bind(name, slice.buffer, slice.offset);
    }
}
void ScheduleContext::meshes(const void *table, const void *parameters, std::uint32_t count) {
    std::lock_guard lock(mutex);
    const auto *bodies = static_cast<const simulation::BodyParameters *>(parameters);
    std::size_t size = 1;
    for (std::uint32_t i = 0; i < count; ++i)
        size = std::max(size, std::size_t(bodies[i].mesh.index) + 1);
    table_counts[table] = size;
}
void ScheduleContext::record(compute::Arguments &arguments, const std::string &name,
                             const RelocatedRecord &host) {
    arguments.record(name, relocate_record(arguments.kernel()->parameter(name), host));
}
void ScheduleContext::paint_fields(const void *table, std::uint32_t count, const void *meshes) {
    std::lock_guard lock(mutex);
    table_counts[table] = count;
    // Painting may reference a UV-bearing mesh distinct from collision geometry.
    // Include these resources in the relocated table, not only body mesh IDs.
    const auto *fields = static_cast<const simulation::PaintFieldResource *>(table);
    for (std::uint32_t i = 0; i < count; ++i)
        if (fields[i].alive && fields[i].options.mesh.generation != 0)
            table_counts[meshes] = std::max(table_counts[meshes],
                std::size_t(fields[i].options.mesh.index) + 1);
}
compute::Record ScheduleContext::relocate_record(compute::Field layout, const RelocatedRecord &host) {
    if (layout.pointer) {
        layout.size = layout.pointee_size;
        layout.alignment = layout.pointee_alignment;
        layout.pointer = false;
    }
    const std::string name = "resource";
    if (host.bytes.size() != layout.size)
        throw std::invalid_argument("Metal host record size mismatch: " + name);
    for (const auto &member : host.members) {
        const compute::Field *field = &layout;
        std::string_view path(member.path);
        std::size_t offset = 0;
        while (!path.empty()) {
            const auto dot = path.find('.');
            field = &field->fields.at(std::string(path.substr(0, dot)));
            offset += field->offset;
            if (dot == std::string_view::npos) break;
            path.remove_prefix(dot + 1);
        }
        if (offset != member.offset || field->size != member.size)
            throw std::invalid_argument("Metal host member layout mismatch: " + name + "." + member.path);
    }
    compute::Record translated(layout, host.bytes);
    const std::function<void(const compute::Field &, const std::string &, std::size_t)> relocate =
        [&](const compute::Field &field, const std::string &path, std::size_t offset) {
            if (field.pointer) {
                if (offset > host.bytes.size() || sizeof(void *) > host.bytes.size() - offset)
                    throw std::invalid_argument("Metal nested pointer outside host record");
                const void *pointer{};
                std::memcpy(&pointer, host.bytes.data() + offset, sizeof(pointer));
                const auto resource = find(pointer, field.pointee_size);
                if (pointer && (!resource.buffer || resource.owner != this))
                    throw std::invalid_argument("unregistered Metal record pointer: " + name + "." + path);
                translated.bind(path, resource.buffer, resource.offset);
            } else if (field.element) {
                for (std::size_t i = 0; i < field.element_count; ++i)
                    relocate(*field.element, path + "[" + std::to_string(i) + "]",
                             offset + i * field.element_stride);
            } else {
                for (const auto &[child_name, child] : field.fields)
                    relocate(child, path.empty() ? child_name : path + "." + child_name, offset + child.offset);
            }
        };
    relocate(layout, "", 0);
    return translated;
}
void ScheduleContext::dispatch(const compute::Arguments &arguments, dim3 groups) {
    if (!command_) command_ = device_->commands();
    command_->dispatch(arguments, {groups.x, groups.y, groups.z});
    ++dispatch_count;
}
compute::SubmissionPtr ScheduleContext::submit() {
    std::erase_if(pending_, [](const auto &completion) { return completion->ready(); });
    if (!command_) command_ = device_->commands();
    // Allocate the scheduler's tracking slot before GPU acceptance too. A
    // post-submit vector growth failure must never lose work from wait().
    pending_.emplace_back();
    try {
        auto completion = command_->submit();
        pending_.back() = completion;
        command_.reset();
        return completion;
    } catch (...) {
        pending_.pop_back();
        throw;
    }
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

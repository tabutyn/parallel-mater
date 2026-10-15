// SPDX-License-Identifier: MIT
#pragma once

// Private reference-runtime execution boundary. No native API types and no
// physics equations. A command list owns its encoding state; submissions own
// all referenced resources through GPU completion.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace parallel_mater::device::compute {

struct Extent {
    std::uint32_t x{1}, y{1}, z{1};
};

class Buffer {
public:
    virtual ~Buffer() = default;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    // Explicit host access; caller must wait for conflicting GPU work first.
    // Device-only allocations return nullptr. Never encode this as a GPU pointer.
    [[nodiscard]] virtual void *host_address() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t device_address() const noexcept = 0;
    [[nodiscard]] virtual void *native_handle() const noexcept = 0;
};
using BufferPtr = std::shared_ptr<Buffer>;

struct Field {
    std::size_t offset{}, size{}, alignment{1};
    bool pointer{};
    std::size_t pointee_size{}, pointee_alignment{};
    // Native record layout, including nested records. Pointer members carry
    // pointee sizes but are not recursively expanded (recursive types allowed).
    std::map<std::string, Field, std::less<>> fields;
    std::shared_ptr<Field> element;
    std::size_t element_count{}, element_stride{};
};

[[nodiscard]] inline bool contains_pointers(const Field &field) {
    if (field.pointer || (field.element && contains_pointers(*field.element))) return true;
    for (const auto &[name, child] : field.fields) {
        (void)name;
        if (contains_pointers(child)) return true;
    }
    return false;
}

[[nodiscard]] inline std::uint64_t address_for(const Field &field,
    const BufferPtr &buffer, std::size_t offset) {
    if (!field.pointer || field.size != sizeof(std::uint64_t) || !field.pointee_alignment)
        throw std::invalid_argument("invalid device-pointer field");
    if (!buffer) {
        if (offset != 0) throw std::invalid_argument("null device pointer with nonzero offset");
        return 0;
    }
    if (offset >= buffer->size() || field.pointee_size > buffer->size() - offset ||
        offset > std::numeric_limits<std::uint64_t>::max() - buffer->device_address() ||
        (buffer->device_address() + offset) % field.pointee_alignment != 0)
        throw std::invalid_argument("device-pointer range/alignment invalid");
    return buffer->device_address() + offset;
}

[[nodiscard]] inline bool same_storage(const Field &left, const Field &right) {
    if (left.size != right.size || left.alignment != right.alignment || left.pointer != right.pointer ||
        (left.pointer && (left.pointee_size != right.pointee_size ||
                         left.pointee_alignment != right.pointee_alignment)) ||
        left.fields.size() != right.fields.size() || bool(left.element) != bool(right.element) ||
        left.element_count != right.element_count || left.element_stride != right.element_stride)
        return false;
    if (left.element && !same_storage(*left.element, *right.element)) return false;
    for (const auto &[name, field] : left.fields) {
        const auto found = right.fields.find(name);
        if (found == right.fields.end() || field.offset != found->second.offset ||
            !same_storage(field, found->second)) return false;
    }
    return true;
}

// Copy host record values, explicitly relocate every embedded pointer, retain
// the reachable allocations. Never mutate the host's original resource record.
class Record {
public:
    Record(Field layout, std::span<const std::byte> host) : layout_(std::move(layout)) {
        // A pointer parameter's fields describe its pointee record.
        if (layout_.pointer) {
            layout_.size = layout_.pointee_size;
            layout_.alignment = layout_.pointee_alignment;
            layout_.pointer = false;
        }
        if (host.size() != layout_.size || layout_.fields.empty())
            throw std::invalid_argument("host/native record layout mismatch");
        bytes_.assign(host.begin(), host.end());
        collect(layout_, "", 0);
        for (const auto &[name, field] : pointers_) {
            (void)name;
            std::memset(bytes_.data() + field.offset, 0, field.size);
        }
    }
    template<class T> Record(Field layout, const T &host)
        : Record(std::move(layout), std::as_bytes(std::span{&host, 1})) {
        static_assert(std::is_trivially_copyable_v<T>);
    }
    void bind(std::string_view path, BufferPtr buffer, std::size_t offset = 0) {
        const auto found = pointers_.find(path);
        if (found == pointers_.end()) throw std::invalid_argument("unknown record pointer: " + std::string(path));
        const auto address = address_for(found->second, buffer, offset);
        std::memcpy(bytes_.data() + found->second.offset, &address, sizeof(address));
        resources_[std::string(path)] = std::move(buffer);
    }
    [[nodiscard]] std::span<const std::byte> bytes() const {
        if (resources_.size() != pointers_.size())
            throw std::invalid_argument("record has unrelocated device pointers");
        return bytes_;
    }
    [[nodiscard]] bool matches(const Field &field) const { return same_storage(layout_, field); }
    [[nodiscard]] std::vector<BufferPtr> resources() const {
        (void)bytes();
        std::vector<BufferPtr> result;
        for (const auto &[name, buffer] : resources_) {
            (void)name;
            if (buffer) result.push_back(buffer);
        }
        return result;
    }
private:
    void collect(const Field &field, const std::string &path, std::size_t offset) {
        if (field.pointer) {
            if (offset > bytes_.size() || field.size > bytes_.size() - offset)
                throw std::invalid_argument("record pointer outside storage");
            auto pointer = field;
            pointer.offset = offset;
            pointers_.emplace(path, std::move(pointer));
        } else if (field.element) {
            for (std::size_t i = 0; i < field.element_count; ++i)
                collect(*field.element, path + "[" + std::to_string(i) + "]",
                    offset + i * field.element_stride);
        } else {
            for (const auto &[name, child] : field.fields)
                collect(child, path.empty() ? name : path + "." + name, offset + child.offset);
        }
    }
    Field layout_;
    std::vector<std::byte> bytes_;
    std::map<std::string, Field, std::less<>> pointers_;
    std::map<std::string, BufferPtr, std::less<>> resources_;
};

class Kernel {
public:
    virtual ~Kernel() = default;
    std::string name;
    std::size_t argument_size{};
    Extent threads;
    std::uint32_t simd_width{};
    std::map<std::string, Field, std::less<>> parameters;

    [[nodiscard]] const Field &parameter(std::string_view name) const {
        const auto found = parameters.find(name);
        if (found == parameters.end())
            throw std::invalid_argument("unknown kernel parameter: " + std::string(name));
        return found->second;
    }
};
using KernelPtr = std::shared_ptr<const Kernel>;

class Arguments {
public:
    explicit Arguments(KernelPtr kernel) : kernel_(std::move(kernel)) {
        if (!kernel_) throw std::invalid_argument("null compute kernel");
        bytes_.resize(kernel_->argument_size);
    }
    template<class T> void set(std::string_view name, const T &value) {
        static_assert(std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>);
        const auto &field = kernel_->parameter(name);
        if (contains_pointers(field))
            throw std::invalid_argument("pointer-bearing parameters require bind()/record()");
        write(field, &value, sizeof(value));
        initialized_[std::string(name)] = true;
    }
    void bind(std::string_view name, const BufferPtr &buffer, std::size_t offset = 0) {
        const auto &field = kernel_->parameter(name);
        const auto address = address_for(field, buffer, offset);
        write(field, &address, sizeof(address));
        pointers_[std::string(name)] = buffer;
        initialized_[std::string(name)] = true;
    }
    void record(std::string_view name, const Record &value) {
        const auto &field = kernel_->parameter(name);
        if (field.pointer || field.fields.empty() || !value.matches(field))
            throw std::invalid_argument("record() requires a by-value record parameter");
        const auto bytes = value.bytes();
        write(field, bytes.data(), bytes.size());
        records_[std::string(name)] = value.resources();
        initialized_[std::string(name)] = true;
    }
    // Register all allocations reachable through pointers embedded in records.
    // Native adapters use these for residency, hazards and lifetime retention.
    void reference(BufferPtr buffer) {
        if (!buffer) throw std::invalid_argument("null indirect resource");
        indirect_.push_back(std::move(buffer));
    }
    void validate() const {
        for (const auto &[name, field] : kernel_->parameters) {
            (void)field;
            if (!initialized_.contains(name))
                throw std::invalid_argument("unset kernel parameter: " + name);
        }
    }
    [[nodiscard]] const KernelPtr &kernel() const noexcept { return kernel_; }
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept { return bytes_; }
    [[nodiscard]] std::vector<BufferPtr> resources() const {
        auto result = indirect_;
        for (const auto &[name, buffers] : records_) {
            (void)name;
            result.insert(result.end(), buffers.begin(), buffers.end());
        }
        for (const auto &[name, buffer] : pointers_) {
            (void)name;
            if (buffer) result.push_back(buffer);
        }
        return result;
    }
private:
    void write(const Field &field, const void *data, std::size_t size) {
        if (size != field.size || field.offset > bytes_.size() ||
            size > bytes_.size() - field.offset)
            throw std::invalid_argument("kernel argument layout mismatch");
        std::memcpy(bytes_.data() + field.offset, data, size);
    }
    KernelPtr kernel_;
    std::vector<std::byte> bytes_;
    std::map<std::string, bool, std::less<>> initialized_;
    std::map<std::string, BufferPtr, std::less<>> pointers_;
    std::map<std::string, std::vector<BufferPtr>, std::less<>> records_;
    std::vector<BufferPtr> indirect_;
};

class Submission {
public:
    virtual ~Submission() = default;
    // GPU failures are deferred exceptions from ready()/wait()/elapsed_seconds().
    // Destruction never waits or throws. Driver ABI entrypoints translate errors.
    [[nodiscard]] virtual bool ready() const = 0;
    virtual void wait() const = 0;
    [[nodiscard]] virtual double elapsed_seconds() const = 0;
    [[nodiscard]] virtual void *native_handle() const noexcept = 0;
};
using SubmissionPtr = std::shared_ptr<Submission>;

class CommandList {
public:
    virtual ~CommandList() = default;
    // Dispatch boundaries guarantee visibility of prior writes on this queue,
    // including indirectly addressed buffers. No implicit CPU wait/readback.
    virtual void dispatch(const Arguments &arguments, Extent groups) = 0;
    virtual void copy(const BufferPtr &destination, std::size_t destination_offset,
                      const BufferPtr &source, std::size_t source_offset,
                      std::size_t size) = 0;
    virtual void fill(const BufferPtr &destination, std::size_t offset,
                      std::size_t size, std::uint8_t value) = 0;
    // Snapshot caller memory during encoding, retain upload through completion.
    virtual void upload(const BufferPtr &destination, std::size_t offset,
                        std::span<const std::byte> bytes) = 0;
    [[nodiscard]] virtual SubmissionPtr submit() = 0;
};

class Device {
public:
    virtual ~Device() = default;
    [[nodiscard]] virtual std::string name() const = 0;
    [[nodiscard]] virtual BufferPtr allocate(std::size_t size, bool host_visible) = 0;
    [[nodiscard]] virtual KernelPtr load_kernel(const std::string &library,
        const std::string &reflection, const std::string &entry) = 0;
    [[nodiscard]] virtual std::unique_ptr<CommandList> commands() = 0;
    [[nodiscard]] virtual void *native_device() const noexcept = 0;
    [[nodiscard]] virtual void *native_queue() const noexcept = 0;
};

// nullptr means no supported hardware; never substitutes serial CPU execution.
[[nodiscard]] std::unique_ptr<Device> create_metal_device();
} // namespace parallel_mater::device::compute

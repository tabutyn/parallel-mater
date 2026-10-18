// SPDX-License-Identifier: MIT
#pragma once

// Private reference-runtime execution boundary. No native API types and no
// physics equations. A command list owns its encoding state; submissions own
// all referenced resources through GPU completion.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <list>
#include <map>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace parallel_mater::device::compute {

struct Extent {
    std::uint32_t x{1}, y{1}, z{1};
};

// Allocation identities are meaningful only together: two native APIs may
// reuse the same allocation value on different devices. Values are opaque to
// common code and stable for the lifetime of the owning allocation.
struct ResourceIdentity {
    std::uint64_t device{}, allocation{};
    [[nodiscard]] bool valid() const noexcept { return device != 0 && allocation != 0; }
    friend bool operator==(const ResourceIdentity &, const ResourceIdentity &) = default;
};

enum class ResourceAccess : std::uint8_t {
    read = 1,
    write = 2,
    read_write = 3,
};

enum class ResourceBindingKind : std::uint8_t {
    parameter,
    record_pointer,
    indirect,
};

class Buffer {
public:
    virtual ~Buffer() = default;
    [[nodiscard]] virtual ResourceIdentity identity() const noexcept = 0;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    // Explicit host access; caller must wait for conflicting GPU work first.
    // Device-only allocations return nullptr. Never encode this as a GPU pointer.
    [[nodiscard]] virtual void *host_address() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t device_address() const noexcept = 0;
    [[nodiscard]] virtual void *native_handle() const noexcept = 0;
};
using BufferPtr = std::shared_ptr<Buffer>;

inline constexpr std::size_t remaining_resource = std::numeric_limits<std::size_t>::max();

struct ResourceRange {
    BufferPtr buffer;
    std::size_t offset{}, size{};
    ResourceAccess access{ResourceAccess::read_write};

    [[nodiscard]] ResourceIdentity identity() const noexcept {
        return buffer ? buffer->identity() : ResourceIdentity{};
    }
};

struct ResourceBinding {
    std::string name;
    ResourceBindingKind kind{ResourceBindingKind::parameter};
    ResourceRange range;
};

[[nodiscard]] inline bool valid(ResourceAccess access) noexcept {
    return access == ResourceAccess::read || access == ResourceAccess::write ||
        access == ResourceAccess::read_write;
}
[[nodiscard]] inline bool valid(ResourceBindingKind kind) noexcept {
    return kind == ResourceBindingKind::parameter ||
        kind == ResourceBindingKind::record_pointer ||
        kind == ResourceBindingKind::indirect;
}

[[nodiscard]] inline ResourceRange resource_range(BufferPtr buffer, std::size_t offset = 0,
    std::size_t size = remaining_resource, ResourceAccess access = ResourceAccess::read_write) {
    if (!buffer || !buffer->identity().valid())
        throw std::invalid_argument("null or unidentified compute resource");
    if (!valid(access)) throw std::invalid_argument("invalid compute resource access");
    if (offset > buffer->size()) throw std::invalid_argument("compute resource range out of bounds");
    if (size == remaining_resource) size = buffer->size() - offset;
    if (size == 0 || size > buffer->size() - offset)
        throw std::invalid_argument("compute resource range out of bounds");
    return {std::move(buffer), offset, size, access};
}

// Aliases are allocation-identity based, never address-only. Half-open byte
// ranges may overlap; dispatch metadata preserves every use and leaves hazard
// scheduling to the adapter rather than silently merging access intent.
[[nodiscard]] inline bool aliases(const ResourceRange &left, const ResourceRange &right) noexcept {
    return left.identity().valid() && left.identity() == right.identity();
}
[[nodiscard]] inline bool overlaps(const ResourceRange &left, const ResourceRange &right) noexcept {
    if (!aliases(left, right) || left.size == 0 || right.size == 0) return false;
    return left.offset <= right.offset ? right.offset - left.offset < left.size
                                       : left.offset - right.offset < right.size;
}

inline void validate_resource_bindings(std::span<const ResourceBinding> bindings) {
    for (std::size_t i = 0; i < bindings.size(); ++i) {
        const auto &left = bindings[i].range;
        if (bindings[i].name.empty() || !valid(bindings[i].kind) || !left.buffer ||
            !left.identity().valid() || !valid(left.access) || left.size == 0 ||
            left.offset > left.buffer->size() || left.size > left.buffer->size() - left.offset)
            throw std::invalid_argument("invalid compute resource binding");
        for (std::size_t j = 0; j < i; ++j) {
            const auto &right = bindings[j].range;
            if (aliases(left, right) &&
                (left.buffer->size() != right.buffer->size() ||
                 left.buffer->device_address() != right.buffer->device_address()))
                throw std::invalid_argument("inconsistent aliased compute resource");
        }
    }
}

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
    void bind(std::string_view path, BufferPtr buffer, std::size_t offset = 0,
              std::size_t size = remaining_resource,
              ResourceAccess access = ResourceAccess::read_write) {
        const auto found = pointers_.find(path);
        if (found == pointers_.end()) throw std::invalid_argument("unknown record pointer: " + std::string(path));
        std::optional<ResourceRange> resource;
        if (buffer) {
            resource = resource_range(buffer, offset, size, access);
            if (resource->size < found->second.pointee_size)
                throw std::invalid_argument("record resource range smaller than pointee");
        } else if (offset != 0 || size != remaining_resource) {
            throw std::invalid_argument("null device pointer with explicit range");
        }
        const auto address = address_for(found->second, buffer, offset);
        std::memcpy(bytes_.data() + found->second.offset, &address, sizeof(address));
        resources_[std::string(path)] = std::move(resource);
    }
    [[nodiscard]] std::span<const std::byte> bytes() const {
        if (resources_.size() != pointers_.size())
            throw std::invalid_argument("record has unrelocated device pointers");
        return bytes_;
    }
    [[nodiscard]] bool matches(const Field &field) const { return same_storage(layout_, field); }
    [[nodiscard]] std::vector<ResourceBinding> bindings() const {
        (void)bytes();
        std::vector<ResourceBinding> result;
        for (const auto &[name, resource] : resources_) {
            if (resource) result.push_back({name, ResourceBindingKind::record_pointer, *resource});
        }
        validate_resource_bindings(result);
        return result;
    }
    [[nodiscard]] std::vector<BufferPtr> resources() const {
        std::vector<BufferPtr> result;
        for (const auto &binding : bindings()) result.push_back(binding.range.buffer);
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
    std::map<std::string, std::optional<ResourceRange>, std::less<>> resources_;
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
    void bind(std::string_view name, const BufferPtr &buffer, std::size_t offset = 0,
              std::size_t size = remaining_resource,
              ResourceAccess access = ResourceAccess::read_write) {
        const auto &field = kernel_->parameter(name);
        std::optional<ResourceRange> resource;
        if (buffer) {
            resource = resource_range(buffer, offset, size, access);
            if (resource->size < field.pointee_size)
                throw std::invalid_argument("parameter resource range smaller than pointee");
        } else if (offset != 0 || size != remaining_resource) {
            throw std::invalid_argument("null device pointer with explicit range");
        }
        const auto address = address_for(field, buffer, offset);
        write(field, &address, sizeof(address));
        pointers_[std::string(name)] = std::move(resource);
        initialized_[std::string(name)] = true;
    }
    void record(std::string_view name, const Record &value) {
        const auto &field = kernel_->parameter(name);
        if (field.pointer || field.fields.empty() || !value.matches(field))
            throw std::invalid_argument("record() requires a by-value record parameter");
        const auto bytes = value.bytes();
        write(field, bytes.data(), bytes.size());
        records_[std::string(name)] = value.bindings();
        initialized_[std::string(name)] = true;
    }
    // Register allocations reached through device-memory records. The caller
    // supplies the complete range because reflection cannot inspect records
    // that are already resident on the device.
    void reference(std::string_view name, BufferPtr buffer, std::size_t offset = 0,
                   std::size_t size = remaining_resource,
                   ResourceAccess access = ResourceAccess::read_write) {
        if (name.empty()) throw std::invalid_argument("unnamed indirect resource");
        indirect_.push_back({std::string(name), ResourceBindingKind::indirect,
                             resource_range(std::move(buffer), offset, size, access)});
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
    [[nodiscard]] std::vector<ResourceBinding> bindings() const {
        auto result = indirect_;
        for (const auto &[name, bindings] : records_) {
            for (auto binding : bindings) {
                binding.name = name + "." + binding.name;
                binding.kind = ResourceBindingKind::record_pointer;
                result.push_back(std::move(binding));
            }
        }
        for (const auto &[name, resource] : pointers_) {
            if (resource) result.push_back({name, ResourceBindingKind::parameter, *resource});
        }
        validate_resource_bindings(result);
        return result;
    }
    [[nodiscard]] std::vector<BufferPtr> resources() const {
        std::vector<BufferPtr> result;
        for (const auto &binding : bindings()) result.push_back(binding.range.buffer);
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
    std::map<std::string, std::optional<ResourceRange>, std::less<>> pointers_;
    std::map<std::string, std::vector<ResourceBinding>, std::less<>> records_;
    std::vector<ResourceBinding> indirect_;
};

class Submission {
public:
    virtual ~Submission() = default;
    // GPU failures are deferred exceptions from ready()/wait()/elapsed_seconds().
    // Destruction never waits or throws. Driver ABI entrypoints translate errors.
    // ready() may throw only after native work is terminal; wait() must drain
    // native work before returning OR throwing. Ownership relies on this rule.
    [[nodiscard]] virtual bool ready() const = 0;
    virtual void wait() const = 0;
    [[nodiscard]] virtual double elapsed_seconds() const = 0;
    [[nodiscard]] virtual void *native_handle() const noexcept = 0;
};
using SubmissionPtr = std::shared_ptr<Submission>;

// Native encoding only. Common CommandList owns every referenced allocation.
// Encoding never submits work; destruction cancels unsubmitted native commands.
class CommandEncoder {
public:
    virtual ~CommandEncoder() = default;
    // Dispatch boundaries guarantee visibility of prior writes on this queue,
    // including indirectly addressed buffers. No implicit CPU wait/readback.
    virtual void dispatch(const Arguments &arguments, Extent groups) = 0;
    virtual void copy(const BufferPtr &destination, std::size_t destination_offset,
                      const BufferPtr &source, std::size_t source_offset,
                      std::size_t size) = 0;
    virtual void fill(const BufferPtr &destination, std::size_t offset,
                      std::size_t size, std::uint8_t value) = 0;
    // Snapshot caller memory during encoding; return the owning staging buffer
    // for common retention. nullptr is allowed only when no staging is needed.
    virtual BufferPtr upload(const BufferPtr &destination, std::size_t offset,
                             std::span<const std::byte> bytes) = 0;
    // Allocate the native completion token BEFORE enqueueing. A throw/null
    // result means nothing was enqueued. Once accepted, always return a token
    // and report execution errors through it, never via a submit exception.
    [[nodiscard]] virtual SubmissionPtr submit() = 0;
};

namespace detail {
struct ResourceOwners {
    std::unordered_set<BufferPtr> buffers;
    std::unordered_set<KernelPtr> kernels;
};

class RetainedSubmission final : public Submission {
public:
    explicit RetainedSubmission(std::shared_ptr<ResourceOwners> owners)
        : owners_(std::move(owners)) {}
    void attach(SubmissionPtr native) noexcept { native_ = std::move(native); }
    bool ready() const override {
        // Another thread may be inside explicit wait(). Polling must not turn
        // that wait into an implicit queue-wide GPU synchronization point.
        std::unique_lock lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) return false;
        return ready_locked();
    }
    void wait() const override {
        std::lock_guard lock(mutex_);
        try { native_->wait(); }
        catch (...) { owners_.reset(); throw; }
        owners_.reset();
    }
    double elapsed_seconds() const override {
        std::unique_lock lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock() || !ready_locked())
            throw std::logic_error("GPU timestamps requested before completion");
        return native_->elapsed_seconds();
    }
    void *native_handle() const noexcept override { return native_->native_handle(); }
private:
    bool ready_locked() const {
        try {
            if (!native_->ready()) return false;
        } catch (...) { owners_.reset(); throw; }
        owners_.reset();
        return true;
    }
    SubmissionPtr native_;
    mutable std::mutex mutex_;
    mutable std::shared_ptr<ResourceOwners> owners_;
};

class SubmissionState {
public:
    void require_open() const {
        std::lock_guard lock(mutex_);
        if (closed_) throw std::logic_error("compute submission queue is closed");
    }
    SubmissionPtr submit(CommandEncoder &encoder, const std::shared_ptr<ResourceOwners> &owners) {
        auto completion = std::make_shared<RetainedSubmission>(owners);
        std::lock_guard lock(mutex_);
        if (closed_) throw std::logic_error("compute submission queue is closed");
        collect_locked();
        // Allocate the pending slot before native acceptance. Everything after
        // successful enqueue is noexcept, even if the client drops its token.
        pending_.push_back(completion);
        try {
            auto native = encoder.submit();
            if (!native) throw std::runtime_error("native submit returned no completion");
            completion->attach(std::move(native));
        } catch (...) {
            pending_.pop_back();
            throw;
        }
        return completion;
    }
    void collect() {
        std::lock_guard lock(mutex_);
        collect_locked();
    }
    void close() noexcept {
        std::list<std::shared_ptr<RetainedSubmission>> pending;
        {
            std::lock_guard lock(mutex_);
            closed_ = true;
            pending.swap(pending_);
        }
        // No queue lock while waiting. A deferred failure is terminal, but
        // must not prevent later submissions from draining during teardown.
        for (const auto &completion : pending) {
            try { completion->wait(); } catch (...) {}
        }
    }
private:
    void collect_locked() {
        for (auto it = pending_.begin(); it != pending_.end();) {
            bool terminal;
            try { terminal = (*it)->ready(); } catch (...) { terminal = true; }
            if (terminal) it = pending_.erase(it);
            else ++it;
        }
    }
    mutable std::mutex mutex_;
    bool closed_{};
    std::list<std::shared_ptr<RetainedSubmission>> pending_;
};
} // namespace detail

class SubmissionQueue;

// Single-writer recording object. Validation errors leave prior commands
// intact; an encoding/submit failure cancels this list. Pending work is owned
// by the device's SubmissionQueue, not by the caller's completion token.
class CommandList final {
public:
    CommandList(const CommandList &) = delete;
    CommandList &operator=(const CommandList &) = delete;
    void dispatch(const Arguments &arguments, Extent groups) {
        ensure_open();
        arguments.validate();
        if (!groups.x || !groups.y || !groups.z)
            throw std::invalid_argument("empty compute dispatch");
        const auto bindings = arguments.bindings();
        encode([&] {
            owners_->kernels.insert(arguments.kernel());
            for (const auto &binding : bindings) owners_->buffers.insert(binding.range.buffer);
            encoder_->dispatch(arguments, groups);
        });
    }
    void copy(const BufferPtr &destination, std::size_t destination_offset,
              const BufferPtr &source, std::size_t source_offset, std::size_t size) {
        ensure_open();
        range(destination, destination_offset, size);
        range(source, source_offset, size);
        if (!size) return;
        if (overlaps({destination, destination_offset, size}, {source, source_offset, size}))
            throw std::invalid_argument("overlapping GPU copy ranges");
        encode([&] {
            owners_->buffers.insert(destination);
            owners_->buffers.insert(source);
            encoder_->copy(destination, destination_offset, source, source_offset, size);
        });
    }
    void fill(const BufferPtr &destination, std::size_t offset, std::size_t size, std::uint8_t value) {
        ensure_open();
        range(destination, offset, size);
        if (!size) return;
        encode([&] {
            owners_->buffers.insert(destination);
            encoder_->fill(destination, offset, size, value);
        });
    }
    void upload(const BufferPtr &destination, std::size_t offset, std::span<const std::byte> bytes) {
        ensure_open();
        range(destination, offset, bytes.size());
        if (bytes.empty()) return;
        encode([&] {
            owners_->buffers.insert(destination);
            auto staging = encoder_->upload(destination, offset, bytes);
            if (staging) owners_->buffers.insert(std::move(staging));
        });
    }
    [[nodiscard]] SubmissionPtr submit() {
        ensure_open();
        try {
            auto completion = state_->submit(*encoder_, owners_);
            cancel(); // Native token and queue now own the accepted work.
            return completion;
        } catch (...) { cancel(); throw; }
    }
private:
    friend class SubmissionQueue;
    CommandList(std::shared_ptr<detail::SubmissionState> state, std::unique_ptr<CommandEncoder> encoder)
        : state_(std::move(state)), owners_(std::make_shared<detail::ResourceOwners>()),
          encoder_(std::move(encoder)) {
        if (!encoder_) throw std::invalid_argument("null compute command encoder");
        ensure_open();
    }
    static void range(const BufferPtr &buffer, std::size_t offset, std::size_t size) {
        if (!buffer || !buffer->identity().valid() || offset > buffer->size() ||
            size > buffer->size() - offset)
            throw std::invalid_argument("invalid compute transfer range");
    }
    void ensure_open() const {
        if (!encoder_) throw std::logic_error("compute command list already submitted or cancelled");
        state_->require_open();
    }
    template<class Function> void encode(Function &&function) {
        try { function(); } catch (...) { cancel(); throw; }
    }
    void cancel() noexcept {
        encoder_.reset();
        owners_.reset();
    }
    std::shared_ptr<detail::SubmissionState> state_;
    std::shared_ptr<detail::ResourceOwners> owners_;
    // Destroy the unsubmitted native encoder before releasing its allocations.
    std::unique_ptr<CommandEncoder> encoder_;
};

// Device-owned lifetime domain. Declare after native device/queue members so
// destruction drains accepted work before those native objects are released.
// Command lists may outlive it, but cannot encode/submit after shutdown.
class SubmissionQueue final {
public:
    SubmissionQueue() : state_(std::make_shared<detail::SubmissionState>()) {}
    ~SubmissionQueue() { state_->close(); }
    SubmissionQueue(const SubmissionQueue &) = delete;
    SubmissionQueue &operator=(const SubmissionQueue &) = delete;
    [[nodiscard]] std::unique_ptr<CommandList> commands(std::unique_ptr<CommandEncoder> encoder) {
        collect();
        return std::unique_ptr<CommandList>(new CommandList(state_, std::move(encoder)));
    }
    // Nonblocking reclamation, also performed when recording/submitting again.
    // Errors remain observable through a retained completion token.
    void collect() { state_->collect(); }
private:
    std::shared_ptr<detail::SubmissionState> state_;
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

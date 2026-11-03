// SPDX-License-Identifier: MIT
#if !__has_feature(objc_arc)
#error "Metal compute requires ARC"
#endif
#include "compute.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace parallel_mater::device::compute {
namespace {
void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}
NSString *native(const std::string &value) {
    NSString *result = [NSString stringWithUTF8String:value.c_str()];
    require(result != nil, "invalid UTF-8 Metal resource name");
    return result;
}
std::string error_text(NSError *error) {
    if (!error) return "no native error detail";
    std::ostringstream result;
    result << error.localizedDescription.UTF8String;
    NSArray<id<MTLCommandBufferEncoderInfo>> *encoders =
        error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
    for (id<MTLCommandBufferEncoderInfo> encoder in encoders) {
        if (encoder.errorState != MTLCommandEncoderErrorStateFaulted)
            continue;
        result << "; encoder="
               << (encoder.label.length ? encoder.label.UTF8String : "<unlabeled>")
               << " state=" << static_cast<unsigned long>(encoder.errorState);
    }
    return result.str();
}
std::string field_name(NSString *native_name) {
    std::string name = native_name.UTF8String;
    const auto suffix = name.find_last_of('_');
    if (suffix != std::string::npos && suffix + 1 < name.size() &&
        name.find_first_not_of("0123456789", suffix + 1) == std::string::npos)
        name.resize(suffix);
    return name;
}

class MetalBuffer final : public Buffer {
public:
    explicit MetalBuffer(id<MTLBuffer> value)
        : value(value), device(value.device), size_(value.length),
          host_address_(value.storageMode == MTLStorageModeShared ? value.contents : nullptr),
          device_address_(value.gpuAddress), identity_{
              static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>((__bridge void *)device)),
              static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>((__bridge void *)value))} {}
    id<MTLBuffer> value;
    id<MTLDevice> device;
    // Native buffer metadata is immutable for the allocation's lifetime. The
    // dependency tracker reads these fields repeatedly for every dispatch.
    ResourceIdentity identity() const noexcept override { return identity_; }
    std::size_t size() const noexcept override { return size_; }
    void *host_address() const noexcept override { return host_address_; }
    std::uint64_t device_address() const noexcept override { return device_address_; }
    void *native_handle() const noexcept override { return (__bridge void *)value; }
private:
    const std::size_t size_;
    void *const host_address_;
    const std::uint64_t device_address_;
    const ResourceIdentity identity_;
};

class MetalKernel final : public Kernel {
public:
    id<MTLComputePipelineState> pipeline;
    NSUInteger argument_index{};
    NSUInteger dispatch_index{};
    bool has_dispatch_uniform{};
};

Field scalar_layout(MTLDataType type) {
    Field result;
    switch (type) {
    case MTLDataTypeBool: case MTLDataTypeChar: case MTLDataTypeUChar:
        result.size = result.alignment = 1; break;
    case MTLDataTypeShort: case MTLDataTypeUShort: case MTLDataTypeHalf:
        result.size = result.alignment = 2; break;
    case MTLDataTypeInt: case MTLDataTypeUInt: case MTLDataTypeFloat:
        result.size = result.alignment = 4; break;
    case MTLDataTypeLong: case MTLDataTypeULong:
        result.size = result.alignment = 8; break;
    case MTLDataTypeFloat2: case MTLDataTypeInt2: case MTLDataTypeUInt2:
        result.size = result.alignment = 8; break;
    case MTLDataTypeFloat3: case MTLDataTypeFloat4:
    case MTLDataTypeInt3: case MTLDataTypeInt4:
    case MTLDataTypeUInt3: case MTLDataTypeUInt4:
        result.size = result.alignment = 16; break;
    default: throw std::runtime_error("unsupported native Metal argument layout: " +
        std::to_string(static_cast<unsigned>(type)));
    }
    return result;
}

Field record_layout(MTLStructType *record, bool expand_pointers);
Field pointer_layout(MTLPointerType *pointer, bool expand_pointers) {
    require(pointer != nil && pointer.alignment != 0, "missing native pointer layout");
    Field result;
    result.size = result.alignment = 8;
    result.pointer = true;
    result.pointee_size = pointer.dataSize;
    result.pointee_alignment = pointer.alignment;
    if (expand_pointers && pointer.elementType == MTLDataTypeStruct)
        result.fields = record_layout(pointer.elementStructType, false).fields;
    return result;
}
Field array_layout(MTLArrayType *array) {
    require(array != nil, "missing native array layout");
    // Metal reflection reports zero stride for arrays of raw device pointers.
    // Their native ABI is an array of 64-bit addresses, not an argument table.
    const auto stride = array.elementType == MTLDataTypePointer && array.stride == 0
        ? sizeof(std::uint64_t) : array.stride;
    require(stride != 0 && array.arrayLength <=
        std::numeric_limits<std::size_t>::max() / stride, "invalid native array layout");
    Field result;
    result.size = stride * array.arrayLength;
    Field element;
    if (array.elementType == MTLDataTypeStruct)
        element = record_layout(array.elementStructType, false);
    else if (array.elementType == MTLDataTypeArray)
        element = array_layout(array.elementArrayType);
    else if (array.elementType == MTLDataTypePointer)
        element = pointer_layout(array.elementPointerType, false);
    else
        element = scalar_layout(array.elementType);
    result.alignment = element.alignment;
    result.element_count = array.arrayLength;
    result.element_stride = stride;
    result.element = std::make_shared<Field>(std::move(element));
    return result;
}
Field member_layout(MTLStructMember *member, bool expand_pointers) {
    Field result;
    switch (member.dataType) {
    case MTLDataTypeStruct:
        result = record_layout(member.structType, expand_pointers); break;
    case MTLDataTypeArray:
        result = array_layout(member.arrayType); break;
    case MTLDataTypePointer: {
        result = pointer_layout(member.pointerType, expand_pointers);
        break;
    }
    default: result = scalar_layout(member.dataType); break;
    }
    result.offset = member.offset;
    return result;
}
Field record_layout(MTLStructType *record, bool expand_pointers) {
    require(record != nil, "missing native record layout");
    Field result;
    for (MTLStructMember *member in record.members) {
        auto field = member_layout(member, expand_pointers);
        result.size = std::max(result.size, field.offset + field.size);
        result.alignment = std::max(result.alignment, field.alignment);
        require(result.fields.emplace(field_name(member.name), std::move(field)).second,
            "ambiguous native record field");
    }
    result.size = (result.size + result.alignment - 1) / result.alignment * result.alignment;
    // metal::array<T, N> is reflected as its implementation record containing
    // one __elems array. Expose the source array, not this native wrapper.
    const auto elements = result.fields.find("__elems");
    if (result.fields.size() == 1 && elements != result.fields.end() &&
        elements->second.element && elements->second.offset == 0 &&
        elements->second.size == result.size)
        return elements->second;
    return result;
}

class MetalSubmission final : public Submission {
public:
    MetalSubmission(id<MTLCommandBuffer> command, QueueIdentity identity,
                    id<MTLComputePipelineState> transfer_pipeline)
        : command_(command), identity_(identity), transfer_pipeline_(transfer_pipeline) {}
    bool ready() const override {
        check_error();
        return command_.status == MTLCommandBufferStatusCompleted;
    }
    void wait() const override {
        [command_ waitUntilCompleted];
        check_error();
        require(command_.status == MTLCommandBufferStatusCompleted,
            "Metal command failed to complete");
    }
    double elapsed_seconds() const override {
        if (!ready()) throw std::logic_error("GPU timestamps requested before completion");
        return command_.GPUEndTime - command_.GPUStartTime;
    }
    double completion_seconds() const override {
        if (!ready()) throw std::logic_error("GPU timestamps requested before completion");
        return command_.GPUEndTime;
    }
    void *native_handle() const noexcept override { return (__bridge void *)command_; }
    QueueIdentity queue_identity() const noexcept override { return identity_; }
private:
    void check_error() const {
        if (command_.status == MTLCommandBufferStatusError)
            throw std::runtime_error("Metal execution failed: " + error_text(command_.error));
    }
    id<MTLCommandBuffer> command_;
    QueueIdentity identity_;
    // Unretained command buffers require utility pipelines to outlive GPU work.
    [[maybe_unused]] id<MTLComputePipelineState> transfer_pipeline_;
};

class MetalCommandList final : public CommandEncoder {
public:
    MetalCommandList(id<MTLDevice> device, id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> transfer_pipeline)
        : device_(device), queue_(queue), transfer_pipeline_(transfer_pipeline) {
        // A caller may run thousands of physics submissions inside one outer
        // autorelease pool. Do not retain every completed command until then.
        @autoreleasepool {
            MTLCommandBufferDescriptor *descriptor =
                [MTLCommandBufferDescriptor new];
            // Common submission ownership already retains every referenced
            // buffer and kernel through completion. Avoid duplicating hundreds
            // of per-dispatch Objective-C retains in the native command buffer.
            descriptor.retainedReferences = NO;
            const char *diagnostics = std::getenv(
                "PM_METAL_ENCODER_DIAGNOSTICS");
            if (diagnostics && std::strcmp(diagnostics, "1") == 0) {
                descriptor.errorOptions =
                    MTLCommandBufferErrorOptionEncoderExecutionStatus;
            }
            command_ = [queue commandBufferWithDescriptor:descriptor];
            require(command_ != nil, "Metal command buffer allocation failed");
        }
    }
    ~MetalCommandList() override { finish_encoders(); }
    QueueIdentity queue_identity() const noexcept override {
        return {NativeApi::metal,
                reinterpret_cast<std::uintptr_t>((__bridge void *)device_),
                reinterpret_cast<std::uintptr_t>((__bridge void *)queue_)};
    }
    void wait(std::span<const SubmissionPtr> prior) override {
        @autoreleasepool {
            ensure_open();
            for (const auto &completion : prior) {
                if (!completion || completion->queue_identity().api != NativeApi::metal ||
                    completion->queue_identity().device != queue_identity().device)
                    throw std::invalid_argument("foreign Metal prior completion");
                if (completion->queue_identity().queue != queue_identity().queue)
                    throw std::invalid_argument(
                        "Metal cross-queue handoff requires shared-event adapter support");
            }
            // Submitted command buffers on one MTLCommandQueue execute in order.
            // No host wait or additional native command is required here.
        }
    }
    void barrier(std::span<const ResourceDependency> dependencies) override {
        @autoreleasepool {
            ensure_open();
            bool requires_transfer_flush = false;
            for (const auto &dependency : dependencies) {
                if (!aliases(dependency.prior, dependency.next) ||
                    !overlaps(dependency.prior, dependency.next) ||
                    (!writes(dependency.prior.access) && !writes(dependency.next.access)) ||
                    dependency.prior.offset != dependency.next.offset ||
                    dependency.prior.size != dependency.next.size)
                    throw std::invalid_argument("invalid Metal resource dependency");
                auto prior = buffer(dependency.prior.buffer);
                auto next = buffer(dependency.next.buffer);
                requires_transfer_flush |= std::find(pending_resources_.begin(),
                    pending_resources_.end(), prior) != pending_resources_.end();
                require(prior == next, "inconsistent Metal resource dependency");
                range(dependency.prior.buffer, dependency.prior.offset,
                      dependency.prior.size);
                range(dependency.next.buffer, dependency.next.offset,
                      dependency.next.size);
            }
            if (requires_transfer_flush) flush_transfers();
            // Serial compute encoders order physics and byte-transfer kernels;
            // tracked resources make writes visible to the next operation.
            // The common tracker still validates exact conflicting ranges
            // without adding a duplicate native barrier.
        }
    }
    void dispatch(const Arguments &arguments, Extent groups) override {
        @autoreleasepool {
            ensure_open();
            auto kernel = std::dynamic_pointer_cast<const MetalKernel>(arguments.kernel());
            require(kernel && kernel->pipeline.device == device_, "foreign compute kernel");
            if (groups.x == 0 || groups.y == 0 || groups.z == 0)
                throw std::invalid_argument("empty compute dispatch");
            // Validate before opening an encoder, so failed validation leaves the
            // command list usable and cannot strand an active native encoder.
            const auto &bindings = arguments.bindings();
            for (const auto &binding : bindings) (void)buffer(binding.range.buffer);
            flush_transfers();
            const std::uint32_t dimensions[4]{groups.x, groups.y, groups.z, kernel->threads.x};
            id<MTLComputeCommandEncoder> encoder = compute_encoder();
            [encoder setComputePipelineState:kernel->pipeline];
            if (kernel->has_dispatch_uniform)
                [encoder setBytes:dimensions length:sizeof(dimensions) atIndex:kernel->dispatch_index];
            [encoder setBytes:arguments.bytes().data() length:arguments.bytes().size()
                atIndex:kernel->argument_index];
            for (const auto &binding : bindings)
                use_resource(encoder, buffer(binding.range.buffer),
                    usage(binding.range.access));
            [encoder dispatchThreadgroups:MTLSizeMake(groups.x, groups.y, groups.z)
                threadsPerThreadgroup:MTLSizeMake(kernel->threads.x, kernel->threads.y, kernel->threads.z)];
        }
    }
    void copy(const BufferPtr &destination, std::size_t destination_offset,
              const BufferPtr &source, std::size_t source_offset, std::size_t size) override {
        @autoreleasepool {
            ensure_open();
            auto dst = buffer(destination), src = buffer(source);
            range(destination, destination_offset, size);
            range(source, source_offset, size);
            if (size == 0) return;
            if (dst == src && destination_offset < source_offset + size &&
                source_offset < destination_offset + size)
                throw std::invalid_argument("overlapping GPU copy ranges");
            queue_transfer(destination, destination_offset, source, source_offset, size, 0);
        }
    }
    void fill(const BufferPtr &destination, std::size_t offset,
              std::size_t size, std::uint8_t value) override {
        @autoreleasepool {
            ensure_open();
            (void)buffer(destination);
            range(destination, offset, size);
            if (size == 0) return;
            queue_transfer(destination, offset, {}, 0, size, value);
        }
    }
    BufferPtr upload(const BufferPtr &destination, std::size_t offset,
                     std::span<const std::byte> bytes) override {
        @autoreleasepool {
            ensure_open();
            (void)buffer(destination);
            range(destination, offset, bytes.size());
            if (bytes.empty()) return {};
            id<MTLBuffer> staging = [device_ newBufferWithBytes:bytes.data()
                length:bytes.size() options:MTLResourceStorageModeShared];
            require(staging != nil, "Metal upload staging allocation failed");
            auto owner = std::make_shared<MetalBuffer>(staging);
            copy(destination, offset, owner, 0, bytes.size());
            return owner;
        }
    }
    SubmissionPtr submit() override {
        @autoreleasepool {
            ensure_open();
            flush_transfers();
            finish_encoders();
            auto result = std::make_shared<MetalSubmission>(command_, queue_identity(),
                transfer_pipeline_);
            [command_ commit];
            submitted_ = true;
            return result;
        }
    }
private:
    // Independent transfers share a launch. Flush before dependent accesses,
    // physics dispatches, submission, or the 4 KiB argument-buffer limit.
    struct Transfer {
        std::uint64_t destination, source, size;
        std::uint32_t flags, first_group;
    };
    static_assert(sizeof(Transfer) == 32);
    void queue_transfer(const BufferPtr &destination, std::size_t destination_offset,
                        const BufferPtr &source, std::size_t source_offset,
                        std::size_t size, std::uint8_t value) {
        const auto groups = (size + 2047U) / 2048U;
        require(groups <= UINT32_MAX, "Metal transfer dispatch too large");
        if (transfers_.size() == 128 || groups > UINT32_MAX - transfer_groups_)
            flush_transfers();
        auto encoder = compute_encoder();
        auto dst = buffer(destination);
        use_resource(encoder, dst, MTLResourceUsageWrite);
        pending_resources_.push_back(dst);
        const auto destination_address = destination->device_address() + destination_offset;
        std::uint64_t source_address = 0;
        if (source) {
            auto src = buffer(source);
            use_resource(encoder, src, MTLResourceUsageRead);
            pending_resources_.push_back(src);
            source_address = source->device_address() + source_offset;
        }
        const std::uint32_t flags = value | (source ? 256U : 0U) |
            (((destination_address | source_address) & 15U) == 0 ? 512U : 0U);
        transfers_.push_back({destination_address, source_address, size, flags, transfer_groups_});
        transfer_groups_ += static_cast<std::uint32_t>(groups);
    }
    void flush_transfers() {
        if (transfers_.empty()) return;
        auto encoder = compute_encoder();
        const std::uint32_t count = static_cast<std::uint32_t>(transfers_.size());
        [encoder setComputePipelineState:transfer_pipeline_];
        [encoder setBytes:transfers_.data() length:transfers_.size() * sizeof(Transfer) atIndex:0];
        [encoder setBytes:&count length:sizeof(count) atIndex:1];
        [encoder dispatchThreadgroups:MTLSizeMake(transfer_groups_, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        transfers_.clear();
        pending_resources_.clear();
        transfer_groups_ = 0;
    }
    id<MTLComputeCommandEncoder> compute_encoder() {
        if (compute_encoder_ == nil) {
            compute_encoder_ = [command_
                computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
            require(compute_encoder_ != nil, "Metal compute encoder allocation failed");
            compute_encoder_.label = @"ParallelMater compute batch";
            compute_resources_.clear();
        }
        return compute_encoder_;
    }
    void finish_compute_encoder() noexcept {
        if (compute_encoder_ == nil) return;
        [compute_encoder_ endEncoding];
        compute_encoder_ = nil;
        compute_resources_.clear();
    }
    void finish_encoders() noexcept {
        finish_compute_encoder();
    }
    void use_resource(id<MTLComputeCommandEncoder> encoder,
                      id<MTLBuffer> resource, MTLResourceUsage next) {
        void *key = (__bridge void *)resource;
        auto [found, inserted] = compute_resources_.try_emplace(key, next);
        if (!inserted && (found->second & next) == next) return;
        if (!inserted) found->second |= next;
        [encoder useResource:resource usage:found->second];
    }
    static MTLResourceUsage usage(ResourceAccess access) {
        switch (access) {
        case ResourceAccess::read: return MTLResourceUsageRead;
        case ResourceAccess::write: return MTLResourceUsageWrite;
        case ResourceAccess::read_write: return MTLResourceUsageRead | MTLResourceUsageWrite;
        }
        throw std::invalid_argument("invalid Metal resource access");
    }
    void ensure_open() const {
        if (submitted_) throw std::logic_error("compute command list already submitted");
    }
    id<MTLBuffer> buffer(const BufferPtr &value) const {
        const auto *native = dynamic_cast<const MetalBuffer *>(value.get());
        if (!native || native->device != device_)
            throw std::invalid_argument("foreign or null compute buffer");
        return native->value;
    }
    static void range(const BufferPtr &buffer, std::size_t offset, std::size_t size) {
        if (offset > buffer->size() || size > buffer->size() - offset)
            throw std::invalid_argument("compute buffer range out of bounds");
    }
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> transfer_pipeline_;
    id<MTLCommandBuffer> command_;
    id<MTLComputeCommandEncoder> compute_encoder_{};
    std::unordered_map<void *, MTLResourceUsage> compute_resources_;
    std::vector<Transfer> transfers_;
    std::vector<id<MTLBuffer>> pending_resources_;
    std::uint32_t transfer_groups_{};
    bool submitted_{};
};

class MetalDevice final : public Device {
public:
    explicit MetalDevice(id<MTLDevice> device) : device_(device), queue_([device newCommandQueue]) {
        require(queue_ != nil, "Metal command queue allocation failed");
        // Adapter memory operations contain no physics. Keep copies and fills
        // in the serial compute encoder to avoid a compute/blit transition for
        // every small transfer. Vector stores retain exact byte semantics,
        // including unaligned ranges and tails; no floating-point work occurs.
        NSString *source = @R"(
            #include <metal_stdlib>
            using namespace metal;
            struct Transfer {
                device uchar *destination;
                device const uchar *source;
                ulong size;
                uint flags, first_group;
            };
            kernel void pm_transfer_bytes(constant Transfer *transfers [[buffer(0)]],
                constant uint &count [[buffer(1)]], uint group [[threadgroup_position_in_grid]],
                uint lane [[thread_position_in_threadgroup]]) {
                uint operation = 0;
                for (uint next = 1; next < count; ++next) {
                    if (group < transfers[next].first_group) break;
                    operation = next;
                }
                const Transfer value = transfers[operation];
                const ulong first = ulong(group - value.first_group) * 2048 + ulong(lane) * 16;
                if (first >= value.size) return;
                const bool copy = (value.flags & 256) != 0;
                if ((value.flags & 512) != 0 && value.size - first >= 16) {
                    const uint4 bytes = copy
                        ? reinterpret_cast<device const uint4 *>(value.source + first)[0]
                        : uint4((value.flags & 255) * 0x01010101u);
                    reinterpret_cast<device uint4 *>(value.destination + first)[0] = bytes;
                } else {
                    for (ulong byte = first; byte < min(first + 16, value.size); ++byte)
                        value.destination[byte] = copy ? value.source[byte] : uchar(value.flags);
                }
            }
        )";
        NSError *error = nil;
        auto library = [device_ newLibraryWithSource:source options:nil error:&error];
        require(library != nil, "Metal byte-transfer library: " + error_text(error));
        auto function = [library newFunctionWithName:@"pm_transfer_bytes"];
        MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = function;
        descriptor.maxTotalThreadsPerThreadgroup = 128;
        transfer_pipeline_ = [device_ newComputePipelineStateWithDescriptor:descriptor
            options:MTLPipelineOptionNone reflection:nil error:&error];
        require(transfer_pipeline_ != nil, "Metal byte-transfer pipeline: " + error_text(error));
    }
    std::string name() const override { return device_.name.UTF8String; }
    BufferPtr allocate(std::size_t size, bool host_visible) override {
        @autoreleasepool {
            if (size == 0 || size > device_.maxBufferLength)
                throw std::invalid_argument("Metal allocation size outside device limits");
            id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:
                host_visible ? MTLResourceStorageModeShared : MTLResourceStorageModePrivate];
            if (buffer == nil) throw std::bad_alloc();
            return std::make_shared<MetalBuffer>(buffer);
        }
    }
    KernelPtr load_kernel(const std::string &path, const std::string &reflection_path,
                         const std::string &entry) override try {
        @autoreleasepool {
            NSError *error = nil;
            auto library = [device_ newLibraryWithURL:[NSURL fileURLWithPath:native(path)] error:&error];
            require(library != nil, entry + ": " + error_text(error));
            auto function = [library newFunctionWithName:native(entry)];
            require(function != nil, entry + ": native entry not found");
            NSData *data = [NSData dataWithContentsOfFile:native(reflection_path)];
            require(data != nil, entry + ": reflection file missing");
            id json = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
            require([json isKindOfClass:NSDictionary.class], entry + ": invalid reflection JSON");
            NSArray *entries = json[@"entryPoints"];
            require([entries isKindOfClass:NSArray.class] && entries.count == 1,
                entry + ": expected single-entry compilation");
            NSDictionary *reflected_entry = entries[0];
            require([reflected_entry[@"name"] isEqualToString:native(entry)], entry + ": wrong entry reflection");
            NSArray *dimensions = reflected_entry[@"threadGroupSize"];
            require(dimensions.count == 3, entry + ": invalid threadgroup dimensions");
            const Extent threads{[dimensions[0] unsignedIntValue],
                [dimensions[1] unsignedIntValue], [dimensions[2] unsignedIntValue]};
            require(threads.x > 0 && threads.y == 1 && threads.z == 1,
                entry + ": unsupported threadgroup dimensions");
            MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
            descriptor.label = native(entry);
            descriptor.computeFunction = function;
            // Dispatch always uses the Slang entry's fixed group size. Expose
            // that bound to Metal compilation instead of reserving resources
            // for a larger group that this adapter will never launch.
            descriptor.maxTotalThreadsPerThreadgroup = threads.x;
            MTLComputePipelineReflection *reflection = nil;
            auto pipeline = [device_ newComputePipelineStateWithDescriptor:descriptor
                options:MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo
                reflection:&reflection error:&error];
            require(pipeline != nil, entry + ": " + error_text(error));
            auto result = std::make_shared<MetalKernel>();
            result->name = entry;
            result->pipeline = pipeline;
            result->threads = threads;
            result->simd_width = static_cast<std::uint32_t>(pipeline.threadExecutionWidth);
            require(result->threads.x > 0 && result->threads.y == 1 && result->threads.z == 1 &&
                result->threads.x <= pipeline.maxTotalThreadsPerThreadgroup,
                entry + ": unsupported threadgroup dimensions");
            require(pipeline.staticThreadgroupMemoryLength <= device_.maxThreadgroupMemoryLength,
                entry + ": threadgroup storage exceeds device limit");
            bool found_arguments = false;
            for (id<MTLBinding> binding in reflection.bindings) {
                if (binding.type != MTLBindingTypeBuffer) continue;
                id<MTLBufferBinding> argument = (id<MTLBufferBinding>)binding;
                if ([binding.name hasPrefix:@"entryPointParams"]) {
                    found_arguments = true;
                    result->argument_index = argument.index;
                    result->argument_size = argument.bufferDataSize;
                    result->parameters = record_layout(argument.bufferStructType, true).fields;
                } else if ([binding.name containsString:@"pmMetalDispatch"]) {
                    result->has_dispatch_uniform = true;
                    result->dispatch_index = argument.index;
                    const auto dispatch = record_layout(argument.bufferStructType, false);
                    require(argument.bufferDataSize == 16 && dispatch.fields.size() == 4,
                        entry + ": invalid dispatch uniform layout");
                    std::size_t offset = 0;
                    for (const auto *name : {"groupCountX", "groupCountY", "groupCountZ", "groupSizeX"}) {
                        const auto found = dispatch.fields.find(name);
                        require(found != dispatch.fields.end() && found->second.offset == offset &&
                            found->second.size == 4, entry + ": invalid dispatch uniform field");
                        offset += 4;
                    }
                } else {
                    throw std::runtime_error("unsupported native Metal buffer binding: " +
                        std::string(binding.name.UTF8String));
                }
            }
            require(found_arguments && result->argument_size > 0 &&
                result->argument_size <= 4096, entry + ": missing/oversized argument buffers");
            NSArray *parameters = reflected_entry[@"parameters"];
            require(parameters.count == result->parameters.size(), entry + ": native argument count mismatch");
            for (NSDictionary *field in parameters) {
                const auto &actual = result->parameter([field[@"name"] UTF8String]);
                require(actual.offset == [field[@"binding"][@"offset"] unsignedLongLongValue],
                    entry + ": native/Slang argument offset mismatch");
                // Pointee records must use NATIVE reflection: Slang 2026.18
                // incorrectly reports nested bool storage as four bytes.
            }
            return result;
        }
    }
    catch (const std::exception &error) {
        throw std::runtime_error(entry + ": " + error.what());
    }
    std::unique_ptr<CommandList> commands(std::span<const SubmissionPtr> prior = {}) override {
        @autoreleasepool {
            return submissions_.commands(
                std::make_unique<MetalCommandList>(device_, queue_, transfer_pipeline_), prior);
        }
    }
    NativeObject native_device() const override {
        return NativeObject::borrowed(NativeApi::metal, NativeObjectKind::device,
            reinterpret_cast<std::uintptr_t>((__bridge void *)device_));
    }
    NativeObject native_queue() const override {
        return NativeObject::borrowed(NativeApi::metal, NativeObjectKind::queue,
            reinterpret_cast<std::uintptr_t>((__bridge void *)queue_));
    }
private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> transfer_pipeline_;
    SubmissionQueue submissions_;
};
} // namespace

std::unique_ptr<Device> create_metal_device() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyApple7]) return nullptr;
        return std::make_unique<MetalDevice>(device);
    }
}
} // namespace parallel_mater::device::compute

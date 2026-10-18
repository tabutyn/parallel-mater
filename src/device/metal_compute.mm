// SPDX-License-Identifier: MIT
#if !__has_feature(objc_arc)
#error "Metal compute requires ARC"
#endif
#include "compute.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <limits>
#include <utility>

namespace parallel_mater::device::compute {
namespace {
void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
NSString *native(const std::string &value) {
    NSString *result = [NSString stringWithUTF8String:value.c_str()];
    require(result != nil, "invalid UTF-8 Metal resource name");
    return result;
}
std::string error_text(NSError *error) {
    return error ? error.localizedDescription.UTF8String : "no native error detail";
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
    explicit MetalBuffer(id<MTLBuffer> value) : value(value) {}
    id<MTLBuffer> value;
    ResourceIdentity identity() const noexcept override {
        return {
            static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>((__bridge void *)value.device)),
            static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>((__bridge void *)value)),
        };
    }
    std::size_t size() const noexcept override { return value.length; }
    void *host_address() const noexcept override {
        return value.storageMode == MTLStorageModeShared ? value.contents : nullptr;
    }
    std::uint64_t device_address() const noexcept override { return value.gpuAddress; }
    void *native_handle() const noexcept override { return (__bridge void *)value; }
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
    MetalSubmission(id<MTLCommandBuffer> command, QueueIdentity identity)
        : command_(command), identity_(identity) {}
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
    void *native_handle() const noexcept override { return (__bridge void *)command_; }
    QueueIdentity queue_identity() const noexcept override { return identity_; }
private:
    void check_error() const {
        if (command_.status == MTLCommandBufferStatusError)
            throw std::runtime_error("Metal execution failed: " + error_text(command_.error));
    }
    id<MTLCommandBuffer> command_;
    QueueIdentity identity_;
};

class MetalCommandList final : public CommandEncoder {
public:
    MetalCommandList(id<MTLDevice> device, id<MTLCommandQueue> queue)
        : device_(device), queue_(queue) {
        // A caller may run thousands of physics submissions inside one outer
        // autorelease pool. Do not retain every completed command until then.
        @autoreleasepool {
            command_ = [queue commandBuffer];
            require(command_ != nil, "Metal command buffer allocation failed");
        }
    }
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
            for (const auto &dependency : dependencies) {
                if (!aliases(dependency.prior, dependency.next) ||
                    !overlaps(dependency.prior, dependency.next) ||
                    (!writes(dependency.prior.access) && !writes(dependency.next.access)) ||
                    dependency.prior.offset != dependency.next.offset ||
                    dependency.prior.size != dependency.next.size)
                    throw std::invalid_argument("invalid Metal resource dependency");
                auto prior = buffer(dependency.prior.buffer);
                auto next = buffer(dependency.next.buffer);
                require(prior == next, "inconsistent Metal resource dependency");
                range(dependency.prior.buffer, dependency.prior.offset,
                      dependency.prior.size);
                range(dependency.next.buffer, dependency.next.offset,
                      dependency.next.size);
            }
            // Every operation below closes its native encoder. Metal orders
            // those encoders within one command buffer and tracked resources
            // make their writes visible, so this adapter needs no extra native
            // command. The callback still proves common scheduling selected the
            // dependency; it never turns that dependency into a CPU wait.
        }
    }
    void dispatch(const Arguments &arguments, Extent groups) override {
        @autoreleasepool {
            ensure_open();
            arguments.validate();
            auto kernel = std::dynamic_pointer_cast<const MetalKernel>(arguments.kernel());
            require(kernel && kernel->pipeline.device == device_, "foreign compute kernel");
            if (groups.x == 0 || groups.y == 0 || groups.z == 0)
                throw std::invalid_argument("empty compute dispatch");
            // Validate before opening an encoder, so failed validation leaves the
            // command list usable and cannot strand an active native encoder.
            auto bindings = arguments.bindings();
            for (const auto &binding : bindings) (void)buffer(binding.range.buffer);
            const std::uint32_t dimensions[4]{groups.x, groups.y, groups.z, kernel->threads.x};
            id<MTLComputeCommandEncoder> encoder = [command_ computeCommandEncoder];
            require(encoder != nil, "Metal compute encoder allocation failed");
            [encoder setComputePipelineState:kernel->pipeline];
            if (kernel->has_dispatch_uniform)
                [encoder setBytes:dimensions length:sizeof(dimensions) atIndex:kernel->dispatch_index];
            [encoder setBytes:arguments.bytes().data() length:arguments.bytes().size()
                atIndex:kernel->argument_index];
            for (const auto &binding : bindings)
                [encoder useResource:buffer(binding.range.buffer) usage:usage(binding.range.access)];
            [encoder dispatchThreadgroups:MTLSizeMake(groups.x, groups.y, groups.z)
                threadsPerThreadgroup:MTLSizeMake(kernel->threads.x, kernel->threads.y, kernel->threads.z)];
            [encoder endEncoding];
            // Separate encoders + tracked resources implement dependencies
            // selected by the common scheduler. Never a group or host barrier.
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
            id<MTLBlitCommandEncoder> encoder = [command_ blitCommandEncoder];
            require(encoder != nil, "Metal blit encoder allocation failed");
            [encoder copyFromBuffer:src sourceOffset:source_offset toBuffer:dst
                destinationOffset:destination_offset size:size];
            [encoder endEncoding];
        }
    }
    void fill(const BufferPtr &destination, std::size_t offset,
              std::size_t size, std::uint8_t value) override {
        @autoreleasepool {
            ensure_open();
            auto dst = buffer(destination);
            range(destination, offset, size);
            if (size == 0) return;
            id<MTLBlitCommandEncoder> encoder = [command_ blitCommandEncoder];
            require(encoder != nil, "Metal blit encoder allocation failed");
            [encoder fillBuffer:dst range:NSMakeRange(offset, size) value:value];
            [encoder endEncoding];
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
            auto result = std::make_shared<MetalSubmission>(command_, queue_identity());
            [command_ commit];
            submitted_ = true;
            return result;
        }
    }
private:
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
        auto native = std::dynamic_pointer_cast<MetalBuffer>(value);
        if (!native || native->value.device != device_)
            throw std::invalid_argument("foreign or null compute buffer");
        return native->value;
    }
    static void range(const BufferPtr &buffer, std::size_t offset, std::size_t size) {
        if (offset > buffer->size() || size > buffer->size() - offset)
            throw std::invalid_argument("compute buffer range out of bounds");
    }
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLCommandBuffer> command_;
    bool submitted_{};
};

class MetalDevice final : public Device {
public:
    explicit MetalDevice(id<MTLDevice> device) : device_(device), queue_([device newCommandQueue]) {
        require(queue_ != nil, "Metal command queue allocation failed");
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
            MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
            descriptor.label = native(entry);
            descriptor.computeFunction = function;
            MTLComputePipelineReflection *reflection = nil;
            auto pipeline = [device_ newComputePipelineStateWithDescriptor:descriptor
                options:MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo
                reflection:&reflection error:&error];
            require(pipeline != nil, entry + ": " + error_text(error));
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
            auto result = std::make_shared<MetalKernel>();
            result->name = entry;
            result->pipeline = pipeline;
            result->threads = {[dimensions[0] unsignedIntValue], [dimensions[1] unsignedIntValue],
                               [dimensions[2] unsignedIntValue]};
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
            return submissions_.commands(std::make_unique<MetalCommandList>(device_, queue_), prior);
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

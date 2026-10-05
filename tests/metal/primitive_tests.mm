// SPDX-License-Identifier: MIT
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {

class PrimitiveHarness {
  public:
    bool create() {
        device_ = MTLCreateSystemDefaultDevice();
        queue_ = [device_ newMTL4CommandQueue];
        event_ = [device_ newSharedEvent];
        NSError *error = nil;
        NSURL *url = [NSURL
            fileURLWithPath:[NSString
                                stringWithUTF8String:
                                    PARALLEL_MATER_TEST_METALLIB_PATH]];
        library_ = [device_ newLibraryWithURL:url error:&error];
        if (device_ == nil || queue_ == nil || event_ == nil || library_ == nil) {
            std::cerr << "Could not initialize primitive harness: "
                      << (error == nil ? "unknown"
                                       : error.localizedDescription.UTF8String)
                      << '\n';
            return false;
        }
        return true;
    }

    id<MTLBuffer> buffer(const void *bytes, NSUInteger length) {
        if (bytes != nullptr) {
            return [device_ newBufferWithBytes:bytes
                                        length:length
                                       options:MTLResourceStorageModeShared];
        }
        return [device_ newBufferWithLength:length
                                    options:MTLResourceStorageModeShared];
    }

    bool dispatch(NSString *function_name,
                  NSArray<id<MTLBuffer>> *buffers) {
        NSError *error = nil;
        id<MTLFunction> function =
            [library_ newFunctionWithName:function_name];
        id<MTLComputePipelineState> pipeline =
            [device_ newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            std::cerr << "Could not create primitive pipeline: "
                      << error.localizedDescription.UTF8String << '\n';
            return false;
        }

        MTL4ArgumentTableDescriptor *argument_descriptor =
            [[MTL4ArgumentTableDescriptor alloc] init];
        argument_descriptor.maxBufferBindCount = buffers.count;
        argument_descriptor.initializeBindings = YES;
        id<MTL4ArgumentTable> arguments =
            [device_ newArgumentTableWithDescriptor:argument_descriptor
                                               error:&error];
        MTLResidencySetDescriptor *residency_descriptor =
            [[MTLResidencySetDescriptor alloc] init];
        residency_descriptor.initialCapacity = buffers.count;
        id<MTLResidencySet> residency =
            [device_ newResidencySetWithDescriptor:residency_descriptor
                                             error:&error];
        if (arguments == nil || residency == nil) {
            std::cerr << "Could not bind primitive resources\n";
            return false;
        }
        for (NSUInteger index = 0; index < buffers.count; ++index) {
            id<MTLBuffer> buffer = buffers[index];
            [arguments setAddress:buffer.gpuAddress atIndex:index];
            [residency addAllocation:buffer];
        }
        [residency commit];

        id<MTL4CommandAllocator> allocator = [device_ newCommandAllocator];
        id<MTL4CommandBuffer> command_buffer = [device_ newCommandBuffer];
        [command_buffer beginCommandBufferWithAllocator:allocator];
        [command_buffer useResidencySet:residency];
        id<MTL4ComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];
        [encoder setArgumentTable:arguments];
        [encoder setComputePipelineState:pipeline];
        const NSUInteger maximum_group_size =
            std::min<NSUInteger>(256, pipeline.maxTotalThreadsPerThreadgroup);
        NSUInteger group_size = 1;
        while ((group_size << 1U) <= maximum_group_size)
            group_size <<= 1U;
        [encoder dispatchThreads:MTLSizeMake(group_size, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
        [encoder endEncoding];
        [command_buffer endCommandBuffer];
        const id<MTL4CommandBuffer> command_buffers[] = {command_buffer};
        [queue_ commit:command_buffers count:1];
        const std::uint64_t value = next_event_value_++;
        [queue_ signalEvent:event_ value:value];
        if (![event_ waitUntilSignaledValue:value timeoutMS:5000]) {
            std::cerr << "Primitive dispatch timed out\n";
            return false;
        }
        return true;
    }

  private:
    id<MTLDevice> device_{nil};
    id<MTL4CommandQueue> queue_{nil};
    id<MTLSharedEvent> event_{nil};
    id<MTLLibrary> library_{nil};
    std::uint64_t next_event_value_{1};
};

template <typename T, std::size_t Size>
bool equal_buffer(id<MTLBuffer> buffer,
                  const std::array<T, Size> &expected) {
    return std::memcmp(buffer.contents, expected.data(), sizeof(expected)) == 0;
}

template <typename T>
bool equal_buffer(id<MTLBuffer> buffer, const std::vector<T> &expected) {
    return std::memcmp(buffer.contents, expected.data(),
                       expected.size() * sizeof(T)) == 0;
}

} // namespace

int main() {
    PrimitiveHarness harness;
    if (!harness.create()) {
        return 1;
    }

    constexpr std::uint32_t element_count = 1025;
    id<MTLBuffer> count = harness.buffer(&element_count, sizeof(element_count));

    std::vector<std::uint32_t> scan_input(element_count);
    std::vector<std::uint32_t> scan_expected(element_count);
    std::uint32_t running_sum = 0;
    for (std::uint32_t index = 0; index < element_count; ++index) {
        scan_input[index] = (index * 17U + 3U) % 11U;
        scan_expected[index] = running_sum;
        running_sum += scan_input[index];
    }
    id<MTLBuffer> scan_source =
        harness.buffer(scan_input.data(), scan_input.size() * sizeof(std::uint32_t));
    id<MTLBuffer> scan_output =
        harness.buffer(nullptr, scan_input.size() * sizeof(std::uint32_t));
    if (!harness.dispatch(@"pm_exclusive_scan_u32",
                          @[ scan_source, scan_output, count ]) ||
        !equal_buffer(scan_output, scan_expected)) {
        std::cerr << "Exclusive scan failed\n";
        return 1;
    }

    std::vector<std::uint32_t> select_input(element_count);
    std::vector<std::uint8_t> select_flags(element_count);
    std::vector<std::uint32_t> select_expected;
    for (std::uint32_t index = 0; index < element_count; ++index) {
        select_input[index] = index * 13U + 11U;
        select_flags[index] =
            static_cast<std::uint8_t>((index % 3U) == 1U || (index % 17U) == 0U);
        if (select_flags[index] != 0U)
            select_expected.push_back(select_input[index]);
    }
    id<MTLBuffer> select_source =
        harness.buffer(select_input.data(),
                       select_input.size() * sizeof(std::uint32_t));
    id<MTLBuffer> flags =
        harness.buffer(select_flags.data(), select_flags.size() * sizeof(std::uint8_t));
    id<MTLBuffer> select_output =
        harness.buffer(nullptr, select_input.size() * sizeof(std::uint32_t));
    id<MTLBuffer> select_count =
        harness.buffer(nullptr, sizeof(std::uint32_t));
    if (!harness.dispatch(@"pm_select_flagged_u32",
                          @[ select_source, flags, select_output, select_count,
                             count ]) ||
        *static_cast<const std::uint32_t *>(select_count.contents) !=
            select_expected.size() ||
        !equal_buffer(select_output, select_expected)) {
        std::cerr << "Flagged selection failed\n";
        return 1;
    }

    std::vector<std::uint64_t> keys(element_count);
    std::vector<std::uint32_t> values(element_count);
    for (std::uint32_t index = 0; index < element_count; ++index) {
        keys[index] =
            (std::uint64_t{(index * 73U) % 29U} << 40U) |
            std::uint64_t{(index * 19U) % 17U};
        values[index] = index;
    }
    std::vector<std::uint32_t> sorted_values = values;
    std::stable_sort(sorted_values.begin(), sorted_values.end(),
                     [&](std::uint32_t first, std::uint32_t second) {
                         return keys[first] < keys[second];
                     });
    std::vector<std::uint64_t> sorted_keys(element_count);
    for (std::uint32_t index = 0; index < element_count; ++index)
        sorted_keys[index] = keys[sorted_values[index]];

    id<MTLBuffer> keys_a =
        harness.buffer(keys.data(), keys.size() * sizeof(std::uint64_t));
    id<MTLBuffer> keys_b =
        harness.buffer(nullptr, keys.size() * sizeof(std::uint64_t));
    id<MTLBuffer> values_a =
        harness.buffer(values.data(), values.size() * sizeof(std::uint32_t));
    id<MTLBuffer> values_b =
        harness.buffer(nullptr, values.size() * sizeof(std::uint32_t));
    for (std::uint32_t pass = 0; pass < 8; ++pass) {
        const std::uint32_t shift = pass * 8;
        id<MTLBuffer> shift_buffer = harness.buffer(&shift, sizeof(shift));
        const bool even = pass % 2 == 0;
        if (!harness.dispatch(
                @"pm_radix_sort_u64_pass",
                even ? @[ keys_a, values_a, keys_b, values_b, count,
                          shift_buffer ]
                     : @[ keys_b, values_b, keys_a, values_a, count,
                          shift_buffer ])) {
            return 1;
        }
    }
    if (!equal_buffer(keys_a, sorted_keys) ||
        !equal_buffer(values_a, sorted_values)) {
        std::cerr << "Stable 64-bit radix sort failed\n";
        return 1;
    }

    std::vector<std::uint64_t> segment_keys;
    std::vector<float> segment_values;
    std::vector<std::uint64_t> expected_segment_keys;
    std::vector<float> expected_sums;
    std::uint32_t produced = 0;
    std::uint64_t segment_key = 7;
    while (produced < element_count) {
        const std::uint32_t length =
            std::min<std::uint32_t>(1U + (segment_key % 11U),
                                    element_count - produced);
        expected_segment_keys.push_back(segment_key);
        float sum = 0.0F;
        for (std::uint32_t offset = 0; offset < length; ++offset) {
            const float value =
                static_cast<float>(((produced + offset) % 7U) + 1U) * 0.25F;
            segment_keys.push_back(segment_key);
            segment_values.push_back(value);
            sum += value;
        }
        expected_sums.push_back(sum);
        produced += length;
        segment_key += 3;
    }
    id<MTLBuffer> segment_key_buffer =
        harness.buffer(segment_keys.data(),
                       segment_keys.size() * sizeof(std::uint64_t));
    id<MTLBuffer> segment_value_buffer =
        harness.buffer(segment_values.data(),
                       segment_values.size() * sizeof(float));
    id<MTLBuffer> output_segment_keys =
        harness.buffer(nullptr, segment_keys.size() * sizeof(std::uint64_t));
    id<MTLBuffer> output_segment_sums =
        harness.buffer(nullptr, segment_values.size() * sizeof(float));
    id<MTLBuffer> output_segment_count =
        harness.buffer(nullptr, sizeof(std::uint32_t));
    if (!harness.dispatch(
            @"pm_segmented_sum_f32",
            @[ segment_key_buffer, segment_value_buffer, output_segment_keys,
               output_segment_sums, output_segment_count, count ]) ||
        *static_cast<const std::uint32_t *>(output_segment_count.contents) !=
            expected_segment_keys.size() ||
        !equal_buffer(output_segment_keys, expected_segment_keys) ||
        !equal_buffer(output_segment_sums, expected_sums)) {
        std::cerr << "Segmented reduction failed\n";
        return 1;
    }

    std::vector<float> minmax_input(element_count);
    for (std::uint32_t index = 0; index < element_count; ++index)
        minmax_input[index] = static_cast<float>(index % 101U) - 37.0F;
    minmax_input[13] = -123.5F;
    minmax_input[987] = 456.25F;
    const std::array<float, 2> large_minmax_expected{-123.5F, 456.25F};
    id<MTLBuffer> minmax_source =
        harness.buffer(minmax_input.data(), minmax_input.size() * sizeof(float));
    id<MTLBuffer> minmax_output =
        harness.buffer(nullptr, sizeof(large_minmax_expected));
    if (!harness.dispatch(@"pm_minmax_f32",
                          @[ minmax_source, minmax_output, count ]) ||
        !equal_buffer(minmax_output, large_minmax_expected)) {
        std::cerr << "Min/max reduction failed\n";
        return 1;
    }

    const std::uint32_t zero = 0;
    id<MTLBuffer> count_zero = harness.buffer(&zero, sizeof(zero));
    const std::array<float, 2> empty_minmax_expected{
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    if (!harness.dispatch(@"pm_segmented_sum_f32",
                          @[ segment_key_buffer, segment_value_buffer,
                             output_segment_keys, output_segment_sums,
                             output_segment_count, count_zero ]) ||
        *static_cast<const std::uint32_t *>(output_segment_count.contents) != 0U ||
        !harness.dispatch(@"pm_minmax_f32",
                          @[ minmax_source, minmax_output, count_zero ]) ||
        !equal_buffer(minmax_output, empty_minmax_expected)) {
        std::cerr << "Empty primitive contract failed\n";
        return 1;
    }
    return 0;
}

// SPDX-License-Identifier: MIT
#include <parallel_mater/vulkan.hpp>

#include <concepts>
#include <type_traits>
#include <utility>

using namespace parallel_mater;
using namespace parallel_mater::vulkan;

static_assert(std::is_standard_layout_v<NativeContext>);
static_assert(std::is_standard_layout_v<BufferSpan<const RigidBodyState>>);
static_assert(std::is_standard_layout_v<NativeCompletion>);
static_assert(std::is_move_constructible_v<FrameToken>);
static_assert(!std::is_copy_constructible_v<FrameToken>);
static_assert(std::same_as<decltype(World::create(WorldOptions{},
                                                  std::declval<World &>())),
                           Status>);
static_assert(std::same_as<decltype(std::declval<World &>().step_async(
                                      StepOptions{},
                                      std::declval<FrameToken &>())),
                           Status>);

int main() {
    Status failure{StatusCode::vulkan_failure, VK_ERROR_DEVICE_LOST, "lost"};
    return failure.ok() || failure.vulkan_result != VK_ERROR_DEVICE_LOST;
}

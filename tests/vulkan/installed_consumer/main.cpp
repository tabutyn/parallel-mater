#include <parallel_mater/vulkan.hpp>

int main() {
    parallel_mater::vulkan::World world;
    return world.native_context().device == VK_NULL_HANDLE ? 0 : 1;
}

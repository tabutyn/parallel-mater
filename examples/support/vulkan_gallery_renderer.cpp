// SPDX-License-Identifier: MIT
#include <parallel_mater_gallery/frame_timing.hpp>
#include <parallel_mater_gallery/gpu_renderer.hpp>
#include <parallel_mater_scene/command_scene.hpp>

#include "vector_math.hpp"
#include "vulkan_gallery_shaders.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace parallel_mater::gallery {
namespace {

template <typename Handle>
[[nodiscard]] Handle decode_handle(PmNativeHandle handle) noexcept {
    if constexpr (std::is_pointer_v<Handle>)
        return reinterpret_cast<Handle>(static_cast<std::uintptr_t>(handle));
    else
        return static_cast<Handle>(handle);
}

[[noreturn]] void fail(const char *operation, VkResult result) {
    throw std::runtime_error(std::string(operation) + " failed (VkResult " +
                             std::to_string(static_cast<int>(result)) + ")");
}

void check(VkResult result, const char *operation) {
    if (result != VK_SUCCESS) fail(operation, result);
}

struct DrawUniforms {
    std::array<float, 4> eye{}, right{}, up{}, forward{}, color{};
    std::array<std::uint32_t, 4> draw{};
    std::array<std::uint32_t, 4> paint{};
    std::array<float, 4> settings{};
};
static_assert(sizeof(DrawUniforms) == 128U);
static_assert(sizeof(Vertex) == 32U && offsetof(Vertex, normal) == 12U &&
              offsetof(Vertex, uv) == 24U);
static_assert(sizeof(PmRigidState) == 52U &&
              sizeof(PmDomainOutputHeader) == 16U);

[[nodiscard]] std::array<float, 4> vector(Vec3 value,
                                          float fourth = 0.0F) noexcept {
    return {value.x, value.y, value.z, fourth};
}

class VulkanGalleryRenderer final : public GpuGalleryRenderer {
    struct Buffer {
        VkBuffer handle{};
        VkDeviceMemory memory{};
        VkDeviceSize size{};
        void *mapped{};
    };

    struct Image {
        VkImage handle{};
        VkDeviceMemory memory{};
        VkImageView view{};
        VkFormat format{};
    };

    struct Mesh {
        Buffer vertices{};
        Buffer indices{};
        std::uint32_t index_count{};
        bool visible{};
    };

    struct Frame {
        VkCommandBuffer commands{};
        VkFence fence{};
        VkSemaphore image_available{};
        VkSemaphore render_complete{};
        VkDescriptorSet descriptors{};
        Buffer uniforms{};
        Buffer overlay_upload{};
        Buffer readback{};
        Image color{};
        Image depth{};
        Image overlay{};
        VkFramebuffer framebuffer{};
        bool overlay_initialized{};
    };

    static constexpr std::uint32_t frame_count = 2U;
    static constexpr VkFormat color_format = VK_FORMAT_R8G8B8A8_UNORM;
    static constexpr VkFormat depth_format = VK_FORMAT_D32_SFLOAT;

    GpuRendererContractState contract_;
    VkInstance instance_{};
    VkPhysicalDevice physical_{};
    VkDevice device_{};
    VkQueue queue_{};
    VkBuffer physics_{};
    VkDeviceSize physics_offset_{};
    VkDeviceSize physics_size_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    VkPhysicalDeviceProperties properties_{};
    std::uint32_t queue_family_{};
    VkCommandPool command_pool_{};
    VkDescriptorSetLayout descriptor_layout_{};
    VkDescriptorPool descriptor_pool_{};
    VkPipelineLayout pipeline_layout_{};
    VkRenderPass render_pass_{};
    VkPipeline scene_pipeline_{};
    VkPipeline overlay_pipeline_{};
    VkSampler overlay_sampler_{};
    std::array<Frame, frame_count> frames_{};
    std::vector<Mesh> meshes_;
    std::vector<GpuDrawCommand> draws_;
    Buffer dummy_vertices_{};
    std::uint32_t uniform_stride_{};
    std::uint32_t uniform_slots_{};
    std::uint32_t frame_index_{};
    std::uint32_t width_{};
    std::uint32_t height_{};

    VkSurfaceKHR surface_{};
    VkSwapchainKHR swapchain_{};
    VkFormat swapchain_format_{};
    VkExtent2D swapchain_extent_{};
    std::vector<VkImage> swapchain_images_;
    bool swapchain_stale_{};

    [[nodiscard]] std::uint32_t memory_type(
        std::uint32_t bits, VkMemoryPropertyFlags required) const {
        for (std::uint32_t index = 0U; index < memory_.memoryTypeCount; ++index)
            if ((bits & (1U << index)) != 0U &&
                (memory_.memoryTypes[index].propertyFlags & required) == required)
                return index;
        throw std::runtime_error("Vulkan Gallery could not find required memory");
    }

    [[nodiscard]] Buffer create_buffer(VkDeviceSize size,
                                       VkBufferUsageFlags usage,
                                       VkMemoryPropertyFlags properties,
                                       const void *initial = nullptr) {
        Buffer result{};
        result.size = std::max<VkDeviceSize>(size, 4U);
        try {
            VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            info.size = result.size;
            info.usage = usage;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            check(vkCreateBuffer(device_, &info, nullptr, &result.handle),
                  "create Gallery buffer");
            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(device_, result.handle, &requirements);
            VkMemoryAllocateInfo allocation{
                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex =
                memory_type(requirements.memoryTypeBits, properties);
            check(vkAllocateMemory(device_, &allocation, nullptr, &result.memory),
                  "allocate Gallery buffer");
            check(vkBindBufferMemory(device_, result.handle, result.memory, 0U),
                  "bind Gallery buffer");
            if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0U) {
                check(vkMapMemory(device_, result.memory, 0U, result.size, 0U,
                                  &result.mapped),
                      "map Gallery buffer");
                if (initial != nullptr) std::memcpy(result.mapped, initial, size);
            }
            return result;
        } catch (...) {
            destroy(result);
            throw;
        }
    }

    void destroy(Buffer &buffer) noexcept {
        if (buffer.mapped && buffer.memory)
            vkUnmapMemory(device_, buffer.memory);
        if (buffer.handle) vkDestroyBuffer(device_, buffer.handle, nullptr);
        if (buffer.memory) vkFreeMemory(device_, buffer.memory, nullptr);
        buffer = {};
    }

    [[nodiscard]] Image create_image(std::uint32_t width, std::uint32_t height,
                                     VkFormat format,
                                     VkImageUsageFlags usage,
                                     VkImageAspectFlags aspect) {
        Image result{};
        result.format = format;
        try {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = format;
            info.extent = {width, height, 1U};
            info.mipLevels = 1U;
            info.arrayLayers = 1U;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = usage;
            info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            check(vkCreateImage(device_, &info, nullptr, &result.handle),
                  "create Gallery image");
            VkMemoryRequirements requirements{};
            vkGetImageMemoryRequirements(device_, result.handle, &requirements);
            VkMemoryAllocateInfo allocation{
                VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = memory_type(
                requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            check(vkAllocateMemory(device_, &allocation, nullptr, &result.memory),
                  "allocate Gallery image");
            check(vkBindImageMemory(device_, result.handle, result.memory, 0U),
                  "bind Gallery image");
            VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view.image = result.handle;
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = format;
            view.subresourceRange.aspectMask = aspect;
            view.subresourceRange.levelCount = 1U;
            view.subresourceRange.layerCount = 1U;
            check(vkCreateImageView(device_, &view, nullptr, &result.view),
                  "create Gallery image view");
            return result;
        } catch (...) {
            destroy(result);
            throw;
        }
    }

    void destroy(Image &image) noexcept {
        if (image.view) vkDestroyImageView(device_, image.view, nullptr);
        if (image.handle) vkDestroyImage(device_, image.handle, nullptr);
        if (image.memory) vkFreeMemory(device_, image.memory, nullptr);
        image = {};
    }

    [[nodiscard]] VkShaderModule shader(const std::uint8_t *bytes,
                                        std::size_t size) const {
        if (size % sizeof(std::uint32_t) != 0U)
            throw std::runtime_error("Gallery SPIR-V size is invalid");
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = size;
        info.pCode = reinterpret_cast<const std::uint32_t *>(bytes);
        VkShaderModule result{};
        check(vkCreateShaderModule(device_, &info, nullptr, &result),
              "create Gallery shader");
        return result;
    }

    void create_descriptors() {
        const std::array<VkDescriptorSetLayoutBinding, 5U> bindings{{
            {0U, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1U,
             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U,
             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1U,
             VK_SHADER_STAGE_VERTEX_BIT, nullptr},
            {3U, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1U,
             VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
            {4U, VK_DESCRIPTOR_TYPE_SAMPLER, 1U,
             VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        }};
        VkDescriptorSetLayoutCreateInfo layout{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout.pBindings = bindings.data();
        check(vkCreateDescriptorSetLayout(
                  device_, &layout, nullptr, &descriptor_layout_),
              "create Gallery descriptor layout");

        const std::array<VkDescriptorPoolSize, 4U> sizes{{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, frame_count},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, frame_count * 2U},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, frame_count},
            {VK_DESCRIPTOR_TYPE_SAMPLER, frame_count},
        }};
        VkDescriptorPoolCreateInfo pool{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = frame_count;
        pool.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
        pool.pPoolSizes = sizes.data();
        check(vkCreateDescriptorPool(device_, &pool, nullptr, &descriptor_pool_),
              "create Gallery descriptor pool");
        std::array<VkDescriptorSetLayout, frame_count> layouts{};
        layouts.fill(descriptor_layout_);
        std::array<VkDescriptorSet, frame_count> sets{};
        VkDescriptorSetAllocateInfo allocate{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = descriptor_pool_;
        allocate.descriptorSetCount = frame_count;
        allocate.pSetLayouts = layouts.data();
        check(vkAllocateDescriptorSets(device_, &allocate, sets.data()),
              "allocate Gallery descriptor sets");
        for (std::uint32_t index = 0U; index < frame_count; ++index)
            frames_[index].descriptors = sets[index];
    }

    void create_render_pass() {
        const std::array<VkAttachmentDescription, 2U> attachments{{
            {0U, color_format, VK_SAMPLE_COUNT_1_BIT,
             VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
             VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
            {0U, depth_format, VK_SAMPLE_COUNT_1_BIT,
             VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE,
             VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
             VK_IMAGE_LAYOUT_UNDEFINED,
             VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
        }};
        const VkAttachmentReference color{0U,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        const VkAttachmentReference depth{1U,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1U;
        subpass.pColorAttachments = &color;
        subpass.pDepthStencilAttachment = &depth;
        const std::array<VkSubpassDependency, 2U> dependencies{{
            {VK_SUBPASS_EXTERNAL, 0U,
             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
             VK_ACCESS_TRANSFER_READ_BIT,
             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
             VK_DEPENDENCY_BY_REGION_BIT},
            {0U, VK_SUBPASS_EXTERNAL,
             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
             VK_PIPELINE_STAGE_TRANSFER_BIT,
             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
             VK_ACCESS_TRANSFER_READ_BIT,
             VK_DEPENDENCY_BY_REGION_BIT},
        }};
        VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        info.attachmentCount = static_cast<std::uint32_t>(attachments.size());
        info.pAttachments = attachments.data();
        info.subpassCount = 1U;
        info.pSubpasses = &subpass;
        info.dependencyCount = static_cast<std::uint32_t>(dependencies.size());
        info.pDependencies = dependencies.data();
        check(vkCreateRenderPass(device_, &info, nullptr, &render_pass_),
              "create Gallery render pass");
    }

    [[nodiscard]] VkPipeline create_pipeline(const std::uint8_t *vertex_bytes,
                                             std::size_t vertex_size,
                                             const char *vertex_entry,
                                             const std::uint8_t *fragment_bytes,
                                             std::size_t fragment_size,
                                             const char *fragment_entry,
                                             bool overlay) {
        VkShaderModule vertex{};
        VkShaderModule fragment{};
        VkPipeline result{};
        try {
            vertex = shader(vertex_bytes, vertex_size);
            fragment = shader(fragment_bytes, fragment_size);
            const std::array<VkPipelineShaderStageCreateInfo, 2U> stages{{
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0U,
                 VK_SHADER_STAGE_VERTEX_BIT, vertex, vertex_entry, nullptr},
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0U,
                 VK_SHADER_STAGE_FRAGMENT_BIT, fragment, fragment_entry, nullptr},
            }};
            VkVertexInputBindingDescription vertex_binding{
                0U, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
            const std::array<VkVertexInputAttributeDescription, 3U> attributes{{
                {0U, 0U, VK_FORMAT_R32G32B32_SFLOAT,
                 static_cast<std::uint32_t>(offsetof(Vertex, position))},
                {1U, 0U, VK_FORMAT_R32G32B32_SFLOAT,
                 static_cast<std::uint32_t>(offsetof(Vertex, normal))},
                {2U, 0U, VK_FORMAT_R32G32_SFLOAT,
                 static_cast<std::uint32_t>(offsetof(Vertex, uv))},
            }};
            VkPipelineVertexInputStateCreateInfo input{
                VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
            if (!overlay) {
                input.vertexBindingDescriptionCount = 1U;
                input.pVertexBindingDescriptions = &vertex_binding;
                input.vertexAttributeDescriptionCount =
                    static_cast<std::uint32_t>(attributes.size());
                input.pVertexAttributeDescriptions = attributes.data();
            }
            VkPipelineInputAssemblyStateCreateInfo assembly{
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
            assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkPipelineViewportStateCreateInfo viewport{
                VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
            viewport.viewportCount = 1U;
            viewport.scissorCount = 1U;
            VkPipelineRasterizationStateCreateInfo raster{
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_NONE;
            raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster.lineWidth = 1.0F;
            VkPipelineMultisampleStateCreateInfo multisample{
                VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            VkPipelineDepthStencilStateCreateInfo depth{
                VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
            depth.depthTestEnable = overlay ? VK_FALSE : VK_TRUE;
            depth.depthWriteEnable = overlay ? VK_FALSE : VK_TRUE;
            depth.depthCompareOp = VK_COMPARE_OP_LESS;
            VkPipelineColorBlendAttachmentState blend{};
            blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                VK_COLOR_COMPONENT_A_BIT;
            if (overlay) {
                blend.blendEnable = VK_TRUE;
                blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
                blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                blend.colorBlendOp = VK_BLEND_OP_ADD;
                blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
                blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                blend.alphaBlendOp = VK_BLEND_OP_ADD;
            }
            VkPipelineColorBlendStateCreateInfo blending{
                VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
            blending.attachmentCount = 1U;
            blending.pAttachments = &blend;
            const std::array<VkDynamicState, 2U> dynamic_states{
                VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
            VkPipelineDynamicStateCreateInfo dynamic{
                VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
            dynamic.dynamicStateCount =
                static_cast<std::uint32_t>(dynamic_states.size());
            dynamic.pDynamicStates = dynamic_states.data();
            VkGraphicsPipelineCreateInfo pipeline{
                VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
            pipeline.stageCount = static_cast<std::uint32_t>(stages.size());
            pipeline.pStages = stages.data();
            pipeline.pVertexInputState = &input;
            pipeline.pInputAssemblyState = &assembly;
            pipeline.pViewportState = &viewport;
            pipeline.pRasterizationState = &raster;
            pipeline.pMultisampleState = &multisample;
            pipeline.pDepthStencilState = &depth;
            pipeline.pColorBlendState = &blending;
            pipeline.pDynamicState = &dynamic;
            pipeline.layout = pipeline_layout_;
            pipeline.renderPass = render_pass_;
            check(vkCreateGraphicsPipelines(
                      device_, VK_NULL_HANDLE, 1U, &pipeline, nullptr, &result),
                  "create Gallery graphics pipeline");
        } catch (...) {
            if (fragment) vkDestroyShaderModule(device_, fragment, nullptr);
            if (vertex) vkDestroyShaderModule(device_, vertex, nullptr);
            throw;
        }
        vkDestroyShaderModule(device_, fragment, nullptr);
        vkDestroyShaderModule(device_, vertex, nullptr);
        return result;
    }

    void create_pipelines() {
        VkPipelineLayoutCreateInfo layout{
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1U;
        layout.pSetLayouts = &descriptor_layout_;
        check(vkCreatePipelineLayout(device_, &layout, nullptr,
                                     &pipeline_layout_),
              "create Gallery pipeline layout");
        scene_pipeline_ = create_pipeline(
            vulkan_shaders::vertex, sizeof(vulkan_shaders::vertex),
            "vertexMain", vulkan_shaders::fragment,
            sizeof(vulkan_shaders::fragment), "fragmentMain", false);
        overlay_pipeline_ = create_pipeline(
            vulkan_shaders::overlay_vertex,
            sizeof(vulkan_shaders::overlay_vertex), "overlayVertex",
            vulkan_shaders::overlay_fragment,
            sizeof(vulkan_shaders::overlay_fragment), "overlayFragment", true);
    }

    void create_meshes(const SceneDefinition &scene,
                       const CommandSceneInstance &instance) {
        GpuSceneDrawList draw_list;
        std::string error;
        if (!make_gpu_scene_draw_list(scene, instance, draw_list, error))
            throw std::runtime_error(error);
        draws_ = std::move(draw_list.commands);
        meshes_.resize(scene.meshes.size());
        for (std::size_t index = 0U; index < scene.meshes.size(); ++index) {
            const TriangleMesh &source = scene.meshes[index];
            Mesh &mesh = meshes_[index];
            mesh.visible = source.visible && !source.indices.empty();
            if (!mesh.visible) continue;
            std::vector<Vertex> vertices = source.vertices;
            std::vector<std::uint32_t> indices = source.indices;
            const bool corners = std::any_of(
                draws_.begin(), draws_.end(), [&](const GpuDrawCommand &draw) {
                    return draw.mesh_index == index && draw.use_triangle_corners;
                });
            if (corners) {
                vertices.clear();
                vertices.reserve(indices.size());
                for (std::uint32_t corner = 0U; corner < indices.size(); ++corner) {
                    vertices.push_back(source.vertices.at(indices[corner]));
                    indices[corner] = corner;
                }
            }
            mesh.index_count = static_cast<std::uint32_t>(indices.size());
            mesh.vertices = create_buffer(
                vertices.size() * sizeof(Vertex),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                vertices.data());
            mesh.indices = create_buffer(
                indices.size() * sizeof(std::uint32_t),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                indices.data());
        }
        const std::array<Vertex, 6U> dummy{};
        dummy_vertices_ = create_buffer(
            sizeof(dummy), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            dummy.data());
        uniform_slots_ = std::max<std::uint32_t>(
            1U, static_cast<std::uint32_t>(draws_.size()));
    }

    void create_frame_objects() {
        uniform_stride_ = static_cast<std::uint32_t>(
            (sizeof(DrawUniforms) +
             properties_.limits.minUniformBufferOffsetAlignment - 1U) /
            properties_.limits.minUniformBufferOffsetAlignment *
            properties_.limits.minUniformBufferOffsetAlignment);
        if (uniform_stride_ == 0U) uniform_stride_ = sizeof(DrawUniforms);
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = queue_family_;
        check(vkCreateCommandPool(device_, &pool, nullptr, &command_pool_),
              "create Gallery command pool");
        std::array<VkCommandBuffer, frame_count> commands{};
        VkCommandBufferAllocateInfo allocate{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = command_pool_;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = frame_count;
        check(vkAllocateCommandBuffers(device_, &allocate, commands.data()),
              "allocate Gallery command buffers");
        for (std::uint32_t index = 0U; index < frame_count; ++index) {
            Frame &frame = frames_[index];
            frame.commands = commands[index];
            VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            check(vkCreateFence(device_, &fence, nullptr, &frame.fence),
                  "create Gallery frame fence");
            VkSemaphoreCreateInfo semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            check(vkCreateSemaphore(device_, &semaphore, nullptr,
                                    &frame.image_available),
                  "create Gallery acquire semaphore");
            check(vkCreateSemaphore(device_, &semaphore, nullptr,
                                    &frame.render_complete),
                  "create Gallery present semaphore");
            frame.uniforms = create_buffer(
                static_cast<VkDeviceSize>(uniform_stride_) * uniform_slots_,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler.magFilter = sampler.minFilter = VK_FILTER_NEAREST;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = sampler.addressModeV = sampler.addressModeW =
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.maxLod = 0.0F;
        check(vkCreateSampler(device_, &sampler, nullptr, &overlay_sampler_),
              "create Gallery overlay sampler");
    }

    void destroy_frame_targets(Frame &frame) noexcept {
        if (frame.framebuffer)
            vkDestroyFramebuffer(device_, frame.framebuffer, nullptr);
        frame.framebuffer = {};
        destroy(frame.overlay);
        destroy(frame.depth);
        destroy(frame.color);
        destroy(frame.overlay_upload);
        destroy(frame.readback);
        frame.overlay_initialized = false;
    }

    void create_frame_targets(Frame &frame, std::uint32_t width,
                              std::uint32_t height) {
        frame.color = create_image(
            width, height, color_format,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        frame.depth = create_image(
            width, height, depth_format,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT);
        frame.overlay = create_image(
            width, height, color_format,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(width) * height * 4U;
        frame.overlay_upload = create_buffer(
            bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        frame.readback = create_buffer(
            bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        const std::array<VkImageView, 2U> attachments{
            frame.color.view, frame.depth.view};
        VkFramebufferCreateInfo framebuffer{
            VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebuffer.renderPass = render_pass_;
        framebuffer.attachmentCount =
            static_cast<std::uint32_t>(attachments.size());
        framebuffer.pAttachments = attachments.data();
        framebuffer.width = width;
        framebuffer.height = height;
        framebuffer.layers = 1U;
        check(vkCreateFramebuffer(
                  device_, &framebuffer, nullptr, &frame.framebuffer),
              "create Gallery framebuffer");
        update_descriptors(frame);
    }

    void update_descriptors(Frame &frame) {
        const VkDescriptorBufferInfo uniforms{
            frame.uniforms.handle, 0U, sizeof(DrawUniforms)};
        const VkDescriptorBufferInfo physics{
            physics_, physics_offset_, physics_size_};
        const VkDescriptorImageInfo overlay{
            VK_NULL_HANDLE, frame.overlay.view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        const VkDescriptorImageInfo sampler{overlay_sampler_, VK_NULL_HANDLE,
                                            VK_IMAGE_LAYOUT_UNDEFINED};
        const std::array<VkWriteDescriptorSet, 5U> writes{{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, frame.descriptors,
             0U, 0U, 1U, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr,
             &uniforms, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, frame.descriptors,
             1U, 0U, 1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &physics,
             nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, frame.descriptors,
             2U, 0U, 1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &physics,
             nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, frame.descriptors,
             3U, 0U, 1U, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &overlay, nullptr,
             nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, frame.descriptors,
             4U, 0U, 1U, VK_DESCRIPTOR_TYPE_SAMPLER, &sampler, nullptr,
             nullptr},
        }};
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()),
                               writes.data(), 0U, nullptr);
    }

    void ensure_extent(std::uint32_t width, std::uint32_t height) {
        if (width_ == width && height_ == height) return;
        check(vkDeviceWaitIdle(device_), "wait before Gallery resize");
        for (Frame &frame : frames_) destroy_frame_targets(frame);
        width_ = width;
        height_ = height;
        for (Frame &frame : frames_) create_frame_targets(frame, width, height);
    }

    void destroy_swapchain() noexcept {
        swapchain_images_.clear();
        if (swapchain_)
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = {};
        swapchain_format_ = {};
        swapchain_extent_ = {};
    }

    void reset_presentation_semaphores() {
        std::array<VkSemaphore, frame_count * 2U> replacements{};
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        try {
            for (VkSemaphore &semaphore : replacements)
                check(vkCreateSemaphore(device_, &info, nullptr, &semaphore),
                      "recreate Gallery presentation semaphore");
        } catch (...) {
            for (VkSemaphore semaphore : replacements)
                if (semaphore) vkDestroySemaphore(device_, semaphore, nullptr);
            throw;
        }
        for (std::uint32_t index = 0U; index < frame_count; ++index) {
            Frame &frame = frames_[index];
            vkDestroySemaphore(device_, frame.image_available, nullptr);
            vkDestroySemaphore(device_, frame.render_complete, nullptr);
            frame.image_available = replacements[index * 2U];
            frame.render_complete = replacements[index * 2U + 1U];
        }
    }

    [[nodiscard]] VkSurfaceFormatKHR choose_surface_format(
        VkSurfaceKHR surface) const {
        std::uint32_t count{};
        check(vkGetPhysicalDeviceSurfaceFormatsKHR(
                  physical_, surface, &count, nullptr),
              "query Gallery surface formats");
        if (count == 0U)
            throw std::runtime_error("Vulkan Gallery surface has no formats");
        std::vector<VkSurfaceFormatKHR> formats(count);
        check(vkGetPhysicalDeviceSurfaceFormatsKHR(
                  physical_, surface, &count, formats.data()),
              "query Gallery surface formats");
        const auto preferred = std::find_if(
            formats.begin(), formats.end(), [](const VkSurfaceFormatKHR &format) {
                return format.format == VK_FORMAT_B8G8R8A8_UNORM &&
                    format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
            });
        return preferred == formats.end() ? formats.front() : *preferred;
    }

    void create_swapchain(VkSurfaceKHR surface) {
        VkBool32 present_supported{};
        check(vkGetPhysicalDeviceSurfaceSupportKHR(
                  physical_, queue_family_, surface, &present_supported),
              "query Gallery presentation support");
        if (!present_supported)
            throw std::runtime_error(
                "Vulkan physics queue cannot present to this surface");
        VkSurfaceCapabilitiesKHR capabilities{};
        check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                  physical_, surface, &capabilities),
              "query Gallery surface capabilities");
        if ((capabilities.supportedUsageFlags &
             VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0U)
            throw std::runtime_error(
                "Vulkan Gallery surface cannot receive rendered images");
        const VkSurfaceFormatKHR format = choose_surface_format(surface);
        VkExtent2D extent = capabilities.currentExtent;
        if (extent.width == std::numeric_limits<std::uint32_t>::max()) {
            extent.width = std::clamp(width_, capabilities.minImageExtent.width,
                                      capabilities.maxImageExtent.width);
            extent.height = std::clamp(height_, capabilities.minImageExtent.height,
                                       capabilities.maxImageExtent.height);
        }
        if (extent.width == 0U || extent.height == 0U)
            throw std::runtime_error("Vulkan Gallery surface is minimized");
        std::uint32_t image_count = capabilities.minImageCount + 1U;
        if (capabilities.maxImageCount != 0U)
            image_count = std::min(image_count, capabilities.maxImageCount);
        VkCompositeAlphaFlagBitsKHR alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        if ((capabilities.supportedCompositeAlpha & alpha) == 0U) {
            constexpr std::array<VkCompositeAlphaFlagBitsKHR, 3U> alternatives{
                VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
                VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR};
            const auto found = std::find_if(
                alternatives.begin(), alternatives.end(), [&](auto candidate) {
                    return (capabilities.supportedCompositeAlpha & candidate) != 0U;
                });
            if (found == alternatives.end())
                throw std::runtime_error(
                    "Vulkan Gallery surface has no composite-alpha mode");
            alpha = *found;
        }
        VkSwapchainCreateInfoKHR info{
            VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        info.surface = surface;
        info.minImageCount = image_count;
        info.imageFormat = format.format;
        info.imageColorSpace = format.colorSpace;
        info.imageExtent = extent;
        info.imageArrayLayers = 1U;
        info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.preTransform = capabilities.currentTransform;
        info.compositeAlpha = alpha;
        info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        info.clipped = VK_TRUE;
        info.oldSwapchain = swapchain_;
        VkSwapchainKHR replacement{};
        check(vkCreateSwapchainKHR(device_, &info, nullptr, &replacement),
              "create Gallery swapchain");
        if (swapchain_)
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = replacement;
        swapchain_format_ = format.format;
        swapchain_extent_ = extent;
        check(vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr),
              "query Gallery swapchain images");
        swapchain_images_.resize(image_count);
        check(vkGetSwapchainImagesKHR(
                  device_, swapchain_, &image_count, swapchain_images_.data()),
              "query Gallery swapchain images");
        swapchain_stale_ = false;
    }

    void ensure_surface(VkSurfaceKHR surface) {
        if (surface_ != surface) {
            check(vkDeviceWaitIdle(device_), "wait before Gallery surface change");
            destroy_swapchain();
            reset_presentation_semaphores();
            surface_ = surface;
        }
        if (!surface_) {
            destroy_swapchain();
            return;
        }
        if (!swapchain_ || swapchain_stale_) {
            check(vkDeviceWaitIdle(device_), "wait before Gallery swapchain rebuild");
            create_swapchain(surface_);
        }
    }

    [[nodiscard]] bool acquire(Frame &frame, std::uint32_t &image,
                               std::string &error) {
        for (unsigned attempt = 0U; attempt < 2U; ++attempt) {
            const VkResult result = vkAcquireNextImageKHR(
                device_, swapchain_, UINT64_MAX, frame.image_available,
                VK_NULL_HANDLE, &image);
            if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
                if (result == VK_SUBOPTIMAL_KHR) swapchain_stale_ = true;
                return true;
            }
            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                swapchain_stale_ = true;
                try {
                    ensure_surface(surface_);
                } catch (const std::exception &exception) {
                    error = exception.what();
                    return false;
                }
                continue;
            }
            if (result == VK_ERROR_SURFACE_LOST_KHR) {
                (void)vkDeviceWaitIdle(device_);
                destroy_swapchain();
                surface_ = {};
                error = "Vulkan Gallery presentation surface was lost";
                return false;
            }
            error = "Vulkan Gallery image acquisition failed (VkResult " +
                std::to_string(static_cast<int>(result)) + ")";
            return false;
        }
        error = "Vulkan Gallery swapchain remained out of date";
        return false;
    }

    void image_barrier(VkCommandBuffer commands, VkImage image,
                       VkImageLayout old_layout, VkImageLayout new_layout,
                       VkAccessFlags source_access, VkAccessFlags target_access,
                       VkPipelineStageFlags source_stage,
                       VkPipelineStageFlags target_stage) const {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcAccessMask = source_access;
        barrier.dstAccessMask = target_access;
        barrier.oldLayout = old_layout;
        barrier.newLayout = new_layout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1U;
        barrier.subresourceRange.layerCount = 1U;
        vkCmdPipelineBarrier(commands, source_stage, target_stage, 0U, 0U,
                             nullptr, 0U, nullptr, 1U, &barrier);
    }

    void physics_barrier(VkCommandBuffer commands, bool acquire) const {
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = acquire ? VK_ACCESS_MEMORY_WRITE_BIT
                                        : VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = acquire ? VK_ACCESS_SHADER_READ_BIT
                                        : VK_ACCESS_MEMORY_READ_BIT |
                                              VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = physics_;
        barrier.offset = physics_offset_;
        barrier.size = physics_size_;
        vkCmdPipelineBarrier(
            commands,
            acquire ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
                    : VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            acquire ? VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                    : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0U, 0U, nullptr, 1U, &barrier, 0U, nullptr);
    }

    [[nodiscard]] DrawUniforms base_uniforms(const Camera &camera) const {
        using namespace math;
        const Vec3 forward = normalize_or(subtract(camera.target, camera.eye),
                                          {0.0F, 0.0F, -1.0F});
        const Vec3 right = normalize_or(cross(forward, camera.up),
                                        {1.0F, 0.0F, 0.0F});
        const Vec3 up = normalize_or(cross(right, forward),
                                     {0.0F, 1.0F, 0.0F});
        const float vertical = std::tan(
            camera.vertical_field_of_view_degrees * 0.00872664626F);
        DrawUniforms result{};
        result.eye = vector(camera.eye);
        result.forward = vector(forward);
        result.right = vector(
            right, vertical * static_cast<float>(width_) /
                       static_cast<float>(height_));
        result.up = vector(up, vertical);
        result.settings[1] = 1.0F;
        return result;
    }

    [[nodiscard]] std::uint32_t checked_offset(PmDeviceSize value) const {
        if (value > std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error(
                "Gallery output offset exceeds Slang graphics address range");
        return static_cast<std::uint32_t>(value);
    }

    void draw_scene(Frame &frame, const GpuRenderRequest &request) {
        DrawUniforms values = base_uniforms(request.camera);
        std::uint32_t slot{};
        const VkDeviceSize zero{};
        for (const GpuDrawCommand &draw : draws_) {
            values.color = vector(draw.color, 1.0F);
            values.draw = {
                static_cast<std::uint32_t>(draw.kind),
                checked_offset(draw.output.offset), 0U,
                checked_offset(draw.previous_output.offset)};
            values.paint = {};
            values.settings[0] = draw.radius;
            values.settings[3] =
                draw.domain == GpuOutputDomain::fluid ? 1.0F : 0.0F;
            if (draw.kind == GpuDrawKind::particles)
                values.draw[2] = draw.output_stride;
            if (draw.paint_output.size != 0U) {
                if (draw.paint_output.size < sizeof(PmDomainOutputHeader))
                    throw std::runtime_error(
                        "Gallery paint output region is invalid");
                values.paint[0] = checked_offset(draw.paint_output.offset);
                values.paint[1] = draw.paint_resolution;
                values.paint[2] = static_cast<std::uint32_t>(
                    std::min<PmDeviceSize>(
                        (draw.paint_output.size - sizeof(PmDomainOutputHeader)) /
                            sizeof(std::uint32_t),
                        std::numeric_limits<std::uint32_t>::max()));
            }
            if (slot >= uniform_slots_)
                throw std::runtime_error("Gallery uniform capacity was exceeded");
            std::memcpy(static_cast<std::byte *>(frame.uniforms.mapped) +
                            static_cast<std::size_t>(slot) * uniform_stride_,
                        &values, sizeof(values));
            const std::uint32_t dynamic = slot++ * uniform_stride_;
            vkCmdBindDescriptorSets(
                frame.commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                pipeline_layout_, 0U, 1U, &frame.descriptors, 1U, &dynamic);
            if (draw.kind == GpuDrawKind::particles) {
                vkCmdBindVertexBuffers(frame.commands, 0U, 1U,
                                       &dummy_vertices_.handle, &zero);
                if (draw.output.size < sizeof(PmDomainOutputHeader) ||
                    draw.output_stride == 0U)
                    throw std::runtime_error(
                        "Gallery particle output region is invalid");
                const PmDeviceSize capacity =
                    (draw.output.size - sizeof(PmDomainOutputHeader)) /
                    draw.output_stride;
                if (capacity > std::numeric_limits<std::uint32_t>::max())
                    throw std::runtime_error(
                        "Gallery particle capacity exceeds Vulkan draw limits");
                vkCmdDraw(frame.commands, 6U,
                          static_cast<std::uint32_t>(capacity), 0U, 0U);
                continue;
            }
            if (draw.mesh_index >= meshes_.size())
                throw std::runtime_error("Gallery draw references unknown mesh");
            const Mesh &mesh = meshes_[draw.mesh_index];
            if (!mesh.visible || mesh.index_count == 0U) continue;
            vkCmdBindVertexBuffers(frame.commands, 0U, 1U,
                                   &mesh.vertices.handle, &zero);
            vkCmdBindIndexBuffer(frame.commands, mesh.indices.handle, 0U,
                                 VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(frame.commands, mesh.index_count, 1U, 0U, 0, 0U);
        }
    }

    void upload_overlay(Frame &frame,
                        std::span<const std::uint32_t> overlay) {
        if (overlay.empty()) return;
        std::memcpy(frame.overlay_upload.mapped, overlay.data(),
                    overlay.size_bytes());
        image_barrier(
            frame.commands, frame.overlay.handle,
            frame.overlay_initialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                      : VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            frame.overlay_initialized
                ? static_cast<VkAccessFlags>(VK_ACCESS_SHADER_READ_BIT)
                : VkAccessFlags{},
            VK_ACCESS_TRANSFER_WRITE_BIT,
            frame.overlay_initialized ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                      : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1U;
        copy.imageExtent = {width_, height_, 1U};
        vkCmdCopyBufferToImage(
            frame.commands, frame.overlay_upload.handle, frame.overlay.handle,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1U, &copy);
        image_barrier(frame.commands, frame.overlay.handle,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        frame.overlay_initialized = true;
    }

    void copy_readback(Frame &frame) const {
        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1U;
        copy.imageExtent = {width_, height_, 1U};
        vkCmdCopyImageToBuffer(frame.commands, frame.color.handle,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               frame.readback.handle, 1U, &copy);
    }

    void copy_present(Frame &frame, VkImage image) const {
        image_barrier(frame.commands, image, VK_IMAGE_LAYOUT_UNDEFINED,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0U,
                      VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.layerCount = 1U;
        blit.srcOffsets[1] = {static_cast<std::int32_t>(width_),
                              static_cast<std::int32_t>(height_), 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.layerCount = 1U;
        blit.dstOffsets[1] = {
            static_cast<std::int32_t>(swapchain_extent_.width),
            static_cast<std::int32_t>(swapchain_extent_.height), 1};
        vkCmdBlitImage(frame.commands, frame.color.handle,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1U, &blit,
                       VK_FILTER_NEAREST);
        image_barrier(frame.commands, image,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                      VK_ACCESS_TRANSFER_WRITE_BIT, 0U,
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    }

    [[nodiscard]] bool submit(Frame &frame, bool presenting,
                              std::uint32_t image_index,
                              GpuDiagnosticReadback readback,
                              std::string &error) {
        check(vkEndCommandBuffer(frame.commands),
              "finish Gallery command buffer");
        const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        if (presenting) {
            submit.waitSemaphoreCount = 1U;
            submit.pWaitSemaphores = &frame.image_available;
            submit.pWaitDstStageMask = &wait_stage;
            submit.signalSemaphoreCount = 1U;
            submit.pSignalSemaphores = &frame.render_complete;
        }
        submit.commandBufferCount = 1U;
        submit.pCommandBuffers = &frame.commands;
        check(vkResetFences(device_, 1U, &frame.fence),
              "reset Gallery frame fence");
        check(vkQueueSubmit(queue_, 1U, &submit, frame.fence),
              "submit Gallery graphics");
        if (presenting) {
            VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
            present.waitSemaphoreCount = 1U;
            present.pWaitSemaphores = &frame.render_complete;
            present.swapchainCount = 1U;
            present.pSwapchains = &swapchain_;
            present.pImageIndices = &image_index;
            const VkResult result = vkQueuePresentKHR(queue_, &present);
            if (result == VK_ERROR_OUT_OF_DATE_KHR ||
                result == VK_SUBOPTIMAL_KHR) {
                swapchain_stale_ = true;
            } else if (result == VK_ERROR_SURFACE_LOST_KHR) {
                (void)vkDeviceWaitIdle(device_);
                destroy_swapchain();
                surface_ = {};
                error = "Vulkan Gallery presentation surface was lost";
                return false;
            } else if (result != VK_SUCCESS) {
                error = "Vulkan Gallery presentation failed (VkResult " +
                    std::to_string(static_cast<int>(result)) + ")";
                return false;
            }
        }
        if (readback.rgba) {
            check(vkWaitForFences(device_, 1U, &frame.fence, VK_TRUE, UINT64_MAX),
                  "wait for Gallery image readback");
            readback.rgba->resize(static_cast<std::size_t>(width_) * height_);
            const auto *source = static_cast<const std::uint32_t *>(
                frame.readback.mapped);
            for (std::uint32_t y = 0U; y < height_; ++y)
                std::copy_n(source + static_cast<std::size_t>(y) * width_, width_,
                            readback.rgba->data() +
                                static_cast<std::size_t>(height_ - 1U - y) * width_);
        }
        return true;
    }

    [[nodiscard]] bool render_frame(const GpuRenderRequest &request,
                                    std::string &error) {
        if (request.synchronization.mode != GpuSynchronizationMode::queue_idle ||
            pmQueueWaitIdle(request.synchronization.physics_queue) != PM_SUCCESS) {
            error = "Vulkan graphics could not acquire completed physics output";
            return false;
        }
        try {
            ensure_extent(request.target.width, request.target.height);
            ensure_surface(decode_handle<VkSurfaceKHR>(request.target.handle));
            Frame &frame = frames_[frame_index_++ % frame_count];
            check(vkWaitForFences(device_, 1U, &frame.fence, VK_TRUE, UINT64_MAX),
                  "wait for Gallery frame slot");
            check(vkResetCommandBuffer(frame.commands, 0U),
                  "reset Gallery command buffer");

            std::uint32_t swapchain_image{};
            const bool presenting = surface_ != VK_NULL_HANDLE;
            if (presenting && !acquire(frame, swapchain_image, error)) return false;

            VkCommandBufferBeginInfo begin{
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vkBeginCommandBuffer(frame.commands, &begin),
                  "begin Gallery command buffer");
            physics_barrier(frame.commands, true);
            upload_overlay(frame, request.overlay);

            const std::array<VkClearValue, 2U> clear{{
                {.color = {{0.055F, 0.08F, 0.11F, 1.0F}}},
                {.depthStencil = {1.0F, 0U}},
            }};
            VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            pass.renderPass = render_pass_;
            pass.framebuffer = frame.framebuffer;
            pass.renderArea.extent = {width_, height_};
            pass.clearValueCount = static_cast<std::uint32_t>(clear.size());
            pass.pClearValues = clear.data();
            vkCmdBeginRenderPass(frame.commands, &pass, VK_SUBPASS_CONTENTS_INLINE);
            const VkViewport viewport{0.0F, 0.0F, static_cast<float>(width_),
                                      static_cast<float>(height_), 0.0F, 1.0F};
            const VkRect2D scissor{{0, 0}, {width_, height_}};
            vkCmdSetViewport(frame.commands, 0U, 1U, &viewport);
            vkCmdSetScissor(frame.commands, 0U, 1U, &scissor);
            vkCmdBindPipeline(frame.commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              scene_pipeline_);
            draw_scene(frame, request);
            if (!request.overlay.empty()) {
                vkCmdBindPipeline(frame.commands,
                                  VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  overlay_pipeline_);
                const std::uint32_t offset{};
                vkCmdBindDescriptorSets(
                    frame.commands, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    pipeline_layout_, 0U, 1U, &frame.descriptors, 1U, &offset);
                vkCmdDraw(frame.commands, 3U, 1U, 0U, 0U);
            }
            vkCmdEndRenderPass(frame.commands);
            if (request.readback.rgba) copy_readback(frame);
            if (presenting)
                copy_present(frame, swapchain_images_.at(swapchain_image));
            physics_barrier(frame.commands, false);
            return submit(frame, presenting, swapchain_image, request.readback,
                          error);
        } catch (const std::exception &exception) {
            error = exception.what();
            return false;
        }
    }

  public:
    explicit VulkanGalleryRenderer(GpuRendererDevice renderer_device)
        : contract_(renderer_device) {}

    void initialize(const SceneDefinition &scene,
                    const CommandSceneInstance &instance) {
        const GpuRendererDevice &renderer_device = contract_.device();
        const PmNativeDeviceInfo native = renderer_device.native;
        if (native.backend != PM_NATIVE_BACKEND_VULKAN ||
            native.instance == 0U || native.physicalDevice == 0U ||
            native.device == 0U)
            throw std::runtime_error(
                "GPU Gallery rendering requires the native Vulkan physics device");
        instance_ = decode_handle<VkInstance>(native.instance);
        physical_ = decode_handle<VkPhysicalDevice>(native.physicalDevice);
        device_ = decode_handle<VkDevice>(native.device);
        const PmNativeBufferInfo output = [&] {
            PmNativeBufferInfo value{};
            if (pmGetNativeBuffer(renderer_device.api_device,
                                  instance.render_buffers.output,
                                  &value) != PM_SUCCESS || value.buffer == 0U)
                throw std::runtime_error(
                    "Vulkan Gallery could not export the physics output buffer");
            return value;
        }();
        if ((output.nativeUsage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) == 0U)
            throw std::runtime_error(
                "Vulkan physics output lacks storage-buffer usage");
        physics_ = decode_handle<VkBuffer>(output.buffer);
        physics_offset_ = output.offset;
        physics_size_ = output.size;
        vkGetPhysicalDeviceMemoryProperties(physical_, &memory_);
        vkGetPhysicalDeviceProperties(physical_, &properties_);

        const PmNativeQueueInfo native_queue = renderer_device.native_queue;
        queue_ = decode_handle<VkQueue>(native_queue.queue);
        queue_family_ = native_queue.familyIndex;
        std::uint32_t family_count{};
        vkGetPhysicalDeviceQueueFamilyProperties(
            physical_, &family_count, nullptr);
        if (!queue_ || queue_family_ >= family_count)
            throw std::runtime_error(
                "Vulkan Gallery received an invalid native physics queue");
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(
            physical_, &family_count, families.data());
        if (native_queue.queueIndex >= families[queue_family_].queueCount)
            throw std::runtime_error(
                "Vulkan Gallery received an invalid native queue index");
        if ((families[queue_family_].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0U)
            throw std::runtime_error(
                "Selected Vulkan physics queue does not support graphics");

        create_meshes(scene, instance);
        create_frame_objects();
        create_descriptors();
        create_render_pass();
        create_pipelines();
    }

    ~VulkanGalleryRenderer() override {
        if (!device_) return;
        (void)vkDeviceWaitIdle(device_);
        destroy_swapchain();
        for (Frame &frame : frames_) {
            destroy_frame_targets(frame);
            destroy(frame.uniforms);
            if (frame.image_available)
                vkDestroySemaphore(device_, frame.image_available, nullptr);
            if (frame.render_complete)
                vkDestroySemaphore(device_, frame.render_complete, nullptr);
            if (frame.fence) vkDestroyFence(device_, frame.fence, nullptr);
        }
        destroy(dummy_vertices_);
        for (Mesh &mesh : meshes_) {
            destroy(mesh.indices);
            destroy(mesh.vertices);
        }
        if (overlay_sampler_)
            vkDestroySampler(device_, overlay_sampler_, nullptr);
        if (overlay_pipeline_)
            vkDestroyPipeline(device_, overlay_pipeline_, nullptr);
        if (scene_pipeline_)
            vkDestroyPipeline(device_, scene_pipeline_, nullptr);
        if (render_pass_)
            vkDestroyRenderPass(device_, render_pass_, nullptr);
        if (pipeline_layout_)
            vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        if (descriptor_pool_)
            vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        if (descriptor_layout_)
            vkDestroyDescriptorSetLayout(device_, descriptor_layout_, nullptr);
        if (command_pool_)
            vkDestroyCommandPool(device_, command_pool_, nullptr);
    }

    const GpuRendererDevice &renderer_device() const noexcept override {
        return contract_.device();
    }

    bool render(const GpuRenderRequest &request,
                std::string &error) override {
        if (!contract_.begin(request, error)) return false;
        if (!render_frame(request, error)) {
            contract_.abandon(request.synchronization.frame_generation);
            return false;
        }
        return contract_.finish(request.synchronization.frame_generation,
                                error);
    }
};

} // namespace

std::unique_ptr<GpuGalleryRenderer> create_gpu_gallery_renderer(
    GpuRendererDevice device, const SceneDefinition &scene,
    const CommandSceneInstance &instance, std::string &error) {
    error.clear();
    try {
        auto renderer = std::make_unique<VulkanGalleryRenderer>(device);
        renderer->initialize(scene, instance);
        return renderer;
    } catch (const std::exception &exception) {
        error = exception.what();
        return {};
    }
}

} // namespace parallel_mater::gallery

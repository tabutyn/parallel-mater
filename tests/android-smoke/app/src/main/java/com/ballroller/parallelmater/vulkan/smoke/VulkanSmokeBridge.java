package com.ballroller.parallelmater.vulkan.smoke;

public final class VulkanSmokeBridge {
    static {
        System.loadLibrary("parallel_mater_vulkan_smoke");
    }

    private VulkanSmokeBridge() {}

    public static native String run();
}

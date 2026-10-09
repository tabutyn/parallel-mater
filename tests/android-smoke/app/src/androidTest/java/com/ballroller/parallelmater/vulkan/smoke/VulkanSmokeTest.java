package com.ballroller.parallelmater.vulkan.smoke;

import static org.junit.Assert.assertEquals;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(AndroidJUnit4.class)
public final class VulkanSmokeTest {
    @Test
    public void borrowedContextAsyncRigidFixture() {
        assertEquals("", VulkanSmokeBridge.run());
    }
}

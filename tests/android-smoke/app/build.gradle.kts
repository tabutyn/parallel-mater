plugins {
    id("com.android.application")
}

val slangcPath = providers.gradleProperty("parallelMaterSlangc").orNull
    ?: System.getenv("PARALLEL_MATER_SLANGC_EXECUTABLE")
    ?: ""
val spirvValPath = providers.gradleProperty("parallelMaterSpirvVal").orNull
    ?: System.getenv("PARALLEL_MATER_SPIRV_VAL_EXECUTABLE")
    ?: ""
val spirvOptPath = providers.gradleProperty("parallelMaterSpirvOpt").orNull
    ?: System.getenv("PARALLEL_MATER_SPIRV_OPT_EXECUTABLE")
    ?: ""

android {
    namespace = "com.ballroller.parallelmater.vulkan.smoke"
    compileSdk = 36
    ndkVersion = "28.2.13676358"

    defaultConfig {
        applicationId = "com.ballroller.parallelmater.vulkan.smoke"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DPM_ROOT=${file("../../..").canonicalPath}",
                    "-DPARALLEL_MATER_SLANGC_EXECUTABLE=$slangcPath",
                    "-DPARALLEL_MATER_SPIRV_VAL_EXECUTABLE=$spirvValPath",
                    "-DPARALLEL_MATER_SPIRV_OPT_EXECUTABLE=$spirvOptPath"
                )
                cppFlags += listOf(
                    "-std=c++20", "-Wall", "-Wextra",
                    "-Wno-missing-field-initializers"
                )
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "4.1.2"
        }
    }

    testOptions {
        execution = "ANDROIDX_TEST_ORCHESTRATOR"
    }
}

dependencies {
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestUtil("androidx.test:orchestrator:1.5.1")
}

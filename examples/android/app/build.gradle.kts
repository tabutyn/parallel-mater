plugins { id("com.android.application") }

val slangCompiler = providers.gradleProperty("slangc")
    .orElse(providers.environmentVariable("SLANGC"))
val cgltfSource = providers.gradleProperty("cgltfSource")
val sourceRevision = providers.gradleProperty("sourceRevision")
val sourceDirty = providers.gradleProperty("sourceDirty").orElse("false").get().toBoolean()
val sourcePatchSha256 = providers.gradleProperty("sourcePatchSha256").orElse("clean").get()
require(sourcePatchSha256 == "clean" || sourcePatchSha256.matches(Regex("[0-9a-f]{64}"))) {
    "sourcePatchSha256 must be clean or a lowercase SHA-256"
}
val physicsBackend = providers.gradleProperty("physicsBackend").orElse("vulkan").get()
require(physicsBackend in listOf("vulkan", "cpu")) {
    "physicsBackend must be vulkan or cpu"
}
val useVulkan = physicsBackend == "vulkan"
val qualification = providers.gradleProperty("qualification").orElse("false").get().toBoolean()
require(!qualification || useVulkan) { "Android qualification targets currently require Vulkan" }

android {
    namespace = "org.parallelmater.gallery"
    compileSdk = 36
    ndkVersion = "29.0.14206865"
    defaultConfig {
        applicationId = "org.parallelmater.gallery"
        minSdk = if (useVulkan) 33 else 29
        targetSdk = 36
        versionCode = 2
        versionName = "0.1-$physicsBackend-preview"
        buildConfigField("String", "PHYSICS_BACKEND", "\"$physicsBackend\"")
        buildConfigField("String", "EXECUTION_CONTRACT", "\"fixed-tick-60hz-v1\"")
        buildConfigField("String", "SOURCE_REVISION", "\"${sourceRevision.orNull ?: "unknown"}\"")
        buildConfigField("boolean", "SOURCE_DIRTY", sourceDirty.toString())
        buildConfigField("String", "SOURCE_PATCH_SHA256", "\"$sourcePatchSha256\"")
        buildConfigField("String", "ANDROID_GRADLE_PLUGIN_VERSION", "\"9.0.1\"")
        buildConfigField("String", "GRADLE_VERSION", "\"9.1.0\"")
        buildConfigField("String", "NDK_VERSION", "\"29.0.14206865\"")
        buildConfigField("String", "CMAKE_VERSION", "\"4.1.2\"")
        buildConfigField("String", "SLANG_VERSION", "\"2026.18\"")
        manifestPlaceholders["vulkanRequired"] = useVulkan.toString()
        ndk { abiFilters += providers.gradleProperty("abis")
            .orElse("arm64-v8a,x86_64").get().split(",") }
        externalNativeBuild {
            cmake {
                cppFlags += "-O2"
                arguments += listOf(
                    "-DBUILD_TESTING=OFF",
                    "-DPARALLEL_MATER_BUILD_CUDA=OFF",
                    "-DPARALLEL_MATER_BUILD_D3D12=OFF",
                    "-DPARALLEL_MATER_BUILD_VULKAN=${if (useVulkan) "ON" else "OFF"}",
                    "-DPARALLEL_MATER_BUILD_SLANG_CPU_DRIVER=${if (useVulkan) "OFF" else "ON"}",
                    "-DPARALLEL_MATER_SLANG_CPU_LIQUIDS=${if (useVulkan) "OFF" else "ON"}",
                    "-DPARALLEL_MATER_ANDROID_BACKEND=$physicsBackend",
                    "-DPARALLEL_MATER_ANDROID_QUALIFICATION=${if (qualification) "ON" else "OFF"}",
                    "-DPARALLEL_MATER_ENABLE_ANDROID_QUALIFICATION=${if (qualification) "ON" else "OFF"}",
                    "-DPARALLEL_MATER_BUILD_GALLERY=ON",
                    "-DPARALLEL_MATER_BUILD_SLANG_GEOMETRY_CONSTRAINTS=OFF",
                    "-DPARALLEL_MATER_BUILD_SLANG_MIRRORS=OFF",
                    "-DCMAKE_JOB_POOLS=gallery_compile=1",
                    "-DCMAKE_JOB_POOL_COMPILE=gallery_compile",
                    "-DCMAKE_JOB_POOL_LINK=gallery_compile",
                    "-DANDROID_STL=c++_shared")
                if (slangCompiler.isPresent)
                    arguments += "-DPARALLEL_MATER_SLANGC_EXECUTABLE=${slangCompiler.get()}"
                if (cgltfSource.isPresent)
                    arguments += "-DFETCHCONTENT_SOURCE_DIR_PARALLEL_MATER_CGLTF=${cgltfSource.get()}"
                if (sourceRevision.isPresent)
                    arguments += "-DPARALLEL_MATER_SOURCE_REVISION=${sourceRevision.get()}"
                targets += "parallel-mater-gallery-android"
                if (qualification) targets += listOf("pm-android-command-tests",
                    "pm-android-async-tests", "pm-android-liquid-tests",
                    "pm-android-realtime-runtime-tests",
                    "pm-android-conformance", "pm-android-device-contracts")
            }
        }
    }
    externalNativeBuild {
        cmake {
            path = file("../../../CMakeLists.txt")
            version = "4.1.2"
        }
    }
    buildTypes {
        debug { isJniDebuggable = true }
    }
    // The loader resolves the driver via a manifest in app-private storage.
    buildFeatures { buildConfig = true }
    packaging { jniLibs { useLegacyPackaging = true } }
    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("gallery-assets").get().asFile)
}

val galleryAssets by tasks.registering(Sync::class) {
    from("../../../examples/assets") { include("*.glb") }
    into(layout.buildDirectory.dir("gallery-assets/scenes"))
}
val galleryProfiles by tasks.registering(Copy::class) {
    from("../../../config/device-profiles.json")
    into(layout.buildDirectory.dir("gallery-assets/config"))
}
tasks.named("preBuild") { dependsOn(galleryAssets, galleryProfiles) }

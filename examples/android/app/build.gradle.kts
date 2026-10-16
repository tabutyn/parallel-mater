plugins { id("com.android.application") }

val slangCompiler = providers.gradleProperty("slangc")
    .orElse(providers.environmentVariable("SLANGC"))
val cgltfSource = providers.gradleProperty("cgltfSource")

android {
    namespace = "org.parallelmater.gallery"
    compileSdk = 36
    ndkVersion = "29.0.14206865"
    defaultConfig {
        applicationId = "org.parallelmater.gallery"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "0.1-preview"
        ndk { abiFilters += providers.gradleProperty("abis")
            .orElse("arm64-v8a,x86_64").get().split(",") }
        externalNativeBuild {
            cmake {
                cppFlags += "-O2"
                arguments += listOf(
                    "-DBUILD_TESTING=OFF",
                    "-DPARALLEL_MATER_BUILD_CUDA=OFF",
                    "-DPARALLEL_MATER_BUILD_D3D12=OFF",
                    "-DPARALLEL_MATER_BUILD_SLANG_CPU_DRIVER=ON",
                    "-DPARALLEL_MATER_SLANG_CPU_LIQUIDS=ON",
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
                targets += "parallel-mater-gallery-android"
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

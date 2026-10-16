# Android is a gallery client of the same loader, scene builder and renderer.
add_library(parallel-mater-gallery-android SHARED gallery_jni.cpp)
target_compile_features(parallel-mater-gallery-android PRIVATE cxx_std_20)
set(PARALLEL_MATER_ANDROID_BACKEND "vulkan" CACHE STRING "Android physics driver")
if(PARALLEL_MATER_ANDROID_BACKEND STREQUAL "vulkan")
    set(android_driver parallel-mater-driver-slang-vulkan)
    if(NOT PARALLEL_MATER_BUILD_VULKAN)
        message(FATAL_ERROR "Android Vulkan packaging requires PARALLEL_MATER_BUILD_VULKAN")
    endif()
elseif(PARALLEL_MATER_ANDROID_BACKEND STREQUAL "cpu")
    set(android_driver parallel-mater-driver-slang-cpu)
    if(NOT PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER)
        message(FATAL_ERROR "Android CPU packaging requires PARALLEL_MATER_BUILD_SLANG_CPU_DRIVER")
    endif()
else()
    message(FATAL_ERROR "Unknown Android physics backend: ${PARALLEL_MATER_ANDROID_BACKEND}")
endif()
target_compile_definitions(parallel-mater-gallery-android PRIVATE
    PM_ANDROID_VULKAN=$<STREQUAL:${PARALLEL_MATER_ANDROID_BACKEND},vulkan>)
target_link_libraries(parallel-mater-gallery-android PRIVATE
    parallel-mater-gallery-renderer ${android_driver}
    parallel-mater-layer-validation android log)
set_target_properties(parallel-mater-gallery-android PROPERTIES
    OUTPUT_NAME parallel_mater_gallery)
parallel_mater_enable_warnings(parallel-mater-gallery-android)

# These are device executables, never host CTest jobs or APK runtime dependencies.
option(PARALLEL_MATER_ANDROID_QUALIFICATION "Build native Android qualification tools" OFF)
if(PARALLEL_MATER_ANDROID_QUALIFICATION)
    if(NOT PARALLEL_MATER_ANDROID_BACKEND STREQUAL "vulkan")
        message(FATAL_ERROR "Android qualification targets require the Vulkan driver")
    endif()
    add_executable(pm-android-command-tests ${PROJECT_SOURCE_DIR}/tests/api/native_command_api_tests.cpp)
    target_compile_definitions(pm-android-command-tests PRIVATE PM_TEST_NATIVE_BACKEND=PM_NATIVE_BACKEND_VULKAN)
    add_executable(pm-android-async-tests ${PROJECT_SOURCE_DIR}/tests/api/async_submission_tests.cpp)
    foreach(target pm-android-command-tests pm-android-async-tests)
        target_compile_features(${target} PRIVATE cxx_std_20)
        target_link_libraries(${target} PRIVATE ParallelMater::parallel_mater)
    endforeach()
    add_executable(pm-android-liquid-tests ${PROJECT_SOURCE_DIR}/tests/gallery/mobile_liquid_tests.cpp)
    target_link_libraries(pm-android-liquid-tests PRIVATE parallel-mater-gallery-scene)
    target_compile_definitions(pm-android-liquid-tests PRIVATE
        PM_TEST_BACKEND="vulkan"
        PM_PROFILE_CATALOG="/data/local/tmp/parallel-mater-qualification/device-profiles.json"
        PM_SCENE_ROOT="/data/local/tmp/parallel-mater-qualification/scenes")
endif()

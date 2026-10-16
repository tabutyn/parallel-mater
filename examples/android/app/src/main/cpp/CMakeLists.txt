# Android is a gallery client of the same loader, scene builder and renderer.
add_library(parallel-mater-gallery-android SHARED gallery_jni.cpp)
target_compile_features(parallel-mater-gallery-android PRIVATE cxx_std_20)
target_link_libraries(parallel-mater-gallery-android PRIVATE
    parallel-mater-gallery-renderer parallel-mater-driver-slang-cpu
    parallel-mater-layer-validation android log)
set_target_properties(parallel-mater-gallery-android PROPERTIES
    OUTPUT_NAME parallel_mater_gallery)
parallel_mater_enable_warnings(parallel-mater-gallery-android)

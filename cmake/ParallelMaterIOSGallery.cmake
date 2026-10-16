# SPDX-License-Identifier: MIT
# UIKit is presentation only; keep the loader and shared Slang Metal driver.
if(NOT CMAKE_GENERATOR STREQUAL "Xcode")
    message(FATAL_ERROR "The iPad Gallery requires the Xcode generator for framework embedding and signing")
endif()
if(NOT TARGET parallel-mater-driver-slang-metal)
    message(FATAL_ERROR "The iPad Gallery requires PARALLEL_MATER_BUILD_SLANG_METAL_PREVIEW")
endif()
set(PARALLEL_MATER_IOS_BUNDLE_ID "com.ballrollergames.parallelmater.gallery" CACHE STRING "iPad Gallery bundle identifier")
foreach(target parallel-mater parallel-mater-layer-validation parallel-mater-driver-slang-metal)
    if(target STREQUAL "parallel-mater")
        set(framework_name ParallelMater)
    elseif(target STREQUAL "parallel-mater-layer-validation")
        set(framework_name ParallelMaterValidation)
    else()
        set(framework_name ParallelMaterMetal)
    endif()
    set_target_properties(${target} PROPERTIES FRAMEWORK TRUE
        OUTPUT_NAME ${framework_name}
        MACOSX_FRAMEWORK_IDENTIFIER "${PARALLEL_MATER_IOS_BUNDLE_ID}.${framework_name}"
        XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "${PARALLEL_MATER_IOS_BUNDLE_ID}.${framework_name}"
        MACOSX_FRAMEWORK_BUNDLE_VERSION "1"
        MACOSX_FRAMEWORK_SHORT_VERSION_STRING "${PROJECT_VERSION}"
        BUILD_WITH_INSTALL_RPATH YES INSTALL_RPATH "@loader_path/.."
        XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Apple Development")
endforeach()

add_executable(parallel-mater-ipad-gallery MACOSX_BUNDLE
    examples/ios/gallery_main.mm examples/ios/gallery_session.cpp)
target_compile_features(parallel-mater-ipad-gallery PRIVATE cxx_std_20)
target_compile_options(parallel-mater-ipad-gallery PRIVATE
    "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>" -Wall -Wextra -Wpedantic)
target_include_directories(parallel-mater-ipad-gallery PRIVATE examples/gallery examples/support)
target_link_libraries(parallel-mater-ipad-gallery PRIVATE
    parallel-mater-gallery-renderer parallel-mater-gallery-frame-timing
    "-framework UIKit" "-framework QuartzCore" "-framework Foundation" "-framework CoreGraphics")
target_compile_definitions(parallel-mater-ipad-gallery PRIVATE
    PM_IPAD_REVISION="${PARALLEL_MATER_CONFORMANCE_SOURCE_COMMIT}")
set_target_properties(parallel-mater-ipad-gallery PROPERTIES
    OUTPUT_NAME "ParallelMater Gallery"
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/examples/ios/Info.plist.in"
    MACOSX_BUNDLE_GUI_IDENTIFIER "${PARALLEL_MATER_IOS_BUNDLE_ID}"
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "${PARALLEL_MATER_IOS_BUNDLE_ID}"
    MACOSX_BUNDLE_BUNDLE_NAME "ParallelMater Gallery"
    MACOSX_BUNDLE_BUNDLE_VERSION "1"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${PROJECT_VERSION}"
    XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "2"
    XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Automatic"
    XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Apple Development"
    XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
    XCODE_ATTRIBUTE_SUPPORTS_MACCATALYST NO
    XCODE_EMBED_FRAMEWORKS "parallel-mater;parallel-mater-layer-validation;parallel-mater-driver-slang-metal"
    XCODE_EMBED_FRAMEWORKS_CODE_SIGN_ON_COPY YES
    XCODE_EMBED_FRAMEWORKS_REMOVE_HEADERS_ON_COPY YES
    BUILD_WITH_INSTALL_RPATH YES INSTALL_RPATH "@executable_path/Frameworks")
add_dependencies(parallel-mater-ipad-gallery parallel-mater-driver-slang-metal parallel-mater-layer-validation)

file(GLOB assets CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/*.glb")
target_sources(parallel-mater-ipad-gallery PRIVATE ${assets} config/device-profiles.json)
set_source_files_properties(${assets} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources/assets")
set_source_files_properties(config/device-profiles.json PROPERTIES MACOSX_PACKAGE_LOCATION "Resources/config")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/ios/slang-metal.json" CONTENT
"{\"file_format_version\":\"1.0.0\",\"PMD\":{\"library_path\":\"../Frameworks/ParallelMaterMetal.framework/ParallelMaterMetal\",\"api_version\":\"0.2.0\",\"library_arch\":\"64\"}}\n")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/ios/validation.json" CONTENT
"{\"file_format_version\":\"1.0.0\",\"layer\":{\"name\":\"PM_LAYER_KHRONOS_validation\",\"library_path\":\"../Frameworks/ParallelMaterValidation.framework/ParallelMaterValidation\",\"api_version\":\"0.2.0\"}}\n")
foreach(manifest slang-metal validation)
    set(path "${CMAKE_CURRENT_BINARY_DIR}/generated/ios/${manifest}.json")
    target_sources(parallel-mater-ipad-gallery PRIVATE "${path}")
    set_source_files_properties("${path}" PROPERTIES MACOSX_PACKAGE_LOCATION "Resources/manifests")
endforeach()

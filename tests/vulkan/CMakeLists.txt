# SPDX-License-Identifier: MIT
add_executable(parallel-mater-vulkan-api-contract api_contract.cpp)
target_link_libraries(parallel-mater-vulkan-api-contract
                      PRIVATE ParallelMater::vulkan)
target_compile_features(parallel-mater-vulkan-api-contract PRIVATE cxx_std_20)
target_compile_options(parallel-mater-vulkan-api-contract
                       PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME parallel-mater-vulkan-api-contract
         COMMAND parallel-mater-vulkan-api-contract)

add_executable(parallel-mater-vulkan-foundation-tests foundation_tests.cpp)
target_link_libraries(parallel-mater-vulkan-foundation-tests
                      PRIVATE ParallelMater::vulkan)
target_compile_features(parallel-mater-vulkan-foundation-tests PRIVATE cxx_std_20)
target_compile_options(parallel-mater-vulkan-foundation-tests
                       PRIVATE -Wall -Wextra -Wpedantic
                               -Wno-missing-field-initializers)
add_test(NAME parallel-mater-vulkan-foundation-tests
         COMMAND parallel-mater-vulkan-foundation-tests)
set_tests_properties(parallel-mater-vulkan-foundation-tests PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 120
    ENVIRONMENT "VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation"
    LABELS "vulkan;validation")

add_test(NAME parallel-mater-vulkan-installed-consumer
    COMMAND ${CMAKE_COMMAND}
        -DPM_BUILD_DIR=${CMAKE_BINARY_DIR}
        -DPM_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/installed_consumer
        -DPM_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/installed
        -DPM_CONSUMER_BUILD=${CMAKE_CURRENT_BINARY_DIR}/installed-consumer-build
        -P ${CMAKE_CURRENT_SOURCE_DIR}/installed_consumer_test.cmake)
set_tests_properties(parallel-mater-vulkan-installed-consumer PROPERTIES
    TIMEOUT 120 LABELS "vulkan;install")

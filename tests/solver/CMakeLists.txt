cmake_minimum_required(VERSION 3.20)

# Also configure standalone: no CUDA toolkit, GPU or public World header needed.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(ParallelMaterContactTests LANGUAGES CXX)
    include(CTest)
endif()

add_executable(parallel-mater-contact-core-tests contact_tests.cpp)
target_compile_features(parallel-mater-contact-core-tests PRIVATE cxx_std_17)
target_include_directories(parallel-mater-contact-core-tests PRIVATE ../../include)
add_test(NAME parallel-mater-contact-core-tests COMMAND parallel-mater-contact-core-tests)

add_executable(parallel-mater-avbd-core-tests avbd_tests.cpp)
target_compile_features(parallel-mater-avbd-core-tests PRIVATE cxx_std_17)
target_include_directories(parallel-mater-avbd-core-tests PRIVATE ../../include)
add_test(NAME parallel-mater-avbd-core-tests COMMAND parallel-mater-avbd-core-tests)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_EXECUTABLE)
    add_test(NAME parallel-mater-slang-kernel-mirror-coverage
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/check_slang_kernel_mirrors.py)
    set_tests_properties(parallel-mater-slang-kernel-mirror-coverage
        PROPERTIES LABELS "slang;coverage")
endif()

get_property(contact_test_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if("CUDA" IN_LIST contact_test_languages)
    add_executable(parallel-mater-contact-core-cuda-tests contact_cuda_tests.cu)
    target_include_directories(parallel-mater-contact-core-cuda-tests PRIVATE ../../include)
    set_target_properties(parallel-mater-contact-core-cuda-tests PROPERTIES CUDA_STANDARD 17)
    add_test(NAME parallel-mater-contact-core-cuda-tests COMMAND parallel-mater-contact-core-cuda-tests)
    set_tests_properties(parallel-mater-contact-core-cuda-tests PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-avbd-core-cuda-tests avbd_cuda_tests.cu)
    target_include_directories(parallel-mater-avbd-core-cuda-tests PRIVATE ../../include)
    set_target_properties(parallel-mater-avbd-core-cuda-tests PROPERTIES CUDA_STANDARD 17)
    add_test(NAME parallel-mater-avbd-core-cuda-tests COMMAND parallel-mater-avbd-core-cuda-tests)
    set_tests_properties(parallel-mater-avbd-core-cuda-tests PROPERTIES SKIP_RETURN_CODE 77)

    if(TARGET parallel-mater-slang-avbd AND PARALLEL_MATER_SLANG_AVBD_PTX)
        add_executable(parallel-mater-avbd-slang-cuda-tests avbd_slang_cuda_tests.cu)
        add_dependencies(parallel-mater-avbd-slang-cuda-tests parallel-mater-slang-avbd)
        target_include_directories(parallel-mater-avbd-slang-cuda-tests PRIVATE ../../include)
        target_link_libraries(parallel-mater-avbd-slang-cuda-tests
                              PRIVATE CUDA::cuda_driver CUDA::cudart)
        set_target_properties(parallel-mater-avbd-slang-cuda-tests
                              PROPERTIES CUDA_STANDARD 17)
        file(TO_CMAKE_PATH "${PARALLEL_MATER_SLANG_AVBD_PTX}" slang_avbd_ptx_path)
        target_compile_definitions(parallel-mater-avbd-slang-cuda-tests PRIVATE
            PARALLEL_MATER_SLANG_AVBD_PTX="${slang_avbd_ptx_path}")
        add_test(NAME parallel-mater-avbd-slang-cuda-tests
                 COMMAND parallel-mater-avbd-slang-cuda-tests)
        set_tests_properties(parallel-mater-avbd-slang-cuda-tests
                             PROPERTIES SKIP_RETURN_CODE 77 LABELS "cuda;slang;avbd")
    endif()
endif()

if(TARGET parallel-mater-slang-avbd AND Python3_EXECUTABLE)
    add_test(NAME parallel-mater-avbd-slang-reflection-tests
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/check_slang_reflection.py
            ${PARALLEL_MATER_SLANG_AVBD_CUDA_REFLECTION}
            ${PARALLEL_MATER_SLANG_AVBD_METAL_REFLECTION}
            ${PARALLEL_MATER_SLANG_AVBD_HLSL_REFLECTION})
    set_tests_properties(parallel-mater-avbd-slang-reflection-tests
                         PROPERTIES LABELS "slang;avbd")
endif()

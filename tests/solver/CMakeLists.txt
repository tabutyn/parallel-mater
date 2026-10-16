cmake_minimum_required(VERSION 3.20)

# Also configure standalone: no CUDA toolkit, GPU or private runtime needed.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(ParallelMaterSolverTests LANGUAGES CXX)
    include(CTest)
endif()

add_executable(parallel-mater-compute-contract-tests compute_contract_tests.cpp)
target_compile_features(parallel-mater-compute-contract-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-compute-contract-tests PRIVATE ../../src)
add_test(NAME parallel-mater-compute-contract-tests COMMAND parallel-mater-compute-contract-tests)

add_executable(parallel-mater-memory-arena-tests memory_arena_tests.cpp ../../src/simulation/memory_arena.cpp)
target_compile_features(parallel-mater-memory-arena-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-memory-arena-tests PRIVATE ../../src ../../include)
add_test(NAME parallel-mater-memory-arena-tests COMMAND parallel-mater-memory-arena-tests)

if(APPLE AND TARGET parallel-mater-slang-metal-kernels)
    add_executable(parallel-mater-slang-metal-kernel-tests metal_kernel_tests.cpp)
    target_compile_features(parallel-mater-slang-metal-kernel-tests PRIVATE cxx_std_20)
    target_include_directories(parallel-mater-slang-metal-kernel-tests PRIVATE ../../src ../../include)
    target_compile_options(parallel-mater-slang-metal-kernel-tests PRIVATE -Wall -Wextra)
    target_link_libraries(parallel-mater-slang-metal-kernel-tests PRIVATE
        parallel-mater-device-metal)
    add_dependencies(parallel-mater-slang-metal-kernel-tests parallel-mater-slang-metal-kernels)
    add_test(NAME parallel-mater-slang-metal-kernel-tests
        COMMAND parallel-mater-slang-metal-kernel-tests "${PARALLEL_MATER_SLANG_METAL_MANIFEST}")
    set_tests_properties(parallel-mater-slang-metal-kernel-tests PROPERTIES
        SKIP_RETURN_CODE 77 LABELS "slang;metal;gpu" TIMEOUT 300)
endif()

add_executable(parallel-mater-contact-friction-tests contact_friction_tests.cpp)
target_compile_features(parallel-mater-contact-friction-tests PRIVATE cxx_std_17)
target_include_directories(parallel-mater-contact-friction-tests PRIVATE ../../src)
add_test(NAME parallel-mater-contact-friction-tests
         COMMAND parallel-mater-contact-friction-tests)

add_executable(parallel-mater-avbd-core-tests geometry_constraint_tests.cpp)
target_compile_features(parallel-mater-avbd-core-tests PRIVATE cxx_std_17)
target_include_directories(parallel-mater-avbd-core-tests PRIVATE ../../src)
add_test(NAME parallel-mater-avbd-core-tests COMMAND parallel-mater-avbd-core-tests)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_EXECUTABLE)
    add_test(NAME parallel-mater-slang-architecture
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/check_slang_architecture.py)
    set_tests_properties(parallel-mater-slang-architecture
        PROPERTIES LABELS "slang;architecture")
endif()

if(TARGET parallel-mater-slang-metal-kernels AND Python3_EXECUTABLE)
    add_test(NAME parallel-mater-slang-metal-reflection-tests
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_metal_reflection.py
            "${PARALLEL_MATER_SLANG_METAL_MANIFEST}")
    set_tests_properties(parallel-mater-slang-metal-reflection-tests
        PROPERTIES LABELS "slang;metal;reflection")
endif()

get_property(contact_test_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if("CUDA" IN_LIST contact_test_languages)
    add_executable(parallel-mater-contact-friction-cuda-tests
                   contact_friction_cuda_tests.cu)
    target_include_directories(parallel-mater-contact-friction-cuda-tests
                               PRIVATE ../../src)
    set_target_properties(parallel-mater-contact-friction-cuda-tests
                          PROPERTIES CUDA_STANDARD 17)
    add_test(NAME parallel-mater-contact-friction-cuda-tests
             COMMAND parallel-mater-contact-friction-cuda-tests)
    set_tests_properties(parallel-mater-contact-friction-cuda-tests
                         PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(parallel-mater-avbd-core-cuda-tests
                   geometry_constraint_cuda_tests.cu)
    target_include_directories(parallel-mater-avbd-core-cuda-tests PRIVATE ../../src)
    set_target_properties(parallel-mater-avbd-core-cuda-tests PROPERTIES CUDA_STANDARD 17)
    add_test(NAME parallel-mater-avbd-core-cuda-tests COMMAND parallel-mater-avbd-core-cuda-tests)
    set_tests_properties(parallel-mater-avbd-core-cuda-tests PROPERTIES SKIP_RETURN_CODE 77)

    if(TARGET parallel-mater-slang-geometry-constraints AND
       PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_PTX)
        add_executable(parallel-mater-avbd-slang-cuda-tests
                       geometry_constraint_slang_cuda_tests.cu)
        add_dependencies(parallel-mater-avbd-slang-cuda-tests
                         parallel-mater-slang-geometry-constraints)
        target_include_directories(parallel-mater-avbd-slang-cuda-tests PRIVATE ../../src)
        target_link_libraries(parallel-mater-avbd-slang-cuda-tests
                              PRIVATE CUDA::cuda_driver CUDA::cudart)
        set_target_properties(parallel-mater-avbd-slang-cuda-tests
                              PROPERTIES CUDA_STANDARD 17)
        file(TO_CMAKE_PATH
             "${PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_PTX}"
             slang_avbd_ptx_path)
        target_compile_definitions(parallel-mater-avbd-slang-cuda-tests PRIVATE
            PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_PTX="${slang_avbd_ptx_path}")
        add_test(NAME parallel-mater-avbd-slang-cuda-tests
                 COMMAND parallel-mater-avbd-slang-cuda-tests)
        set_tests_properties(parallel-mater-avbd-slang-cuda-tests
                             PROPERTIES SKIP_RETURN_CODE 77 LABELS "cuda;slang;avbd")
    endif()
endif()

if(TARGET parallel-mater-slang-geometry-constraints AND Python3_EXECUTABLE)
    set(parallel_mater_slang_reflections
        ${PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_CUDA_REFLECTION}
        ${PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_METAL_REFLECTION}
        ${PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_HLSL_REFLECTION}
        ${PARALLEL_MATER_SLANG_GEOMETRY_CONSTRAINTS_WGSL_REFLECTION})
    if(WIN32)
        list(APPEND parallel_mater_slang_reflections
            ${PARALLEL_MATER_SLANG_AVBD_PREPARE_DXIL_REFLECTION}
            ${PARALLEL_MATER_SLANG_AVBD_SOLVE_DXIL_REFLECTION})
    endif()
    add_test(NAME parallel-mater-avbd-slang-reflection-tests
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/check_slang_reflection.py
            ${parallel_mater_slang_reflections})
    set_tests_properties(parallel-mater-avbd-slang-reflection-tests
                         PROPERTIES LABELS "slang;avbd")
endif()

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

add_executable(parallel-mater-compute-dependency-tests compute_dependency_tests.cpp)
target_compile_features(parallel-mater-compute-dependency-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-compute-dependency-tests PRIVATE ../../src)
add_test(NAME parallel-mater-compute-dependency-tests COMMAND parallel-mater-compute-dependency-tests)

add_executable(parallel-mater-compute-lifetime-tests compute_lifetime_tests.cpp)
find_package(Threads REQUIRED)
target_compile_features(parallel-mater-compute-lifetime-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-compute-lifetime-tests PRIVATE ../../src)
target_link_libraries(parallel-mater-compute-lifetime-tests PRIVATE Threads::Threads)
add_test(NAME parallel-mater-compute-lifetime-tests COMMAND parallel-mater-compute-lifetime-tests)
set_tests_properties(parallel-mater-compute-lifetime-tests PROPERTIES TIMEOUT 30)

add_executable(parallel-mater-compute-handoff-contract-tests compute_handoff_contract_tests.cpp)
target_compile_features(parallel-mater-compute-handoff-contract-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-compute-handoff-contract-tests PRIVATE ../../src)
add_test(NAME parallel-mater-compute-handoff-contract-tests
    COMMAND parallel-mater-compute-handoff-contract-tests)

add_executable(parallel-mater-engine-handoff-mock-tests engine_handoff_mock_tests.cpp)
target_compile_features(parallel-mater-engine-handoff-mock-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-engine-handoff-mock-tests PRIVATE ../../src)
add_test(NAME parallel-mater-engine-handoff-mock-tests
    COMMAND parallel-mater-engine-handoff-mock-tests)

if(APPLE)
    add_executable(parallel-mater-metal-schedule-recovery-tests
        metal_schedule_recovery_tests.cpp ../../src/driver/metal/schedule_context.cpp)
    target_compile_features(parallel-mater-metal-schedule-recovery-tests PRIVATE cxx_std_20)
    target_include_directories(parallel-mater-metal-schedule-recovery-tests PRIVATE
        ../../src/device/metal_compat ../../src ../../include)
    target_link_libraries(parallel-mater-metal-schedule-recovery-tests PRIVATE Threads::Threads)
    set_target_properties(parallel-mater-metal-schedule-recovery-tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/schedule-recovery/$<CONFIG>")
    # The real scheduler resolves its manifest relative to its binary. No GPU
    # kernels are needed by this non-installed failure-injection test device.
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/schedule-recovery/$<CONFIG>/slang-metal/manifest.txt"
        CONTENT "")
    add_test(NAME parallel-mater-metal-schedule-recovery-tests
        COMMAND parallel-mater-metal-schedule-recovery-tests)
    set_tests_properties(parallel-mater-metal-schedule-recovery-tests PROPERTIES TIMEOUT 30)
endif()

add_executable(parallel-mater-memory-arena-tests memory_arena_tests.cpp ../../src/simulation/memory_arena.cpp)
target_compile_features(parallel-mater-memory-arena-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-memory-arena-tests PRIVATE ../../src ../../include)
add_test(NAME parallel-mater-memory-arena-tests COMMAND parallel-mater-memory-arena-tests)

add_executable(parallel-mater-owner-slot-tests owner_slot_tests.cpp)
target_compile_features(parallel-mater-owner-slot-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-owner-slot-tests PRIVATE ../../src)
add_test(NAME parallel-mater-owner-slot-tests COMMAND parallel-mater-owner-slot-tests)

add_executable(parallel-mater-spatial-build-primitives-tests
    spatial_build_primitives_tests.cpp)
target_compile_features(parallel-mater-spatial-build-primitives-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-spatial-build-primitives-tests
    PRIVATE ../../src ../../include)
add_test(NAME parallel-mater-spatial-build-primitives-tests
    COMMAND parallel-mater-spatial-build-primitives-tests)
set_tests_properties(parallel-mater-spatial-build-primitives-tests PROPERTIES
    LABELS "slang;spatial;f8")

add_executable(parallel-mater-spatial-service-tests
    spatial_service_tests.cpp)
target_compile_features(parallel-mater-spatial-service-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-spatial-service-tests
    PRIVATE ../../src ../../include)
add_test(NAME parallel-mater-spatial-service-tests
    COMMAND parallel-mater-spatial-service-tests)
set_tests_properties(parallel-mater-spatial-service-tests PROPERTIES
    LABELS "spatial;lifecycle;f8")

add_executable(parallel-mater-rigid-spatial-tests
    rigid_spatial_tests.cpp)
target_compile_features(parallel-mater-rigid-spatial-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-rigid-spatial-tests
    PRIVATE ../../src ../../include)
add_test(NAME parallel-mater-rigid-spatial-tests
    COMMAND parallel-mater-rigid-spatial-tests)
set_tests_properties(parallel-mater-rigid-spatial-tests PROPERTIES
    LABELS "rigid;spatial;lifecycle;f8")

add_executable(parallel-mater-rigid-realtime-tests
    rigid_realtime_tests.cpp)
target_compile_features(parallel-mater-rigid-realtime-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-rigid-realtime-tests
    PRIVATE ../../src ../../include)
add_test(NAME parallel-mater-rigid-realtime-tests
    COMMAND parallel-mater-rigid-realtime-tests)
set_tests_properties(parallel-mater-rigid-realtime-tests PROPERTIES
    LABELS "rigid;realtime;f8")

add_executable(parallel-mater-fluid-spatial-tests
    fluid_spatial_tests.cpp)
target_compile_features(parallel-mater-fluid-spatial-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-fluid-spatial-tests
    PRIVATE ../../src ../../include)
add_test(NAME parallel-mater-fluid-spatial-tests
    COMMAND parallel-mater-fluid-spatial-tests)
set_tests_properties(parallel-mater-fluid-spatial-tests PROPERTIES
    LABELS "fluid;spatial;lifecycle;f8")

add_executable(parallel-mater-deformable-spatial-tests
    deformable_spatial_tests.cpp)
target_compile_features(parallel-mater-deformable-spatial-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-deformable-spatial-tests
    PRIVATE ../../src ../../include)
add_test(NAME parallel-mater-deformable-spatial-tests
    COMMAND parallel-mater-deformable-spatial-tests)
set_tests_properties(parallel-mater-deformable-spatial-tests PROPERTIES
    LABELS "deformable;smoke;spatial;lifecycle;f8")

add_executable(parallel-mater-geometry-resource-plan-tests
    geometry_resource_plan_tests.cpp ../../src/simulation/memory_arena.cpp)
target_compile_features(parallel-mater-geometry-resource-plan-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-geometry-resource-plan-tests
    PRIVATE ../../src ../../include)
add_test(NAME parallel-mater-geometry-resource-plan-tests
    COMMAND parallel-mater-geometry-resource-plan-tests)

add_executable(parallel-mater-paint-resource-plan-tests
    paint_resource_plan_tests.cpp ../../src/simulation/memory_arena.cpp)
target_compile_features(parallel-mater-paint-resource-plan-tests PRIVATE cxx_std_20)
target_include_directories(parallel-mater-paint-resource-plan-tests
    PRIVATE ../../src ../../include)
add_test(NAME parallel-mater-paint-resource-plan-tests
    COMMAND parallel-mater-paint-resource-plan-tests)

add_executable(parallel-mater-fluid-rope-resource-plan-tests
    fluid_rope_resource_plan_tests.cpp ../../src/simulation/memory_arena.cpp)
target_compile_features(parallel-mater-fluid-rope-resource-plan-tests
    PRIVATE cxx_std_20)
target_include_directories(parallel-mater-fluid-rope-resource-plan-tests
    PRIVATE ../../src ../../include)
add_test(NAME parallel-mater-fluid-rope-resource-plan-tests
    COMMAND parallel-mater-fluid-rope-resource-plan-tests)

if(PARALLEL_MATER_SLANG_CPU_LIQUIDS)
    add_executable(parallel-mater-fluid-pair-neighbor-index-tests
        fluid_pair_neighbor_index_tests.cpp
        ../../src/simulation/cpu_fluid_dispatch.cpp
        ../../src/simulation/cpu_primitives.cpp)
    add_dependencies(parallel-mater-fluid-pair-neighbor-index-tests
        parallel-mater-driver-slang-cpu)
    target_compile_features(parallel-mater-fluid-pair-neighbor-index-tests
        PRIVATE cxx_std_20)
    target_include_directories(parallel-mater-fluid-pair-neighbor-index-tests PRIVATE
        ../../src/device/cpu_compat ../../src ../../include
        "${PARALLEL_MATER_SLANG_CPU_GENERATED_DIR}"
        "${PARALLEL_MATER_SLANG_CPP_INCLUDE_DIR}")
    add_test(NAME parallel-mater-fluid-pair-neighbor-index-tests
        COMMAND parallel-mater-fluid-pair-neighbor-index-tests)
    set_tests_properties(parallel-mater-fluid-pair-neighbor-index-tests
        PROPERTIES LABELS "slang;fluid;coupling")

    add_executable(parallel-mater-fluid-interface-reconstruction-tests
        fluid_interface_reconstruction_tests.cpp
        ../../src/simulation/cpu_fluid_dispatch.cpp
        ../../src/simulation/cpu_primitives.cpp)
    add_dependencies(parallel-mater-fluid-interface-reconstruction-tests
        parallel-mater-driver-slang-cpu)
    target_compile_features(parallel-mater-fluid-interface-reconstruction-tests
        PRIVATE cxx_std_20)
    target_include_directories(
        parallel-mater-fluid-interface-reconstruction-tests PRIVATE
        ../../src/device/cpu_compat ../../src ../../include
        "${PARALLEL_MATER_SLANG_CPU_GENERATED_DIR}"
        "${PARALLEL_MATER_SLANG_CPP_INCLUDE_DIR}")
    add_test(NAME parallel-mater-fluid-interface-reconstruction-tests
        COMMAND parallel-mater-fluid-interface-reconstruction-tests)
    set_tests_properties(parallel-mater-fluid-interface-reconstruction-tests
        PROPERTIES LABELS "slang;fluid;coupling;immiscible")

    add_executable(parallel-mater-fluid-immiscible-pressure-capillarity-tests
        fluid_immiscible_pressure_capillarity_tests.cpp
        ../../src/simulation/cpu_fluid_dispatch.cpp
        ../../src/simulation/cpu_primitives.cpp)
    add_dependencies(parallel-mater-fluid-immiscible-pressure-capillarity-tests
        parallel-mater-driver-slang-cpu)
    target_compile_features(
        parallel-mater-fluid-immiscible-pressure-capillarity-tests
        PRIVATE cxx_std_20)
    target_include_directories(
        parallel-mater-fluid-immiscible-pressure-capillarity-tests PRIVATE
        ../../src/device/cpu_compat ../../src ../../include
        "${PARALLEL_MATER_SLANG_CPU_GENERATED_DIR}"
        "${PARALLEL_MATER_SLANG_CPP_INCLUDE_DIR}")
    add_test(NAME parallel-mater-fluid-immiscible-pressure-capillarity-tests
        COMMAND parallel-mater-fluid-immiscible-pressure-capillarity-tests)
    set_tests_properties(
        parallel-mater-fluid-immiscible-pressure-capillarity-tests
        PROPERTIES LABELS "slang;fluid;coupling;immiscible")
endif()

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

    if(TARGET parallel-mater-driver-slang-cuda)
        add_executable(parallel-mater-spatial-build-cuda-tests
            spatial_build_cuda_tests.cu)
        add_dependencies(parallel-mater-spatial-build-cuda-tests
            parallel-mater-driver-slang-cuda)
        target_include_directories(parallel-mater-spatial-build-cuda-tests PRIVATE
            ../../src ../../include "${PARALLEL_MATER_SLANG_GENERATED_DIR}")
        target_link_libraries(parallel-mater-spatial-build-cuda-tests PRIVATE
            CUDA::cudart)
        set_target_properties(parallel-mater-spatial-build-cuda-tests PROPERTIES
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
        add_test(NAME parallel-mater-spatial-build-cuda-tests
            COMMAND parallel-mater-spatial-build-cuda-tests)
        set_tests_properties(parallel-mater-spatial-build-cuda-tests PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "cuda;slang;spatial;f8" TIMEOUT 300)

        add_executable(parallel-mater-rigid-spatial-cuda-tests
            rigid_spatial_cuda_tests.cu)
        add_dependencies(parallel-mater-rigid-spatial-cuda-tests
            parallel-mater-driver-slang-cuda)
        target_include_directories(parallel-mater-rigid-spatial-cuda-tests PRIVATE
            ../../src ../../include "${PARALLEL_MATER_SLANG_GENERATED_DIR}")
        target_link_libraries(parallel-mater-rigid-spatial-cuda-tests PRIVATE
            CUDA::cudart)
        set_target_properties(parallel-mater-rigid-spatial-cuda-tests PROPERTIES
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
        add_test(NAME parallel-mater-rigid-spatial-cuda-tests
            COMMAND parallel-mater-rigid-spatial-cuda-tests)
        set_tests_properties(parallel-mater-rigid-spatial-cuda-tests PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "cuda;slang;rigid;spatial;f8"
            TIMEOUT 300)

        add_executable(parallel-mater-rigid-realtime-cuda-tests
            rigid_realtime_cuda_tests.cu)
        add_dependencies(parallel-mater-rigid-realtime-cuda-tests
            parallel-mater-driver-slang-cuda)
        target_include_directories(parallel-mater-rigid-realtime-cuda-tests PRIVATE
            ../../src ../../include "${PARALLEL_MATER_SLANG_GENERATED_DIR}")
        target_link_libraries(parallel-mater-rigid-realtime-cuda-tests PRIVATE
            CUDA::cudart)
        set_target_properties(parallel-mater-rigid-realtime-cuda-tests PROPERTIES
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
        add_test(NAME parallel-mater-rigid-realtime-cuda-tests
            COMMAND parallel-mater-rigid-realtime-cuda-tests)
        set_tests_properties(parallel-mater-rigid-realtime-cuda-tests PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "cuda;slang;rigid;realtime;f8"
            TIMEOUT 300)

        add_executable(parallel-mater-fluid-spatial-cuda-tests
            fluid_spatial_cuda_tests.cu)
        add_dependencies(parallel-mater-fluid-spatial-cuda-tests
            parallel-mater-driver-slang-cuda)
        target_include_directories(parallel-mater-fluid-spatial-cuda-tests PRIVATE
            ../../src ../../include "${PARALLEL_MATER_SLANG_GENERATED_DIR}")
        target_link_libraries(parallel-mater-fluid-spatial-cuda-tests PRIVATE
            CUDA::cudart)
        set_target_properties(parallel-mater-fluid-spatial-cuda-tests PROPERTIES
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
        add_test(NAME parallel-mater-fluid-spatial-cuda-tests
            COMMAND parallel-mater-fluid-spatial-cuda-tests)
        set_tests_properties(parallel-mater-fluid-spatial-cuda-tests PROPERTIES
            SKIP_RETURN_CODE 77 LABELS "cuda;slang;fluid;spatial;f8"
            TIMEOUT 300)

        add_executable(parallel-mater-deformable-spatial-cuda-tests
            deformable_spatial_cuda_tests.cu)
        add_dependencies(parallel-mater-deformable-spatial-cuda-tests
            parallel-mater-driver-slang-cuda)
        target_include_directories(
            parallel-mater-deformable-spatial-cuda-tests PRIVATE
            ../../src ../../include "${PARALLEL_MATER_SLANG_GENERATED_DIR}")
        target_link_libraries(parallel-mater-deformable-spatial-cuda-tests
            PRIVATE CUDA::cudart)
        set_target_properties(
            parallel-mater-deformable-spatial-cuda-tests PROPERTIES
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
        add_test(NAME parallel-mater-deformable-spatial-cuda-tests
            COMMAND parallel-mater-deformable-spatial-cuda-tests)
        set_tests_properties(
            parallel-mater-deformable-spatial-cuda-tests PROPERTIES
            SKIP_RETURN_CODE 77
            LABELS "cuda;slang;deformable;smoke;spatial;f8" TIMEOUT 300)
    endif()

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

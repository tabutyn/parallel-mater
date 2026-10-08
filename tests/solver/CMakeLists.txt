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

if(CMAKE_CUDA_COMPILER)
    add_executable(parallel-mater-contact-core-cuda-tests contact_cuda_tests.cu)
    target_include_directories(parallel-mater-contact-core-cuda-tests PRIVATE ../../include)
    set_target_properties(parallel-mater-contact-core-cuda-tests PROPERTIES CUDA_STANDARD 17)
    add_test(NAME parallel-mater-contact-core-cuda-tests COMMAND parallel-mater-contact-core-cuda-tests)
    set_tests_properties(parallel-mater-contact-core-cuda-tests PROPERTIES SKIP_RETURN_CODE 77)
endif()

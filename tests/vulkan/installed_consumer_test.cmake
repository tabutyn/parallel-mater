file(REMOVE_RECURSE "${PM_PREFIX}" "${PM_CONSUMER_BUILD}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${PM_BUILD_DIR}" --prefix "${PM_PREFIX}"
    RESULT_VARIABLE install_result)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "ParallelMater install failed: ${install_result}")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${PM_SOURCE_DIR}" -B "${PM_CONSUMER_BUILD}"
            "-DCMAKE_PREFIX_PATH=${PM_PREFIX}"
    RESULT_VARIABLE configure_result)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Installed consumer configure failed: ${configure_result}")
endif()
execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${PM_CONSUMER_BUILD}"
    RESULT_VARIABLE build_result)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Installed consumer build failed: ${build_result}")
endif()

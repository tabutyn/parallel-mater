# SPDX-License-Identifier: MIT

if(NOT DEFINED SOURCE_APP OR NOT DEFINED DESTINATION_APP)
    message(FATAL_ERROR "SOURCE_APP and DESTINATION_APP are required")
endif()
if(NOT IS_DIRECTORY "${SOURCE_APP}")
    message(FATAL_ERROR "Built application bundle not found: ${SOURCE_APP}")
endif()

set(SOURCE_BUILD_INFO "${SOURCE_APP}/Contents/Resources/build-info.json")
if(NOT EXISTS "${SOURCE_BUILD_INFO}")
    message(FATAL_ERROR "Built application has no build identity: ${SOURCE_BUILD_INFO}")
endif()
file(READ "${SOURCE_BUILD_INFO}" SOURCE_BUILD_INFO_JSON)
string(JSON BUILD_SCHEMA ERROR_VARIABLE BUILD_SCHEMA_ERROR
       GET "${SOURCE_BUILD_INFO_JSON}" schema_version)
if(BUILD_SCHEMA_ERROR OR NOT BUILD_SCHEMA EQUAL 2)
    message(FATAL_ERROR "Built application has an invalid build identity schema")
endif()
string(JSON BUILD_REVISION ERROR_VARIABLE BUILD_REVISION_ERROR
       GET "${SOURCE_BUILD_INFO_JSON}" revision)
string(JSON BUILD_BACKEND ERROR_VARIABLE BUILD_BACKEND_ERROR
       GET "${SOURCE_BUILD_INFO_JSON}" backend)
string(JSON BUILD_DIRTY ERROR_VARIABLE BUILD_DIRTY_ERROR
       GET "${SOURCE_BUILD_INFO_JSON}" dirty)
string(JSON BUILD_CONTRACT ERROR_VARIABLE BUILD_CONTRACT_ERROR
       GET "${SOURCE_BUILD_INFO_JSON}" execution_contract)
if(BUILD_REVISION_ERROR OR BUILD_BACKEND_ERROR OR BUILD_DIRTY_ERROR OR
   BUILD_CONTRACT_ERROR)
    message(FATAL_ERROR "Built application has an incomplete build identity")
endif()
if(NOT BUILD_CONTRACT STREQUAL "fixed-tick-60hz-v1")
    message(FATAL_ERROR
        "Built application uses unexpected execution contract: ${BUILD_CONTRACT}")
endif()
if(DEFINED EXPECTED_BACKEND AND NOT EXPECTED_BACKEND STREQUAL "" AND
   NOT BUILD_BACKEND STREQUAL EXPECTED_BACKEND)
    message(FATAL_ERROR
        "Built application backend is ${BUILD_BACKEND}, expected ${EXPECTED_BACKEND}")
endif()

set(STAGING_APP "${DESTINATION_APP}.staging")
set(PREVIOUS_APP "${DESTINATION_APP}.previous")
file(REMOVE_RECURSE "${STAGING_APP}" "${PREVIOUS_APP}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${SOURCE_APP}" "${STAGING_APP}"
    RESULT_VARIABLE COPY_RESULT)
if(NOT COPY_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to stage ${SOURCE_APP}")
endif()

execute_process(
    COMMAND /usr/bin/codesign --force --deep --sign - "${STAGING_APP}"
    RESULT_VARIABLE SIGN_RESULT)
if(NOT SIGN_RESULT EQUAL 0)
    file(REMOVE_RECURSE "${STAGING_APP}")
    message(FATAL_ERROR "Failed to sign staged application")
endif()

if(EXISTS "${DESTINATION_APP}")
    # Replacing files cannot update a running process. Quit only this bundle,
    # then let the verified deployment wrapper relaunch the installed revision.
    execute_process(COMMAND /usr/bin/osascript -e
        "if application id \"com.tabutyn.parallel-mater.gallery\" is running then tell application id \"com.tabutyn.parallel-mater.gallery\" to quit"
        RESULT_VARIABLE QUIT_RESULT ERROR_VARIABLE QUIT_ERROR)
    # GLFW returns NSTerminateCancel while asking its windows to close. The
    # Apple event can report -128 even though the main loop exits normally.
    # Verify actual exit; never force-terminate a still-running application.
    set(GALLERY_RUNNING true)
    foreach(attempt RANGE 1 40)
        execute_process(COMMAND /usr/bin/osascript -e
            "application id \"com.tabutyn.parallel-mater.gallery\" is running"
            RESULT_VARIABLE CHECK_RESULT OUTPUT_VARIABLE GALLERY_RUNNING
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_VARIABLE CHECK_ERROR)
        if(NOT CHECK_RESULT EQUAL 0)
            message(FATAL_ERROR "Cannot verify Gallery shutdown: ${CHECK_ERROR}")
        endif()
        if(GALLERY_RUNNING STREQUAL "false")
            break()
        endif()
        execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 0.25)
    endforeach()
    if(NOT GALLERY_RUNNING STREQUAL "false")
        message(FATAL_ERROR "Quit the installed Gallery before deployment (still running): ${QUIT_ERROR}")
    endif()
    file(RENAME "${DESTINATION_APP}" "${PREVIOUS_APP}"
         RESULT MOVE_PREVIOUS_RESULT)
    if(MOVE_PREVIOUS_RESULT)
        file(REMOVE_RECURSE "${STAGING_APP}")
        message(FATAL_ERROR
            "Failed to preserve installed application: ${MOVE_PREVIOUS_RESULT}")
    endif()
endif()
file(RENAME "${STAGING_APP}" "${DESTINATION_APP}" RESULT INSTALL_RESULT)
if(INSTALL_RESULT)
    if(EXISTS "${PREVIOUS_APP}")
        file(RENAME "${PREVIOUS_APP}" "${DESTINATION_APP}")
    endif()
    file(REMOVE_RECURSE "${STAGING_APP}")
    message(FATAL_ERROR "Failed to install application: ${INSTALL_RESULT}")
endif()
file(REMOVE_RECURSE "${PREVIOUS_APP}")

set(INSTALLED_BUILD_INFO "${DESTINATION_APP}/Contents/Resources/build-info.json")
if(NOT EXISTS "${INSTALLED_BUILD_INFO}")
    message(FATAL_ERROR "Installed application has no build identity")
endif()
file(READ "${INSTALLED_BUILD_INFO}" INSTALLED_BUILD_INFO_JSON)
if(NOT INSTALLED_BUILD_INFO_JSON STREQUAL SOURCE_BUILD_INFO_JSON)
    message(FATAL_ERROR "Installed application identity differs from the built application")
endif()
execute_process(
    COMMAND /usr/bin/codesign --verify --deep --strict "${DESTINATION_APP}"
    RESULT_VARIABLE VERIFY_SIGNATURE_RESULT
    ERROR_VARIABLE VERIFY_SIGNATURE_ERROR)
if(NOT VERIFY_SIGNATURE_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Installed application signature verification failed: ${VERIFY_SIGNATURE_ERROR}")
endif()

set(LAUNCH_SERVICES_REGISTER
    "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister")
if(EXISTS "${LAUNCH_SERVICES_REGISTER}")
    execute_process(COMMAND "${LAUNCH_SERVICES_REGISTER}" -f "${DESTINATION_APP}"
                    RESULT_VARIABLE REGISTER_RESULT)
    if(NOT REGISTER_RESULT EQUAL 0)
        message(FATAL_ERROR "Launch Services refresh failed: ${REGISTER_RESULT}")
    endif()
endif()

if(UPDATE_DOCK)
    find_program(DOCKUTIL_EXECUTABLE dockutil
        PATHS /opt/homebrew/bin /usr/local/bin)
    if(NOT DOCKUTIL_EXECUTABLE)
        message(FATAL_ERROR
            "dockutil is required to verify the canonical Gallery Dock item")
    endif()
    execute_process(
        COMMAND "${DOCKUTIL_EXECUTABLE}" --find "ParallelMater Metal Gallery"
        RESULT_VARIABLE LEGACY_DOCK_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(LEGACY_DOCK_RESULT EQUAL 0)
        execute_process(
            COMMAND "${DOCKUTIL_EXECUTABLE}" --remove
                    "ParallelMater Metal Gallery" --no-restart
            RESULT_VARIABLE REMOVE_LEGACY_RESULT)
        if(NOT REMOVE_LEGACY_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to remove the legacy Gallery Dock item")
        endif()
    endif()
    execute_process(
        COMMAND "${DOCKUTIL_EXECUTABLE}" --add "${DESTINATION_APP}"
                --replacing "ParallelMater Gallery"
        RESULT_VARIABLE ADD_DOCK_RESULT)
    if(NOT ADD_DOCK_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to replace the canonical Gallery Dock item")
    endif()
    execute_process(
        COMMAND "${DOCKUTIL_EXECUTABLE}" --find "${DESTINATION_APP}"
        RESULT_VARIABLE VERIFY_DOCK_RESULT OUTPUT_QUIET ERROR_QUIET)
    if(NOT VERIFY_DOCK_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Dock does not point to the installed Gallery: ${DESTINATION_APP}")
    endif()
endif()

message(STATUS
    "Installed ${DESTINATION_APP} (revision=${BUILD_REVISION}, backend=${BUILD_BACKEND}, dirty=${BUILD_DIRTY})")

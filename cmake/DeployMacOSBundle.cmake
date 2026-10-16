# SPDX-License-Identifier: MIT

if(NOT DEFINED SOURCE_APP OR NOT DEFINED DESTINATION_APP)
    message(FATAL_ERROR "SOURCE_APP and DESTINATION_APP are required")
endif()
if(NOT IS_DIRECTORY "${SOURCE_APP}")
    message(FATAL_ERROR "Built application bundle not found: ${SOURCE_APP}")
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

set(LAUNCH_SERVICES_REGISTER
    "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister")
if(EXISTS "${LAUNCH_SERVICES_REGISTER}")
    execute_process(COMMAND "${LAUNCH_SERVICES_REGISTER}" -f "${DESTINATION_APP}"
                    RESULT_VARIABLE REGISTER_RESULT)
    if(NOT REGISTER_RESULT EQUAL 0)
        message(WARNING "Launch Services refresh failed: ${REGISTER_RESULT}")
    endif()
endif()

if(UPDATE_DOCK)
    find_program(DOCKUTIL_EXECUTABLE dockutil)
    if(DOCKUTIL_EXECUTABLE)
        execute_process(
            COMMAND "${DOCKUTIL_EXECUTABLE}" --find "ParallelMater Metal Gallery"
            RESULT_VARIABLE LEGACY_DOCK_RESULT OUTPUT_QUIET ERROR_QUIET)
        if(LEGACY_DOCK_RESULT EQUAL 0)
            execute_process(
                COMMAND "${DOCKUTIL_EXECUTABLE}" --remove
                        "ParallelMater Metal Gallery" --no-restart)
        endif()
        execute_process(
            COMMAND "${DOCKUTIL_EXECUTABLE}" --find "ParallelMater Gallery"
            RESULT_VARIABLE CURRENT_DOCK_RESULT OUTPUT_QUIET ERROR_QUIET)
        if(NOT CURRENT_DOCK_RESULT EQUAL 0)
            execute_process(
                COMMAND "${DOCKUTIL_EXECUTABLE}" --add "${DESTINATION_APP}"
                RESULT_VARIABLE ADD_DOCK_RESULT)
            if(NOT ADD_DOCK_RESULT EQUAL 0)
                message(WARNING "Gallery installed, but Dock update failed")
            endif()
        endif()
    else()
        message(WARNING "dockutil is unavailable; the Gallery was not Dock-pinned")
    endif()
endif()

message(STATUS "Installed ${DESTINATION_APP}")

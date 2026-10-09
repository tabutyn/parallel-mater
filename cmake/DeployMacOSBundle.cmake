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
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
            "${SOURCE_APP}" "${STAGING_APP}"
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
# Notify Launch Services about changed bundle resources, including the Dock icon.
set(LAUNCH_SERVICES_REGISTER
    "/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister")
if(EXISTS "${LAUNCH_SERVICES_REGISTER}")
    execute_process(
        COMMAND "${LAUNCH_SERVICES_REGISTER}" -f "${DESTINATION_APP}"
        RESULT_VARIABLE REGISTER_RESULT)
    if(NOT REGISTER_RESULT EQUAL 0)
        message(WARNING "Installed app, but Launch Services refresh failed: ${REGISTER_RESULT}")
    endif()
endif()
message(STATUS "Installed ${DESTINATION_APP}")

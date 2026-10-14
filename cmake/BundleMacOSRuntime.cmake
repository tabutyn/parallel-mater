# SPDX-License-Identifier: MIT

if(NOT DEFINED APPLICATION_EXECUTABLE OR NOT DEFINED GLFW_LIBRARY)
    message(FATAL_ERROR "APPLICATION_EXECUTABLE and GLFW_LIBRARY are required")
endif()
execute_process(COMMAND /usr/bin/otool -L "${APPLICATION_EXECUTABLE}"
                OUTPUT_VARIABLE LINKED_LIBRARIES
                RESULT_VARIABLE OTOOL_RESULT)
if(NOT OTOOL_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not inspect ${APPLICATION_EXECUTABLE}")
endif()
string(REGEX MATCH "[^ \t\n]*libglfw[^ \t\n]*\\.dylib"
       LINKED_GLFW "${LINKED_LIBRARIES}")
if(NOT LINKED_GLFW)
    message(FATAL_ERROR "The Gallery executable has no linked GLFW runtime")
endif()
get_filename_component(GLFW_NAME "${GLFW_LIBRARY}" NAME)
execute_process(
    COMMAND /usr/bin/install_name_tool -change "${LINKED_GLFW}"
            "@rpath/${GLFW_NAME}" "${APPLICATION_EXECUTABLE}"
    RESULT_VARIABLE CHANGE_RESULT)
if(NOT CHANGE_RESULT EQUAL 0)
    message(FATAL_ERROR "Could not make the Gallery GLFW dependency portable")
endif()

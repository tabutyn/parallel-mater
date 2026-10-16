# SPDX-License-Identifier: MIT
# Package the approved artwork; no replacement logo or checked-in generated PNGs.
file(MAKE_DIRECTORY "${OUTPUT}/AppIcon.appiconset")
set(icon "${OUTPUT}/AppIcon.appiconset/AppIcon.png")
if(NOT EXISTS "${icon}" OR "${SOURCE}" IS_NEWER_THAN "${icon}")
    execute_process(COMMAND /usr/bin/sips -z 1024 1024 "${SOURCE}" --out "${icon}"
        RESULT_VARIABLE result OUTPUT_QUIET ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Could not package iPad logo: ${error}")
    endif()
endif()
file(WRITE "${OUTPUT}/Contents.json" "{\"info\":{\"author\":\"xcode\",\"version\":1}}\n")
file(WRITE "${OUTPUT}/AppIcon.appiconset/Contents.json"
    "{\"images\":[{\"filename\":\"AppIcon.png\",\"idiom\":\"universal\",\"platform\":\"ios\",\"size\":\"1024x1024\"}],\"info\":{\"author\":\"xcode\",\"version\":1}}\n")

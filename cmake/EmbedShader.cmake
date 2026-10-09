# SPDX-License-Identifier: MIT
# Embed FXC binary output in one buffered write. FXC /Fh emits tiny writes and
# is exceptionally slow on some monitored Windows workstations.
if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT OR NOT SYMBOL MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR "INPUT, OUTPUT and a valid SYMBOL are required")
endif()
file(READ "${INPUT}" shader_hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," shader_bytes "${shader_hex}")
string(REGEX REPLACE "(0x..,0x..,0x..,0x..,0x..,0x..,0x..,0x..,)" "\\1\n" shader_lines "${shader_bytes}")
file(WRITE "${OUTPUT}" "// Generated from FXC DXBC. Do not edit.\nalignas(4) const unsigned char ${SYMBOL}[] = {\n${shader_lines}\n};\n")

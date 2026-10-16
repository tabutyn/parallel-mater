# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.25)
project(ParallelMaterHeadlessExample LANGUAGES C)
find_package(ParallelMater 0.2 CONFIG REQUIRED COMPONENTS loader)
add_executable(pm-devices devices.c)
target_compile_features(pm-devices PRIVATE c_std_11)
target_link_libraries(pm-devices PRIVATE ParallelMater::api)

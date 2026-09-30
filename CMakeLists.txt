cmake_minimum_required(VERSION 3.20)
project(cpp_nn_from_scratch LANGUAGES CXX)

# C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Recursively find all implementation files under src/ (excluding main files)
file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

list(FILTER CORE_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")

add_library(nn_core STATIC ${CORE_SOURCES})

# Public include directory: any target linking against nn_core inherits this
target_include_directories(nn_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Compiler warnings and optimizations
target_compile_options(nn_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        $<$<CONFIG:Release>:-O3 -march=native>
        $<$<CONFIG:Debug>:-g -O0>
    >
)

# Uses src/math/main.cpp
add_executable(main.out "${CMAKE_CURRENT_SOURCE_DIR}/src/math/main.cpp")

target_link_libraries(main.out PRIVATE nn_core)

target_compile_options(main.out PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:
        -Wall
        -Wextra
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-g>
    >
)

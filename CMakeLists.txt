cmake_minimum_required(VERSION 3.20)
project(cpp_nn_from_scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add executable from source files
add_executable(main.out
    source/main.cpp
    source/Matrix.cpp
)

# Tell the compiler where to find headers (#include "Matrix.hpp")
target_include_directories(main.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Standard warnings
target_compile_options(main.out PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

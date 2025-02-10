cmake_minimum_required(VERSION 3.15)
project(spherical_harmonics)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Add pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Add source files
set(SOURCES
    src/cpp/binding.cpp
)

# Create the Python module
pybind11_add_module(spherical_harmonics ${SOURCES})

# Include directories
target_include_directories(spherical_harmonics PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
)

# Enable SSE2 instructions
if(MSVC)
    target_compile_options(spherical_harmonics PRIVATE /arch:SSE2)
else()
    target_compile_options(spherical_harmonics PRIVATE -msse2)
endif()
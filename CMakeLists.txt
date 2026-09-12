cmake_minimum_required(VERSION 3.15)
project(fruitcraft CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # mini-openbwapi links into a Python module

find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# headless engine library from the pinned OpenSnowstorm submodule (no SDL/UI)
add_subdirectory(third_party/opensnowstorm/mini-openbwapi mini-openbwapi)

pybind11_add_module(_engine bridge/engine_bindings.cpp)
target_include_directories(_engine PRIVATE
    third_party/opensnowstorm
    third_party/opensnowstorm/mini-openbwapi)
target_link_libraries(_engine PRIVATE mini-openbwapi Threads::Threads)

# place the module where the fruitcraft package can import it
set_target_properties(_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/fruitcraft)

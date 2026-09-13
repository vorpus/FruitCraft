cmake_minimum_required(VERSION 3.15)
project(fruitcraft CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # mini-openbwapi links into a Python module

find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# Optional SDL renderer (the engine's real game graphics). Headless remains the
# runtime default; Game::setGUI(true) opens the window. Point SDL2_INCLUDE_DIR
# at the vendored headers and SDL2_LIBRARY at any libSDL2 (e.g. the pysdl2-dll
# wheel's), plus FRUITCRAFT_SDL_RPATH for its directory.
option(FRUITCRAFT_UI "Compile in the OpenSnowstorm SDL renderer" OFF)
if(FRUITCRAFT_UI)
    set(OPENBW_ENABLE_UI ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(third_party/opensnowstorm/mini-openbwapi mini-openbwapi)

pybind11_add_module(_engine bridge/engine_bindings.cpp)
target_include_directories(_engine PRIVATE
    third_party/opensnowstorm
    third_party/opensnowstorm/mini-openbwapi)
target_link_libraries(_engine PRIVATE mini-openbwapi Threads::Threads)
if(FRUITCRAFT_UI AND FRUITCRAFT_SDL_RPATH)
    set_target_properties(_engine PROPERTIES
        BUILD_RPATH "${FRUITCRAFT_SDL_RPATH}"
        INSTALL_RPATH "${FRUITCRAFT_SDL_RPATH}")
endif()

# place the module where the fruitcraft package can import it
set_target_properties(_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/fruitcraft)

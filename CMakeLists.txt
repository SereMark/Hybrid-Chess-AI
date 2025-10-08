cmake_minimum_required(VERSION 3.21)

project(chesscore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

set(CHESSCORE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/core")
set(CHESSCORE_SOURCES
    "${CHESSCORE_SRC_DIR}/bindings.cpp"
    "${CHESSCORE_SRC_DIR}/chess.cpp"
    "${CHESSCORE_SRC_DIR}/mcts.cpp"
)

set(pybind11_DIR "$ENV{VIRTUAL_ENV}/Lib/site-packages/pybind11/share/cmake/pybind11" CACHE PATH "" FORCE)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(chesscore ${CHESSCORE_SOURCES})
target_include_directories(chesscore PRIVATE "${CHESSCORE_SRC_DIR}")
target_compile_features(chesscore PRIVATE cxx_std_23)

set_target_properties(chesscore PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/python"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/python"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/src/python"
    INTERPROCEDURAL_OPTIMIZATION TRUE
)

target_compile_definitions(chesscore PRIVATE
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
    WIN32_LEAN_AND_MEAN
)

target_compile_options(chesscore PRIVATE
    /MP16
    /arch:AVX2
    /favor:AMD64
    /O2
    /Oi
    /Gy
    /GF
    /guard:cf
    /GL
    /Zc:inline
    /EHsc
    /fp:fast
)

target_link_options(chesscore PRIVATE
    /LTCG
    /OPT:REF
    /OPT:ICF
    /guard:cf
)

set_property(TARGET chesscore PROPERTY POSITION_INDEPENDENT_CODE ON)

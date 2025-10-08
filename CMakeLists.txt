cmake_minimum_required(VERSION 3.21)

project(chesscore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

set(CHESSCORE_SRC_DIR "${PROJECT_SOURCE_DIR}/src/core")
set(CHESSCORE_PYTHON_DIR "${PROJECT_SOURCE_DIR}/src/python")
set(CHESSCORE_BUILD_RUNTIME_DIR "${PROJECT_BINARY_DIR}/python")
set(CHESSCORE_BUILD_ARCHIVE_DIR "${PROJECT_BINARY_DIR}/lib")

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
    RUNTIME_OUTPUT_DIRECTORY "${CHESSCORE_BUILD_RUNTIME_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CHESSCORE_BUILD_RUNTIME_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CHESSCORE_BUILD_ARCHIVE_DIR}"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    POSITION_INDEPENDENT_CODE ON
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

add_custom_command(TARGET chesscore POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CHESSCORE_PYTHON_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:chesscore>" "${CHESSCORE_PYTHON_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E rm -f
        "${CHESSCORE_PYTHON_DIR}/chesscore.lib"
        "${CHESSCORE_PYTHON_DIR}/chesscore.exp"
)

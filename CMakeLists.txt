cmake_minimum_required(VERSION 3.18)
project(chesscore LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(chesscore core/chess.cpp core/mcts.cpp core/encoder.cpp core/replay_buffer.cpp core/bindings.cpp)
set_target_properties(
    chesscore
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
)

target_compile_options(
    chesscore
    PRIVATE
        /O2
        /W4
        /DNOMINMAX
        /D_CRT_SECURE_NO_WARNINGS
        /EHsc
)

target_link_options(
    chesscore
    PRIVATE
        /OPT:REF
        /OPT:ICF
)

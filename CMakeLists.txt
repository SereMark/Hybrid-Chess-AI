cmake_minimum_required(VERSION 3.18)
project(ChessAI LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(pybind11 REQUIRED)
pybind11_add_module(chessai
    chess.cpp
    mcts.cpp
    bindings.cpp
)
target_compile_options(chessai PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -funroll-loops -fomit-frame-pointer>
    $<$<CONFIG:Debug>:-O0 -g>
)
set_target_properties(chessai PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    CXX_VISIBILITY_PRESET hidden
)

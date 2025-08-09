cmake_minimum_required(VERSION 3.25)
project(ChessCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(chesscore chess.cpp mcts.cpp encoder.cpp bindings.cpp)

set_target_properties(chesscore PROPERTIES CXX_VISIBILITY_PRESET hidden)

target_compile_options(chesscore PRIVATE
  $<$<CONFIG:Release>:-O3 -DNDEBUG -march=native -mtune=native -ffast-math -fomit-frame-pointer -fno-plt -flto>
  $<$<CONFIG:Debug>:-O0 -g -Wall -Wextra -Wpedantic>
)

target_link_options(chesscore PRIVATE $<$<CONFIG:Release>:-flto>)
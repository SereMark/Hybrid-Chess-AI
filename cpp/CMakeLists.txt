cmake_minimum_required(VERSION 3.18)
project(chesscore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(chesscore
  chess.cpp
  mcts.cpp
  encoder.cpp
  bindings.cpp
)

set_target_properties(chesscore PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
)

target_compile_options(chesscore PRIVATE -Wall -Wextra -Wpedantic)

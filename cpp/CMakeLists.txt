cmake_minimum_required(VERSION 3.25)
project(ChessCore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

pybind11_add_module(chesscore
  chess.cpp
  mcts.cpp
  encoder.cpp
  bindings.cpp
)

set_target_properties(chesscore PROPERTIES
  CXX_VISIBILITY_PRESET hidden
)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
  set_property(TARGET chesscore PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

target_compile_options(chesscore PRIVATE
  $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math -fomit-frame-pointer -fno-plt -flto>
  $<$<CONFIG:Debug>:-O0 -g -Wall -Wextra -Wpedantic>
)

target_link_options(chesscore PRIVATE
  $<$<CONFIG:Release>:-flto>
)
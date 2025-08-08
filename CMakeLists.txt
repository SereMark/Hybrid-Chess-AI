cmake_minimum_required(VERSION 3.18)
project(ChessAI LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

include(CheckCXXCompilerFlag)
option(CHESSAI_NATIVE_OPT "Enable native CPU optimizations (-march=native)" ON)

pybind11_add_module(chessai
    chess.cpp
    mcts.cpp
    bindings.cpp
)

set_target_properties(chessai PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    CXX_VISIBILITY_PRESET hidden
)

if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(chessai PRIVATE -Wall -Wextra -Wpedantic)
endif()
if (CHESSAI_NATIVE_OPT)
  check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  if (CHESSAI_NATIVE_OPT AND COMPILER_SUPPORTS_MARCH_NATIVE)
    target_compile_options(chessai PRIVATE -O3 -march=native -funroll-loops -fomit-frame-pointer)
  else()
    target_compile_options(chessai PRIVATE -O3 -fomit-frame-pointer)
  endif()
else()
  target_compile_options(chessai PRIVATE -O0 -g)
endif()

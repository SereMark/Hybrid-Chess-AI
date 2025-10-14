cmake_minimum_required(VERSION 3.21)

project(chesscore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pybind11,sys; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(PREPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
find_package(pybind11 CONFIG REQUIRED)

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/core")
set(PY_DST  "${CMAKE_CURRENT_SOURCE_DIR}/src/python")

pybind11_add_module(chesscore
  "${SRC_DIR}/bindings.cpp"
  "${SRC_DIR}/chess.cpp"
  "${SRC_DIR}/mcts.cpp"
)
target_include_directories(chesscore PRIVATE "${SRC_DIR}")

add_custom_command(TARGET chesscore POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${PY_DST}"
  COMMAND "${CMAKE_COMMAND}" -E copy $<TARGET_FILE:chesscore> "${PY_DST}"
)
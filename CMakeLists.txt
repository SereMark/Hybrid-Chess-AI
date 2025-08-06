cmake_minimum_required(VERSION 3.20)
project(ChessAI)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG -flto=auto")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=native -mtune=native -g -DNDEBUG")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -funroll-loops")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fomit-frame-pointer")
endif()


set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)


set(CHESS_SOURCES
    chess.cpp
    mcts.cpp
    bindings.cpp
)

pybind11_add_module(chessai ${CHESS_SOURCES})


set_property(TARGET chessai PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

target_compile_features(chessai PRIVATE cxx_std_23)
target_compile_options(chessai PRIVATE
    $<$<CONFIG:Release>:-Ofast>
    $<$<CONFIG:RelWithDebInfo>:-O3>
)

set_target_properties(chessai PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN TRUE
)

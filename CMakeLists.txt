cmake_minimum_required(VERSION 3.18)
project(graphsack CUDA CXX)
set (CMAKE_CXX_STANDARD 17)

add_executable(graphsack)
add_subdirectory(src)
target_sources(graphsack PRIVATE src/graphs/adjacency_matrix.cpp)

set_target_properties(graphsack PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
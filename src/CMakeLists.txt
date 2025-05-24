file(GLOB src_sources CONFIGURE_DEPENDS *.cpp)
target_sources(graphsack PRIVATE ${src_sources})
file(GLOB src_cu_sources CONFIGURE_DEPENDS *.cu)
target_sources(graphsack PRIVATE ${src_cu_sources})

add_subdirectory(solvers)
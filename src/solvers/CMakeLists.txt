file(GLOB solvers_sources CONFIGURE_DEPENDS *.cu)
target_sources(graphsack PRIVATE ${solvers_sources})
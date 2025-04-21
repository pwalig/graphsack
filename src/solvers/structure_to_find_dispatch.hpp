#pragma once
#include <stdexcept>
#include "../requirements.hpp"

#define structure_to_find_dispatch() \
switch (instance.structure_to_find()) {\
case structure::none:\
	return solve(instance, [](const instance_t&, const solution_t&) {return true; });\
	break;\
case structure::path:\
	return solve(instance, is_path_possible);\
	break;\
case structure::cycle:\
	return solve(instance, is_cycle_possible);\
	break;\
default:\
	throw std::logic_error("invalid structure");\
	break;\
}

#define solve_with_structure_to_find_dispatch() inline static solution_t solve(const instance_t& instance) { structure_to_find_dispatch() }

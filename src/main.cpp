#include <iostream>

#include "weight_value_vector.hpp"
#include "bit_vector.hpp"
#include "solvers/Greedy.hpp"
#include "structure_check.hpp"
#include "graphs/nexts_list.hpp"

void main(int argc, char** argv) {
	const const int a = 5;
	std::cout << a << " A\n";

	gs::weight_value_vector<> inst(
		{ 11, 12, 13 },
		{ { 21, { 1, 2, 3}}, {22, { 4, 5, 6 }} }
	);

	std::cout << inst << "\n";

	gs::weight_value_vector<> inst2("instances/test.txt");
	std::cout << inst2 << "\n";

	std::cout << gs::solver::Greedy<gs::weight_value_vector<>, gs::bit_vector, unsigned int>::solve(inst2) << "\n";

	gs::grahps::nexts_list<unsigned int> nlist({
		{ 1, 2, 3},
		{ 0, 3 },
		{ 1 },
		{ 0, 1, 2 }
		});

	std::cout << nlist.view;
}
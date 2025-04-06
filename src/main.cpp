#include <iostream>

#include "inst/weight_value_vector.hpp"
#include "bit_vector.hpp"
#include "solvers/Greedy.hpp"
#include "structure_check.hpp"
#include "graphs/nexts_list.hpp"
#include "inst/itemlocal_nlist.hpp"

int main(int argc, char** argv) {
	const int a = 5;
	std::cout << a << " A\n";

	gs::weight_value_vector<> inst(
		{ 11, 12, 13 },
		{ { 21, { 1, 2, 3}}, {22, { 4, 5, 6 }} }
	);

	std::cout << inst << "\n";

	gs::weight_value_vector<> inst2("instances/test.txt");
	std::cout << inst2 << "\n";

	//std::cout << gs::solver::Greedy<gs::weight_value_vector<>, gs::bit_vector, unsigned int>::solve(inst2) << "\n";

	std::vector<unsigned int> storage(13);
	gs::grahps::nexts_list_view<unsigned int> nlist(slice<unsigned int>(storage.data(), storage.size()), {
		{ 1, 2, 3},
		{ 0, 3 },
		{ 1 },
		{ 0, 1, 2 }
		});

	std::cout << nlist;

	gs::grahps::nexts_list<std::vector<unsigned int>> nlist2("instances/nexts_list_test.txt");
	std::cout << nlist2 << "\n";


	gs::inst::itemlocal_nlist<unsigned int> ilnl(
		{ 11, 12, 13 },
		{ 21, 22, 23 },
		{ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} },
		{ {1}, {2}, {0, 1}}
	);

	for (auto val : ilnl.item_data_slice()) {
		std::cout << val / sizeof(unsigned int) << " ";
	}

	std::cout << "\n" << ilnl << "\n";
	std::cout << gs::solver::Greedy<gs::inst::itemlocal_nlist<unsigned int>, gs::bit_vector>::solve<gs::metric::ValueWeightRatio<float>>(ilnl) << "\n";

}
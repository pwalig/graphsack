#include <iostream>

#include "inst/weight_value_vector.hpp"
#include "bit_vector.hpp"
#include "solvers/Greedy.hpp"
#include "solvers/Dynamic.hpp"
#include "solvers/BruteForce.hpp"
#include "structure_check.hpp"
#include "graphs/nexts_list.hpp"
#include "inst/itemlocal_nlist.hpp"
#include "SolverRunner.hpp"
#include "graphs/adjacency_matrix.hpp"
#include "inst/gs_random.hpp"

#include "cuda_test.h"
#include "solvers/CudaBrutforce.hpp"

const std::string gs::solver::cuda::BruteForce<gs::inst::itemlocal_nlist<uint32_t, uint32_t, uint32_t>, gs::bit_vector>::name = "CudaBruteForce";

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
	gs::graphs::nexts_list_view<unsigned int> nlist(slice<unsigned int>(storage.data(), storage.size()), {
		{ 1, 2, 3},
		{ 0, 3 },
		{ 1 },
		{ 0, 1, 2 }
		});

	std::cout << nlist;

	gs::graphs::nexts_list<std::vector<unsigned int>> nlist2("instances/nexts_list_test.txt");
	std::cout << nlist2 << "\n";


	gs::inst::itemlocal_nlist<unsigned int> ilnl(
		{ 11, 12, 13 },
		{ 21, 22, 23 },
		{ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} },
		{ {1}, {2}, {0, 1} },
		gs::structure::cycle
	);
	gs::inst::itemlocal_nlist<unsigned int> ilnl2("instances/itemlocal_nlist_test.txt");

	for (auto val : ilnl.item_data_slice()) {
		std::cout << val / sizeof(unsigned int) << " ";
	}

	std::cout << "\n" << ilnl << "\n";
	std::cout << "\n" << ilnl2 << "\n";
	std::cout << gs::solver::Greedy<gs::inst::itemlocal_nlist<unsigned int>, gs::bit_vector>::solve(ilnl) << "\n";

	std::string format = "result: {result}\ntime: {time}s\nvalue: {result value}\nweights: {result weights}/ {limits}\nstructure: {structure valid}\nfitting: {fitting}\n";
	gs::SolverRunner<gs::solver::Greedy<gs::inst::itemlocal_nlist<unsigned int>, gs::bit_vector>>::run(ilnl2, format, std::cout);



	gs::graphs::adjacency_matrix am({
		{true, true, false},
		{true, true, false},
		{true, true, false}
		});
	std::cout << am;

	am = gs::graphs::adjacency_matrix::from_graph6("SeaLsGRWR{TjcoJYK`hqCYRz@FfnMuhSG");
	std::cout << am << am.graph6() << "\n";

	std::ifstream fin("instances/1.g6");
	if (fin.is_open()) {
		std::string graph6_string;
		fin >> graph6_string;
		am = gs::graphs::adjacency_matrix::from_graph6(graph6_string);
		std::cout << am << am.graph6() << "\n";
	}

	std::random_device randomDevice;
	std::knuth_b gen(randomDevice());
	am = gs::graphs::adjacency_matrix::from_gnp(8, 0.5, gen, true, true);
	std::cout << am;

	std::vector<unsigned int> randomValues(3 + 10 + (3 * 10));
	auto randomValueGen = std::bind(std::uniform_int_distribution<unsigned int>(1, 10), std::ref(gen));
	auto randomLimitGen = std::bind(std::uniform_int_distribution<unsigned int>(30, 50), std::ref(gen));
	gs::random::into<unsigned int>(randomValues.begin(), randomValues.begin() + 3, randomLimitGen);
	gs::random::into<unsigned int>(randomValues.begin() + 3, randomValues.end(), randomValueGen);
	for (auto i : randomValues) std::cout << i << " ";
	std::cout << "\n";
	gs::inst::itemlocal_nlist<uint32_t, uint32_t, uint32_t> randomItemlocalNlist(
		randomValues.begin(), randomValues.begin() + 3,
		randomValues.begin() + 3, randomValues.begin() + 13,
		randomValues.begin() + 13, randomValues.end(),
		gs::graphs::adjacency_matrix::from_gnp(10, 0.2, gen)
	);
	//std::cout << randomItemlocalNlist << "\n";
	gs::SolverRunner<gs::solver::Greedy<gs::inst::itemlocal_nlist<uint32_t, uint32_t, uint32_t>, gs::bit_vector>>::run(randomItemlocalNlist, format, std::cout);
	gs::SolverRunner<gs::solver::BruteForce<gs::inst::itemlocal_nlist<uint32_t, uint32_t, uint32_t>, gs::bit_vector>>::run(randomItemlocalNlist, format, std::cout);
	randomItemlocalNlist.weight_treatment() = gs::weight_treatment::first_only;
	randomItemlocalNlist.structure_to_find() = gs::structure::none;
	gs::SolverRunner<gs::solver::Dynamic<gs::inst::itemlocal_nlist<uint32_t, uint32_t, uint32_t>, gs::bit_vector>>::run(randomItemlocalNlist, format, std::cout);

	gs::cuda::info::print();
	gs::cuda::test();
	gs::SolverRunner<gs::solver::cuda::BruteForce<gs::inst::itemlocal_nlist<uint32_t, uint32_t, uint32_t>, gs::bit_vector>>::run(randomItemlocalNlist, format, std::cout);
}
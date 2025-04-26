#include <iostream>

#include "inst/weight_value_vector.hpp"
#include "res/bit_vector.hpp"
#include "solvers/Greedy.hpp"
#include "solvers/GRASP.hpp"
#include "solvers/Dynamic.hpp"
#include "solvers/MultiRun.hpp"
#include "solvers/BruteForce.hpp"
#include "structure_check.hpp"
#include "graphs/nexts_list.hpp"
#include "inst/itemlocal_nlist.hpp"
#include "SolverRunner.hpp"
#include "graphs/adjacency_matrix.hpp"
#include "inst/gs_random.hpp"
#include "solvers/CudaBrutforce.hpp"

#ifndef NDEBUG
#include "cuda_test.h"
#endif

using namespace gs;

int main(int argc, char** argv) {
	weight_value_vector<> inst(
		{ 11, 12, 13 },
		{ { 21, { 1, 2, 3}}, {22, { 4, 5, 6 }} }
	);

	std::cout << inst << "\n";

	weight_value_vector<> inst2("instances/test.txt");
	std::cout << inst2 << "\n";

	//std::cout << solver::Greedy<weight_value_vector<>, res::bit_vector, unsigned int>::solve(inst2) << "\n";

	std::vector<unsigned int> storage(13);
	graphs::nexts_list_view<unsigned int> nlist(slice<unsigned int>(storage.data(), storage.size()), {
		{ 1, 2, 3},
		{ 0, 3 },
		{ 1 },
		{ 0, 1, 2 }
		});

	std::cout << nlist;

	graphs::nexts_list<std::vector<unsigned int>> nlist2("instances/nexts_list_test.txt");
	std::cout << nlist2 << "\n";


	inst::itemlocal_nlist<unsigned int> ilnl(
		{ 11, 12, 13 },
		{ 21, 22, 23 },
		{ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} },
		{ {1}, {2}, {0, 1} },
		structure::cycle
	);
	inst::itemlocal_nlist<unsigned int> ilnl2("instances/itemlocal_nlist_test.txt");

	for (auto val : ilnl.item_data_slice()) {
		std::cout << val / sizeof(unsigned int) << " ";
	}

	std::cout << "\n" << ilnl << "\n";
	std::cout << "\n" << ilnl2 << "\n";
	std::cout << solver::Greedy<inst::itemlocal_nlist<unsigned int>, res::bit_vector>::solve(ilnl) << "\n";

	std::string format = "{solver name}\nresult: {result}\ntime: {time}s\nvalue: {result value}\nweights: {result weights}/ {limits}\nstructure: {structure valid}\nfitting: {fitting}\n";
	SolverRunner<solver::Greedy<inst::itemlocal_nlist<unsigned int>, res::bit_vector>>::run(ilnl2, format, std::cout);


#ifndef NDEBUG
	// adjacency matrix
	gs::graphs::adjacency_matrix::test::all();

	// cuda
	cuda::info::print();
	cuda::test();
#endif


	std::random_device randomDevice;
	std::knuth_b gen(randomDevice());

	std::vector<unsigned int> randomValues(3 + 10 + (3 * 10));
	auto randomValueGen = std::bind(std::uniform_int_distribution<unsigned int>(1, 10), std::ref(gen));
	auto randomLimitGen = std::bind(std::uniform_int_distribution<unsigned int>(30, 50), std::ref(gen));
	random::into<unsigned int>(randomValues.begin(), randomValues.begin() + 3, randomLimitGen);
	random::into<unsigned int>(randomValues.begin() + 3, randomValues.end(), randomValueGen);
	for (auto i : randomValues) std::cout << i << " ";
	std::cout << "\n";
	inst::itemlocal_nlist<uint32_t, uint32_t, uint32_t> randomItemlocalNlist(
		randomValues.begin(), randomValues.begin() + 3,
		randomValues.begin() + 3, randomValues.begin() + 13,
		randomValues.begin() + 13, randomValues.end(),
		graphs::adjacency_matrix::from_gnp(10, 0.2, gen)
	);
	//std::cout << randomItemlocalNlist << "\n";
	SolverRunner<solver::Greedy<inst::itemlocal_nlist<uint32_t, uint32_t, uint32_t>, res::bit_vector>>::run(randomItemlocalNlist, format, std::cout);
	SolverRunner<solver::Greedy<inst::itemlocal_nlist<uint32_t, uint32_t, uint32_t>, res::bit_vector, metric::NextsCountValueWeightRatio<float>>>::run(randomItemlocalNlist, format, std::cout);
	SolverRunner<solver::MultiRun<solver::GRASP<inst::itemlocal_nlist<uint32_t, uint32_t, uint32_t>, res::bit_vector, std::knuth_b>>>::run<float, std::knuth_b, float>(randomItemlocalNlist, format, std::cout, 1.0f, gen, 0.5f);
	SolverRunner<solver::BruteForce<inst::itemlocal_nlist<uint32_t, uint32_t, uint32_t>, res::bit_vector>>::run(randomItemlocalNlist, format, std::cout);
	randomItemlocalNlist.weight_treatment() = weight_treatment::first_only;
	randomItemlocalNlist.structure_to_find() = structure::none;
	SolverRunner<solver::Dynamic<inst::itemlocal_nlist<uint32_t, uint32_t, uint32_t>, res::bit_vector>>::run(randomItemlocalNlist, format, std::cout);

	SolverRunner<solver::cuda::BruteForce<inst::itemlocal_nlist<uint32_t, uint32_t, uint32_t>, res::bit_vector>>::run(randomItemlocalNlist, format, std::cout);
}
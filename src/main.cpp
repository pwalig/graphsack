#include <iostream>

#include "inst/weight_value_vector.hpp"
#include "res/bit_vector.hpp"
#include "solvers/Greedy.hpp"
#include "solvers/GRASP.hpp"
#include "solvers/ompGRASP.hpp"
#include "solvers/GHS.hpp"
#include "solvers/Dynamic.hpp"
#include "solvers/BruteForce.hpp"
#include "solvers/ompBruteForce.hpp"
#include "solvers/PathBruteForce.hpp"
#include "structure_check.hpp"
#include "graphs/nexts_list.hpp"
#include "inst/itemlocal_nlist.hpp"
#include "SolverRunner.hpp"
#include "graphs/adjacency_matrix.hpp"
#include "inst/gs_random.hpp"
#include "solvers/CudaBrutforce.hpp"
#include "solvers/CudaGRASP.hpp"
#include "inst/inst_generator.hpp"
#include "inst/cuda_instance.hpp"
#include "cuda_init.h"

#ifndef NDEBUG
#include "cuda_test.h"
#endif

using namespace gs;

template <typename value_type, typename weight_type>
void test(
	size_t itemsCount, size_t weightsDim,
	weight_type MinLimit, weight_type MaxLimit,
	value_type MinValue, value_type MaxValue,
	weight_type MinWeight, weight_type MaxWeight,
	double density, bool unidirectional = false, bool selfArches = true
) {
	using Rand = std::mt19937;
	std::random_device randomDevice;
	Rand gen(randomDevice());

	using cpu_instance = inst::itemlocal_nlist<value_type, weight_type>;
	using gpu_instance = cuda::inst::instance64<value_type, weight_type>;

	using cpu_result = res::bit_vector;
	using gpu_result = cuda::res::solution64;
	
	auto ld = [&gen, MinLimit, MaxLimit]() { return gs::random::function<Rand>::from_uniform_distribution(gen, MinLimit, MaxLimit); };
	auto vd = [&gen, MinValue, MaxValue]() { return gs::random::function<Rand>::from_uniform_distribution(gen, MinValue, MaxValue); };
	auto wd = [&gen, MinWeight, MaxWeight]() { return gs::random::function<Rand>::from_uniform_distribution(gen, MinWeight, MaxWeight); };
	
	std::vector<weight_type> randomWeights(itemsCount * weightsDim + weightsDim);
	std::vector<value_type> randomValues(itemsCount);

	random::into(randomWeights.begin(), randomWeights.begin() + weightsDim, ld);
	random::into(randomValues.begin(), randomValues.end(), vd);
	random::into(randomWeights.begin() + weightsDim, randomWeights.end(), wd);

	 cpu_instance cpui(
		randomWeights.begin(), randomWeights.begin() + weightsDim,
		randomValues.begin(), randomValues.end(),
		randomWeights.begin() + weightsDim, randomWeights.end(),
		gs::graphs::adjacency_matrix::from_gnp(itemsCount, density, gen, unidirectional, selfArches),
		structure::cycle, weight_treatment::full
	);
	gpu_instance gpui(
		randomWeights.begin(), randomWeights.begin() + weightsDim,
		randomValues.begin(), randomValues.end(),
		randomWeights.begin() + weightsDim, randomWeights.end(),
		gs::graphs::adjacency_matrix::from_gnp(itemsCount, density, gen, unidirectional, selfArches),
		structure::cycle, weight_treatment::full
	);

	std::string format = "{solver name}\nresult: {result}\ntime: {time}s\nvalue: {result value}\nweights: {result weights}/ {limits}\nstructure: {structure valid}\nfitting: {fitting}\n\n";

	SolverRunner<solver::Greedy<cpu_instance, cpu_result>>::run(cpui, format, std::cout);
	SolverRunner<solver::Greedy<cpu_instance, cpu_result, metric::NextsCountValueWeightRatio<>>>::run(cpui, format, std::cout);
	SolverRunner<solver::GHS<cpu_instance, cpu_result, metric::NextsCountValueWeightRatio<>>>::run(cpui, format, std::cout, size_t(5), true);

	SolverRunner<solver::BruteForce<cpu_instance, cpu_result>>::run(cpui, format, std::cout);
	SolverRunner<solver::ompBruteForce<cpu_instance, cpu_result>>::run(cpui, format, std::cout);
	SolverRunner<cuda::solver::BruteForce<gpu_instance>>::run(gpui, format, std::cout, 0, 1);

	SolverRunner<solver::GRASP<cpu_instance, cpu_result, Rand>>::run(cpui, format, std::cout, gen, 0.5f, size_t(256));
	SolverRunner<solver::ompGRASP<cpu_instance, cpu_result, Rand>>::run(cpui, format, std::cout, gen, 0.5f, size_t(256));
	SolverRunner<cuda::solver::GRASP<gpu_instance>>::run(gpui, format, std::cout, 64, gpui.size() / 2);
}

int main(int argc, char** argv) {
	cuda::device_properties_t cuda_capability = cuda::init();
	std::cout << "threads per block: " << cuda_capability.maxThreadsPerBlock << '\n';
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

	std::string format = "{solver name}\nresult: {result}\ntime: {time}s\nvalue: {result value}\nweights: {result weights}/ {limits}\nstructure: {structure valid}\nfitting: {fitting}\n\n";
	SolverRunner<solver::Greedy<inst::itemlocal_nlist<unsigned int>, res::bit_vector>>::run(ilnl2, format, std::cout);


#ifndef NDEBUG
	// adjacency matrix
	gs::graphs::adjacency_matrix::test::all();

	// cuda
	cuda::info::print();
	cuda::test();
#endif

	test<uint32_t, uint32_t>(
		16, 3,
		30, 50,
		1, 10,
		1, 10,
		0.2, false, true
	);

	std::random_device randomDevice;
	std::knuth_b gen(randomDevice());

	std::vector<size_t> path = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0 };
	graphs::adjacency_matrix fp = graphs::adjacency_matrix::from_path(10, path.begin(), path.end(), true);
	std::cout << fp << "\n";
	fp.gnp_fill(0.5, gen, true);
	std::cout << fp << "\n";

	cuda::inst::instance64<uint32_t, uint32_t> cudaInstance64 = inst::Generator<cuda::inst::instance64<uint32_t, uint32_t>>::random(
		10, 3, 0.2, gen, 30, 50, 1, 10, 1, 10, false, true, structure::cycle, weight_treatment::full
	);
	std::cout << cudaInstance64 << "\n";

	auto known_res = inst::Generator<inst::itemlocal_nlist<uint32_t, uint32_t, uint32_t>>::known_path_or_cycle_gnp(
		10, 3, 5, gen, 1, 10, 1, 10, false, true, structure::path, weight_treatment::full
	);
	std::cout << known_res.first << "\noptimum: " << known_res.second << "\n\n";
	SolverRunner<solver::BruteForce<inst::itemlocal_nlist<uint32_t, uint32_t, uint32_t>, res::bit_vector>>::run(known_res.first, format, std::cout);
}
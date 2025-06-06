#include <iostream>

#include "inst/weight_value_vector.hpp"
#include "inst/naive_item_vector.hpp"
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
#include "json.hpp"

#include "cuda_test.h"

using namespace gs;

namespace gs::test {
	using Rand = std::mt19937;

	template <typename value_type, typename weight_type>
	struct params {
		size_t itemsCount;
		size_t weightsDim;
		weight_type MinLimit;
		weight_type MaxLimit;
		value_type MinValue;
		value_type MaxValue;
		weight_type MinWeight;
		weight_type MaxWeight;
		double density;
		bool unidirectional = false;
		bool selfArches = true;
		structure structureToFind = structure::cycle;
		weight_treatment weightTreatment = weight_treatment::full;
	};

	template <typename value_type, typename weight_type>
	void test(
		const params<value_type, weight_type>& p,
		Rand& gen,
		uint32_t runs = 1
	) {

		using cpu_instance = inst::itemlocal_nlist<value_type, weight_type, uint32_t, std::vector<uint32_t>>;
		//using cpu_instance = inst::naive_item_vector<value_type, weight_type>;
		using gpu_instance = cuda::inst::instance64<value_type, weight_type>;

		using cpu_result = res::bit_vector;
		using gpu_result = cuda::res::solution64;

		using graph_t = graphs::adjacency_matrix;
		
		auto ld = [&gen, MinLimit = p.MinLimit, MaxLimit = p.MaxLimit]()
			{ return gs::random::function<Rand>::from_uniform_distribution(gen, MinLimit, MaxLimit); };
		auto vd = [&gen, MinValue = p.MinValue, MaxValue = p.MaxValue]()
			{ return gs::random::function<Rand>::from_uniform_distribution(gen, MinValue, MaxValue); };
		auto wd = [&gen, MinWeight = p.MinWeight, MaxWeight = p.MaxWeight]()
			{ return gs::random::function<Rand>::from_uniform_distribution(gen, MinWeight, MaxWeight); };
		
		std::vector<weight_type> randomWeights(p.itemsCount * p.weightsDim + p.weightsDim);
		std::vector<value_type> randomValues(p.itemsCount);

		std::vector<cpu_instance> cpui;
		std::vector<gpu_instance> gpui;
		cpui.reserve(runs);
		gpui.reserve(runs);
		for (uint32_t i = 0; i < runs; ++i) {

			random::into(randomWeights.begin(), randomWeights.begin() + p.weightsDim, ld);
			random::into(randomValues.begin(), randomValues.end(), vd);
			random::into(randomWeights.begin() + p.weightsDim, randomWeights.end(), wd);

			graph_t graph = graph_t::from_gnp(p.itemsCount, p.density, gen, p.unidirectional, p.selfArches);

			cpui.push_back(cpu_instance(
				randomWeights.begin(), randomWeights.begin() + p.weightsDim,
				randomValues.begin(), randomValues.end(),
				randomWeights.begin() + p.weightsDim, randomWeights.end(),
				graph, p.structureToFind, p.weightTreatment
			));
			gpui.push_back(gpu_instance(
				randomWeights.begin(), randomWeights.begin() + p.weightsDim,
				randomValues.begin(), randomValues.end(),
				randomWeights.begin() + p.weightsDim, randomWeights.end(),
				graph, p.structureToFind, p.weightTreatment
			));
		}

		std::string format = "result: {result}\ttime: {time}s\tvalue: {result value}\tweights: {result weights} / {limits}\tstructure: {structure valid}\tfitting: {fitting}\n";
		std::string avg_format = "{solver name}\ttime: {time}s\tvalue: {result value}\tweights: {result weights} / {limits}\tstructure: {structure valid}\tfitting: {fitting}\n";
		std::string single_format = "{solver name}\nresult: {result}\ntime: {time}s\nvalue: {result value}\nweights: {result weights}/ {limits}\nstructure: {structure valid}\nfitting: {fitting}\n\n";
		std::vector<std::pair<std::string, std::ostream&>> outputs;
		std::vector<std::pair<std::string, std::ostream&>> avg_outputs;
		if (runs == 1) {
			outputs.push_back({ single_format, std::cout });
		}
		else {
			outputs.push_back({ format, std::cout });
			avg_outputs.push_back({ avg_format, std::cout });
		}

		//SolverRunner<solver::Greedy<cpu_instance, cpu_result>>::run(cpui, outputs, avg_outputs);
		//SolverRunner<solver::Greedy<cpu_instance, cpu_result, metric::NextsCountValueWeightRatio<>>>::run(cpui, outputs, avg_outputs);
		//SolverRunner<solver::GHS<cpu_instance, cpu_result, metric::NextsCountValueWeightRatio<>>>::run(cpui, outputs, avg_outputs, size_t(5), true);

		SolverRunner<cuda::solver::BruteForce<gpu_instance>>::run(gpui, outputs, avg_outputs, 0, 1);
		SolverRunner<solver::ompBruteForce<cpu_instance, cpu_result>>::run(cpui, outputs, avg_outputs);
		SolverRunner<solver::BruteForce<cpu_instance, cpu_result>>::run(cpui, outputs, avg_outputs);

		SolverRunner<cuda::solver::GRASP<gpu_instance>>::run(gpui, outputs, avg_outputs, 1, gpui.size() / 2);
		SolverRunner<solver::ompGRASP<cpu_instance, cpu_result, Rand>>::run(cpui, outputs, avg_outputs, gen, 0.5f, size_t(256));
		SolverRunner<solver::GRASP<cpu_instance, cpu_result, Rand>>::run(cpui, outputs, avg_outputs, gen, 0.5f, size_t(256));
	}

	template <typename value_type, typename weight_type>
	void test(
		const params<value_type, weight_type>& p,
		uint32_t runs = 1
	) {
		std::random_device randomDevice;
		Rand gen(randomDevice());
		test<value_type, weight_type>(p, gen, runs);
	}
}


int main(int argc, char** argv) {
	std::random_device randomDevice;
	cuda::info::print_json();
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
	json::pretty_writer<inst::itemlocal_nlist<uint32_t>>::write(
		ilnl,
		{json::key::limits, json::key::weights, json::key::values,
		json::key::weight_value_items, json::key::weight_treatment, json::key::structure},
		"instances/itemlocal_nlist_test.json"
	);
	std::cout << "IMPORTED:\n" << json::reader<inst::itemlocal_nlist<uint32_t>>::read("instances/itemlocal_nlist_test.json") << '\n';
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

	test::Rand testGen(randomDevice());
	for (size_t i = 5; i < 5; i += 5) {
		std::cout << i << '\n';
		test::params<uint32_t, float> params = {
			i, 3,
			30, 50,
			1, 10,
			1, 10,
			0.2, false, true
		};
		test::test<uint32_t, float>(
			params, testGen, 1 
		);
	}

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
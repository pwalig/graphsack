#pragma once
#include <vector>
#include <random>
#include <cassert>

#include "GRASP.hpp"
#include "../Validator.hpp"
#include "../structure_check.hpp"
#include "../weight_vector_operations.hpp"
#include "metric.hpp"

namespace gs {
	namespace solver {
		template <typename InstanceT, typename SolutionT, typename RandomEngine, typename metricT = gs::metric::ValueWeightRatio<float>, typename indexT = typename InstanceT::index_type>
		class ompGRASP {
		public:
			using instance_t = InstanceT;
			using solution_t = SolutionT;
			inline static const std::string name = "ompGRASP<" + metricT::name + ">";

			inline static solution_t solve(
				const instance_t& instance,
				RandomEngine& randomEngine,
				size_t choose_from,
				size_t runs,
				bool (*structure_check) (const instance_t&, const solution_t&)
			) {
				// sort elements
				auto sorted = metric::sorted_indexes<metricT, instance_t, indexT>(instance);

				// best results
				solution_t best_solution(instance.size());
				typename instance_t::value_type best_value = 0;

				#pragma omp parallel
				{
					// local best results
					solution_t local_best_solution(instance.size());
					typename instance_t::value_type local_best_value = 0;

					// working memory
					std::vector<typename instance_t::weight_type> remaining(instance.dim());
					solution_t solution(instance.size());

					// main loop
					#pragma omp for
					for (long long i = 0; i < runs; ++i) {
						typename instance_t::value_type value = GRASP<InstanceT, SolutionT, RandomEngine, metricT, indexT>::solve_one(
							instance, solution, randomEngine, sorted, remaining.data(), choose_from, structure_check
						);
						if (value > local_best_value) {
							local_best_value = value;
							local_best_solution = solution;
						}
					}

					#pragma omp critical
					{
						if (local_best_value > best_value) {
							best_value = local_best_value;
							best_solution = local_best_solution;
						}
					}
				}

				return best_solution;
			}

			inline static solution_t solve(
				const instance_t& instance,
				RandomEngine& randomEngine,
				size_t choose_from,
				size_t runs
			) {
				switch (instance.structure_to_find()) {
				case structure::none:
					return solve(instance, randomEngine, choose_from, runs, [](const instance_t&, const solution_t&) {return true; });
					break;
				case structure::path:
					return solve(instance, randomEngine, choose_from, runs, is_path_possible);
					break;
				case structure::cycle:
					return solve(instance, randomEngine, choose_from, runs, is_cycle_possible);
					break;
				default:
					throw std::logic_error("invalid structure");
					break;
				}
			}

			inline static solution_t solve(
				const instance_t& instance,
				RandomEngine& randomEngine,
				float choose_from,
				size_t runs
			) {
				return solve(instance, randomEngine, (size_t)(instance.size() * choose_from), runs);
			}

			inline static solution_t solve(
				const instance_t& instance,
				RandomEngine& randomEngine,
				float choose_from,
				float coverage
			) {
				return solve(instance, randomEngine, (size_t)(instance.size() * choose_from), (size_t)(instance.size() * coverage));
			}

			inline static solution_t solve(
				const instance_t& instance,
				RandomEngine& randomEngine,
				size_t choose_from,
				float coverage
			) {
				return solve(instance, randomEngine, choose_from, (size_t)(instance.size() * coverage));
			}
		};
	}
}

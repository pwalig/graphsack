#pragma once
#include <vector>
#include <random>
#include <cassert>

#include "../Validator.hpp"
#include "../structure_check.hpp"
#include "../weight_vector_operations.hpp"
#include "metric.hpp"

namespace gs {
	namespace solver {
		template <typename InstanceT, typename SolutionT, typename RandomEngine, typename metricT = gs::metric::ValueWeightRatio<float>, typename indexT = typename InstanceT::index_type>
		class GRASP {
		public:
			using instance_t = InstanceT;
			using solution_t = SolutionT;
			inline static const std::string name = "GRASP<" + metricT::name + ">";

			inline static typename instance_t::value_type solve_one(
				const instance_t& instance,
				solution_t& res,
				RandomEngine& randomEngine,
				std::vector<indexT> sorted,
				typename instance_t::weight_type* weight_memory,
				size_t choose_from,
				bool (*structure_check) (const instance_t&, const solution_t&)
			) {
				assert(choose_from > 0);
				assert(res.size() == instance.size());

				res.clear();
				typename instance_t::value_type value = 0;
				for (typename instance_t::size_type i = 0; i < instance.dim(); ++i)
					weight_memory[i] = instance.limit(i);

				// solve
				while (sorted.size() > 0) {
					std::uniform_int_distribution<indexT> distrib(0, (indexT)std::min(choose_from, sorted.size() - 1));
					indexT pick = distrib(randomEngine);
					indexT itemId = sorted[pick];

					// fit check
					typename instance_t::size_type i = 0;
					for (; i < instance.dim(); ++i) {
						if (instance.weight(itemId, i) > weight_memory[i]) break;
					}
					if (i == instance.dim()) {
						res.add(itemId);
						if (!structure_check(instance, res)) res.remove(itemId); // structure check
						else {
							for (i = 0; i < instance.dim(); ++i) weight_memory[i] -= instance.weight(itemId, i);
							value += instance.value(itemId);
						}
					}

					sorted.erase(sorted.begin() + pick);
				}
				return value;
			}
			
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

				// working memory
				std::vector<typename instance_t::weight_type> remaining(instance.dim());
				solution_t solution(instance.size());

				// main loop
				for (size_t i = 0; i < runs; ++i) {
					typename instance_t::value_type value = solve_one(
						instance, solution, randomEngine, sorted, remaining.data(), choose_from, structure_check
					);
					if (value > best_value) {
						best_value = value;
						best_solution = solution;
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

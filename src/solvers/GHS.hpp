#pragma once
#include <vector>
#include <random>
#include <cassert>

#include "../structure_check.hpp"
#include "../weight_vector_operations.hpp"
#include "metric.hpp"

namespace gs {
	namespace solver {
		template <typename InstanceT, typename SolutionT, typename metricT = gs::metric::ValueWeightRatio<float>, typename indexT = typename InstanceT::index_type>
		class GHS {
		public:
			using instance_t = InstanceT;
			using solution_t = SolutionT;
			inline static const std::string name = "GHS";

			using metric_function = metric::function<metricT, instance_t>;

			inline static std::pair<solution_t, typename instance_t::value_type> fast(
				const instance_t& instance,
				const solution_t& current_solution,
				std::vector<indexT> sorted, // in reverse
				const std::vector<typename instance_t::weight_type>& remaining,
				typename instance_t::value_type current_value,
				size_t to_visit,
				bool (*structure_check) (const instance_t&, const solution_t&)
			) {
				solution_t best_solution(current_solution);
				typename instance_t::value_type best_value = current_value;
				while (to_visit > 0 && sorted.size() > 0) {
					--to_visit;

					indexT itemId = sorted.back();
					sorted.pop_back();

					if (fits(instance.weights(itemId), remaining)) continue;

					solution_t temp_solution(current_solution);
					temp_solution.add(itemId);
					if (!structure_check(instance, temp_solution)) continue;

					std::vector<typename instance_t::weight_type> temp_remaining(remaining);
					sub_from_weights(temp_remaining, instance.weights(itemId));

					auto temp_res = fast(instance, temp_solution, sorted, temp_remaining,
						current_value + instance.value(itemId), std::max<size_t>(to_visit, 1),
						structure_check
					);
					if (temp_res.second > best_value) {
						best_value = temp_res.second;
						best_solution = temp_res.first;
					}
				}
				return std::make_pair(best_solution, current_value);
			}

			inline static solution_t solve(
				const instance_t& instance,
				size_t to_visit,
				bool (*structure_check) (const instance_t&, const solution_t&)
			) {
				solution_t res(instance.size());
				std::vector<typename instance_t::weight_type> remaining(instance.limits().begin(), instance.limits().end());

				// get sorted elements
				std::vector<indexT> sorted = metric::sorted_indexes<metricT, instance_t, indexT>(instance);

				// solve
				return fast(instance, res, sorted, remaining, 0, to_visit, structure_check).first;
			}
			
			inline static solution_t solve(
				const instance_t& instance,
				size_t choose_from
			) {
				switch (instance.structure_to_find()) {
				case structure::none:
					return solve(instance, choose_from, [](const instance_t&, const solution_t&) {return true; });
					break;
				case structure::path:
					return solve(instance, choose_from, is_path_possible);
					break;
				case structure::cycle:
					return solve(instance, choose_from, is_cycle_possible);
					break;
				default:
					throw std::logic_error("invalid structure");
					break;
				}
			}

			inline static solution_t solve(
				const instance_t& instance,
				float coverage
			) {
				return solve(instance, (size_t)(instance.size() * coverage));
			}
		};
	}
}

#pragma once
#include <vector>
#include <random>
#include <cassert>

#include "../structure_check.hpp"
#include "../weight_vector_operations.hpp"
#include "metric.hpp"
#include "../Validator.hpp"

namespace gs {
	namespace solver {
		template <typename InstanceT, typename SolutionT, typename metricT = gs::metric::ValueWeightRatio<float>, typename indexT = typename InstanceT::index_type>
		class GHS {
		public:
			using instance_t = InstanceT;
			using solution_t = SolutionT;
			inline static const std::string name = "GHS<" + metricT::name + ">";

			using metric_function = metric::function<metricT, instance_t>;

			inline static solution_t fast(
				const instance_t& instance,
				const solution_t& current_solution,
				const std::vector<typename instance_t::weight_type>& remaining,
				typename instance_t::value_type maxValue,
				std::vector<indexT> sorted,
				size_t to_visit,
				bool (*structure_check) (const instance_t&, const solution_t&)
			) {
				solution_t best = current_solution;
				while ((to_visit--) > 0  && sorted.size() > 0) {
					indexT itemId = sorted.back();
					sorted.pop_back();

					if (fits(instance.weights(itemId), remaining)) {
						solution_t tmp_solution = current_solution;
						tmp_solution.add(itemId);
						if (structure_check(instance, tmp_solution)) {
							std::vector<typename instance_t::weight_type> tmp_remaining = remaining;
							sub_from_weights(tmp_remaining, instance.weights(itemId));
							tmp_solution = fast(instance, tmp_solution, tmp_remaining, maxValue, sorted, std::max<size_t>(to_visit, 1), structure_check);
							typename instance_t::value_type value = Validator<instance_t, solution_t>::getResultValue(instance, tmp_solution);
							if (value > maxValue) {
								maxValue = value;
								best = tmp_solution;
							}
						}
					}
				}
				return best;
			}

			inline static solution_t accurate(
				const instance_t& instance,
				const solution_t& current_solution,
				const std::vector<typename instance_t::weight_type>& remaining,
				typename instance_t::value_type maxValue,
				std::vector<indexT> sorted,
				size_t to_visit,
				bool (*structure_check) (const instance_t&, const solution_t&)
			) {
				solution_t best = current_solution;
				while (to_visit > 0  && sorted.size() > 0) {
					indexT itemId = sorted.back();
					sorted.pop_back();

					if (fits(instance.weights(itemId), remaining)) {
						solution_t tmp_solution = current_solution;
						tmp_solution.add(itemId);
						if (structure_check(instance, tmp_solution)) {
							std::vector<typename instance_t::weight_type> tmp_remaining = remaining;
							sub_from_weights(tmp_remaining, instance.weights(itemId));
							tmp_solution = accurate(instance, tmp_solution, tmp_remaining, maxValue, sorted, std::max<size_t>(--to_visit, 1), structure_check);
							typename instance_t::value_type value = Validator<instance_t, solution_t>::getResultValue(instance, tmp_solution);
							if (value > maxValue) {
								maxValue = value;
								best = tmp_solution;
							}
						}
					}
				}
				return best;
			}

			inline static solution_t solve(
				const instance_t& instance,
				size_t to_visit,
				bool Accurate,
				bool (*structure_check) (const instance_t&, const solution_t&)
			) {
				assert(to_visit > 0);
				if (Accurate) return accurate(
					instance,
					solution_t(instance.size()),
					std::vector<typename instance_t::weight_type>(instance.limits().begin(), instance.limits().end()),
					0,
					metric::sorted_indexes<metricT, instance_t, indexT>(instance, true),
					to_visit,
					structure_check
				);
				else return fast(
					instance,
					solution_t(instance.size()),
					std::vector<typename instance_t::weight_type>(instance.limits().begin(), instance.limits().end()),
					0,
					metric::sorted_indexes<metricT, instance_t, indexT>(instance, true),
					to_visit,
					structure_check
				);
			}
			
			inline static solution_t solve(
				const instance_t& instance,
				size_t to_visit,
				bool Accurate
			) {
				switch (instance.structure_to_find()) {
				case structure::none:
					return solve(instance, to_visit, Accurate, [](const instance_t&, const solution_t&) {return true; });
					break;
				case structure::path:
					return solve(instance, to_visit, Accurate, is_path_possible);
					break;
				case structure::cycle:
					return solve(instance, to_visit, Accurate, is_cycle_possible);
					break;
				default:
					throw std::logic_error("invalid structure");
					break;
				}
			}

			inline static solution_t solve(
				const instance_t& instance,
				float coverage,
				bool Accurate
			) {
				return solve(instance, (size_t)(instance.size() * coverage), Accurate);
			}
		};
	}
}

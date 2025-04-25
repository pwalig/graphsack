#pragma once
#include <vector>
#include <algorithm>

#include "../structure_check.hpp"
#include "../weight_vector_operations.hpp"
#include "structure_to_find_dispatch.hpp"
#include "metric.hpp"

namespace gs {
	namespace solver {
		template <typename InstanceT, typename SolutionT, typename metricT = gs::metric::ValueWeightRatio<float>, typename indexT = typename InstanceT::index_type>
		class GRASP {
		public:
			using instance_t = InstanceT;
			using solution_t = SolutionT;
			inline static const std::string name = "GRASP";

			using metric_function = metric::function<metricT, instance_t>;

			inline static solution_t solve(
				const instance_t& instance,
				float choose_from,
				bool (*structure_check) (const instance_t&, const solution_t&)
			) {
				solution_t res(instance.size());
				std::vector<typename instance_t::weight_type> remaining(instance.limits().begin(), instance.limits().end());

				// get sorted elements
				std::vector<indexT> sorted = metric::sorted_indexes<metricT, instance_t, indexT>(instance);

				// solve
				size_t amount_to_chose_from = instance.size() * choose_from;
				while (sorted.size() > 0) {
					size_t pick = std::rand() % std::min(amount_to_chose_from, sorted.size());
					indexT itemId = sorted[pick];

					if (fits(instance.weights(itemId), remaining)) {
						res.add(itemId);
						if (!structure_check(instance, res)) res.remove(itemId);
						else sub_from_weights(remaining, instance.weights(itemId));
					}

					sorted.erase(sorted.begin() + pick);
				}
				return res;
			}
			
			inline static solution_t solve(const instance_t& instance, float choose_from = 0.25f) {
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
		};
	}
}

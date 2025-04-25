#pragma once
#include <vector>

#include "../structure_check.hpp"
#include "../weight_vector_operations.hpp"
#include "structure_to_find_dispatch.hpp"
#include "metric.hpp"

namespace gs {
	namespace solver {
		template <typename InstanceT, typename SolutionT, typename metricT = gs::metric::ValueWeightRatio<float>, typename indexT = typename InstanceT::index_type>
		class Greedy {
		public:
			using instance_t = InstanceT;
			using solution_t = SolutionT;
			inline static const std::string name = "Greedy";

			using metric_function = metric::function<metricT, instance_t>;

			inline static solution_t solve(
				const instance_t& instance,
				bool (*structure_check) (const instance_t&, const solution_t&)
			) {
				// prepare storage
				solution_t res(instance.size());
				std::vector<typename instance_t::weight_type> remaining(instance.limits().begin(), instance.limits().end());

				// get sorted elements
				std::vector<indexT> sorted = metric::sorted_indexes<metricT, instance_t, indexT>(instance);

				// solve
				for (indexT itemId : sorted) {
					if (fits(instance.weights(itemId), remaining)) {
						res.add(itemId);
						if (!structure_check(instance, res)) res.remove(itemId);
						else sub_from_weights(remaining, instance.weights(itemId));
					}
				}

				// return
				return res;
			}
			solve_with_structure_to_find_dispatch()
		};
	}
}

#pragma once
#include <vector>
#include <algorithm>

#include "../structure_check.hpp"
#include "../weight_vector_operations.hpp"
#include "../requirements.hpp"
#include "metric.hpp"

namespace gs {
	namespace solver {
		template <typename InstanceT, typename SolutionT, typename metricT = gs::metric::ValueWeightRatio<float>, typename indexT = typename InstanceT::size_type>
		class Greedy {
		public:
			using instance_t = InstanceT;
			using solution_t = SolutionT;
			inline static const std::string name = "Greedy";

			using metric_function = metric::function<metricT, instance_t>;

			inline static std::vector<indexT> getSortedByMetric(
				const instance_t& instance
			) {
				std::vector<indexT> res(instance.size());
				for (indexT i = 0; i < instance.size(); ++i) {
					res[i] = i;
				}
				std::vector<typename metricT::value_type> metric = metric::calculate<metricT, instance_t>(instance);
				std::sort(res.begin(), res.end(), [&metric](indexT a, indexT b) { return metric[a] > metric[b]; });
				return res;
			}

			inline static solution_t solve(
				const instance_t& instance,
				bool (*structure_check) (const instance_t&, const solution_t&)
			) {
				// prepare storage
				solution_t res(instance.size());
				std::vector<typename instance_t::weight_type> remaining(instance.dim());
				element_wise::in_place_operate(remaining, instance.limits(), element_wise::copy);

				// get sorted elements
				std::vector<indexT> sorted = getSortedByMetric(instance);

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
			inline static solution_t solve(
				const instance_t& instance
			) {
				switch (instance.structure_to_find())
				{
				case structure::none:
					return solve(instance, [](const instance_t&, const solution_t&) {return true; });
					break;
				case structure::path:
					return solve(instance, is_path_possible);
					break;
				case structure::cycle:
					return solve(instance, is_cycle_possible);
					break;
				default:
					break;
				}
			}
		};
	}
}

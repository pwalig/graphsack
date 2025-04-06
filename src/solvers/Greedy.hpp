#pragma once
#include <vector>
#include <algorithm>

#include "../structure_check.hpp"
#include "../weight_vector_operations.hpp"
#include "metric.hpp"

namespace gs {
	namespace solver {
		template <typename InstanceT, typename SolutionT, typename indexT = typename InstanceT::size_type>
		class Greedy {
		public:
			using instance_t = InstanceT;
			using solution_t = SolutionT;

			template <typename metricT>
			using metric_function = metric::function<metricT, instance_t>;

			template <typename metricT>
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

			template <typename metricT>
			inline static solution_t solve(
				const instance_t& instance
			) {
				// prepare storage
				solution_t res(instance.size());
				std::vector<typename instance_t::weight_type> remaining(instance.dim());
				element_wise::operate(remaining, instance.limits(), element_wise::copy);

				// get sorted elements
				std::vector<indexT> sorted = getSortedByMetric<metricT>(instance);

				// solve
				for (indexT itemId : sorted) {
					if (fits(instance.weights(itemId), remaining)) {
						res.add(itemId);
						if (!is_path_possible(instance, res)) res.remove(itemId);
						else sub_from_weights(remaining, instance.weights(itemId));
					}
				}

				// return
				return res;
			}
		};
	}
}

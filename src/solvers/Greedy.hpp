#pragma once
#include <vector>
#include <algorithm>

#include "../structure_check.hpp"
#include "../weight_vector_operations.hpp"

namespace gs {
	namespace solver {
		template <typename InstanceT, typename SolutionT, typename indexT = size_t>
		class Greedy {
		public:
			using instance_t = InstanceT;
			using solution_t = SolutionT;

			inline static solution_t solve(const instance_t& instance) {
				// prepare storage
				solution_t res(instance.size());
				std::vector<typename instance_t::weight_type> weights(instance.dim(), 0);

				// get sorted elements
				struct elem {
					indexT id;
					float score;
				};
				std::vector<elem> sorted(instance.size());
				for (indexT i = 0; i < instance.size(); ++i) {
					sorted[i] = elem{ i, 
						static_cast<float>(instance.value(i)) / static_cast<float>(instance.item(i).total_weight())
					};
				}
				std::sort(sorted.begin(), sorted.end(), [](elem a, elem b) { return a.score > b.score; });

				// solve
				for (const auto& item : sorted) {
					if (would_fit(weights, instance.weights(item.id), instance.limits())) {
						res.add(item.id);
						if (!is_path_possible(instance, res)) res.remove(item.id);
						else add_weights(weights, instance.weights(item.id));
					}
				}

				// return
				return res;
			}
		};
	}
}

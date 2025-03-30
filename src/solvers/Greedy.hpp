#pragma once
#include <vector>
#include <algorithm>

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
				std::vector<typename instance_t::weight_t> weights(instance.dim(), 0);

				// get sorted elements
				struct elem {
					indexT id;
					float score;
				};
				std::vector<elem> sorted(instance.size());
				for (indexT i = 0; i < instance.size(); ++i) {
					sorted[i] = elem{ i, 
						static_cast<float>(instance.value(i)) / static_cast<float>(instance[i].weights.total())
					};
				}
				std::sort(sorted.begin(), sorted.end(), [](elem a, elem b) { return a.score > b.score; });

				// solve
				for (const auto& item : sorted) {
					for (size_t i = 0; i < instance.dim(); ++i) {
						if (weights[i] + instance.weight(item.id, i) > instance.limit(i)) return res;
						weights[i] += instance.weight(item.id, i);
					}
					res.add(item.id);
				}

				// return
				return res;
			}
		};
	}
}

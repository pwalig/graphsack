#pragma once
#include <vector>
#include <random>
#include <cassert>

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

			using metric_function = metric::function<metricT, instance_t>;

			inline static solution_t solve(
				const instance_t& instance,
				RandomEngine& randomEngine,
				size_t choose_from,
				bool (*structure_check) (const instance_t&, const solution_t&)
			) {
				assert(choose_from > 0);

				solution_t res(instance.size());
				std::vector<typename instance_t::weight_type> remaining(instance.limits().begin(), instance.limits().end());

				// get sorted elements
				std::vector<indexT> sorted = metric::sorted_indexes<metricT, instance_t, indexT>(instance);

				// solve
				while (sorted.size() > 0) {
					std::uniform_int_distribution<indexT> distrib(0, std::min(choose_from, sorted.size() - 1));
					indexT pick = distrib(randomEngine);
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
			
			inline static solution_t solve(
				const instance_t& instance,
				RandomEngine& randomEngine,
				size_t choose_from
			) {
				switch (instance.structure_to_find()) {
				case structure::none:
					return solve(instance, randomEngine, choose_from, [](const instance_t&, const solution_t&) {return true; });
					break;
				case structure::path:
					return solve(instance, randomEngine, choose_from, is_path_possible);
					break;
				case structure::cycle:
					return solve(instance, randomEngine, choose_from, is_cycle_possible);
					break;
				default:
					throw std::logic_error("invalid structure");
					break;
				}
			}

			inline static solution_t solve(
				const instance_t& instance,
				RandomEngine& randomEngine,
				float choose_from
			) {
				return solve(instance, randomEngine, (size_t)(instance.size() * choose_from));
			}
		};
	}
}

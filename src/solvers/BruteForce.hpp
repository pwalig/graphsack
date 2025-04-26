#pragma once
#include <vector>

#include "../structure_check.hpp"
#include "../weight_vector_operations.hpp"
#include "structure_to_find_dispatch.hpp"

namespace gs {
	namespace solver {
		template <typename InstanceT, typename SolutionT>
		class BruteForce {
		public:
			using instance_t = InstanceT;
			using solution_t = SolutionT;
			inline static const std::string name = "BruteForce";

			inline static solution_t solutionFromNumber(
				const instance_t& instance,
				size_t n
			) {
				solution_t res(instance.size());
				typename instance_t::size_type i = instance.size();
				while (n > 0) {
					--i;
					if (n % 2 == 1) res.add(i);
					n /= 2;
				}
				return res;
			}

			inline static solution_t solve(
				const instance_t& instance,
				bool (*structure_check) (const instance_t&, const solution_t&)
			) {
				using value_type = typename instance_t::value_type;

				solution_t best_solution(instance.size());
				value_type best_value = 0;

				size_t n = (size_t)std::pow(2, instance.size());

				for (size_t i = 0; i < n; ++i) {
					auto solution = solutionFromNumber(instance, i);
					typename instance_t::value_type value = 0;
					std::vector<typename instance_t::value_type> weights(instance.dim(), 0);
					for (size_t i = 0; i < instance.size(); ++i) {
						if (solution.has(i)) {
							value += instance.value(i);
							add_to_weights(weights, instance.weights(i));
						}
					}
					if (value > best_value && structure_check(instance, solution) && fits(weights, instance.limits())) {
						best_solution = solution;
						best_value = value;
					}
				}

				return best_solution;
			}

			solve_with_structure_to_find_dispatch()
		};
	}
}

#pragma once
#include <vector>
#include <string>

#include "../structure_check.hpp"
#include "../weight_vector_operations.hpp"

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
				using weight_type = typename instance_t::weight_type;
				using index_type = typename instance_t::index_type;

				size_t best_solution = 0;
				value_type best_value = 0;

				size_t n = size_t(1) << instance.size();

				for (size_t solution = 0; solution < n; ++solution) {
					value_type value = 0;
					std::vector<weight_type> weights(instance.dim(), 0);
					
					index_type itemId = instance.size();
					size_t number = solution;
					while (number > 0) {
						--itemId;
						if (number % 2 == 1) {
							value += instance.value(itemId);
							add_to_weights(weights, instance.weights(itemId));
						}
						number /= 2;
					}

					if (value > best_value && fits(weights, instance.limits()) && structure_check(instance, solution)) {
						best_solution = solution;
						best_value = value;
					}
				}

				return solutionFromNumber(instance, best_solution);
			}

			inline static solution_t solve(const instance_t& instance, bool iterative_structure_check = false) {
				if (instance.weight_treatment() != weight_treatment::full)
					throw std::invalid_argument("BruteForce can only solve for full weight treatment");

				switch (instance.structure_to_find()) {
				case structure::none:
					return solve(instance, [](const instance_t&, const solution_t&) {return true; });
					break;
				case structure::path:
					return solve(instance, is_path);
					break;
				case structure::cycle:
					if (iterative_structure_check) return solve(instance, is_cycle_iterative);
					else return solve(instance, is_cycle_recursive);
					break;
				default:
					throw std::logic_error("invalid structure");
					break;
				}
			}
		};
	}
}

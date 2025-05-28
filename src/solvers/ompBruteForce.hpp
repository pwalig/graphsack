#pragma once
#include <vector>
#include <string>

#include "../structure_check.hpp"
#include "../weight_vector_operations.hpp"

namespace gs {
	namespace solver {
		template <typename InstanceT, typename SolutionT>
		class ompBruteForce {
		public:
			using instance_t = InstanceT;
			using solution_t = SolutionT;
			inline static const std::string name = "ompBruteForce";

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

				size_t n = (size_t)std::pow(2, instance.size());
				solution_t best_solution(instance.size());
				value_type best_value = 0;

				#pragma omp parallel
				{
					solution_t local_best_solution(instance.size());
					value_type local_best_value = 0;

					#pragma omp for
					for (long long i = 0; i < n; ++i) {
						auto solution = solutionFromNumber(instance, i);
						typename instance_t::value_type value = 0;
						std::vector<typename instance_t::value_type> weights(instance.dim(), 0);
						for (size_t i = 0; i < instance.size(); ++i) {
							if (solution.has(i)) {
								value += instance.value(i);
								add_to_weights(weights, instance.weights(i));
							}
						}
						if (value > local_best_value && fits(weights, instance.limits()) && structure_check(instance, solution)) {
							local_best_solution = solution;
							local_best_value = value;
						}
					}

					#pragma omp critical
					{
						if (local_best_value > best_value) {
							best_value = local_best_value;
							best_solution = local_best_solution;
						}
					}
				}


				return best_solution;
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

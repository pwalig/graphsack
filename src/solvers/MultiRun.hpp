#pragma once
#include <string>

#include "../Validator.hpp"

namespace gs {
	namespace solver {
		template <typename Solver>
		class MultiRun {
		public:
			using instance_t = typename Solver::instance_t;
			using solution_t = typename Solver::solution_t;
			inline static const std::string name = "Multirun<" + Solver::name + ">";

			template <typename ...Args>
			inline static solution_t solve(
				const instance_t& instance,
				size_t iterations,
				Args... args
			) {
				solution_t best(instance.size());
				typename instance_t::value_type best_value = std::numeric_limits<typename instance_t::value_type>::min();
				for (size_t i = 0; i < iterations; ++i) {
					solution_t solution = Solver::solve(instance, args...);
					typename instance_t::value_type value = Validator<instance_t, solution_t>::getResultValue(instance, solution);
					if (value > best_value) {
						best_value = value;
						best = solution;
					}
				}
				return best;
			}

			template <typename ...Args>
			inline static solution_t solve(
				const instance_t& instance,
				float coverage,
				Args... args
			) {
				return solve<Args...>(instance, (size_t)(instance.size() * coverage), args...);
			}
		};
	}
}

#pragma once
#include <vector>
#include <stdexcept>

#include "../Validator.hpp"
#include "../weight_vector_operations.hpp"

namespace gs {
	namespace solver {
		template <typename InstanceT, typename SolutionT>
		class PathBruteForce {
		public:
			using instance_t = InstanceT;
			using solution_t = SolutionT;
			inline static const std::string name = "PathBruteForce";

			inline static solution_t dfs(
				const instance_t& instance,
				const solution_t& current_solution,
				const std::vector<typename instance_t::weight_type> remaining,
				typename instance_t::index_type current_item_id
			) {
				solution_t best_solution = current_solution;
				typename instance_t::value_type best_value = Validator<instance_t, solution_t>::getResultValue(instance, current_solution);

				for (typename instance_t::index_type itemId : instance.nexts(current_item_id)) {
					if (current_solution.has(itemId)) continue; // skip if already added

					if (fits(instance.weights(itemId), remaining)) {
						solution_t temp_solution = current_solution;
						temp_solution.add(itemId);
						std::vector<typename instance_t::weight_type> temp_remaining = remaining;
						sub_from_weights(temp_remaining, instance.weights(itemId));
						temp_solution = dfs(instance, temp_solution, temp_remaining, itemId);
						typename instance_t::value_type value = Validator<instance_t, solution_t>::getResultValue(instance, temp_solution);
						if (value > best_value) {
							best_value = value;
							best_solution = temp_solution;
						}
					}
				}

				return best_solution;
			}

			inline static solution_t solve(
				const instance_t& instance
			) {
				if (instance.structure_to_find() != structure::path)
					throw std::invalid_argument("PathBruteForce can only solve for path structure requirement");
				solution_t best_solution(instance.size());
				typename instance_t::value_type best_value = 0;
				std::vector<typename instance_t::weight_type> tmp_remaining(instance.limits().begin(), instance.limits().end());

				for (typename instance_t::index_type i = 0; i < instance.size(); ++i) {
					solution_t solution = dfs(
						instance,
						solution_t(instance.size()),
						tmp_remaining, i
					);
					typename instance_t::value_type value = Validator<instance_t, solution_t>::getResultValue(instance, solution);
					if (value > best_value) {
						best_value = value;
						best_solution = solution;
					}
				}
				return best_solution;
			}
		};
	}
}

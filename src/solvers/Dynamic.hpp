#pragma once
#include <string>
#include <stdexcept>
#include <vector>
#include <cassert>

#include "../structure.hpp"
#include "../weight_treatment.hpp"
#include "../Validator.hpp"

namespace gs {
	namespace solver {
		template <typename InstanceT, typename SolutionT>
		class Dynamic {
		public:
			using instance_t = InstanceT;
			using solution_t = SolutionT;
			inline static const std::string name = "Dynamic";

			inline static solution_t solve(
				const instance_t& instance
			) {
				if (instance.structure_to_find() != structure::none)
					throw std::invalid_argument("Dynamic solver can only solve instances with no structure requirements");
				if (instance.weight_treatment() != weight_treatment::first_only || instance.dim() == 1)
					throw std::invalid_argument("Dynamic solver can only solve instances where limit vector is of size 1");

				using size_type = typename instance_t::size_type;
				using value_type = typename instance_t::value_type;
				size_type n = instance.size();
				size_type rl = instance.limit() + 1;

				// Solution
				std::vector<value_type> dp((n + 1) * (instance.limit() + 1), 0); // initialise dynamic programming 2D NxP array with zeros
				for (size_type i = 1; i <= n; ++i){
					for (size_type j = 1; j <= instance.limit(); ++j) {
						if (instance.weight(i-1) <= j)
							dp[i * rl + j] = std::max(instance.value(i-1) + dp[(i-1) * rl + j - instance.weight(i-1)], dp[(i-1) * rl + j]);
						else
							dp[i * rl + j] = dp[(i-1) * rl + j];
					}
				}

				// Back tracking
				solution_t s(instance.size());
				value_type res = dp[n * rl + instance.limit()];

				typename instance_t::weight_type w = instance.limit();
				for (size_t i = n; i > 0 && res > 0; --i) {
					 
					// either the result comes from the top
					// (K[i-1][w]) or from (val[i-1] + K[i-1]
					// [w-wt[i-1]]) as in Knapsack table. If
					// it comes from the latter one/ it means
					// the item is included.
					if (res == dp[(i-1) * rl + w])
						continue;    
					else {
						// This item is included.
						s.add(i - 1);
						 
						// Since this weight is included its
						// value is deducted
						assert(instance.value(i - 1) <= res);
						assert(instance.weight(i - 1) <= w);
						res -= instance.value(i - 1);
						w -= instance.weight(i - 1);
					}
				}

				return s;
			}
		};
	}
}

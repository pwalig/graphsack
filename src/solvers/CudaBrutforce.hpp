#pragma once
#include <vector>

#include "../bit_vector.hpp"

namespace gs {
	namespace solver {
		namespace cuda {
			namespace brute_force {
				// data should be: limits | values | weights
				uint32_t runner_u32_u32(uint32_t* data, uint32_t N, uint32_t M);
			}

			template <typename InstanceT, typename SolutionT>
			class BruteForce {
			public:
				using solution_t = SolutionT;
				using instance_t = InstanceT;
				const static std::string name;

				BruteForce() = delete;
				inline static solution_t solve(const instance_t& instance) 
				{
					assert(instance.size() <= 10);
					std::vector<uint32_t> data(instance.dim() * instance.size() + instance.dim() + instance.size());

					// copy instance to data
					auto it = std::copy(instance.limits().begin(), instance.limits().end(), data.begin());
					for (typename instance_t::size_type i = 0; i < instance.size(); ++i) (*(it++)) = instance.value(i);
					for (typename instance_t::size_type i = 0; i < instance.size(); ++i) it = std::copy(instance.weights(i).begin(), instance.weights(i).end(), it);

					// run kernel
					uint32_t res = brute_force::runner_u32_u32(data.data(), instance.size(), instance.dim());

					// rewrite to solution
					solution_t solution(instance.size());
					uint32_t i = instance.size();
					while (i > 0) {
						--i;
						if (res % 2 == 1) solution.add(i);
						res /= 2;
					}
					return solution;
				}
			};
		}
	}
}

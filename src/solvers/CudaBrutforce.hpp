#pragma once
#include <vector>
#include "../structure.hpp"
#include "../res/cuda_solution.hpp"
#include "../inst/cuda_instance.hpp"

namespace gs {
	namespace cuda {
		namespace solver {
			namespace brute_force {
				// data should be: limits | values | weights
				uint32_t runner_u32_u32(uint32_t* data, uint32_t N, uint32_t M, uint32_t threadsPerBlock, uint32_t share, structure to_find);
				//res::solution64 runner_u32_u32(
				//	const inst::instance64<uint32_t, uint32_t>& data,
				//	uint32_t threadsPerBlock, uint32_t share
				//);
			}

			//class BruteForce64 {
			//public:
			//	using solution_t = res::solution64;
			//	using instance_t = inst::instance64<uint32_t, uint32_t>;
			//	const static std::string name;

			//	BruteForce64() = delete;
			//};

			//const std::string BruteForce64::name = "CudaBruteForce64";

			template <typename InstanceT, typename SolutionT>
			class BruteForce {
			public:
				using solution_t = SolutionT;
				using instance_t = InstanceT;
				const static std::string name;

				BruteForce() = delete;
				inline static solution_t solve(const instance_t& instance, uint32_t threadsPerBlock = 1024, uint32_t share = 1) 
				{
					std::vector<uint32_t> data(instance.dim() * instance.size() + instance.dim() + instance.size() + instance.size(), 0);

					// copy instance to data
					auto it = std::copy(instance.limits().begin(), instance.limits().end(), data.begin());
					for (typename instance_t::size_type i = 0; i < instance.size(); ++i) (*(it++)) = instance.value(i);
					for (typename instance_t::size_type i = 0; i < instance.size(); ++i) it = std::copy(instance.weights(i).begin(), instance.weights(i).end(), it);
					for (typename instance_t::size_type i = 0; i < instance.size(); ++i) {
						for (auto next : instance.nexts(i)) {
							*it |= (1 << next);
						}
						++it;
					}

					// run kernel
					uint32_t res = brute_force::runner_u32_u32(data.data(), (uint32_t)instance.size(), (uint32_t)instance.dim(), threadsPerBlock, share, instance.structure_to_find());

					// rewrite to solution
					solution_t solution(instance.size());
					size_t i = 0;
					while (i < instance.size()) {
						if (res % 2 == 1) solution.add(i);
						res /= 2;
						i++;
					}
					return solution;
				}
			};

			template<typename InstanceT, typename SolutionT>
			const std::string BruteForce<InstanceT, SolutionT>::name = "CudaBruteForce";
		}
	}
}


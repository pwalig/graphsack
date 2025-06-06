#pragma once
#include <vector>
#include "size_string.hpp"
#include "../structure.hpp"
#include "../res/cuda_solution.hpp"
#include "../inst/cuda_instance.hpp"

namespace gs {
	namespace cuda {
		namespace solver {
			namespace brute_force {
				template <typename instance_t>
				res::solution<typename instance_t::adjacency_base_type> runner(
					const instance_t& instance,
					uint32_t threadsPerBlock, uint32_t share
				);

				extern template res::solution32 runner(const inst::instance32<uint32_t, uint32_t>&, uint32_t, uint32_t);
				extern template res::solution32 runner(const inst::instance32<float, uint32_t>&, uint32_t, uint32_t);
				extern template res::solution32 runner(const inst::instance32<uint32_t, float>&, uint32_t, uint32_t);
				extern template res::solution32 runner(const inst::instance32<float, float>&, uint32_t, uint32_t);
				extern template res::solution64 runner(const inst::instance64<uint32_t, uint32_t>&, uint32_t, uint32_t);
				extern template res::solution64 runner(const inst::instance64<float, uint32_t>&, uint32_t, uint32_t);
				extern template res::solution64 runner(const inst::instance64<uint32_t, float>&, uint32_t, uint32_t);
				extern template res::solution64 runner(const inst::instance64<float, float>&, uint32_t, uint32_t);
			}

			// maximum percentage of global memory to use
			extern double global_mem_percent;

			// uint32_t threadsPerBlock = 0
			// uint32_t share = 1 
			template <typename InstanceT>
			class BruteForce {
			public:
				using instance_t = InstanceT;
				using storage_t = typename instance_t::adjacency_base_type;
				using solution_t = res::solution<storage_t>;
				const static std::string name;

				BruteForce() = delete;

				inline static solution_t solve(const instance_t& instance, uint32_t threadsPerBlock = 0, uint32_t share = 1) 
				{
					return brute_force::runner<instance_t>(instance, threadsPerBlock, share);
				}
			};
			template <typename InstanceT>
			const std::string BruteForce<InstanceT>::name = std::string("cudaBruteForce") + size_string<typename BruteForce<InstanceT>::storage_t>();

		}
	}
}


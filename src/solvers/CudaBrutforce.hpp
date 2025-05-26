#pragma once
#include <vector>
#include "../structure.hpp"
#include "../res/cuda_solution.hpp"
#include "../inst/cuda_instance.hpp"

namespace gs {
	namespace cuda {
		namespace solver {
			namespace brute_force {
				res::solution32 runner32(
					const inst::instance32<uint32_t, uint32_t>& instance,
					uint32_t threadsPerBlock, uint32_t share
				);
				res::solution64 runner64(
					const inst::instance64<uint32_t, uint32_t>& instance,
					uint32_t threadsPerBlock, uint32_t share
				);
			}

			class BruteForce32 {
			public:
				using solution_t = res::solution32;
				using instance_t = inst::instance32<uint32_t, uint32_t>;
				const static std::string name;

				BruteForce32() = delete;

				inline static solution_t solve(const instance_t& instance, uint32_t threadsPerBlock = 1024, uint32_t share = 1) 
				{
					return brute_force::runner32(instance, threadsPerBlock, share);
				}
			};


			class BruteForce64 {
			public:
				using solution_t = res::solution64;
				using instance_t = inst::instance64<uint32_t, uint32_t>;
				const static std::string name;

				BruteForce64() = delete;

				inline static solution_t solve(const instance_t& instance, uint32_t threadsPerBlock = 1024, uint32_t share = 1) 
				{
					return brute_force::runner64(instance, threadsPerBlock, share);
				}
			};

		}
	}
}


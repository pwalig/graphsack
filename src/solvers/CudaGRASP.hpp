#pragma once
#include <vector>
#include "../structure.hpp"
#include "../res/cuda_solution.hpp"
#include "../inst/cuda_instance.hpp"

namespace gs {
	namespace cuda {
		namespace solver {
			namespace grasp {
				// data should be: limits | values | weights
				res::solution32 runner32(
					const inst::instance32<uint32_t, uint32_t>& instance,
					uint32_t blocksCount
				);
				res::solution64 runner64(
					const inst::instance64<uint32_t, uint32_t>& instance,
					uint32_t blocksCount
				);
			}

			class GRASP32 {
			public:
				using solution_t = res::solution32;
				using instance_t = inst::instance32<uint32_t, uint32_t>;
				const static std::string name;

				GRASP32() = delete;

				inline static solution_t solve(const instance_t& instance, uint32_t blocksCount = 1) 
				{
					return grasp::runner32(instance, blocksCount);
				}
			};


			class GRASP64 {
			public:
				using solution_t = res::solution64;
				using instance_t = inst::instance64<uint32_t, uint32_t>;
				const static std::string name;

				GRASP64() = delete;

				inline static solution_t solve(const instance_t& instance, uint32_t blocksCount = 1) 
				{
					return grasp::runner64(instance, blocksCount);
				}
			};
		}
	}
}


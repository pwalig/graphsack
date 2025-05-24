#pragma once
#include <device_launch_parameters.h>
#include <cstdint>

namespace gs {
	namespace cuda {
		namespace res {
			__device__ bool has(uint64_t solution, uint32_t itemId);
			__device__ bool not_has(uint64_t solution, uint32_t itemId);
			__device__ void add(uint64_t& solution, uint32_t itemId);
			__device__ void remove(uint64_t& solution, uint32_t itemId);
		}
	}
}

#pragma once
#include <device_launch_parameters.h>
#include <cstdint>

namespace gs {
	namespace cuda {
		__device__  bool has_connection_to(const uint64_t* adjacency, uint32_t from, uint32_t to);
		__device__ bool is_cycle_DFS(
			const uint64_t* adjacency,
			uint32_t N, uint64_t selected, uint64_t visited,
			uint32_t current, uint32_t start, uint32_t length, uint32_t depth
		);
		__device__ bool is_cycle_recursive(
			const uint64_t* adjacency,
			uint64_t selected, uint32_t N
		);
		__device__ bool is_cycle_iterative_helper(
			const uint64_t* adjacency,
			uint32_t* stack_memory,
			uint64_t selected, uint32_t N,
			uint32_t start, uint32_t length
		);
		__device__ bool is_cycle_iterative(
			const uint64_t* adjacency,
			uint32_t* stack_memory,
			uint64_t selected, uint32_t N
		);
	}
}
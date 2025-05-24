#include "cuda_structure_check.cuh"
#include "res/cuda_solution.cuh"


// does not work because cuda refuses to enter if (true) block
__device__ bool gs::cuda::is_cycle_iterative(
	const uint64_t* adjacency,
	uint32_t* stack_memory,
	uint64_t selected, uint32_t N
) {
	// calculate whats the length of the cycle
	uint32_t length = 0;
	for (uint32_t i = 0; i < N; ++i)
		if (res::has(selected, i)) length++;
		
	if (length == 0) return true;

	// check from each starting point
	for (uint32_t start = 0; start < N; ++start) {
		if (res::has(selected, start)) {
			if (is_cycle_iterative_helper(adjacency, stack_memory, selected, N, start, length)) return true;
		}
	}
	return false;
}

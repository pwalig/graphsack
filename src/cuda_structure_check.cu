#include "cuda_structure_check.cuh"
#include "res/cuda_solution.cuh"

#define GS_CUDA_MAX_RECURSION 10

__device__  bool gs::cuda::has_connection_to(const uint64_t* adjacency, uint32_t from, uint32_t to) {
	if (adjacency[from] & (uint64_t(1) << to)) return true;
	else return false;
}

__device__ bool gs::cuda::is_cycle_DFS(
	const uint64_t* adjacency,
	uint32_t N, uint64_t selected, uint64_t visited,
	uint32_t current, uint32_t start, uint32_t length, uint32_t depth
) {
	//if (depth > GS_CUDA_MAX_RECURSION) return false;
	for (uint32_t next = 0; next < N; ++next) {
		if (!has_connection_to(adjacency, current, next)) continue;
		if (res::has(selected, next) && !res::has(visited, next)) { // next item has to be selected and new
			res::add(visited, next);
			if (depth == length && has_connection_to(adjacency, next, start)) return true;
			if (depth > length) return false;
			if (is_cycle_DFS(adjacency, N, selected, visited, next, start, length, depth + 1)) return true;
			res::remove(visited, next);
		}
	}
	return false;
}

__device__ bool gs::cuda::is_cycle_recursive(
	const uint64_t* adjacency,
	uint64_t selected, uint32_t N
) {

	// calculate whats the length of the cycle
	uint32_t length = 0;
	for (uint32_t i = 0; i < N; ++i)
		if (res::has(selected, i)) length++;
		
	if (length == 0) return true;

	// check from each starting point
	uint64_t visited = 0;
	for (uint32_t i = 0; i < N; ++i) {
		if (res::has(selected, i)){
			if (length == 1) return has_connection_to(adjacency, i, i);
			res::add(visited, i);
			if (is_cycle_DFS(adjacency, N, selected, visited, i, i, length, 2)) return true;
			res::remove(visited, i);
		}
	}
	return false;
}

// does not work because cuda refuses to enter if (true) block
__device__ bool gs::cuda::is_cycle_iterative(
	const uint64_t* adjacency,
	uint32_t* stack_memory,
	uint64_t selected, uint32_t N
) {
	uint32_t stack_pointer = 0;

	// calculate whats the length of the cycle
	uint32_t length = 0;
	for (uint32_t i = 0; i < N; ++i)
		if (res::has(selected, i)) length++;
		
	if (length == 0) return true;

	// check from each starting point
	uint64_t visited = 0;
	uint32_t visitedCount = 0;
	for (uint32_t start = 0; start < N; ++start) {
		if (res::has(selected, start)) {
			uint32_t prev = start;
			while (true) {

				// visiting for the first time
				if (!res::has(visited, prev)) {
					res::add(visited, prev);
					++visitedCount;
					if (visitedCount == length && has_connection_to(adjacency, prev, start))
						return true;
					if (visitedCount > length) stack_memory[stack_pointer++] = N;
					else stack_memory[stack_pointer++] = 0;
				}

				uint32_t next = stack_memory[--stack_pointer];
				for (; next < N; ++next) {
					if (!has_connection_to(adjacency, prev, next)) continue;
					if (res::has(selected, next) && !res::has(visited, next)) {
						stack_memory[stack_pointer++] = next + 1;
						prev = next;
						break;
					}
				}

				// visited all nexts
				if (next == N) {
					res::remove(visited, prev);
					--visitedCount;
					if (stack_pointer == 0)
						prev = stack_memory[stack_pointer] - 1;
					else break;
				}
			}
		}
	}
	return false;
}

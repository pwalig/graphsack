#pragma once
#include <cstdint>
#include <device_launch_parameters.h>

#include "res/cuda_solution.cuh"

//#define GS_CUDA_MAX_RECURSION 10

namespace gs {
	namespace cuda {
		template <typename adjacency_base_type, typename index_type>
		inline __device__  bool has_connection_to(const adjacency_base_type* adjacency, index_type from, index_type to) {
			if (adjacency[from] & (adjacency_base_type(1) << to)) return true;
			else return false;
		}

		template <typename adjacency_base_type, typename index_type>
		__device__ bool is_cycle_DFS(
			const adjacency_base_type* adjacency,
			index_type N, adjacency_base_type selected, adjacency_base_type visited,
			index_type current, index_type start, index_type length, index_type depth
		) {
#ifdef GS_CUDA_MAX_RECURSION
			if (depth > GS_CUDA_MAX_RECURSION) return false;
#endif
			for (index_type next = 0; next < N; ++next) {
				if (!has_connection_to(adjacency, current, next)) continue;
				if (res::has(selected, next) && !res::has(visited, next)) { // next item has to be selected and new
					res::add(visited, next);
					if (depth == length && has_connection_to(adjacency, next, start)) return true;
					if (depth > length) return false;
					if (is_cycle_DFS<adjacency_base_type, index_type>(
						adjacency, N, selected, visited, next, start, length, depth + 1
					)) return true;
					res::remove(visited, next);
				}
			}
			return false;
		}

		template <typename adjacency_base_type, typename index_type>
		__device__ bool is_cycle_recursive(
			const adjacency_base_type* adjacency,
			adjacency_base_type selected, index_type N
		) {

			// calculate whats the length of the cycle
			index_type length = 0;
			for (index_type i = 0; i < N; ++i)
				if (res::has(selected, i)) length++;
				
			if (length == 0) return true;

			// check from each starting point
			adjacency_base_type visited = 0;
			for (index_type i = 0; i < N; ++i) {
				if (res::has(selected, i)){
					if (length == 1) return has_connection_to<adjacency_base_type, index_type>(adjacency, i, i);
					res::add(visited, i);
					if (is_cycle_DFS<adjacency_base_type, index_type>(
						adjacency, N, selected, visited, i, i, length, 2
					)) return true;
					res::remove(visited, i);
				}
			}
			return false;
		}

		template <typename adjacency_base_type, typename index_type>
		__device__ bool is_cycle_iterative_helper(
			const adjacency_base_type* adjacency,
			index_type* stack_memory,
			adjacency_base_type selected, index_type N,
			index_type start, index_type length
		) {
			adjacency_base_type visited = 0;
			index_type visitedCount = 0;
			index_type stack_pointer = 0;

			index_type prev = start;
			while (true) {

				// visiting for the first time
				if (!res::has(visited, prev)) {
					res::add(visited, prev);
					++visitedCount;
					if (visitedCount == length && has_connection_to(adjacency, prev, start))
						return true;
					if (visitedCount > length) {
						stack_memory[stack_pointer++] = prev;
						stack_memory[stack_pointer++] = N;
					}
					else {
						stack_memory[stack_pointer++] = prev;
						stack_memory[stack_pointer++] = 0;
					}
				}

				index_type next = stack_memory[--stack_pointer] + 1;
				--stack_pointer;
				for (; next < N; ++next) {
					if (!has_connection_to(adjacency, prev, next)) continue;
					if (res::has(selected, next) && !res::has(visited, next)) {
						stack_memory[stack_pointer++] = prev;
						stack_memory[stack_pointer++] = next;
						prev = next;
						break;
					}
				}

				// visited all nexts
				if (next == N) {
					res::remove(visited, prev);
					--visitedCount;
					if (stack_pointer == 0) break;
					prev = stack_memory[stack_pointer - 2];
				}
			}
			return false;
		}

		template <typename adjacency_base_type, typename index_type>
		__device__ bool is_cycle_iterative(
			const adjacency_base_type* adjacency,
			index_type* stack_memory,
			adjacency_base_type selected, index_type N
		) {
			// calculate whats the length of the cycle
			index_type length = 0;
			for (index_type i = 0; i < N; ++i)
				if (res::has(selected, i)) length++;
				
			if (length == 0) return true;

			// check from each starting point
			for (index_type start = 0; start < N; ++start) {
				if (res::has(selected, start)) {
					if (is_cycle_iterative_helper<adjacency_base_type, index_type>(
						adjacency, stack_memory, selected, N, start, length
					)) return true;
				}
			}
			return false;
		}
	}
}
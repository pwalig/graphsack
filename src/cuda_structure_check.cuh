#pragma once
#include <cstdint>
#include <device_launch_parameters.h>

#include "res/cuda_solution.cuh"
#include "inst/cuda_instance.cuh"

//#define GS_CUDA_MAX_RECURSION 10

namespace gs {
	namespace cuda {
		template <typename result_type, typename weight_type, typename index_type>
		inline __device__ bool is_cycle_possible_DFS(
			const result_type* adjacency, const weight_type* weights,
			result_type selected,
			result_type visited,
			const weight_type _remaining_space[GS_CUDA_INST_MAXM],
			index_type current, index_type start, index_type N, uint32_t M
		) {
			for (index_type next = 0; next < N; ++next) {
				if (!inst::has_connection_to(adjacency, current, next)) continue;
				if (res::has(visited, next)) continue; // next item has to be new (not visited yet)

				bool fit = true;
				weight_type new_remaining_space[GS_CUDA_INST_MAXM];
				for (uint32_t j = 0; j < M; ++j) {
					if (_remaining_space[j] >= weights[next * M + j])
						new_remaining_space[j] = _remaining_space[j] - weights[next * M + j];
					else { fit = false; break; }
				}
				if (!fit) continue;

				res::add(visited, next);
				if (inst::has_connection_to(adjacency, next, start)) { // is it a cycle (closed path)
					bool _found = true; // found some cycle lets check if it has all selected vertices
					for (index_type i = 0; i < N; ++i) {
						if (res::has(selected, i) && !res::has(visited, i)) {
							_found = false;
							break;
						}
					}
					if (_found) return true; // it has - cycle found
				}
				if (is_cycle_possible_DFS<result_type, weight_type, index_type>(
					adjacency, weights, selected, visited, new_remaining_space, next, start, N, M)
				) return true; // cycle found later
				res::remove(visited, next);
			}
			return false; // cycle not found
		}

		template <typename result_type, typename weight_type, typename index_type>
		inline __device__ bool is_cycle_possible_recursive(
			const result_type* adjacency, const weight_type* weights, const weight_type* limits,
			result_type selected, index_type N, uint32_t M
		) {
			result_type visited = 0;
			for (index_type i = 0; i < N; ++i) {

				bool fit = true;
				weight_type _remaining_space[GS_CUDA_INST_MAXM];
				for (uint32_t j = 0; j < M; ++j) {
					if (limits[j] >= weights[i * M + j]) _remaining_space[j] = limits[j] - weights[i * M + j];
					else { fit = false; break; }
				}
				if (!fit) continue;

				res::add(visited, i);
				if (inst::has_connection_to(adjacency, i, i)) { // is it a cycle (closed path)
					bool _found = true; // found some cycle lets check if it has all selected vertices
					for (index_type j = 0; j < N; ++j) {
						if (res::has(selected, j) && j != i) {
							_found = false;
							break;
						}
					}
					if (_found) return true; // it has - cycle found
				}
				if (is_cycle_possible_DFS<result_type, weight_type, index_type>(
					adjacency, weights, selected, visited, _remaining_space, i, i, N, M)
				) return true; // cycle found somewhere
				res::remove(visited, i);
			}
			return false;
		}

		template <typename adjacency_base_type, typename index_type>
		inline __device__ bool is_cycle_DFS(
			const adjacency_base_type* adjacency,
			index_type N, adjacency_base_type selected, adjacency_base_type visited,
			index_type current, index_type start, index_type length, index_type depth
		) {
#ifdef GS_CUDA_MAX_RECURSION
			if (depth > GS_CUDA_MAX_RECURSION) return false;
#endif
			for (index_type next = 0; next < N; ++next) {
				if (!inst::has_connection_to(adjacency, current, next)) continue;
				if (res::has(selected, next) && !res::has(visited, next)) { // next item has to be selected and new
					res::add(visited, next);
					if (depth == length && inst::has_connection_to(adjacency, next, start)) return true;
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
		inline __device__ bool is_cycle_recursive(
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
					if (length == 1) return inst::has_connection_to<adjacency_base_type, index_type>(adjacency, i, i);
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
		inline __device__ bool is_cycle_iterative_helper(
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
					if (visitedCount == length && inst::has_connection_to(adjacency, prev, start))
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
					if (!inst::has_connection_to(adjacency, prev, next)) continue;
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
		inline __device__ bool is_cycle_iterative(
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

		template <typename adjacency_base_type, typename index_type>
		inline __device__ bool is_cycle_iterative(
			index_type* stack_memory,
			adjacency_base_type selected, index_type N
		) {
			is_cycle_iterative<adjacency_base_type, index_type>(inst::adjacency<adjacency_base_type>(), stack_memory, selected, N);
		}
	}
}
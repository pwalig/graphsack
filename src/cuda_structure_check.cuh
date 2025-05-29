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
			result_type selected,
			result_type visited,
			const weight_type* _remaining_space,
			index_type current, index_type start
		) {
			for (index_type next = 0; next < inst::size; ++next) {
				if (!inst::has_connection_to<result_type, index_type>(current, next)) continue;
				if (res::has(visited, next)) continue; // next item has to be new (not visited yet)

				// fit check
				weight_type new_remaining_space[GS_CUDA_INST_MAXM];
				uint32_t j = 0;
				for (; j < inst::dim; ++j) {
					if (_remaining_space[j] >= inst::weights<weight_type>()[next * inst::dim + j])
						new_remaining_space[j] = _remaining_space[j] - inst::weights<weight_type>()[next * inst::dim + j];
					else break;
				}
				if (j != inst::dim) continue;

				res::add(visited, next);

				if (inst::has_connection_to<result_type, index_type>(next, start)) {
					// completness check
					index_type i = 0;
					for (; i < inst::size; ++i) {
						if (res::has(selected, i) && !res::has(visited, i)) break;
					}
					if (i == inst::size) return true; // cycle found
				}

				// DFS call
				if (is_cycle_possible_DFS<result_type, weight_type, index_type>(
					selected, visited, new_remaining_space, next, start)
				) return true; // cycle found later

				res::remove(visited, next);
			}
			return false; // cycle not found
		}

		template <typename result_type, typename weight_type, typename index_type>
		inline __device__ bool is_cycle_possible_recursive(
			result_type selected
		) {
			result_type visited = 0;
			weight_type _remaining_space[GS_CUDA_INST_MAXM];

			for (index_type i = 0; i < inst::size; ++i) {

				// fit check
				uint32_t wid = 0;
				for (; wid < inst::dim; ++wid) {
					if (inst::limits<weight_type>()[wid] >= inst::weights<weight_type>()[i * inst::dim + wid])
						_remaining_space[wid] = inst::limits<weight_type>()[wid] - inst::weights<weight_type>()[i * inst::dim + wid];
					else break;
				}
				if (wid != inst::dim) continue;

				res::add(visited, i);

				if (inst::has_connection_to<result_type, index_type>(i, i)) {
					// completness check
					index_type j = 0;
					for (; j < inst::size; ++j) {
						if (res::has(selected, j) && j != i) break;
					}
					if (j == inst::size) return true; // cycle found
				}

				// DFS call
				if (is_cycle_possible_DFS<result_type, weight_type, index_type>(
					selected, visited, _remaining_space, i, i)
				) return true; // cycle found somewhere

				res::remove(visited, i);
			}
			return false;
		}

		template <typename adjacency_base_type, typename index_type>
		inline __device__ bool is_cycle_DFS(
			adjacency_base_type selected, adjacency_base_type visited,
			index_type current, index_type start, index_type length, index_type depth
		) {
#ifdef GS_CUDA_MAX_RECURSION
			if (depth > GS_CUDA_MAX_RECURSION) return false;
#endif
			for (index_type next = 0; next < inst::size; ++next) {
				if (!inst::has_connection_to<adjacency_base_type, index_type>(current, next)) continue;
				if (res::has(selected, next) && !res::has(visited, next)) { // next item has to be selected and new
					res::add(visited, next);
					if (depth == length && inst::has_connection_to<adjacency_base_type, index_type>(next, start)) return true;
					if (depth > length) return false;
					if (is_cycle_DFS<adjacency_base_type, index_type>(
						selected, visited, next, start, length, depth + 1
					)) return true;
					res::remove(visited, next);
				}
			}
			return false;
		}

		template <typename adjacency_base_type, typename index_type>
		inline __device__ bool is_cycle_recursive(
			adjacency_base_type selected
		) {

			// calculate whats the length of the cycle
			index_type length = 0;
			for (index_type i = 0; i < inst::size; ++i)
				if (res::has(selected, i)) length++;
				
			if (length == 0) return true;

			// check from each starting point
			adjacency_base_type visited = 0;
			for (index_type i = 0; i < inst::size; ++i) {
				if (res::has(selected, i)){
					if (length == 1) return inst::has_connection_to<adjacency_base_type, index_type>(i, i);
					res::add(visited, i);
					if (is_cycle_DFS<adjacency_base_type, index_type>(
						selected, visited, i, i, length, 2
					)) return true;
					res::remove(visited, i);
				}
			}
			return false;
		}

		template <typename adjacency_base_type, typename index_type>
		inline __device__ bool is_cycle_iterative_helper(
			index_type* stack_memory,
			adjacency_base_type selected,
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
					if (visitedCount == length && inst::has_connection_to<adjacency_base_type, index_type>(prev, start))
						return true;
					if (visitedCount > length) {
						stack_memory[stack_pointer++] = prev;
						stack_memory[stack_pointer++] = inst::size;
					}
					else {
						stack_memory[stack_pointer++] = prev;
						stack_memory[stack_pointer++] = 0;
					}
				}

				index_type next = stack_memory[--stack_pointer] + 1;
				--stack_pointer;
				for (; next < inst::size; ++next) {
					if (!inst::has_connection_to<adjacency_base_type, index_type>(prev, next)) continue;
					if (res::has(selected, next) && !res::has(visited, next)) {
						stack_memory[stack_pointer++] = prev;
						stack_memory[stack_pointer++] = next;
						prev = next;
						break;
					}
				}

				// visited all nexts
				if (next == inst::size) {
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
			index_type* stack_memory,
			adjacency_base_type selected
		) {
			// calculate whats the length of the cycle
			index_type length = 0;
			for (index_type i = 0; i < inst::size; ++i)
				if (res::has(selected, i)) length++;
				
			if (length == 0) return true;

			// check from each starting point
			for (index_type start = 0; start < inst::size; ++start) {
				if (res::has(selected, start)) {
					if (is_cycle_iterative_helper<adjacency_base_type, index_type>(
						stack_memory, selected, start, length
					)) return true;
				}
			}
			return false;
		}
	}
}
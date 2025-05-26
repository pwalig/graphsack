#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace gs {
	namespace cuda {
		namespace solver {
			namespace sort {

				template <typename index_type>
				inline __global__ void in_order(index_type* memory, index_type N) {
					index_type id = threadIdx.x;
					if (id < N) memory[id] = id;
				}
				template <typename index_type>
				inline __global__ void reverse_order(index_type* memory, index_type N) {
					index_type id = threadIdx.x;
					if (id < N) memory[id] = N - id - 1;
				}

				// does not work for some reason
				template <typename index_type, typename value_type>
				inline __global__ void by_value(index_type* index_memory, const value_type* value_memory, index_type N) {
					index_type i = blockIdx.x * blockDim.x + threadIdx.x;
					if (i > N) return;
					index_memory[i] = i;

					index_type n = 2;
					while (n < N) n *= 2;

					for (index_type k = 2; k <= n; k *= 2) {// k is doubled every iteration
						for (index_type j = k / 2; j > 0; j /= 2) { // j is halved at every iteration, with truncation of fractional parts
							__syncthreads();
							index_type l = i ^ j;
							if (l > i && l < N) {
								if (
									//((i & k == 0) && (value_memory[index_memory[i]] > value_memory[index_memory[l]])) ||
									//((i & k != 0) && (value_memory[index_memory[i]] < value_memory[index_memory[l]]))
									((i & k == 0) && (index_memory[i] < index_memory[l])) ||
									((i & k != 0) && (index_memory[i] > index_memory[l]))
								) {
									index_type tmp = index_memory[i];
									index_memory[i] = index_memory[l];
									index_memory[l] = tmp;
								}
							}
						}
					}

				}



			}
		}
	}
}
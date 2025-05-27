#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "../inst/cuda_instance.cuh"

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

				// has te be launched with smallest power of 2 thats larger or equal to N
				template <typename index_type, typename value_type>
				inline __global__ void by_value(index_type* index_memory, index_type N) {
					index_type i = blockIdx.x * blockDim.x + threadIdx.x;
					if (i < N) index_memory[i] = i;
					else index_memory[i] = 0;

					for (index_type k = 2; k <= blockDim.x; k *= 2) {// k is doubled every iteration
						for (index_type j = k / 2; j > 0; j /= 2) { // j is halved at every iteration, with truncation of fractional parts
							__syncthreads();
							index_type l = i ^ j;
							if (l > i) {
								index_type val = i & k;
								if (
									//((val == 0) && (inst::values<uint32_t>()[index_memory[i]] < inst::values<uint32_t>()[index_memory[l]])) ||
									//((val != 0) && (inst::values<uint32_t>()[index_memory[i]] > inst::values<uint32_t>()[index_memory[l]]))
									((val == 0) && (index_memory[i] < index_memory[l])) ||
									((val != 0) && (index_memory[i] > index_memory[l]))
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
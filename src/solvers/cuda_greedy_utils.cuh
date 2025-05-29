#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "../inst/cuda_instance.cuh"

namespace gs {
	namespace cuda {
		namespace solver {
			namespace metric {
				template <typename T>
				class Value {
				public:
					using value_type = T;
					inline __device__ static value_type get(size_t itemId) {
						if (itemId < inst::size) return inst::values<value_type>()[itemId];
						else return value_type(0);
					}
				};

				template <typename MetricT, typename index_type>
				inline __global__ void calculate(typename MetricT::value_type* memory) {
					index_type id = threadIdx.x;
					memory[id] = MetricT::get(id);
				}

			}
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
				template <typename index_type, typename metric_type>
				inline __global__ void by_metric_desc(
					index_type* index_memory, metric_type* metric_memory, index_type N
				) {
					index_type i = blockIdx.x * blockDim.x + threadIdx.x;

					// save to shared memory
					__shared__ index_type index[64];
					__shared__ metric_type metric[64];
					index[i] = i;
					metric[i] = metric_memory[i];

					// bitonic sort
					for (index_type k = 2; k <= blockDim.x; k *= 2) { // k is doubled every iteration
						for (index_type j = k / 2; j > 0; j /= 2) { // j is halved at every iteration, with truncation of fractional parts
							__syncthreads();
							index_type l = i ^ j;
							if (l > i) {
								index_type val = i & k;
								if (
									((val == 0) && (metric[index[i]] < metric[index[l]])) ||
									((val != 0) && (metric[index[i]] > metric[index[l]]))
								) {
									index_type tmp = index[i];
									index[i] = index[l];
									index[l] = tmp;
								}
							}
						}
					}

					// bring result back to global memory
					index_memory[i] = index[i];

				}



			}
		}
	}
}
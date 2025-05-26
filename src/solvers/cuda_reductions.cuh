#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define GS_CUDA_REDUCTIONS_PICK(result_type, thread_id, value_memory, result_memory) \
for (result_type n = 1; n < blockDim.x; n *= 2) { \
	__syncthreads(); \
	if (thread_id % (n*2) != 0) return; \
	if (value_memory[thread_id + n] > value_memory[thread_id]) { \
		result_memory[thread_id] = result_memory[thread_id + n]; \
		value_memory[thread_id] = value_memory[thread_id + n]; \
	} \
}

#define GS_CUDA_REDUCTIONS_PICK_STANDARD GS_CUDA_REDUCTIONS_PICK(result_type, id, value_memory, result_memory)

namespace gs {
	namespace cuda {
		namespace reductions {
			// should be called with 1 blocks per grid and `stride` / 2 threads per block
			// stride should be exactly threads per block of solver kernel
			template <typename ResT, typename ValueT>
			__global__ void pick(
				ValueT* value_memory,
				ResT* result_memory,
				uint32_t stride,
				ResT totalThreads
			) {
				ResT id = (ResT)(blockIdx.x * blockDim.x + threadIdx.x) * stride * 2;

				if (id > totalThreads) return;

				for (ResT n = stride; n < totalThreads; n *= 2) {
					__syncthreads();
					if (value_memory[id + n] > value_memory[id]) {
						result_memory[id] = result_memory[id + n];
						value_memory[id] = value_memory[id + n];
					}
					if (id % (n*4) != 0) return;
				}
			}

			// should be called with 1 blocks per grid and at least `stride` amount of threads
			// stride should be exactly threads per block of solver kernel
			template <typename ResT, typename ValueT>
			__global__ void shared_pick(
				ValueT* value_memory,
				ResT* result_memory,
				ResT stride,
				ResT totalThreads
			) {
				ResT id = (ResT)(blockIdx.x * blockDim.x + threadIdx.x) * (totalThreads / blockDim.x);

				if (id > totalThreads) return;

				ResT strideStride = stride;
				for (; stride < totalThreads / blockDim.x; stride += strideStride) {
					if (value_memory[id + stride] > value_memory[id]) {
						result_memory[id] = result_memory[id + stride];
						value_memory[id] = value_memory[id + stride];
					}
				}

				for (; stride < totalThreads; stride *= 2) {
					__syncthreads();
					if (id % (stride*2) != 0) return;
					if (value_memory[id + stride] > value_memory[id]) {
						result_memory[id] = result_memory[id + stride];
						value_memory[id] = value_memory[id + stride];
					}
				}
			}
		}
	}
}
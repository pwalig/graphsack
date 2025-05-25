#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define GS_CUDA_REDUCTIONS_PICK(result_type, thread_id, value_memory, result_memory) \
for (result_type n = 1; n < gridDim.x * blockDim.x; n *= 2) { \
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
			template <typename ResT, typename ValueResT>
			__global__ void pick(
				ValueResT* value_memory,
				ResT* result_memory,
				uint32_t stride, // threads per block of previous kernel
				ResT solutionSpace // threads per block times block count of previous kernel
			) {
				ResT id = (ResT)(blockIdx.x * blockDim.x + threadIdx.x) * stride * 2;

				if (id > solutionSpace) return;

				for (ResT n = stride; n < solutionSpace; n *= 2) {
					__syncthreads();
					if (value_memory[n + id] > value_memory[id]) {
						result_memory[id] = result_memory[n + id];
						value_memory[id] = value_memory[id + n];
					}
					if (id % (n*4) != 0) return;
				}
			}
		}
	}
}
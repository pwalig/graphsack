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
			}
		}
	}
}
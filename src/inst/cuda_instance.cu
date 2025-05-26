#include "cuda_instance.cuh"

namespace gs {
	namespace cuda {
		namespace inst {
			__constant__ uint32_t limits_u32[GS_CUDA_INST_MAXM];
			__constant__ uint32_t values_u32[GS_CUDA_INST_MAXN];
			__constant__ uint32_t weights_u32[GS_CUDA_INST_MAXN * GS_CUDA_INST_MAXM];
			__constant__ uint32_t adjacency32[32];
			__constant__ uint64_t adjacency64[64];


			//template __host__ __device__ const uint32_t* values<uint32_t>();
			//template __host__ __device__ const uint32_t* limits<uint32_t>();
			//template __host__ __device__ const uint32_t* weights<uint32_t>();
			//template __host__ __device__ const uint32_t* adjacency<uint32_t>();
			//template __host__ __device__ const uint64_t* adjacency<uint64_t>();

		}
	}
}
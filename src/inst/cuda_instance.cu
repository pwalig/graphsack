#include "cuda_instance.cuh"

namespace gs {
	namespace cuda {
		namespace inst {
			__constant__ uint32_t limits_u32[GS_CUDA_INST_MAXM];
			__constant__ float limits_f32[GS_CUDA_INST_MAXM];
			__constant__ uint32_t values_u32[GS_CUDA_INST_MAXN];
			__constant__ float values_f32[GS_CUDA_INST_MAXN];
			__constant__ uint32_t weights_u32[GS_CUDA_INST_MAXN * GS_CUDA_INST_MAXM];
			__constant__ float weights_f32[GS_CUDA_INST_MAXN * GS_CUDA_INST_MAXM];
			__constant__ uint32_t adjacency32[32];
			__constant__ uint64_t adjacency64[64];
			__constant__ size_t size;
			__constant__ size_t dim;
		}
	}
}
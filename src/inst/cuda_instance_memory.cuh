#pragma once
#include <device_launch_parameters.h>
#include <cstdint>

#define GS_CUDA_INST_MAXN 64
#define GS_CUDA_INST_MAXM 5
namespace gs {
	namespace cuda {
		namespace inst {
			__constant__ uint32_t limits[GS_CUDA_INST_MAXM];
			__constant__ uint32_t values[GS_CUDA_INST_MAXN];
			__constant__ uint32_t weights[GS_CUDA_INST_MAXN * GS_CUDA_INST_MAXM];
			__constant__ uint64_t adjacency[GS_CUDA_INST_MAXN];

			__device__  bool has_connection_to(const uint64_t* adjacency, uint32_t from, uint32_t to);
		}
	}
}

#pragma once
#include <device_launch_parameters.h>
#include <cstdint>

#define GS_CUDA_INST_MAXN 64
#define GS_CUDA_INST_MAXM 5

#define GS_CUDA_INST_CONSTANTS \
__constant__ uint32_t limits[GS_CUDA_INST_MAXM]; \
__constant__ uint32_t values[GS_CUDA_INST_MAXN]; \
__constant__ uint32_t weights[GS_CUDA_INST_MAXN * GS_CUDA_INST_MAXM]; \
__constant__ uint64_t adjacency[GS_CUDA_INST_MAXN];

#define GS_CUDA_INST_COPY_TO_SYMBOL_FUNCTION \
void copy_to_symbol(const instance64<uint32_t, uint32_t>& inst) { \
	cudaMemcpyToSymbol(limits, inst.limits().data(), inst.dim() * sizeof(uint32_t)); \
	cudaMemcpyToSymbol(values, inst.values().data(), inst.size() * sizeof(uint32_t)); \
	cudaMemcpyToSymbol(weights, inst.weights().data(), inst.size() * inst.dim() * sizeof(uint32_t)); \
	cudaMemcpyToSymbol(adjacency, inst.graph_data(), inst.size() * sizeof(uint64_t)); \
	cudaDeviceSynchronize(); \
}

#define GS_CUDA_INST_COPY_TO_SYMBOL_INLINE(inst) \
assert(inst.size() <= 64);\
cudaMemcpyToSymbol(limits, inst.limits().data(), inst.dim() * sizeof(uint32_t)); \
cudaMemcpyToSymbol(values, inst.values().data(), inst.size() * sizeof(uint32_t)); \
cudaMemcpyToSymbol(weights, inst.weights().data(), inst.size() * inst.dim() * sizeof(uint32_t)); \
cudaMemcpyToSymbol(adjacency, inst.graph_data(), inst.size() * sizeof(uint64_t)); \
cudaDeviceSynchronize();


namespace gs {
	namespace cuda {
		namespace inst {
			extern __constant__ uint32_t limits[GS_CUDA_INST_MAXM];
			extern __constant__ uint32_t values[GS_CUDA_INST_MAXN];
			extern __constant__ uint32_t weights[GS_CUDA_INST_MAXN * GS_CUDA_INST_MAXM];
			extern __constant__ uint64_t adjacency[GS_CUDA_INST_MAXN];

			__device__  bool has_connection_to(const uint64_t* adjacency, uint32_t from, uint32_t to);
		}
	}
}

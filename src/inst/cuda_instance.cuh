#pragma once
#include <device_launch_parameters.h>
#include <cstdint>

#define GS_CUDA_INST_MAXN 64
#define GS_CUDA_INST_MAXM 5

#define GS_CUDA_INST_CONSTANTS \
__constant__ uint32_t limits[GS_CUDA_INST_MAXM]; \
__constant__ uint32_t values[GS_CUDA_INST_MAXN]; \
__constant__ uint32_t weights[GS_CUDA_INST_MAXN * GS_CUDA_INST_MAXM]; \
__constant__ uint32_t adjacency32[32]; \
__constant__ uint64_t adjacency64[64]; \
template <typename StorageBase> \
__device__ const StorageBase* adjacency(); \
template <> \
__device__ const uint32_t* adjacency() { return adjacency32; } \
template <> \
__device__ const uint64_t* adjacency() { return adjacency64; } 


#define GS_CUDA_INST_COPY_TO_SYMBOL_FUNCTION \
void copy_to_symbol(const instance64<uint32_t, uint32_t>& inst) { \
	cudaMemcpyToSymbol(limits, inst.limits().data(), inst.dim() * sizeof(uint32_t)); \
	cudaMemcpyToSymbol(values, inst.values().data(), inst.size() * sizeof(uint32_t)); \
	cudaMemcpyToSymbol(weights, inst.weights().data(), inst.size() * inst.dim() * sizeof(uint32_t)); \
	cudaMemcpyToSymbol(adjacency, inst.graph_data(), inst.size() * sizeof(uint64_t)); \
	cudaDeviceSynchronize(); \
}

#define GS_CUDA_INST_COPY_TO_SYMBOL_INLINE(inst) \
cudaMemcpyToSymbol(limits, inst.limits().data(), inst.dim() * sizeof(uint32_t)); \
cudaMemcpyToSymbol(values, inst.values().data(), inst.size() * sizeof(uint32_t)); \
cudaMemcpyToSymbol(weights, inst.weights().data(), inst.size() * inst.dim() * sizeof(uint32_t)); \
cudaDeviceSynchronize();

namespace gs {
	namespace cuda {
		namespace inst {
			template <typename adjacency_base_type, typename index_type>
			inline __device__  bool has_connection_to(
				const adjacency_base_type* adjacency, index_type from, index_type to
			) {
				if (adjacency[from] & (adjacency_base_type(1) << to)) return true;
				else return false;
			}
		}
	}
}

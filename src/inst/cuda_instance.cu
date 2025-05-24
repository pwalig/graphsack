#include "cuda_instance.hpp"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cuda_instance.cuh"

__constant__ uint32_t gs::cuda::inst::limits[GS_CUDA_INST_MAXM];
__constant__ uint32_t gs::cuda::inst::values[GS_CUDA_INST_MAXN];
__constant__ uint32_t gs::cuda::inst::weights[GS_CUDA_INST_MAXN * GS_CUDA_INST_MAXM];
__constant__ uint64_t gs::cuda::inst::adjacency[GS_CUDA_INST_MAXN];

void gs::cuda::inst::copy_to_symbol(const instance64<uint32_t, uint32_t>& inst)
{
	cudaMemcpyToSymbol(limits, inst.limits().data(), inst.dim() * sizeof(uint32_t));
	cudaMemcpyToSymbol(values, inst.values().data(), inst.size() * sizeof(uint32_t));
	cudaMemcpyToSymbol(weights, inst.weights().data(), inst.size() * inst.dim() * sizeof(uint32_t));
	cudaMemcpyToSymbol(adjacency, inst.graph_data(), inst.size() * sizeof(uint64_t));
	cudaDeviceSynchronize();
}


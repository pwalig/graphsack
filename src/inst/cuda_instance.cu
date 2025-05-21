#include "cuda_instance.hpp"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cuda_instance_memory.cuh"

void gs::cuda::inst::copy_to_symbol(const instance64<uint32_t, uint32_t>& inst)
{
	cudaMemcpyToSymbol(limits, inst.limits_data(), inst.dim() * sizeof(uint32_t));
	cudaMemcpyToSymbol(values, inst.values_data(), inst.size() * sizeof(uint32_t));
	cudaMemcpyToSymbol(weights, inst.weights_data(), inst.size() * inst.dim() * sizeof(uint32_t));
	cudaMemcpyToSymbol(adjacency, inst.graph_data(), inst.size() * sizeof(uint64_t));
	cudaDeviceSynchronize();
}

__device__  bool gs::cuda::inst::has_connection_to(const uint64_t* adjacency, uint32_t from, uint32_t to) {
	if (adjacency[from] & (uint64_t(1) << to)) return true;
	else return false;
}


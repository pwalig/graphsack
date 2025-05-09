#include "cuda_instance.hpp"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cuda_instance_memory.cuh"

void gs::inst::cuda::copy_to_symbol(const instance<uint32_t, uint32_t>& inst)
{
	cudaMemcpyToSymbol(limits, inst.limits_data(), inst.dim() * sizeof(uint32_t));
	cudaMemcpyToSymbol(values, inst.values_data(), inst.size() * sizeof(uint32_t));
	cudaMemcpyToSymbol(weights, inst.weights_data(), inst.size() * inst.dim() * sizeof(uint32_t));
	cudaMemcpyToSymbol(adjacency, inst.graph_data(), inst.size() * sizeof(uint64_t));
	cudaDeviceSynchronize();
}

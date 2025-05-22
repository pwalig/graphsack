#include "cuda_solution.cuh"

__device__ bool gs::cuda::res::has(uint64_t solution, uint32_t itemId) {
	if (solution & (uint64_t(1) << itemId)) return true;
	else return false;
}

__device__ void gs::cuda::res::add(uint64_t& solution, uint32_t itemId)
{
	solution |= (uint64_t(1) << itemId);
}

__device__ void gs::cuda::res::remove(uint64_t& solution, uint32_t itemId)
{
	solution &= ~(uint64_t(1) << itemId);
}

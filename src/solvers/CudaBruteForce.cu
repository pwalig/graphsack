#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <cmath>
#include <vector>

#include "CudaBrutforce.hpp"
#include "../inst/cuda_instance.cuh"
#include "../res/cuda_solution.cuh"
#include "../cuda_structure_check.cuh"

//__constant__ uint32_t gs::cuda::inst::limits[GS_CUDA_INST_MAXM];
//__constant__ uint32_t gs::cuda::inst::values[GS_CUDA_INST_MAXN];
//__constant__ uint32_t gs::cuda::inst::weights[GS_CUDA_INST_MAXN * GS_CUDA_INST_MAXM];
//__constant__ uint64_t gs::cuda::inst::adjacency[GS_CUDA_INST_MAXN];
GS_CUDA_INST_CONSTANTS

__device__  bool has_connection_to(const uint32_t* adjacency, uint32_t from, uint32_t to) {
	if (adjacency[from] & (1 << to)) return true;
	else return false;
}

__device__  bool has(uint32_t selected, uint32_t itemId) {
	if (selected & (1 << itemId)) return true;
	else return false;
}

__device__ bool is_cycle_DFS(
	const uint32_t* adjacency,
	uint32_t N, uint32_t selected, uint32_t visited,
	uint32_t current, uint32_t start, uint32_t length, uint32_t depth
) {
	for (uint32_t next = 0; next < N; ++next) {
		if (!has_connection_to(adjacency, current, next)) continue;
		if (has(selected, next) && !has(visited, next)) { // next item has to be selected and new
			visited |= (1 << next);
			if (depth == length && has_connection_to(adjacency, next, start)) return true;
			if (depth > length) return false;
			if (is_cycle_DFS(adjacency, N, selected, visited, next, start, length, depth + 1)) return true;
			visited &= ~(1 << next);
		}
	}
	return false;
}

__device__ bool is_cycle(
	const uint32_t* adjacency,
	uint32_t selected, uint32_t N
) {

	// calculate whats the length of the cycle
	uint32_t length = 0;
	for (uint32_t i = 0; i < N; ++i)
		if (has(selected, i)) length++;
		
	if (length == 0) return true;
	
	// check from each starting point
	uint32_t visited = 0;
	for (uint32_t i = 0; i < N; ++i) {
		if (has(selected, i)){
			if (length == 1) return has_connection_to(adjacency, i, i);
			visited |= (1 << i);
			if (is_cycle_DFS(adjacency, N, selected, visited, i, i, length, 2)) return true;
			visited &= ~(1 << i);
		}
	}
	return false;
}

__global__ void cycle_kernel(
	const uint32_t* limits, const uint32_t* values, const uint32_t* weights, const uint32_t* adjacency,
	uint32_t* value_memory, uint32_t* weight_memory,
	uint32_t* result,
	uint32_t N, uint32_t M, uint32_t solutionSpace
) {
	const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id > solutionSpace) return;
	result[id] = id;
	value_memory[id] = 0;
	for (unsigned wid = 0; wid < M; ++wid) weight_memory[M * id + wid] = 0;

	uint32_t n = id;
	uint32_t i = 0;
	bool fitting = is_cycle(adjacency, n, N);

	while (i < N && fitting) {
		if (has(n, i)) {
			value_memory[id] += values[i];
			for (uint32_t  wid = 0; wid < M; ++wid) {
				weight_memory[M * id + wid] += weights[M * i + wid];
				
				if (weight_memory[M * id + wid] > limits[wid]) {
					fitting = false;
					value_memory[id] = 0;
					break;
				}
			}
		}
		i++;
	}

	for (n = 1; n < solutionSpace; n *= 2) {
		__syncthreads();
		if (id % (n*2) != 0) return;
		if (value_memory[result[id + n]] > value_memory[result[id]]) result[id] = result[id + n];
	}
}

__global__ void base_kernel(
	uint32_t* limits, uint32_t* values, uint32_t* weights,
	uint32_t* value_memory, uint32_t* weight_memory,
	uint32_t* result,
	uint32_t N, uint32_t M, uint32_t solutionSpace
) {
	const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id > solutionSpace) return;
	result[id] = id;
	value_memory[id] = 0;
	for (unsigned wid = 0; wid < M; ++wid) weight_memory[M * id + wid] = 0;

	size_t n = id;
	uint32_t i = 0;
	bool fitting = true;

	while (i < N && fitting) {
		if (n % 2 == 1) {
			value_memory[id] += values[i];
			for (uint32_t wid = 0; wid < M; ++wid) {
				weight_memory[M * id + wid] += weights[M * i + wid];
				if (weight_memory[M * id + wid] > limits[wid]) {
					fitting = false;
					value_memory[id] = 0;
					break;
				}
			}
		}
		n /= 2;
		i++;
	}

	for (n = 1; n < solutionSpace; n *= 2) {
		__syncthreads();
		if (id % (n*2) != 0) return;
		if (value_memory[result[id + n]] > value_memory[result[id]]) result[id] = result[id + n];
	}
}

__global__ void pick(
	uint32_t* value_memory,
	uint32_t* result_memory,
	uint32_t stride,
	uint32_t solutionSpace
) {
	uint32_t id = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2;

	if (id > solutionSpace) return;

	for (uint32_t n = stride; n < solutionSpace; n *= 2) {
		__syncthreads();
		if (value_memory[result_memory[n + id]] > value_memory[result_memory[id]]) result_memory[id] = result_memory[n + id];
		if (id % (n*4) != 0) return;
	}
}

// data should be: limits | values | weights | adjacency
uint32_t gs::cuda::solver::brute_force::runner_u32_u32(uint32_t* data, uint32_t N, uint32_t M, uint32_t threadsPerBlock, uint32_t share, structure to_find) {
	cudaError_t cudaStatus;
	size_t solutionSpace = (size_t)std::pow(2, N) / share;
	uint32_t* device_memory;
	size_t data_size = N * M + N + M + N;
	size_t memory_size = solutionSpace * M + solutionSpace + data_size;
	cudaStatus = cudaMalloc(&device_memory, (memory_size + solutionSpace) * sizeof(uint32_t));
	if (cudaStatus != cudaSuccess) throw std::runtime_error("failed to allocate GPU memory");
	cudaStatus = cudaMemcpy(device_memory, data, data_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) throw std::runtime_error("failed to memcpy host to GPU memory");
	uint32_t blocksCount = std::max<uint32_t>(1, (uint32_t)(solutionSpace / threadsPerBlock));
	if (to_find == structure::cycle) cycle_kernel<<<blocksCount, threadsPerBlock>>>(device_memory, device_memory + M, device_memory + M + N, device_memory + (N * M) + N + M,
		device_memory + data_size, device_memory + data_size + solutionSpace, device_memory + memory_size,
		N, M, solutionSpace
	);
	else base_kernel<<<blocksCount, threadsPerBlock>>>(device_memory, device_memory + M, device_memory + M + N,
		device_memory + data_size, device_memory + data_size + solutionSpace, device_memory + memory_size,
		N, M, solutionSpace
	);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		cudaFree(device_memory);
		throw std::runtime_error("failed to synch GPU");
	}

	if (blocksCount > 1) {
		pick<<<1, blocksCount>>> (
			device_memory + data_size,
			device_memory + memory_size,
			threadsPerBlock,
			solutionSpace
		);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			cudaFree(device_memory);
			throw std::runtime_error("failed to synch GPU");
		}
	}

	uint32_t result;
	cudaStatus = cudaMemcpy(&result, device_memory + memory_size, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		cudaFree(device_memory);
		throw std::runtime_error("failed to synch GPU");
	}

	cudaFree(device_memory);
	return result;
}

const std::string gs::cuda::solver::BruteForce64::name = "CudaBruteForce64";

namespace gs {
	namespace cuda {
		__global__ void cycle_kernel(
			uint32_t N, uint32_t M, uint64_t solutionSpace,
			uint32_t* value_memory, uint32_t* weight_memory, uint64_t* result_memory
		) {
			const uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
			if (id > solutionSpace) return;
			result_memory[id] = id;
			value_memory[id] = 0;
			for (unsigned wid = 0; wid < M; ++wid) weight_memory[M * id + wid] = 0;

			uint64_t n = id;
			uint32_t i = 0;
			bool fitting = is_cycle(adjacency, n, N);

			while (i < N && fitting) {
				if (res::has(n, i)) {
					value_memory[id] += values[i];
					for (uint32_t  wid = 0; wid < M; ++wid) {
						weight_memory[M * id + wid] += weights[M * i + wid];
						
						if (weight_memory[M * id + wid] > limits[wid]) {
							fitting = false;
							value_memory[id] = 0;
							break;
						}
					}
				}
				i++;
			}

			for (n = 1; n < solutionSpace; n *= 2) {
				__syncthreads();
				if (id % (n*2) != 0) return;
				if (value_memory[result_memory[id + n]] > value_memory[result_memory[id]]) result_memory[id] = result_memory[id + n];
			}
		}
	}
}

gs::cuda::res::solution64 gs::cuda::solver::brute_force::runner_instance64_solution64(
	const inst::instance64<uint32_t, uint32_t>& instance, uint32_t threadsPerBlock, uint32_t share
) {
	cudaError_t cudaStatus;
	GS_CUDA_INST_COPY_TO_SYMBOL_INLINE(instance)
	//inst::copy_to_symbol(instance);

	size_t solutionSpace = (size_t)std::pow(2, instance.size()) / share;

	uint32_t* device_weight_value_memory;
	size_t device_weight_value_memory_size = solutionSpace * (instance.dim() + 1);
	uint64_t* device_result_memory;
	size_t device_result_memory_size = solutionSpace;

	cudaStatus = cudaMalloc(&device_weight_value_memory, device_weight_value_memory_size * sizeof(uint32_t));
	if (cudaStatus != cudaSuccess) throw std::runtime_error("failed to allocate GPU memory");

	cudaStatus = cudaMalloc(&device_result_memory, device_result_memory_size * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess) {
		cudaFree(device_weight_value_memory);
		throw std::runtime_error("failed to allocate GPU memory");
	}

	uint32_t blocksCount = std::max<uint32_t>(1, static_cast<uint32_t>(solutionSpace / threadsPerBlock));
	cycle_kernel<<<blocksCount, threadsPerBlock>>>(
		instance.size(), instance.dim(), solutionSpace,
		device_weight_value_memory, device_weight_value_memory + solutionSpace, device_result_memory
	);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		cudaFree(device_weight_value_memory);
		cudaFree(device_result_memory);
		throw std::runtime_error("failed to synch GPU");
	}

	res::solution64 result(instance.size());

	cudaStatus = cudaMemcpy(&result._data, device_result_memory, sizeof(uint64_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		cudaFree(device_weight_value_memory);
		cudaFree(device_result_memory);
		throw std::runtime_error("failed to copy from GPU do CPU");
	}

	cudaFree(device_weight_value_memory);
	cudaFree(device_result_memory);
	return result;
}

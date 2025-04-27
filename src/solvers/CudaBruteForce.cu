#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <cmath>
#include <vector>

#include "CudaBrutforce.hpp"
/*
bool is_cycle_DFS(
	const uint32_t* limits, const uint32_t* values, const uint32_t* weights,
	const size_t* selected,
	bool* visited,
	uint32_t current, uint32_t start, uint32_t length, uint32_t depth
) {
	for (uint32_t next : instance.nexts(current)) {
		if (selected.has(next) && !visited[next]){ // next item has to be selected and new
			visited[next] = true;
			if (depth == length && has_connection_to(instance, next, start)) return true; // cycle found
			if (depth > length) return false; // cycle would have to be to long
			if (is_cycle_DFS<instance_t, solution_t, indexT>(instance, selected, visited, next, start, length, depth + 1)) return true; // cycle found later
			visited[next] = false;
		}
	}
	return false; // cycle not found
}

bool is_cycle(
	const instance_t& instance,
	const solution_t& selected
) {
	assert(selected.size() == instance.size());

	// calculate whats the length of the cycle
	indexT length = 0;
	for (indexT i = 0; i < selected.size(); ++i)
		if (selected.has(i)) ++length;
	
	if (length == 0) return true;
	
	// check from each starting point
	std::vector<bool> visited(selected.size(), false);
	for (indexT i = 0; i < selected.size(); ++i) {
		if (selected.has(i)){
			if (length == 1) return has_connection_to(instance, i, i);
			visited[i] = true;
			if (is_cycle_DFS<instance_t, solution_t, indexT>(instance, selected, visited, i, i, length, 2)) return true; // cycle found somewhere
			visited[i] = false;
		}
	}
	return false;
}
*/
__global__ void base_kernel(
	uint32_t* limits, uint32_t* values, uint32_t* weights,
	uint32_t* value_memory, uint32_t* weight_memory,
	unsigned* result,
	unsigned N, unsigned M, unsigned solutionSpace
) {
	const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id > solutionSpace) return;
	result[id] = id;
	value_memory[id] = 0;
	for (unsigned wid = 0; wid < M; ++wid) weight_memory[M * id + wid] = 0;

	unsigned n = id;
	unsigned i = N;
	bool fitting = true;

	while (i > 0 && fitting) {
		--i;
		if (n % 2 == 1) {
			value_memory[id] += values[i];
			for (unsigned wid = 0; wid < M; ++wid) {
				weight_memory[M * id + wid] += weights[M * i + wid];
				if (weight_memory[M * id + wid] > limits[wid]) {
					fitting = false;
					value_memory[id] = 0;
					break;
				}
			}
		}
		n /= 2;
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

// data should be: limits | values | weights
uint32_t gs::solver::cuda::brute_force::runner_u32_u32(uint32_t* data, uint32_t N, uint32_t M, uint32_t threadsPerBlock) {
	cudaError_t cudaStatus;
	uint32_t solutionSpace = (uint32_t)std::pow(2, N);
	uint32_t* device_memory;
	uint32_t data_size = N * M + N + M;
	uint32_t memory_size = solutionSpace * M + solutionSpace + data_size;
	cudaStatus = cudaMalloc(&device_memory, (memory_size + solutionSpace) * sizeof(uint32_t));
	if (cudaStatus != cudaSuccess) throw std::runtime_error("failed to allocate GPU memory");
	cudaStatus = cudaMemcpy(device_memory, data, data_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) throw std::runtime_error("failed to memcpy host to GPU memory");
	uint32_t blocksCount = solutionSpace / threadsPerBlock;
	base_kernel<<<blocksCount, threadsPerBlock>>>(device_memory, device_memory + M, device_memory + M + N,
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

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <cmath>
#include <vector>

namespace gs {
	namespace solver {
		namespace cuda {
			namespace brute_force {
				__global__ void kernel(
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

				// data should be: limits | values | weights
				std::vector<uint32_t> runner_u32_u32(uint32_t* data, uint32_t N, uint32_t M) {
					cudaError_t cudaStatus;
					uint32_t solutionSpace = std::pow(2, N);
					uint32_t* device_memory;
					uint32_t data_size = N * M + N + M;
					uint32_t memory_size = solutionSpace * M + solutionSpace + data_size;
					cudaStatus = cudaMalloc(&device_memory, (memory_size + solutionSpace) * sizeof(uint32_t));
					if (cudaStatus != cudaSuccess) throw std::runtime_error("failed to allocate GPU memory");
					cudaStatus = cudaMemcpy(device_memory, data, data_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
					if (cudaStatus != cudaSuccess) throw std::runtime_error("failed to memcpy host to GPU memory");
					kernel<<<1, solutionSpace>>>(device_memory, device_memory + M, device_memory + M + N,
						device_memory + data_size, device_memory + data_size + solutionSpace, device_memory + memory_size,
						N, M, solutionSpace
					);
					cudaStatus = cudaDeviceSynchronize();
					if (cudaStatus != cudaSuccess) throw std::runtime_error("failed to synch GPU");

					std::vector<uint32_t> result(solutionSpace * 2);
					cudaStatus = cudaMemcpy(result.data(), device_memory + memory_size, solutionSpace * sizeof(uint32_t), cudaMemcpyDeviceToHost);
					if (cudaStatus != cudaSuccess) throw std::runtime_error("failed to copy results back to CPU");
					cudaStatus = cudaMemcpy(result.data() + solutionSpace, device_memory + data_size, solutionSpace * sizeof(uint32_t), cudaMemcpyDeviceToHost);
					if (cudaStatus != cudaSuccess) throw std::runtime_error("failed to copy results back to CPU");

					cudaFree(device_memory);
					return result;
				}
			}
		}
	}
}
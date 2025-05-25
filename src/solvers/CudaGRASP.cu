#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>

#include <stdexcept>
#include <cmath>
#include <vector>

#include "CudaGRASP.hpp"
#include "cuda_reductions.cuh"
#include "cuda_greedy_utils.cuh"
#include "../inst/cuda_instance.cuh"
#include "../res/cuda_solution.cuh"
#include "../cuda_structure_check.cuh"

namespace gs {
	namespace cuda {
		namespace solver {
			namespace grasp {
				GS_CUDA_INST_CONSTANTS
			}
		}
	}
}

const std::string gs::cuda::solver::GRASP32::name = "CudaGRASP32";
const std::string gs::cuda::solver::GRASP64::name = "CudaGRASP64";

namespace gs {
	namespace cuda {
		namespace solver {
			namespace grasp {
				template <typename result_type, typename index_type>
				__global__ void cycle_kernel(
					index_type N, uint32_t M, curandStateMtgp32* random_state, 
					uint32_t* value_memory, uint32_t* weight_memory, result_type* result_memory, index_type* stack_memory,
					index_type* sorted, index_type choose_from
				) {
					const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

					// setup
					value_memory[id] = 0;
					result_memory[id] = 0;
					for (uint32_t wid = 0; wid < M; ++wid) weight_memory[M * id + wid] = 0;

					// construct solution
					for (index_type left = N; left > 0; --left) {
						index_type left_index = curand(&random_state[blockIdx.x]) % (left < choose_from ? left : choose_from);

						index_type sorted_index = 0;
						while (res::has(result_memory[id], sorted[sorted_index])) ++sorted_index;

						for (index_type i = 0; i < left_index; ++i) {
							++sorted_index;
							while (res::has(result_memory[id], sorted[sorted_index])) ++sorted_index;
						}

						index_type to_add = sorted[sorted_index];
						bool fitting = true;
						for (uint32_t wid = 0; wid < M; ++wid) {
							if (weight_memory[M * id + wid] + weights[M * to_add + wid] > limits[wid]) {
								fitting = false;
								break;
							}
						}
						if (fitting) {
							for (uint32_t wid = 0; wid < M; ++wid) weight_memory[M * id + wid] += weights[M * to_add + wid];
							value_memory[id] += values[to_add];
							res::add(result_memory[id], to_add);
						}
					}

					// do reduction
					GS_CUDA_REDUCTIONS_PICK(result_type, id)
				}

				template <typename result_type>
				res::solution<result_type> runner(
					const inst::instance<result_type, uint32_t, uint32_t>& instance, uint32_t blocksCount
				) {
					using index_type = typename inst::instance<result_type, uint32_t, uint32_t>::index_type;

					uint32_t threadsPerBlock = 256;
					size_t solutionSpace = threadsPerBlock * blocksCount;

					cudaError_t cudaStatus;
					GS_CUDA_INST_COPY_TO_SYMBOL_INLINE(instance)



					uint32_t* dev_mem;
					size_t dev_mem_siz = solutionSpace * (instance.dim() + 1);
					cudaStatus = cudaMalloc(&dev_mem, dev_mem_siz * sizeof(uint32_t));
					if (cudaStatus != cudaSuccess) throw std::runtime_error("failed to allocate GPU memory");

					index_type* dev_index_mem;
					size_t dev_index_mem_siz = solutionSpace * instance.size() + instance.size();
					cudaStatus = cudaMalloc(&dev_index_mem, dev_index_mem_siz * sizeof(uint32_t));
					if (cudaStatus != cudaSuccess) {
						cudaFree(dev_mem);
						throw std::runtime_error("failed to allocate GPU memory");
					}

					result_type* dev_res_mem;
					cudaStatus = cudaMalloc(&dev_res_mem, solutionSpace * sizeof(result_type));
					if (cudaStatus != cudaSuccess) {
						cudaFree(dev_mem);
						cudaFree(dev_index_mem);
						throw std::runtime_error("failed to allocate GPU memory");
					}

					curandStateMtgp32 *devMTGPStates;
					if (cudaMalloc(&devMTGPStates, blocksCount * sizeof(curandStateMtgp32)) != cudaSuccess) {
						cudaFree(dev_mem);
						cudaFree(dev_index_mem);
						cudaFree(dev_res_mem);
						throw std::runtime_error("failed to allocate GPU memory");
					}

					mtgp32_kernel_params *devKernelParams;
					if (cudaMalloc(&devKernelParams, sizeof(mtgp32_kernel_params)) != cudaSuccess) {
						cudaFree(dev_mem);
						cudaFree(dev_index_mem);
						cudaFree(dev_res_mem);
						throw std::runtime_error("failed to allocate GPU memory");
					}
					curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devKernelParams);
					//curandMakeMTGP32KernelState(devMTGPStates, mtgp32dc_params_fast_11213, devKernelParams, blocksCount, 1234);
					curandMakeMTGP32KernelState(devMTGPStates, mtgp32dc_params_fast_11213, devKernelParams, blocksCount, time(NULL));

					sort::in_order<index_type><<<1, 64>>>(dev_index_mem, instance.size());
					if (cudaDeviceSynchronize()) {
						cudaFree(dev_mem);
						cudaFree(dev_index_mem);
						cudaFree(dev_res_mem);
						throw std::runtime_error("failed to synch GPU");
					}

					cycle_kernel<result_type, index_type><<<blocksCount, threadsPerBlock>>>(
						instance.size(), instance.dim(), devMTGPStates,
						dev_mem, dev_mem + solutionSpace, dev_res_mem, dev_index_mem + instance.size(),
						dev_index_mem, instance.size() / 2
					);
					if (cudaDeviceSynchronize() != cudaSuccess) {
						cudaFree(dev_mem);
						cudaFree(dev_index_mem);
						cudaFree(dev_res_mem);
						throw std::runtime_error("failed to synch GPU");
					}

					if (blocksCount > 1) {
						blocksCount /= 2;
						reductions::pick<result_type, uint32_t><<<1, blocksCount>>> (
							dev_mem,
							dev_res_mem,
							threadsPerBlock,
							solutionSpace
						);
						if (cudaDeviceSynchronize() != cudaSuccess) {
							cudaFree(dev_mem);
							cudaFree(dev_index_mem);
							cudaFree(dev_res_mem);
							throw std::runtime_error("failed to synch GPU");
						}
					}

					res::solution<result_type> result(instance.size());

					if (cudaMemcpy(&result._data, dev_res_mem, sizeof(result_type), cudaMemcpyDeviceToHost) != cudaSuccess) {
						cudaFree(dev_mem);
						cudaFree(dev_index_mem);
						cudaFree(dev_res_mem);
						throw std::runtime_error("failed to copy from GPU do CPU");
					}

					cudaFree(dev_mem);
					cudaFree(dev_index_mem);
					cudaFree(dev_res_mem);
					return result;
				}
			}
		}
	}
}

gs::cuda::res::solution32 gs::cuda::solver::grasp::runner32(
	const inst::instance32<uint32_t, uint32_t>& instance, uint32_t blocksCount
) {
	assert(instance.size() <= 32);
	cudaMemcpyToSymbol(adjacency32, instance.graph_data(), instance.size() * sizeof(uint32_t));
	return runner<uint32_t>(instance, blocksCount);
}
gs::cuda::res::solution64 gs::cuda::solver::grasp::runner64(
	const inst::instance64<uint32_t, uint32_t>& instance, uint32_t blocksCount
) {
	assert(instance.size() <= 64);
	cudaMemcpyToSymbol(adjacency64, instance.graph_data(), instance.size() * sizeof(uint64_t));
	return runner<uint64_t>(instance, blocksCount);
}

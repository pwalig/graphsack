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
#include "cuda_buffer.cuh"
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
					GS_CUDA_REDUCTIONS_PICK(result_type, id, value_memory, result_memory)
				}

				template <typename result_type>
				res::solution<result_type> runner(
					const inst::instance<result_type, uint32_t, uint32_t>& instance, uint32_t blocksCount
				) {
					using index_type = typename inst::instance<result_type, uint32_t, uint32_t>::index_type;

					if (blocksCount > 200) throw std::invalid_argument("cudaGRASP blocksCount limit of 200 exeeded");
					uint32_t threadsPerBlock = 256;
					size_t totalThreads = threadsPerBlock * blocksCount;

					cudaError_t cudaStatus;
					GS_CUDA_INST_COPY_TO_SYMBOL_INLINE(instance)

					buffer<uint32_t> weight_value(totalThreads * (instance.dim() + 1));
					buffer<index_type> index_memory(totalThreads * instance.size() + instance.size());
					buffer<result_type> result_memory(totalThreads);

					buffer<curandStateMtgp32> random_states(blocksCount);
					buffer<mtgp32_kernel_params> kernel_params(1);

					curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, kernel_params.data());
					curandMakeMTGP32KernelState(random_states.data(), mtgp32dc_params_fast_11213, kernel_params.data(), blocksCount, time(NULL));

					sort::in_order<index_type><<<1, 64>>>(index_memory.data(), instance.size());
					if (cudaDeviceSynchronize()) {
						throw std::runtime_error("failed to synch GPU");
					}

					cycle_kernel<result_type, index_type><<<blocksCount, threadsPerBlock>>>(
						instance.size(), instance.dim(), random_states.data(),
						weight_value.data(), weight_value.data() + totalThreads,
						result_memory.data(), index_memory.data() + instance.size(),
						index_memory.data(), instance.size() / 2
					);
					if (cudaDeviceSynchronize() != cudaSuccess) {
						throw std::runtime_error("failed to synch GPU");
					}

					if (blocksCount > 1) {
						blocksCount /= 2;
						reductions::pick<result_type, uint32_t><<<1, blocksCount>>> (
							weight_value.data(),
							result_memory.data(),
							threadsPerBlock,
							totalThreads
						);
						if (cudaDeviceSynchronize() != cudaSuccess) {
							throw std::runtime_error("failed to synch GPU");
						}
					}

					res::solution<result_type> result(instance.size());
					result_memory.get(&result._data);

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

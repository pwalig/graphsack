#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <cmath>
#include <vector>

#include "CudaBrutforce.hpp"
#include "cuda_reductions.cuh"
#include "../cuda/buffer.cuh"
#include "../cuda/error_wrapper.cuh"
#include "../inst/cuda_instance.cuh"
#include "../res/cuda_solution.cuh"
#include "../cuda_structure_check.cuh"
#include "../cuda/device_properties.cuh"


namespace gs {
	namespace cuda {
		namespace solver {
			namespace brute_force {

				double global_mem_percent = 0.8;

				template <typename result_type, typename value_type, typename weight_type, typename index_type>
				__global__ void cycle_kernel(
					result_type totalThreads, uint32_t share,
					value_type* value_memory, weight_type* weight_memory,
					result_type* result_memory, index_type* stack_memory
				) {
					const result_type id = blockIdx.x * blockDim.x + threadIdx.x;
					if (id > totalThreads) return;

					// check share solutions
					value_memory[id] = 0;
					result_memory[id] = id;
					for (result_type resId = 0; resId < share; ++resId) {

						// setup
						result_type n = id * share + resId;
						value_type value = 0;
						for (uint32_t wid = 0; wid < inst::dim; ++wid) {
							weight_memory[inst::dim * id + wid] = 0;
						}

						bool fitting = true;

						// check if valid weight and sum value
						index_type i = 0;
						while (i < inst::size && fitting) {
							if (res::has(n, i)) {
								value += inst::values<value_type>()[i];
								for (uint32_t  wid = 0; wid < inst::dim; ++wid) {
									weight_memory[inst::dim * id + wid] += inst::weights<weight_type>()[inst::dim * i + wid];
									
									if (weight_memory[inst::dim * id + wid] > inst::limits<weight_type>()[wid]) {
										fitting = false;
										value = 0;
										break;
									}
								}
							}
							i++;
						}

						// check if valid structure
						if (value > value_memory[id] && is_cycle_iterative<result_type, index_type>(stack_memory + (2 * inst::size * id), n)) {
						//if (value > value_memory[id] && is_cycle_recursive<result_type, index_type>(n)) {
							value_memory[id] = value;
							result_memory[id] = n;
						}
					}

					// do reduction
					GS_CUDA_REDUCTIONS_PICK(result_type, id, value_memory, result_memory)
				}

//#define GS_CUDA_BRUTE_FORCE_DIAGNOSTIC
				template <typename instance_t, typename result_type>
				res::solution<result_type> runner(
					const inst::instance<result_type, uint32_t, uint32_t>& instance,
					uint32_t threadsPerBlock, uint32_t share
				) {
					using value_type = typename instance_t::value_type;
					using weight_type = typename instance_t::weight_type;
					using index_type = typename instance_t::index_type;

					inst::copy_to_symbol(instance);

					size_t solutionSpace = (size_t(1) << instance.size());
					size_t totalMemory = device_properties.totalGlobalMem * global_mem_percent;
					size_t memoryPerThread = sizeof(value_type) + (instance.dim() * sizeof(weight_type))
						+ (2 * instance.size() * sizeof(index_type)) + sizeof(result_type);
					size_t maxThreads = totalMemory / memoryPerThread;

					if (share == 0) share = 1;
					size_t totalThreads = solutionSpace / share;
					while (totalThreads > maxThreads) {
						totalThreads /= 2;
						share *= 2;
					}

					if (threadsPerBlock == 0 || threadsPerBlock > device_properties.maxThreadsPerBlock)
						threadsPerBlock = device_properties.maxThreadsPerBlock;
					threadsPerBlock = std::min<result_type>(threadsPerBlock, totalThreads);

					buffer<value_type> value_memory(totalThreads);
					buffer<weight_type> weight_memory(totalThreads * instance.dim());
					buffer<index_type> stack_memory(totalThreads * 2 * instance.size());
					buffer<result_type> result_memory(totalThreads);

					size_t blocksCount = std::max<size_t>(size_t(1), totalThreads / threadsPerBlock);
#ifdef GS_CUDA_BRUTE_FORCE_DIAGNOSTIC
					std::cout << "solution space: " << solutionSpace << '\n';
					std::cout << "max global memory: " << totalMemory << '\n';
					std::cout << "used memory: " << memoryPerThread * totalThreads << '\n';
					std::cout << "total threads: " << totalThreads << '\n';
					std::cout << "share: " << share << '\n';
					std::cout << "threads per block: " << threadsPerBlock << '\n';
					std::cout << "blocks count: " << blocksCount << '\n';
#endif
					cycle_kernel<result_type><<<blocksCount, threadsPerBlock>>>(
						totalThreads, share,
						value_memory.data(), weight_memory.data(),
						result_memory.data(), stack_memory.data()
					);

					if (blocksCount > 1) {
						blocksCount /= 2;
						if (blocksCount > device_properties.maxThreadsPerBlock) {
							reductions::shared_pick<result_type, uint32_t><<<1, device_properties.maxThreadsPerBlock>>>(
								value_memory.data(),
								result_memory.data(),
								threadsPerBlock,
								totalThreads
							);
						}
						else {
							reductions::pick<result_type, uint32_t><<<1, blocksCount>>>(
								value_memory.data(),
								result_memory.data(),
								threadsPerBlock,
								totalThreads
							);
						}
					}

					res::solution<result_type> result(instance.size());
					result_memory.get(&result._data);

					return result;
				}
				template res::solution32 runner<inst::instance32<uint32_t, uint32_t>, uint32_t>(const inst::instance32<uint32_t, uint32_t>&, uint32_t, uint32_t);
				template res::solution64 runner<inst::instance64<uint32_t, uint32_t>, uint64_t>(const inst::instance64<uint32_t, uint32_t>&, uint32_t, uint32_t);

			}
		}
	}
}


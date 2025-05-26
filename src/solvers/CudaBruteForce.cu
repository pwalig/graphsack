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


namespace gs {
	namespace cuda {
		namespace solver {
			const std::string BruteForce32::name = "CudaBruteForce32";
			const std::string BruteForce64::name = "CudaBruteForce64";

			namespace brute_force {

				template <typename result_type, typename index_type>
				__global__ void cycle_kernel(
					uint32_t N, uint32_t M, result_type totalThreads, uint32_t share,
					uint32_t* value_memory, uint32_t* weight_memory, result_type* result_memory, index_type* stack_memory
				) {
					const result_type id = blockIdx.x * blockDim.x + threadIdx.x;
					if (id > totalThreads) return;

					// check share solutions
					value_memory[id] = 0;
					result_memory[id] = id;
					for (result_type resId = 0; resId < share; ++resId) {

						// setup
						result_type n = id * share + resId;
						uint32_t value = 0;
						for (uint32_t wid = 0; wid < M; ++wid) {
							weight_memory[M * id + wid] = 0;
						}

						// check if valid structure
						bool fitting = true;

						// check if valid weight and sum value
						index_type i = 0;
						while (i < N && fitting) {
							if (res::has(n, i)) {
								value += inst::values<uint32_t>()[i];
								for (uint32_t  wid = 0; wid < M; ++wid) {
									weight_memory[M * id + wid] += inst::weights<uint32_t>()[M * i + wid];
									
									if (weight_memory[M * id + wid] > inst::limits<uint32_t>()[wid]) {
										fitting = false;
										value = 0;
										break;
									}
								}
							}
							i++;
						}

						if (value > value_memory[id] && is_cycle_iterative<result_type, index_type>(stack_memory + (2 * N * id), n, N)) {
							value_memory[id] = value;
							result_memory[id] = n;
						}
					}

					// do reduction
					GS_CUDA_REDUCTIONS_PICK(result_type, id, value_memory, result_memory)
				}

				template <typename result_type>
				res::solution<result_type> runner(
					const inst::instance<result_type, uint32_t, uint32_t>& instance, uint32_t threadsPerBlock, uint32_t share
				) {
					using index_type = typename inst::instance<result_type, uint32_t, uint32_t>::index_type;

					inst::copy_to_symbol(instance);

					result_type solutionSpace = (size_t(1) << instance.size());
					result_type totalThreads = solutionSpace / share;

					threadsPerBlock = std::min<result_type>(threadsPerBlock, totalThreads);

					buffer<uint32_t> device_memory(totalThreads * (instance.dim() + 1));
					buffer<index_type> stack_memory(totalThreads * 2 * instance.size());
					buffer<result_type> result_memory(totalThreads);

					uint32_t blocksCount = std::max<uint32_t>(1, static_cast<uint32_t>(totalThreads / threadsPerBlock));
#ifdef GS_CUDA_BRUTE_FORCE_DIAGNOSTIC
					printf("solution space: %d\n", solutionSpace);
					printf("share: %d\n", share);
					printf("total threads: %d\n", totalThreads);
					printf("threads per block: %d\n", threadsPerBlock);
					printf("blocks count: %d\n", blocksCount);
#endif
					except::DeviceSynchronize();
					cycle_kernel<result_type><<<blocksCount, threadsPerBlock>>>(
						instance.size(), instance.dim(), totalThreads, share,
						device_memory.data(), // value_memory
						device_memory.data() + totalThreads, // weight_memory
						result_memory.data(),
						stack_memory.data()
					);
					except::DeviceSynchronize();

					if (blocksCount > 1) {
						blocksCount /= 2;
						if (blocksCount > 1024) {
							reductions::shared_pick<result_type, uint32_t><<<1, 1024>>>(
								device_memory.data(),
								result_memory.data(),
								threadsPerBlock,
								totalThreads
							);
						}
						else {
							reductions::pick<result_type, uint32_t><<<1, blocksCount>>>(
								device_memory.data(),
								result_memory.data(),
								threadsPerBlock,
								totalThreads
							);
						}
						except::DeviceSynchronize();
					}

					res::solution<result_type> result(instance.size());
					result_memory.get(&result._data);

					return result;
				}

				res::solution32 runner32(
					const inst::instance32<uint32_t, uint32_t>& instance, uint32_t threadsPerBlock, uint32_t share
				) {
					return runner<uint32_t>(instance, threadsPerBlock, share);
				}

				res::solution64 runner64(
					const inst::instance64<uint32_t, uint32_t>& instance, uint32_t threadsPerBlock, uint32_t share
				) {
					return runner<uint64_t>(instance, threadsPerBlock, share);
				}
			}
		}
	}
}


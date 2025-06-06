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
#include "../cuda/buffer.cuh"
#include "../cuda/error_wrapper.cuh"
#include "../cuda/curand_wrapper.cuh"
#include "../inst/cuda_instance.cuh"
#include "../res/cuda_solution.cuh"
#include "../cuda_structure_check.cuh"


namespace gs {
	namespace cuda {
		namespace solver {
			namespace grasp {
				template <typename result_type, typename value_type, typename weight_type, typename index_type>
				__global__ void cycle_kernel(
					curandStateMtgp32* random_state, 
					value_type* value_memory, weight_type* weight_memory, result_type* result_memory,
					index_type* sorted, index_type choose_from
				) {
					const size_t id = blockIdx.x * blockDim.x + threadIdx.x;

					// setup
					value_memory[id] = 0;
					result_memory[id] = 0;
					for (uint32_t wid = 0; wid < inst::dim; ++wid)
						weight_memory[inst::dim * id + wid] = inst::limits<weight_type>()[wid];

					// construct solution
					for (index_type left = inst::size; left > 0; --left) {
						index_type left_index = ::curand(&random_state[blockIdx.x]) % (left < choose_from ? left : choose_from);

						index_type sorted_index = 0;
						while (res::has(result_memory[id], sorted[sorted_index])) ++sorted_index;

						for (index_type i = 0; i < left_index; ++i) {
							++sorted_index;
							while (res::has(result_memory[id], sorted[sorted_index])) ++sorted_index;
						}

						index_type to_add = sorted[sorted_index];
						uint32_t wid = 0;
						for (; wid < inst::dim; ++wid) {
							if (weight_memory[inst::dim * id + wid] < inst::weights<weight_type>()[inst::dim * to_add + wid])
								break;
						}
						if (wid == inst::dim) {
							res::add(result_memory[id], to_add);
							if (!is_cycle_possible_recursive<result_type, weight_type, index_type>(
								result_memory[id]
							)) res::remove(result_memory[id], to_add);
							else {
								for (uint32_t wid = 0; wid < inst::dim; ++wid)
									weight_memory[inst::dim * id + wid] -= inst::weights<weight_type>()[inst::dim * to_add + wid];
								value_memory[id] += inst::values<value_type>()[to_add];
							}
						}
					}

					// do reduction
					GS_CUDA_REDUCTIONS_PICK(result_type, id, value_memory, result_memory)
				}

				template <typename instance_t>
				res::solution<typename instance_t::adjacency_base_type> runner(
					const instance_t& instance, uint32_t blocksCount, typename instance_t::index_type choose_from
				) {
					using value_type = typename instance_t::value_type;
					using weight_type = typename instance_t::weight_type;
					using index_type = typename instance_t::index_type;
					using result_type = typename instance_t::adjacency_base_type;

					if (blocksCount > 200) throw std::invalid_argument("cudaGRASP blocksCount limit of 200 exeeded");
					uint32_t threadsPerBlock = 256;
					size_t totalThreads = threadsPerBlock * blocksCount;

					inst::copy_to_symbol(instance);

					buffer<value_type> value_memory(totalThreads);
					buffer<weight_type> weight_memory(totalThreads * instance.dim());

					size_t closestPowerOf2 = 1;
					while (closestPowerOf2 < instance.size()) closestPowerOf2 *= 2;
					using MetricT = metric::Value<uint32_t>;
					buffer<typename MetricT::value_type> metric_memory(closestPowerOf2);
					metric::calculate<MetricT, index_type><<<1, closestPowerOf2>>>(metric_memory.data());
					buffer<index_type> index_memory(totalThreads * instance.size() + closestPowerOf2);

					buffer<result_type> result_memory(totalThreads);

					buffer<curandStateMtgp32> random_states(blocksCount);
					buffer<mtgp32_kernel_params> kernel_params(1);

					curand::MakeMTGP32Constants(kernel_params);
					curand::MakeMTGP32KernelState(random_states, kernel_params, blocksCount, time(NULL));

					sort::by_metric_desc<index_type, typename MetricT::value_type><<<1, closestPowerOf2>>>(
						index_memory.data(), metric_memory.data(), static_cast<index_type>(instance.size())
					);
					//index_memory.debug_print(0, instance.size(), 1);

					cycle_kernel<result_type, value_type, weight_type, index_type><<<blocksCount, threadsPerBlock>>>(
						random_states.data(),
						value_memory.data(), weight_memory.data(), result_memory.data(),
						index_memory.data(), instance.size() / 2
					);

					if (blocksCount > 1) {
						blocksCount /= 2;
						reductions::pick<result_type, value_type><<<1, blocksCount>>> (
							value_memory.data(),
							result_memory.data(),
							threadsPerBlock,
							totalThreads
						);
					}

					res::solution<result_type> result(instance.size());
					result_memory.get(&result._data);

					return result;
				}

				template res::solution32 runner(const inst::instance32<uint32_t, uint32_t>&, uint32_t, typename inst::instance32<uint32_t, uint32_t>::index_type );
				template res::solution32 runner(const inst::instance32<float, uint32_t>&, uint32_t, typename inst::instance32<float, uint32_t>::index_type );
				template res::solution32 runner(const inst::instance32<uint32_t, float>&, uint32_t, typename inst::instance32<uint32_t, float>::index_type );
				template res::solution32 runner(const inst::instance32<float, float>&, uint32_t, typename inst::instance32<float, float>::index_type );
				template res::solution64 runner(const inst::instance64<uint32_t, uint32_t>&, uint32_t, typename inst::instance64<uint32_t, uint32_t>::index_type );
				template res::solution64 runner(const inst::instance64<float, uint32_t>&, uint32_t, typename inst::instance64<float, uint32_t>::index_type );
				template res::solution64 runner(const inst::instance64<uint32_t, float>&, uint32_t, typename inst::instance64<uint32_t, float>::index_type );
				template res::solution64 runner(const inst::instance64<float, float>&, uint32_t, typename inst::instance64<float, float>::index_type );
			}
		}
	}
}


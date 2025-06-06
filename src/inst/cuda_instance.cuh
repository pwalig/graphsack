#pragma once
#include <device_launch_parameters.h>
#include <cstdint>

#include "../cuda/error_wrapper.cuh"
#include "cuda_instance.hpp"

#define GS_CUDA_INST_MAXN 64
#define GS_CUDA_INST_MAXM 5

namespace gs {
	namespace cuda {
		namespace inst {
			extern __constant__ uint32_t limits_u32[GS_CUDA_INST_MAXM];
			extern __constant__ float limits_f32[GS_CUDA_INST_MAXM];
			extern __constant__ uint32_t values_u32[GS_CUDA_INST_MAXN];
			extern __constant__ float values_f32[GS_CUDA_INST_MAXN];
			extern __constant__ uint32_t weights_u32[GS_CUDA_INST_MAXN * GS_CUDA_INST_MAXM];
			extern __constant__ float weights_f32[GS_CUDA_INST_MAXN * GS_CUDA_INST_MAXM];
			extern __constant__ uint32_t adjacency32[32];
			extern __constant__ uint64_t adjacency64[64];
			extern __constant__ size_t size;
			extern __constant__ size_t dim;

			template <typename ValueT>
			__host__ __device__ const ValueT* values();
			template <typename WeightT>
			__host__ __device__ const WeightT* limits();
			template <typename WeightT>
			__host__ __device__ const WeightT* weights();
			template <typename adjacency_base_type>
			__host__ __device__ const adjacency_base_type* adjacency();

			template <>
			inline __host__ __device__ const uint32_t* limits() { return limits_u32; }
			template <>
			inline __host__ __device__ const float* limits() { return limits_f32; }
			template <>
			inline __host__ __device__ const uint32_t* values() { return values_u32; }
			template <>
			inline __host__ __device__ const float* values() { return values_f32; }
			template <>
			inline __host__ __device__ const uint32_t* weights() { return weights_u32; }
			template <>
			inline __host__ __device__ const float* weights() { return weights_f32; }
			template <>
			inline __host__ __device__ const uint32_t* adjacency() { return adjacency32; }
			template <>
			inline __host__ __device__ const uint64_t* adjacency() { return adjacency64; }

			template <typename adjacency_base_type, typename index_type>
			inline __device__  bool has_connection_to(
				const adjacency_base_type* adjacency, index_type from, index_type to
			) {
				if (adjacency[from] & (adjacency_base_type(1) << to)) return true;
				else return false;
			}

			template <typename adjacency_base_type, typename index_type>
			inline __device__  bool has_connection_to(
				index_type from, index_type to
			) {
				if (inst::adjacency<adjacency_base_type>()[from] & (adjacency_base_type(1) << to)) return true;
				else return false;
			}

			template <typename InstanceT>
			inline void copy_to_symbol(const InstanceT& instance) {
				using value_type = typename InstanceT::value_type;
				using weight_type = typename InstanceT::weight_type;
				using adjacency_base_type = typename InstanceT::adjacency_base_type;
				size_t hostSize = instance.size();
				size_t hostDim = instance.dim();
				except::MemcpyToSymbol(&size, &hostSize, sizeof(size_t));
				except::MemcpyToSymbol(&dim, &hostDim, sizeof(size_t));
				except::MemcpyToSymbol(inst::limits<weight_type>(), instance.limits().data(), instance.dim() * sizeof(weight_type));
				except::MemcpyToSymbol(inst::values<value_type>(), instance.values().data(), instance.size() * sizeof(value_type));
				except::MemcpyToSymbol(inst::weights<weight_type>(), instance.weights().data(), instance.size() * instance.dim() * sizeof(weight_type));
				except::MemcpyToSymbol(inst::adjacency<adjacency_base_type>(), instance.graph_data(), instance.size() * sizeof(adjacency_base_type));
			}
		}
	}
}

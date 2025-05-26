#pragma once
#include <device_launch_parameters.h>
#include <cstdint>

#include "../cuda/error_wrapper.cuh"
#include "cuda_instance.hpp"

#define GS_CUDA_INST_MAXN 64
#define GS_CUDA_INST_MAXM 5

#define GS_CUDA_INST_WEIGHT_VALUE \
__constant__ uint32_t limits[GS_CUDA_INST_MAXM]; \
__constant__ uint32_t values[GS_CUDA_INST_MAXN]; \
__constant__ uint32_t weights[GS_CUDA_INST_MAXN * GS_CUDA_INST_MAXM];

#define GS_CUDA_INST_ADJACENCY \
__constant__ uint32_t adjacency32[32]; \
__constant__ uint64_t adjacency64[64]; \
template <typename adjacency_base_type> \
__host__ __device__ const adjacency_base_type* adjacency(); \
template <> \
__host__ __device__ const uint32_t* adjacency() { return adjacency32; } \
template <> \
__host__ __device__ const uint64_t* adjacency() { return adjacency64; } 

#define GS_CUDA_INST_CONSTANTS GS_CUDA_INST_WEIGHT_VALUE GS_CUDA_INST_ADJACENCY

#define GS_CUDA_INST_COPY_TO_SYMBOL_INLINE(inst) \
assert(inst.size() <= sizeof(result_type) * 8); \
except::MemcpyToSymbol(limits, inst.limits().data(), inst.dim() * sizeof(uint32_t)); \
except::MemcpyToSymbol(values, inst.values().data(), inst.size() * sizeof(uint32_t)); \
except::MemcpyToSymbol(weights, inst.weights().data(), inst.size() * inst.dim() * sizeof(uint32_t)); \
except::MemcpyToSymbol(adjacency<result_type>(), instance.graph_data(), instance.size() * sizeof(result_type)); \
except::DeviceSynchronize();

namespace gs {
	namespace cuda {
		namespace inst {
			extern __constant__ uint32_t limits_u32[GS_CUDA_INST_MAXM];
			extern __constant__ uint32_t values_u32[GS_CUDA_INST_MAXN];
			extern __constant__ uint32_t weights_u32[GS_CUDA_INST_MAXN * GS_CUDA_INST_MAXM];
			extern __constant__ uint32_t adjacency32[32];
			extern __constant__ uint64_t adjacency64[64];

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
			inline __host__ __device__ const uint32_t* values() { return values_u32; }
			template <>
			inline __host__ __device__ const uint32_t* weights() { return weights_u32; }
			template <>
			inline __host__ __device__ const uint32_t* adjacency() { return adjacency32; }
			template <>
			inline __host__ __device__ const uint64_t* adjacency() { return adjacency64; }

			//extern template __host__ __device__ const uint32_t* limits<uint32_t>();
			//extern template __host__ __device__ const uint32_t* values<uint32_t>();
			//extern template __host__ __device__ const uint32_t* weights<uint32_t>();
			//extern template __host__ __device__ const uint32_t* adjacency<uint32_t>();
			//extern template __host__ __device__ const uint64_t* adjacency<uint64_t>();

			template <typename adjacency_base_type, typename index_type>
			inline __device__  bool has_connection_to(
				const adjacency_base_type* adjacency, index_type from, index_type to
			) {
				if (adjacency[from] & (adjacency_base_type(1) << to)) return true;
				else return false;
			}

			template <typename InstanceT>
			inline void copy_to_symbol(const InstanceT& instance) {
				using value_type = typename InstanceT::value_type;
				using weight_type = typename InstanceT::weight_type;
				using adjacency_base_type = typename InstanceT::adjacency_base_type;
				except::MemcpyToSymbol(inst::limits<weight_type>(), instance.limits().data(), instance.dim() * sizeof(weight_type));
				except::MemcpyToSymbol(inst::values<value_type>(), instance.values().data(), instance.size() * sizeof(value_type));
				except::MemcpyToSymbol(inst::weights<weight_type>(), instance.weights().data(), instance.size() * instance.dim() * sizeof(weight_type));
				except::MemcpyToSymbol(inst::adjacency<adjacency_base_type>(), instance.graph_data(), instance.size() * sizeof(adjacency_base_type));
			}
		}
	}
}

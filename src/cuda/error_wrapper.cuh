#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <assert.h>

#define GS_CUDA_THROW_LAST_ERROR(error_type) throw error_type(cudaGetErrorName(cudaGetLastError()));
#define GS_CUDA_EXCEPT_CALL(...) if (__VA_ARGS__ != cudaSuccess) GS_CUDA_THROW_LAST_ERROR(std::runtime_error)
#define GS_CUDA_ASSERT_CALL(...) { cudaError_t status = __VA_ARGS__; assert(status == cudaSuccess); }

namespace gs {
	namespace cuda {
		namespace except {
			template <typename T>
			inline T* Malloc(size_t size) {
				T* devPtr;
				GS_CUDA_EXCEPT_CALL(cudaMalloc(&devPtr, size))
				return devPtr;
			}

			inline void Free(void* devPtr) {
				GS_CUDA_EXCEPT_CALL(cudaFree(devPtr))
			}

			inline void DeviceSynchronize() {
				GS_CUDA_EXCEPT_CALL(cudaDeviceSynchronize())
			}
		}
		namespace assert {
			template <typename T>
			inline T* Malloc(size_t size) {
				T* devPtr;
				GS_CUDA_ASSERT_CALL(cudaMalloc(devPtr, size))
				return devPtr;
			}

			inline void Free(void* devPtr) {
				GS_CUDA_ASSERT_CALL(cudaFree(devPtr))
			}

			inline void DeviceLSynchronize() {
				GS_CUDA_ASSERT_CALL(cudaDeviceSynchronize())
			}
		}
	}
}
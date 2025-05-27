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
			inline int GetDeviceCount() {
				int deviceCount;
				GS_CUDA_EXCEPT_CALL(cudaGetDeviceCount(&deviceCount))
				return deviceCount;
			}

			inline void SetDevice(int i) {
				GS_CUDA_EXCEPT_CALL(cudaSetDevice(i))
			}

			template <typename T>
			inline T* Malloc(size_t size) {
				T* devPtr;
				GS_CUDA_EXCEPT_CALL(cudaMalloc<T>(&devPtr, size))
				return devPtr;
			}

			inline void Free(void* devPtr) {
				GS_CUDA_EXCEPT_CALL(cudaFree(devPtr))
			}

			inline void DeviceSynchronize() {
				GS_CUDA_EXCEPT_CALL(cudaDeviceSynchronize())
			}

			inline void Memcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
				GS_CUDA_EXCEPT_CALL(cudaMemcpy(dst, src, count, kind))
			}

			inline void MemcpyToSymbol(
				const void* symbol, const void* src, size_t count,
				size_t offset = 0Ui64, cudaMemcpyKind kind = cudaMemcpyHostToDevice
			) {
				GS_CUDA_EXCEPT_CALL(cudaMemcpyToSymbol(symbol, src, count, offset, kind))
			}

			inline void MemcpyToSymbolAsync(
				const void* symbol, const void* src, size_t count,
				size_t offset = 0Ui64, cudaMemcpyKind kind = cudaMemcpyHostToDevice,
				cudaStream_t stream = cudaStream_t(0)
			) {
				GS_CUDA_EXCEPT_CALL(cudaMemcpyToSymbolAsync(symbol, src, count, offset, kind, stream))
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

			inline void DeviceSynchronize() {
				GS_CUDA_ASSERT_CALL(cudaDeviceSynchronize())
			}
		}
	}
}
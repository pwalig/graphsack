#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>

namespace gs {
	namespace cuda {
		template <typename T>
		class buffer {
		public:
			using value_type = T;
			using pointer = T*;
		private:
			pointer ptr;
		public:
			buffer() = delete;
			inline buffer(size_t siz) {
				if (cudaMalloc(&ptr, siz * sizeof(value_type)) != cudaSuccess) {
					throw std::bad_alloc();
				}
#ifdef GS_CUDA_BUFFER_DIAGNOSTIC
				printf("allocated %d bytes of GPU memory\n", siz * sizeof(value_type));
#endif
			}

			buffer(const buffer& other) = delete;
			buffer& operator=(const buffer& other) = delete;

			inline ~buffer() {
				cudaFree(ptr);
#ifdef GS_CUDA_BUFFER_DIAGNOSTIC
				printf("freed GPU memory\n");
#endif
			}

			inline pointer data() { return ptr; }

			inline void get(pointer dst, size_t count = 1, size_t offset = 0) {
				if (cudaMemcpy(dst, ptr + offset, count * sizeof(value_type), cudaMemcpyDeviceToHost) != cudaSuccess) {
					throw std::runtime_error("failed to copy data from GPU do CPU");
				}
			}

			inline void set(const pointer src, size_t count = 1, size_t offset = 0) {
				if (cudaMemcpy(ptr + offset, src, count * sizeof(value_type), cudaMemcpyHostToDevice) != cudaSuccess) {
					throw std::runtime_error("failed to copy data from CPU do GPU");
				}
			}
		};
	}
}
#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <iostream>

#include "error_wrapper.cuh"

//#define GS_CUDA_BUFFER_DIAGNOSTIC
namespace gs {
	namespace cuda {
		template <typename T>
		class buffer {
		public:
			using value_type = T;
			using pointer = T*;
		private:
			pointer ptr;
			size_t siz;
		public:
			buffer() = delete;
			inline buffer(size_t Size) : siz(Size) {
				if (cudaMalloc(&ptr, Size * sizeof(value_type)) != cudaSuccess) {
					throw std::bad_alloc();
				}
#ifdef GS_CUDA_BUFFER_DIAGNOSTIC
				printf("allocated %llu bytes of GPU memory\n", siz * sizeof(value_type));
#endif
			}

			buffer(const buffer& other) = delete;
			buffer& operator=(const buffer& other) = delete;

			inline ~buffer() {
				except::Free(ptr);
#ifdef GS_CUDA_BUFFER_DIAGNOSTIC
				printf("freed %llu bytes of GPU memory\n", siz * sizeof(value_type));
#endif
			}

			inline pointer data() { return ptr; }

			inline void get(pointer dst, size_t count = 1, size_t offset = 0) const {
				except::Memcpy(dst, ptr + offset, count * sizeof(value_type), cudaMemcpyDeviceToHost);
			}

			inline void debug_print(size_t start, size_t stop, size_t step) {
				pointer host = new value_type[siz];
				get(host, siz);
				for (size_t i = start; i < stop; i += step) std::cout << (uint64_t)host[i] << '\n';
				delete[] host;
			}

			inline void set(const pointer src, size_t count = 1, size_t offset = 0) {
				except::Memcpy(ptr + offset, src, count * sizeof(value_type), cudaMemcpyHostToDevice);
			}
		};
	}
}
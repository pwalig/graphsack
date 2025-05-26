#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>

#include <stdexcept>
#include <assert.h>

#include "buffer.cuh"

#define GS_CURAND_THROW(error_type, status) throw error_type(gs::curand::GetErrorString(status));
#define GS_CURAND_EXCEPT_CALL(...) { curandStatus_t status = __VA_ARGS__; if (status != curandStatus::CURAND_STATUS_SUCCESS) GS_CURAND_THROW(std::runtime_error, status) }
#define GS_CURAND_ASSERT_CALL(...) { curandStatus_t status = __VA_ARGS__; assert(status == CURAND_STATUS_SUCCESS) }

namespace gs {
	namespace curand {
        const char* GetErrorString(curandStatus_t error) {
			switch (error)
			{
				case CURAND_STATUS_SUCCESS:
					return "CURAND_STATUS_SUCCESS";

				case CURAND_STATUS_VERSION_MISMATCH:
					return "CURAND_STATUS_VERSION_MISMATCH";

				case CURAND_STATUS_NOT_INITIALIZED:
					return "CURAND_STATUS_NOT_INITIALIZED";

				case CURAND_STATUS_ALLOCATION_FAILED:
					return "CURAND_STATUS_ALLOCATION_FAILED";

				case CURAND_STATUS_TYPE_ERROR:
					return "CURAND_STATUS_TYPE_ERROR";

				case CURAND_STATUS_OUT_OF_RANGE:
					return "CURAND_STATUS_OUT_OF_RANGE";

				case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
					return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

				case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
					return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

				case CURAND_STATUS_LAUNCH_FAILURE:
					return "CURAND_STATUS_LAUNCH_FAILURE";

				case CURAND_STATUS_PREEXISTING_FAILURE:
					return "CURAND_STATUS_PREEXISTING_FAILURE";

				case CURAND_STATUS_INITIALIZATION_FAILED:
					return "CURAND_STATUS_INITIALIZATION_FAILED";

				case CURAND_STATUS_ARCH_MISMATCH:
					return "CURAND_STATUS_ARCH_MISMATCH";

				case CURAND_STATUS_INTERNAL_ERROR:
					return "CURAND_STATUS_INTERNAL_ERROR";
			}

			return "CURAND_UNKNOWN_ERROR";
		}

		inline void MakeMTGP32Constants(cuda::buffer<mtgp32_kernel_params>& kernel_params) {
			GS_CURAND_EXCEPT_CALL(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, kernel_params.data()))

		}

		inline void MakeMTGP32KernelState(
			cuda::buffer<curandStateMtgp32>& random_states,
			cuda::buffer<mtgp32_kernel_params>& kernel_params,
			int n, unsigned long long seed
		) {
			GS_CURAND_EXCEPT_CALL(curandMakeMTGP32KernelState(
				random_states.data(), mtgp32dc_params_fast_11213, kernel_params.data(), n, seed
			))
		}
	}
}
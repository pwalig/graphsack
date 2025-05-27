#pragma once

namespace gs {
	namespace cuda {
		struct device_properties_t {
			size_t       totalGlobalMem;             /**< Global memory available on device in bytes */
			size_t       sharedMemPerBlock;          /**< Shared memory available per block in bytes */
			size_t       totalConstMem;              /**< Constant memory available on device in bytes */
			int          maxThreadsPerBlock;         /**< Maximum number of threads per block */
			int          canMapHostMemory;           /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
		};
		device_properties_t init();
	}
}

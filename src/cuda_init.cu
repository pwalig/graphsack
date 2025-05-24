#include "cuda_init.h"

#include "cuda_runtime.h"
#include <stdexcept>

gs::cuda::device_properties gs::cuda::init() {
    int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		device_properties current = {
			prop.totalGlobalMem,
			prop.sharedMemPerBlock,
			prop.totalConstMem,
			prop.maxThreadsPerBlock,
			prop.canMapHostMemory
		};
		cudaSetDevice(i);
		return current;
	}

	throw std::runtime_error("No devices found");
}
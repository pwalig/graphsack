#include "cuda_init.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#include "cuda/error_wrapper.cuh"

gs::cuda::device_properties gs::cuda::init() {
    int nDevices = except::GetDeviceCount();

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
		std::cout << "selected: " << prop.name << '\n';
		except::SetDevice(i);
		return current;
	}

	throw std::runtime_error("No devices found");
}
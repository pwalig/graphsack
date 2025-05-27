#include "cuda_init.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#include "cuda/error_wrapper.cuh"
#include "cuda/device_properties.cuh"

cudaDeviceProp gs::cuda::device_properties;

gs::cuda::device_properties_t gs::cuda::init() {
    int nDevices = except::GetDeviceCount();

	for (int i = 0; i < nDevices; i++) {
		cudaGetDeviceProperties(&device_properties, i);
		device_properties_t current = {
			device_properties.totalGlobalMem,
			device_properties.sharedMemPerBlock,
			device_properties.totalConstMem,
			device_properties.maxThreadsPerBlock,
			device_properties.canMapHostMemory
		};
		std::cout << "selected: " << device_properties.name << '\n';
		except::SetDevice(i);
		return current;
	}

	throw std::runtime_error("No devices found");
}
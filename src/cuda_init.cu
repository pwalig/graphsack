#include "cuda_init.h"

#include "cuda_runtime.h"

void gs::cuda::init() {
    int nDevices;
	int bestDevice = -1;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		//printf("Device Number: %d\n", i);
		//printf("  Device name: %s\n", prop.name);
		//printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		//printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		//printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		//printf("  Threads Per Block: %d\n", prop.maxThreadsPerBlock);
		//printf("  Can Map Host Memory: %d\n", prop.canMapHostMemory);
		//printf("  totalConstMem: %llu\n", prop.totalConstMem);
		//printf("  totalGlobalMem: %llu\n" ,prop.totalGlobalMem);	
	}

	cudaSetDevice(0);
	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
}
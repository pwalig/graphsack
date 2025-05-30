#include "cuda_test.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void gs::cuda::test()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

void gs::cuda::info::print_json()
{
    cudaDeviceProp  prop;
	int count;
	printf("{\n");
	cudaGetDeviceCount(&count);
	// printf("\"DeviceCount\":\"%d\",\n",count);
	printf("\t\"Devices\":[");
	for (int i=0;i<count;i++) {
		printf("\n\t{\n");
		cudaGetDeviceProperties(&prop,i);
		printf("\t\t\"name\":\"%s\",\n",prop.name);
		printf("\t\t\"major\":\"%d\",\n",prop.major);
		printf("\t\t\"minor\":\"%d\",\n",prop.minor);
		printf("\t\t\"computeMode\":\"%d\",\n",prop.computeMode);
		printf("\t\t\"integrated\":\"%d\",\n",prop.integrated);
		printf("\t\t\"tccDriver\":\"%d\",\n",prop.tccDriver);
		printf("\t\t\"ECCEnabled\":\"%d\",\n",prop.ECCEnabled);
		printf("\t\t\"deviceOverlap\":\"%d\",\n",prop.deviceOverlap); // Przeplatanie urz¹dzeñ
		printf("\t\t\"concurrentKernels\":\"%d\",\n",prop.concurrentKernels);
		printf("\t\t\"kernelExecTimeoutEnabled\":\"%d\",\n",prop.kernelExecTimeoutEnabled);
		printf("\t\t\"canMapHostMemory\":\"%d\",\n",prop.canMapHostMemory);
		printf("\t\t\"multiProcessorCount\":\"%d\",\n",prop.multiProcessorCount);
		printf("\t\t\"warpSize\":\"%d\",\n",prop.warpSize);
		printf("\t\t\"clockRate\":\"%d\",\n",prop.clockRate);
		printf("\t\t\"maxThreadsPerBlock\":\"%d\",\n",prop.maxThreadsPerBlock);
		printf("\t\t\"maxThreadsDim\":[%d,%d,%d],\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("\t\t\"maxGridSize\":[%d,%d,%d],\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\t\t\"maxTexture1D\":\"%d\",\n",prop.maxTexture1D);
		printf("\t\t\"maxTexture2D\":[%d,%d],\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
		printf("\t\t\"maxTexture3D\":[%d,%d,%d],\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
		printf("\t\t\"pciBusID\":\"%d\",\n",prop.pciBusID);
		printf("\t\t\"pciDeviceID\":\"%d\",\n",prop.pciDeviceID);
		printf("\t\t\"regsPerBlock\":\"%d\",\n",prop.regsPerBlock);
		// size_t
		printf("\t\t\"memPitch\":\"%llu\",\n",prop.memPitch);
		printf("\t\t\"surfaceAlignment\":\"%llu\",\n",prop.surfaceAlignment);
		printf("\t\t\"textureAlignment\":\"%llu\",\n",prop.textureAlignment);
		printf("\t\t\"totalConstMem\":\"%llu\",\n",prop.totalConstMem);
		printf("\t\t\"totalGlobalMem\":\"%llu\",\n",prop.totalGlobalMem);	
		printf("\t\t\"sharedMemPerBlock\":\"%llu\",\n",prop.sharedMemPerBlock);
		// size_t
		printf("\t\t\"memPitch\":\"%zu\",\n",prop.memPitch);
		printf("\t\t\"surfaceAlignment\":\"%zu\",\n",prop.surfaceAlignment);
		printf("\t\t\"textureAlignment\":\"%zu\",\n",prop.textureAlignment);
		printf("\t\t\"totalConstMem\":\"%zu\",\n",prop.totalConstMem);
		printf("\t\t\"totalGlobalMem\":\"%zu\",\n",prop.totalGlobalMem);
		printf("\t\t\"sharedMemPerBlock\":\"%zu\",\n",prop.sharedMemPerBlock);
		printf("\t\t\"canMapHostMemory\":\"%d\",\n",prop.canMapHostMemory);
		printf("\t\t\"l2CacheSize\":\"%d\"\n",prop.l2CacheSize);
		printf("\t\t\"persistingL2CacheMaxSize\":\"%d\"\n",prop.persistingL2CacheMaxSize);
		printf("\t\t\"regsPerBlock\":\"%d\"\n",prop.regsPerBlock);
		if (i<count-1) printf("\t},\n"); else printf("\t}\n");
	}
	printf("\t]\n}\n");
}

void gs::cuda::info::print()
{
	printf("Cuda Version: %d\n", CUDART_VERSION);

    int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		printf("  Threads Per Block: %d\n", prop.maxThreadsPerBlock);
		printf("  Can Map Host Memory: %d\n", prop.canMapHostMemory);
		printf("  totalConstMem: %llu\n", prop.totalConstMem);
		printf("  totalGlobalMem: %llu\n" ,prop.totalGlobalMem);	
	}
}

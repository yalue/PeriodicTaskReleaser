#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>

#include "GPUOp.h"

/**
* CUDA Kernel Device code
*
* Computes the vector addition of A and B into C. The 3 vectors have the same
* number of elements numElements.
*/
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}


void run_GPU_Op(int numElements, int sync_level) {
    /*
    * The sync_level parameter is an integer that indicates the desired level of
    * synchronization used by the GPU driver (values defined below).  The
    * specified level is used in cudaSetDeviceFlags() to set the level
    * prior to initialization.
    */
    switch (sync_level)
    {
        case 0:
        cudaSetDeviceFlags(cudaDeviceScheduleSpin);
        break;
        case 1:
        cudaSetDeviceFlags(cudaDeviceScheduleYield);
        break;
        default:
        break;
    }

    // Follow convention and initialize CUDA/GPU
    // used here to invoke initialization of GPU locking
    cudaFree(0);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // create a user defined stream
    cudaStream_t my_stream;
    cudaStreamCreate(&my_stream);

    size_t size = numElements * sizeof(float); // 16,000,000 bytes

    float *h_A, *h_B, *h_C;

    // Host allocations in pinned memory
    // Allocate the host input vector A
    err = cudaMallocHost((void **) &h_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate host vector A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Allocate the host input vector B
    err = cudaMallocHost((void **) &h_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate host vector B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Allocate the host output vector C
    err = cudaMallocHost((void **)&h_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate host vector C (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        return;
    }


    // copy the A and B vectors from Host to Device memory
    // these calls are asynchronous so only the lock of CE can be handled in the wrapper
    err = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, my_stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // synchronize with the stream
    // the wrapper for this function releases any lock held (CE here)
    cudaStreamSynchronize(my_stream);

    err = cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, my_stream);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // synchronize with the stream
    // the wrapper for this function releases any lock held (CE here)
    cudaStreamSynchronize(my_stream);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;

    // lock of EE is handled in wrapper for cudaLaunch()
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, my_stream>>>(d_A, d_B, d_C, numElements);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // synchronize with the stream after kernel execution
    // the wrapper for this function releases any lock held (EE here)
    cudaStreamSynchronize(my_stream);

    // copy the result vector from Device to Host memory
    // this call is asynchronous so only the lock of CE can be handled in the wrapper
    err = cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, my_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // synchronize with the stream
    // the wrapper for this function releases any lock held (CE here)
    cudaStreamSynchronize(my_stream);

    // Free device global memory for inputs A and B and result C
    err = cudaFree(d_A);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(d_B);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(d_C);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Free host memory that was pinned
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    // clean up the user allocated stream
    cudaStreamSynchronize(my_stream);
    cudaStreamDestroy(my_stream);

    // Reset the device and return
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application returns
    err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
    }
}


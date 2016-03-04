/**
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

extern "C" {
#include "mm.h"
}


extern "C"
void mm_init_GPU_Op(int sync_level) {
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
}

extern "C"
void mm_finish_GPU_Op() {
     // Reset the device and return
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application returns
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
    }
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

extern "C"
void mm_run_GPU_Op(int numElements) {
    float *h_A, *h_B, *h_C;
    cudaError_t err = cudaSuccess;
    
    // Stream for the thread's GPU Operations
    cudaStream_t my_stream;

    // create a user defined stream
    cudaStreamCreate(&my_stream);

    // Allocate host memory for matrices A and B
    unsigned int size = sizeof(float) * numElements;
    err = cudaMallocHost((void **) &h_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate host memory A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMallocHost((void **) &h_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate host memory B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    // Allocate host matrix C
    err = cudaMallocHost((void **) &h_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate host memory C (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Initialize host memory
    // const float valB = 0.01f;
    // constantInit(h_A, size_A, 1.0f);
    // constantInit(h_B, size_B, valB);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    err = cudaMalloc((void **) &d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc((void **) &d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMalloc((void **) &d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory C (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // copy the A and B blocks from Host to Device memory
    // these calls are asynchronous so only the lock of CE can be handled in the wrapper
    err = cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, my_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy memory A from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // synchronize with the stream
    // the wrapper for this function releases any lock held (CE here)
    cudaStreamSynchronize(my_stream);

    err = cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, my_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy memory B from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // synchronize with the stream
    // the wrapper for this function releases any lock held (CE here)
    cudaStreamSynchronize(my_stream);

    // Setup execution parameters
    int block_size = 16;
    dim3 dimsA(5*2*block_size, 5*2*block_size, 1);
    dim3 dimsB(5*4*block_size, 5*2*block_size, 1);
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch matrixMul kernel (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    
    // synchronize with the stream after kernel execution
    // the wrapper for this function releases any lock held (EE here)
    cudaStreamSynchronize(my_stream);

    // copy the result memory from Device to Host memory
    // this call is asynchronous so only the lock of CE can be handled in the wrapper
    err = cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, my_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy memory C from device to host (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    // synchronize with the stream
    // the wrapper for this function releases any lock held (CE here)
    cudaStreamSynchronize(my_stream);

    // Free device global memory for inputs A and B and result C
    err = cudaFree(d_A);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device memory A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(d_B);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device memory B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(d_C);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to free device memory C (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    // Free host memory that was pinned
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    // clean up the user allocated stream
    cudaStreamSynchronize(my_stream);
    cudaStreamDestroy(my_stream);
}

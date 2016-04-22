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
#include "../mm.h"
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
  template <int BLOCK_SIZE> __global__ void
      matrixMulCUDA(float *C, float *A, float *B, int wA, int wB) {
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
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
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

    for (int k = 0; k < BLOCK_SIZE; ++k) {
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

void constantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

// Stream for the thread's GPU Operations
cudaStream_t mm_stream;

float *h_mA, *h_mB, *h_mC;
float *d_mA, *d_mB, *d_mC;
unsigned int matrix_size;

dim3 dimsA;
dim3 dimsB;
dim3 threads;
dim3 grid;

extern "C" void mm_init(int sync_level) {
  /*
   * The sync_level parameter is an integer that indicates the desired level of
   * synchronization used by the GPU driver (values defined below).  The
   * specified level is used in cudaSetDeviceFlags() to set the level
   * prior to initialization.
   */
  switch (sync_level) {
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
  
  // create a user defined stream
  cudaStreamCreate(&mm_stream);
}

extern "C" void mm_mallocHost(int numElements) {
  int mem_size = sqrt(numElements) * sqrt(numElements);

  // Allocate host memory for matrices A and B
  matrix_size = sizeof(float) * mem_size;
  cudaError_t err = cudaMallocHost((void **) &h_mA, matrix_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host memory A (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  err = cudaMallocHost((void **) &h_mB, matrix_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host memory B (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  // Allocate host matrix C
  err = cudaMallocHost((void **) &h_mC, matrix_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host memory C (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Initialize host memory
  constantInit(h_mA, mem_size, 1.0f);
  constantInit(h_mB, mem_size, 0.01f);

  // Setup execution parameters
  int block_size = 16;
  dimsA = dim3(5*2*block_size, 5*2*block_size, 1);
  dimsA.x = sqrt(mem_size);
  dimsA.y = sqrt(mem_size);
  dimsB = dim3(5*4*block_size, 5*2*block_size, 1);
  dimsB.x = sqrt(mem_size);
  dimsB.y = sqrt(mem_size);
  threads = dim3(block_size, block_size);
  grid = dim3(ceil(dimsB.x / (float) threads.x), ceil(dimsA.y / (float) threads.y));
}

extern "C" void mm_cudaMalloc(int numElements) {
  int mem_size = sqrt(numElements) * sqrt(numElements);

  // Allocate device memory
  cudaError_t err = cudaMalloc((void **) &d_mA, matrix_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory A (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  err = cudaMalloc((void **) &d_mB, matrix_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory B (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  err = cudaMalloc((void **) &d_mC, matrix_size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory C (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
}

extern "C" void mm_copyin(int numElements) {
  // copy the A and B blocks from Host to Device memory
  // these calls are asynchronous so only the lock of CE can be handled in the wrapper
  cudaError_t err = cudaMemcpyAsync(d_mA, h_mA, matrix_size, cudaMemcpyHostToDevice, mm_stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy memory A from host to device (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(mm_stream);

  err = cudaMemcpyAsync(d_mB, h_mB, matrix_size, cudaMemcpyHostToDevice, mm_stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy memory B from host to device (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(mm_stream);
}

extern "C" void mm_exec(int numElements) {
  cudaError_t err = cudaSuccess;
  matrixMulCUDA<16><<< grid, threads, 0, mm_stream>>>(d_mC, d_mA, d_mB, dimsA.x, dimsB.x);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch matrixMul kernel (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  // synchronize with the stream after kernel execution
  // the wrapper for this function releases any lock held (EE here)
  cudaStreamSynchronize(mm_stream);
}

extern "C" void mm_copyout() {
  // copy the result memory from Device to Host memory
  // this call is asynchronous so only the lock of CE can be handled in the wrapper
  cudaError_t err = cudaMemcpyAsync(h_mC, d_mC, matrix_size, cudaMemcpyDeviceToHost, mm_stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy memory C from device to host (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(mm_stream);
}

extern "C" void mm_cudaFree() {
  // Free device global memory for inputs A and B and result C
  cudaError_t err = cudaFree(d_mA);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device memory A (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  err = cudaFree(d_mB);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device memory B (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  err = cudaFree(d_mC);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device memory C (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
}

extern "C" void mm_freeHost() {
  // Free host memory that was pinned
  cudaFreeHost(h_mA);
  cudaFreeHost(h_mB);
  cudaFreeHost(h_mC);
}

extern "C" void mm_finish() {
  // clean up the user allocated stream
  cudaStreamSynchronize(mm_stream);
  cudaStreamDestroy(mm_stream);
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
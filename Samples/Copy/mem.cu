// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

extern "C" {
#include "../mem.h"
}

// Stream for the thread's GPU Operations
cudaStream_t mm_stream;

float *h_memtest;
float *d_memtest;

extern "C" void memtest_init(int sync_level, int numElements) {
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

  cudaError_t err = cudaMallocHost((void **) &h_memtest, numElements);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host memory (error code %s)!\n", cudaGetErrorString(err));
  }
}

extern "C" void memtest_alloc(int numElements) {
  cudaError_t err = cudaSuccess;

  // Allocate device memory
  err = cudaMalloc((void **) &d_memtest, numElements);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
}

#define SPLITSIZE 8192
extern "C" void memtest_copyin(int numElements) {
  // these calls are asynchronous so only the lock of CE can be handled in the wrapper
  cudaError_t err = cudaMemcpyAsync(d_memtest, h_memtest, numElements, cudaMemcpyHostToDevice, mm_stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy memory from host to device (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(mm_stream);
}

extern "C" void memtest_copyout(int numElements) {
  // this call is asynchronous so only the lock of CE can be handled in the wrapper
  cudaError_t err = cudaMemcpyAsync(h_memtest, d_memtest, numElements, cudaMemcpyDeviceToHost, mm_stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy memory from device to host (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(mm_stream);
}

extern "C" void memtest_cudafree() {
  // Free device global memory for inputs A and B and result C
  cudaError_t err = cudaFree(d_memtest);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device memory A (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
}

extern "C" void memtest_cleanup() {
  // Free host memory that was pinned
  cudaFreeHost(h_memtest);

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

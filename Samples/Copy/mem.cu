// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

extern "C" {
#include "../gpusync.h"
}

// Stream for the thread's GPU Operations
cudaStream_t stream;

float *h;
float *d;
int size;

extern "C" void init(int sync_level) {
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
  cudaStreamCreate(&stream);
}

extern "C" void mallocCPU(int numElements) {
  cudaError_t err = cudaMallocHost((void **) &h, numElements);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host memory (error code %s)!\n", cudaGetErrorString(err));
  }
}

extern "C" void mallocGPU(int numElements) {
  cudaError_t err = cudaSuccess;

  // Allocate device memory
  err = cudaMalloc((void **) &d, numElements);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
}

#define SPLITSIZE 8192
extern "C" void copyin(int numElements) {
  size = numElements;
  // these calls are asynchronous so only the lock of CE can be handled in the wrapper
  cudaError_t err = cudaMemcpyAsync(d, h, numElements, cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy memory from host to device (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(stream);
}

extern "C" void exec(int numElements) {
  // Nothing to do
}

extern "C" void copyout() {
  // this call is asynchronous so only the lock of CE can be handled in the wrapper
  cudaError_t err = cudaMemcpyAsync(h, d, size, cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy memory from device to host (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(stream);
}

extern "C" void freeGPU() {
  // Free device global memory for inputs A and B and result C
  cudaError_t err = cudaFree(d);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device memory A (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
}

extern "C" void freeCPU() {
  // Free host memory that was pinned
  cudaFreeHost(h);
}

extern "C" void finish() {
  // clean up the user allocated stream
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

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

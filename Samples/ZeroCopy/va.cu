#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>

extern "C" {
#include "../va.h"
}

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

// Stream for the thread's GPU Operations
cudaStream_t va_stream;

// Memory regions
float *h_vA, *h_vB, *h_vC;
float *d_vA, *d_vB, *d_vC;
size_t vector_len;
int v_threadsPerBlock;
int v_blocksPerGrid;

extern "C" void va_init(int sync_level) {
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

  // Set up zero copy
  cudaSetDeviceFlags(cudaDeviceMapHost);

  // Follow convention and initialize CUDA/GPU
  // used here to invoke initialization of GPU locking
  cudaFree(0);

  // create a user defined stream
  cudaStreamCreate(&va_stream);
}

extern "C" void va_mallocHost(int numElements) {
  vector_len = numElements * sizeof(float);

  // Host allocations in pinned memory
  // Allocate the host input vector A
  cudaError_t err = cudaHostAlloc((void **) &h_vA, vector_len, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host vector A (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Allocate the host input vector B
  err = cudaHostAlloc((void **) &h_vB, vector_len, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host vector B (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Allocate the host output vector C
  err = cudaHostAlloc((void **)&h_vC, vector_len, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host vector C (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    h_vA[i] = rand()/(float)RAND_MAX;
    h_vB[i] = rand()/(float)RAND_MAX;
  }
  v_threadsPerBlock = 256;
  v_blocksPerGrid = (numElements + v_threadsPerBlock - 1) / v_threadsPerBlock;
}

extern "C" void va_cudaMalloc(int numElements) {
  // Allocate the device input vector A
  cudaError_t err = cudaHostGetDevicePointer((void **)&d_vA, (void *) h_vA, 0);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Allocate the device input vector B
  err = cudaHostGetDevicePointer((void **)&d_vB, (void *) h_vB, 0);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Allocate the device output vector C
  err = cudaHostGetDevicePointer((void **)&d_vC, (void *) h_vC, 0);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(va_stream);
}

extern "C" void va_copyin(int numElements) {
  // NOP
}

extern "C" void va_exec(int numElements) {
  cudaError_t err = cudaSuccess;

  // Launch the Vector Add CUDA Kernel
  // lock of EE is handled in wrapper for cudaLaunch()
  vectorAdd<<<v_blocksPerGrid, v_threadsPerBlock, 0, va_stream>>>(d_vA, d_vB, d_vC, numElements);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  // synchronize with the stream after kernel execution
  // the wrapper for this function releases any lock held (EE here)
  cudaStreamSynchronize(va_stream);
}

extern "C" void va_copyout() {
  // NOP
}

extern "C" void va_cudaFree() {
  // NOP
}

extern "C" void va_freeHost() {
  // Free host memory that was pinned
  cudaFreeHost(h_vA);
  cudaFreeHost(h_vB);
  cudaFreeHost(h_vC);
}
 
extern "C" void va_finish() {
  // clean up the user allocated stream
  cudaStreamSynchronize(va_stream);
  cudaStreamDestroy(va_stream);

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
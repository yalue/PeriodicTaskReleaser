#include <stdio.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>

extern "C" {
#include "../gpusync.h"
}

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements)
  {
    C[i] = A[i] + B[i];
  }
}

// Stream for the thread's GPU Operations
cudaStream_t stream;

// Memory regions
float *hA, *hB, *hC;
float *dA, *dB, *dC;
size_t vector_bytes;
int v_threadsPerBlock;
int v_blocksPerGrid;

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
    case 2:
      cudaSetDeviceFlags(cudaDeviceBlockingSync);
      break;
    default:
      fprintf(stderr, "Unknown sync level: %d\n", sync_level);
      break;
  }

  // Set up zero copy
  cudaSetDeviceFlags(cudaDeviceMapHost);

  // Follow convention and initialize CUDA/GPU
  // used here to invoke initialization of GPU locking
  cudaFree(0);

  // Pin code
  if(!mlockall(MCL_CURRENT | MCL_FUTURE)) {
    fprintf(stderr, "Failed to lock code pages.\n");
    exit(EXIT_FAILURE);
  }
 
  // Set the device context 
  cudaSetDevice(0);

  // create a user defined stream
  cudaStreamCreate(&stream);
}

extern "C" void mallocCPU(int numElements) {
  vector_bytes = numElements * sizeof(float);

  // Host allocations in pinned memory
  // Allocate the host input vector A
  cudaError_t err = cudaHostAlloc((void **) &hA, vector_bytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host vector A (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Allocate the host input vector B
  err = cudaHostAlloc((void **) &hB, vector_bytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host vector B (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Allocate the host output vector C
  err = cudaHostAlloc((void **)&hC, vector_bytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate host vector C (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    hA[i] = rand()/(float)RAND_MAX;
    hB[i] = rand()/(float)RAND_MAX;
  }
  v_threadsPerBlock = 256;
  v_blocksPerGrid = (numElements + v_threadsPerBlock - 1) / v_threadsPerBlock;
}

extern "C" void mallocGPU(int numElements) {
  // Allocate the device input vector A
  cudaError_t err = cudaHostGetDevicePointer((void **)&dA, (void *) hA, 0);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Allocate the device input vector B
  err = cudaHostGetDevicePointer((void **)&dB, (void *) hB, 0);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // Allocate the device output vector C
  err = cudaHostGetDevicePointer((void **)&dC, (void *) hC, 0);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
    return;
  }

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(stream);
}

extern "C" void copyin(int numElements) {
  // NOP
}

extern "C" void exec(int numElements) {
  cudaError_t err = cudaSuccess;

  // Launch the Vector Add CUDA Kernel
  // lock of EE is handled in wrapper for cudaLaunch()
  vectorAdd<<<v_blocksPerGrid, v_threadsPerBlock, 0, stream>>>(dA, dB, dC, numElements);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
    return;
  }
  // synchronize with the stream after kernel execution
  // the wrapper for this function releases any lock held (EE here)
  cudaStreamSynchronize(stream);
}

extern "C" void copyout() {
  // NOP
}

extern "C" void freeGPU() {
  // NOP
}

extern "C" void freeCPU() {
  // Free host memory that was pinned
  cudaFreeHost(hA);
  cudaFreeHost(hB);
  cudaFreeHost(hC);
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

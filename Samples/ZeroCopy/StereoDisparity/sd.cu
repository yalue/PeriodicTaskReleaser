/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* A CUDA program that demonstrates how to compute a stereo disparity map using
 *   SIMD SAD (Sum of Absolute Difference) intrinsics
 */

/*
 * The program's performance is dominated by 
 * the computation on the execution engine (EE) while memory copies 
 * between Host and Device using the copy engine (CE) are significantly
 * less time consuming.
 *
 * This version uses a user allocated stream and asynchronous memory
 * copy operations (cudaMemcpyAsync()).  Cuda kernel invocations on the
 * stream are also asynchronous.  cudaStreamSynchronize() is used to 
 * synchronize with both the copy and kernel executions.  Host pinned
 * memory is not used because the copy operations are not a significant 
 * element of performance.
 *
 * The program depends on two input files containing the image 
 * representations for the left and right stereo images 
 * (stereo.im0.640x533.ppm and stereo.im1.640x533.ppm)
 * which must be in the directory with the executable.
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>

// includes, kernels
// For the CUDA runtime routines (prefixed with "cuda")
#include <cuda_runtime.h>
// the kernel code
#include "sd_kernel.cuh"
// GPUSync header
 extern "C" {
#include "gpusync.h"
 }

// includes, project
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include <helper_cuda.h>       // helper for checking cuda initialization and error checking
#include <helper_string.h>     // helper functions for string parsing

// Stream for the thread's GPU Operations
 cudaStream_t stream;

// Host Memory
 unsigned int *h_odata;
 unsigned char *h_img0;
 unsigned char *h_img1;

// Device memory
 unsigned int *d_odata;
 unsigned int *d_img0;
 unsigned int *d_img1;

// Kernel execution parameters
unsigned int w, h;
size_t offset;
dim3 numThreads;
dim3 numBlocks;
unsigned int numData;
unsigned int memSize;
cudaChannelFormatDesc ca_desc0;
cudaChannelFormatDesc ca_desc1;

// Search parameters
int minDisp = -16;
int maxDisp = 0;

// Relative path to images
char fname0[] = "../Samples/Copy/StereoDisparity/data/stereo.im0.640x533.ppm";
char fname1[] = "../Samples/Copy/StereoDisparity/data/stereo.im1.640x533.ppm";


int iDivUp(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

////////////////////////////////////////////////////////////////////////////////
//! CUDA Sample for calculating depth maps
////////////////////////////////////////////////////////////////////////////////
    //initalize the memory for output data to zeros
    // for (unsigned int i = 0; i < numData; i++)
      // h_odata[i] = 0;

#ifdef PRINT_CHECKSUM
  // calculate sum of resultant GPU image
  // This verification is applied only to the
  // last result computed
  unsigned int checkSum = 0;
  for (unsigned int i=0 ; i<w *h ; i++) {
    checkSum += h_odata[i];
  }
  if (checkSum == 4293895789) //valid checksum only for these two images
    printf("PID %d Test PASSED\n", my_pid);
  else {
    fprintf(stderr, "PID %d verification failed, GPU Checksum = %u, ", my_pid, checkSum);
    exit(-1);
  }
#endif
#ifdef WRITE_DISPARITY
  // write out the resulting disparity image.
  // creates file in directory containing executable
  unsigned char *dispOut = (unsigned char *) malloc(numData);
  int mult = 20;

  char fnameOut[50] = "";
  sprintf(fnameOut,"PID_%d_", my_pid);
  strcat(fnameOut, "output_GPU.pgm");

  for (unsigned int i=0; i<numData; i++) {
    dispOut[i] = (int) h_odata[i]*mult;
  }

  printf("GPU image: <%s>\n", fnameOut);
  sdkSavePGM(fnameOut, dispOut, w, h);
  if (dispOut != NULL) free(dispOut);
#endif

// Override helper_image.h
inline bool loadPPM4ub(const char *file, unsigned char **data,
  unsigned int *w, unsigned int *h) {
  unsigned char *idata = 0;
  unsigned int channels;

  if (__loadPPM(file, &idata, w, h, &channels)) {
    // pad 4th component
    int size = *w **h;
    // keep the original pointer
    unsigned char *idata_orig = idata;
    checkCudaErrors(cudaMallocHost(data, sizeof(unsigned char) * size * 4));
    unsigned char *ptr = *data;

    for (int i=0; i<size; i++) {
      *ptr++ = *idata++;
      *ptr++ = *idata++;
      *ptr++ = *idata++;
      *ptr++ = 0;
    }

    free(idata_orig);
    return true;
  } else {
    free(idata);
    return false;
  }
}



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
  // Load image data
  // functions allocate memory for the images on host side
  // initialize pointers to NULL to request lib call to allocate as needed
  // PPM images are loaded into 4 byte/pixel memory (RGBX)
  h_img0 = NULL;
  h_img1 = NULL;

  if (!loadPPM4ub(fname0, &h_img0, &w, &h)) {
    fprintf(stderr, "Failed to load <%s>\n", fname0);
    exit(-1);
  }

  if (!loadPPM4ub(fname1, &h_img1, &w, &h)) {
    fprintf(stderr, "Failed to load <%s>\n", fname1);
    exit(-1);
  }

  // set up parameters used in the rest of program
  numThreads = dim3(blockSize_x, blockSize_y, 1);
  numBlocks = dim3(iDivUp(w, numThreads.x), iDivUp(h, numThreads.y));
  numData = w * h;
  memSize = sizeof(int) * numData;

  //allocate memory for the result on host side
  checkCudaErrors(cudaMallocHost(&h_odata, memSize));

  // more setup for using the GPU
  offset = 0;
  ca_desc0 = cudaCreateChannelDesc<unsigned int>();
  ca_desc1 = cudaCreateChannelDesc<unsigned int>();

  tex2Dleft.addressMode[0] = cudaAddressModeClamp;
  tex2Dleft.addressMode[1] = cudaAddressModeClamp;
  tex2Dleft.filterMode     = cudaFilterModePoint;
  tex2Dleft.normalized     = false;
  tex2Dright.addressMode[0] = cudaAddressModeClamp;
  tex2Dright.addressMode[1] = cudaAddressModeClamp;
  tex2Dright.filterMode     = cudaFilterModePoint;
  tex2Dright.normalized     = false;
}


extern "C" void mallocGPU(int unused) {
  // allocate device memory for inputs
  checkCudaErrors(cudaMalloc((void **) &d_img0, memSize));
  checkCudaErrors(cudaMalloc((void **) &d_img1, memSize));
  checkCudaErrors(cudaHostGetDevicePointer(&d_odata, h_odata, 0));

  checkCudaErrors(cudaBindTexture2D(&offset, tex2Dleft, d_img0, ca_desc0, w, h, w*4));
  assert(offset == 0);

  checkCudaErrors(cudaBindTexture2D(&offset, tex2Dright, d_img1, ca_desc1, w, h, w*4));
  assert(offset == 0);

}

extern "C" void copyin(int unused) {
  // copy host memory with images to device
  // this call is asynchronous so only the lock of CE can be handled in the wrapper
  checkCudaErrors(cudaMemcpyAsync(d_img0,  h_img0, memSize, cudaMemcpyHostToDevice, stream));

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(stream);

  // this call is asynchronous so only the lock of CE can be handled in the wrapper
  checkCudaErrors(cudaMemcpyAsync(d_img1,  h_img1, memSize, cudaMemcpyHostToDevice, stream));

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(stream);

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(stream);
}

extern "C" void exec(int unused) {
  // launch the stereoDisparity kernel
  // lock of EE is handled in wrapper for cudaLaunch()
  stereoDisparityKernel<<<numBlocks, numThreads, 0, stream>>>(d_img0, d_img1, d_odata, w, h, minDisp, maxDisp);

  // synchronize with the stream after kernel execution
  // the wrapper for this function releases any lock held (EE here)
  cudaStreamSynchronize(stream);

  // Check to make sure the kernel didn't fail
  getLastCudaError("Kernel execution failed");
}

extern "C" void copyout() {
  //Copy result from device to host for verification
  // this call is asynchronous so only the lock of CE can be handled in the wrapper
  checkCudaErrors(cudaMemcpyAsync(h_odata, d_odata, memSize, cudaMemcpyDeviceToHost, stream));

  // synchronize with the stream
  // the wrapper for this function releases any lock held (CE here)
  cudaStreamSynchronize(stream);
}

extern "C" void freeGPU() {
  // cleanup device memory
  checkCudaErrors(cudaFree(d_img0));
  checkCudaErrors(cudaFree(d_img1));
}

extern "C" void freeCPU() {
  // cleanup host memory
  cudaFreeHost(h_odata);
  cudaFreeHost(h_img0);
  cudaFreeHost(h_img1);
}

extern "C" void finish() {
  // clean up the user allocated stream
  cudaStreamSynchronize(stream);
   // finish clean up with deleting the user-created stream
  cudaStreamDestroy(stream);

  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  checkCudaErrors(cudaDeviceReset());
}

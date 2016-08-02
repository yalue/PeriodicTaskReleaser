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

#include <errno.h>
#include <math.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
extern "C" {
#include "gpusync.h"
}
#include "sd_kernel.cuh"

// Relative path to images
static const char fname0[] = "../Samples/Copy/StereoDisparity/data/stereo.im0.640x533.ppm";
static const char fname1[] = "../Samples/Copy/StereoDisparity/data/stereo.im1.640x533.ppm";

// Holds per-thread state for this algorithm.
typedef struct {
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
  dim3 numThreads;
  dim3 numBlocks;
  unsigned int numData;
  unsigned int memSize;
  cudaTextureObject_t texture_right;
  cudaTextureObject_t texture_left;
  // Search parameters
  int minDisp;
  int maxDisp;
} ThreadContext;

int iDivUp(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

// Override helper_image.h
inline bool loadPPM4ub(const char *file, unsigned char **data,
  unsigned int *w, unsigned int *h) {
  unsigned char *idata = 0;
  unsigned int channels;
  if (!__loadPPM(file, &idata, w, h, &channels)) {
    free(idata);
    return false;
  }
  // pad 4th component
  int size = *w * *h;
  // keep the original pointer
  unsigned char *idata_orig = idata;
  checkCudaErrors(cudaMallocHost(data, sizeof(unsigned char) * size * 4));
  unsigned char *ptr = *data;
  for (int i = 0; i < size; i++) {
    *ptr++ = *idata++;
    *ptr++ = *idata++;
    *ptr++ = *idata++;
    *ptr++ = 0;
  }
  free(idata_orig);
  return true;
}

void* Initialize(int sync_level) {
  ThreadContext *g;
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
  checkCudaErrors(cudaMallocHost(&g, sizeof(ThreadContext)));
  g->minDisp = -16;
  g->maxDisp = 0;
  // Follow convention and initialize CUDA/GPU
  // used here to invoke initialization of GPU locking
  cudaFree(0);
  // Pin code
  if (!mlockall(MCL_CURRENT | MCL_FUTURE)) {
    fprintf(stderr, "Failed to lock code pages.\n");
    exit(EXIT_FAILURE);
  }
  cudaSetDevice(0);
  cudaStreamCreate(&(g->stream));
  return g;
}

void MallocCPU(int numElements, void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;
  // Load image data
  // functions allocate memory for the images on host side
  // initialize pointers to NULL to request lib call to allocate as needed
  // PPM images are loaded into 4 byte/pixel memory (RGBX)
  g->h_img0 = NULL;
  g->h_img1 = NULL;
  if (!loadPPM4ub(fname0, &(g->h_img0), &(g->w), &(g->h))) {
    fprintf(stderr, "Failed to load <%s>\n", fname0);
    exit(-1);
  }
  if (!loadPPM4ub(fname1, &(g->h_img1), &(g->w), &(g->h))) {
    fprintf(stderr, "Failed to load <%s>\n", fname1);
    exit(-1);
  }
  // set up parameters used in the rest of program
  g->numThreads = dim3(blockSize_x, blockSize_y, 1);
  g->numBlocks = dim3(iDivUp(g->w, g->numThreads.x), iDivUp(g->h,
    g->numThreads.y));
  g->numData = g->w * g->h;
  g->memSize = sizeof(int) * g->numData;

  // allocate memory for the result on host side
  checkCudaErrors(cudaMallocHost(&(g->h_odata), g->memSize));
}


void MallocGPU(int unused, void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;
  cudaResourceDesc left_resource, right_resource;
  cudaTextureDesc texture_desc;
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned int>();
  // Allocate device memory for inputs and result
  // TODO: Make Texture Object memory zero copy, too?
  checkCudaErrors(cudaHostGetDevicePointer(&(g->d_odata), g->h_odata, 0));
  checkCudaErrors(cudaMalloc(&(g->d_img0), g->memSize));
  checkCudaErrors(cudaMalloc(&(g->d_img1), g->memSize));
  // Initialize texture objects.
  memset(&left_resource, 0, sizeof(left_resource));
  left_resource.resType = cudaResourceTypePitch2D;
  left_resource.res.pitch2D.width = g->w;
  left_resource.res.pitch2D.height = g->h;
  left_resource.res.pitch2D.desc = desc;
  left_resource.res.pitch2D.pitchInBytes = g->w * 4;
  // The only difference between the left and right textures is the image
  memcpy(&right_resource, &left_resource, sizeof(left_resource));
  left_resource.res.pitch2D.devPtr = g->d_img0;
  right_resource.res.pitch2D.devPtr = g->d_img1;
  texture_desc.addressMode[0] = cudaAddressModeClamp;
  texture_desc.addressMode[1] = cudaAddressModeClamp;
  texture_desc.filterMode = cudaFilterModePoint;
  texture_desc.readMode = cudaReadModeElementType;
  checkCudaErrors(cudaCreateTextureObject(&(g->texture_left), &left_resource,
    &texture_desc, NULL));
  checkCudaErrors(cudaCreateTextureObject(&(g->texture_right), &right_resource,
    &texture_desc, NULL));
}

void CopyIn(int unused, void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;
  // copy host memory with images to device
  checkCudaErrors(cudaMemcpyAsync(g->d_img0, g->h_img0, g->memSize,
    cudaMemcpyHostToDevice, g->stream));
  checkCudaErrors(cudaMemcpyAsync(g->d_img1, g->h_img1, g->memSize,
    cudaMemcpyHostToDevice, g->stream));
  // copy host memory that was set to zero to initialize device output
  checkCudaErrors(cudaMemcpyAsync(g->d_odata, g->h_odata, g->memSize,
    cudaMemcpyHostToDevice, g->stream));
  cudaStreamSynchronize(g->stream);
}

void Exec(int unused, void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;
  stereoDisparityKernel<<<g->numBlocks, g->numThreads, 0, g->stream>>>(
    g->d_img0, g->d_img1, g->d_odata, g->w, g->h, g->minDisp, g->maxDisp,
    g->texture_left, g->texture_right);
  cudaStreamSynchronize(g->stream);
  getLastCudaError("Kernel execution failed");
}

void CopyOut(void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;
  checkCudaErrors(cudaMemcpyAsync(g->h_odata, g->d_odata, g->memSize,
    cudaMemcpyDeviceToHost, g->stream));
  cudaStreamSynchronize(g->stream);
}

void FreeGPU(void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;
  checkCudaErrors(cudaFree(g->d_odata));
  checkCudaErrors(cudaFree(g->d_img0));
  checkCudaErrors(cudaFree(g->d_img1));
}

void FreeCPU(void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;
  cudaFreeHost(g->h_odata);
  cudaFreeHost(g->h_img0);
  cudaFreeHost(g->h_img1);
}

void Finish(void *thread_data) {
  ThreadContext *g = (ThreadContext*) thread_data;
  cudaStreamSynchronize(g->stream);
  cudaDestroyTextureObject(g->texture_right);
  cudaDestroyTextureObject(g->texture_left);
  cudaStreamDestroy(g->stream);
  checkCudaErrors(cudaDeviceReset());
}

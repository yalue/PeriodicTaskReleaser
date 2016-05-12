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

// CUDA utilities and system includes
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <cuda_runtime.h>
// includes, project
#include <helper_string.h>
#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>    // includes for cuda initialization and error checking

#include "sf_kernels.h"

extern "C" {
#include "gpusync.h"
}

#define MAX_EPSILON_ERROR 5.0f
#define OFFSET(i) ((char *)NULL + (i))
#define MAX(a,b) ((a > b) ? a : b)
#define RADIUS 1
#ifdef FIXED_BLOCKWIDTH
#define BlockWidth 80
#define SharedPitch 384
#endif

const char *sSDKsample = "CUDA Sobel Edge-Detection";

static int imWidth  = 0;   // Image width
static int imHeight = 0;   // Image height
unsigned int g_Bpp;
unsigned int g_Index = 0;

unsigned char *pixels = NULL;  // Image pixel data on the host
float imageScale = 1.f;    // Image exposure

// Texture reference for reading image
texture<unsigned char, 2> tex;
extern __shared__ unsigned char LocalBlock[];
static cudaArray *array = NULL;

// Stream for the thread's GPU Operations
cudaStream_t stream;

// Device memory location for result
Pixel *d_result;
// Host memory location for result
unsigned char *h_result;

char dump_file[256];
char ref_image_path[] = "../Samples/Copy/SobelFilter/data/ref_shared.pgm";
// Sorry for this. Needs a path to the source image.
char image_path[] = "../Samples/Copy/SobelFilter/data/lena.pgm";

// Kernel execution parameters
dim3 threads;
#ifndef FIXED_BLOCKWIDTH
int BlockWidth;
#endif
dim3 blocks;
int SharedPitch;
int sharedMem;

// Utility functions
inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
        file, line, (int)err, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// override methods in helper_cuda.h
template <class T> inline bool loadPGM(const char *file, T **data, 
  unsigned int *w, unsigned int *h) {
  unsigned char *idata = NULL;
  unsigned int channels;

  if (!__loadPPM(file, &idata, w, h, &channels)) {
    return false;
  }

  unsigned int size = *w **h * channels;

  // initialize mem if necessary
  // the correct size is checked / set in loadPGMc()
  if (NULL == *data) {
    checkCudaErrors(cudaMallocHost(data, sizeof(T) * size));
  }

  // copy and cast data
  std::transform(idata, idata + size, *data, ConverterFromUByte<T>());
  free(idata);
  return true;
}

template <class T> inline bool loadPPM4(const char *file, T **data,
  unsigned int *w,unsigned int *h) {
  unsigned char *idata = 0;
  unsigned int channels;

  if (__loadPPM(file, &idata, w, h, &channels)) {
    // pad 4th component
    int size = *w **h;
    // keep the original pointer
    unsigned char *idata_orig = idata;
    cudaMallocHost(data, sizeof(T) * size * 4);
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

// Kernels
__device__ unsigned char ComputeSobel(
    unsigned char ul, // upper left
    unsigned char um, // upper middle
    unsigned char ur, // upper right
    unsigned char ml, // middle left
    unsigned char mm, // middle (unused)
    unsigned char mr, // middle right
    unsigned char ll, // lower left
    unsigned char lm, // lower middle
    unsigned char lr, // lower right
    float fScale) {
  short Horz = ur + 2 * mr + lr - ul - 2 * ml - ll;
  short Vert = ul + 2 * um + ur - ll - 2 * lm - lr;
  short Sum = (short)(fScale * (abs((int)Horz) + abs((int)Vert)));

  if (Sum < 0) {
    return 0;
  } else if (Sum > 0xff) {
    return 0xff;
  }
  return (unsigned char) Sum;
}

__global__ void SobelShared(uchar4 *pSobelOriginal, unsigned short SobelPitch,
#ifndef FIXED_BLOCKWIDTH
    short BlockWidth, short SharedPitch,
#endif
    short w, short h, float fScale) {
  short u = 4 * blockIdx.x * BlockWidth;
  short v = blockIdx.y * blockDim.y + threadIdx.y;
  short ib;

  int SharedIdx = threadIdx.y * SharedPitch;

  for (ib = threadIdx.x; ib < BlockWidth + 2 * RADIUS; ib += blockDim.x) {
    LocalBlock[SharedIdx + 4 * ib + 0] = tex2D(tex,
       (float)(u + 4 * ib - RADIUS + 0), (float)(v - RADIUS));
    LocalBlock[SharedIdx + 4 * ib + 1] = tex2D(tex,
       (float)(u + 4 * ib - RADIUS + 1), (float)(v - RADIUS));
    LocalBlock[SharedIdx + 4 * ib + 2] = tex2D(tex,
       (float)(u + 4 * ib - RADIUS + 2), (float)(v - RADIUS));
    LocalBlock[SharedIdx + 4 * ib + 3] = tex2D(tex,
       (float)(u + 4 * ib - RADIUS + 3), (float)(v - RADIUS));
  }

  if (threadIdx.y < RADIUS * 2) {
    //
    // copy trailing RADIUS*2 rows of pixels into shared
    //
    SharedIdx = (blockDim.y + threadIdx.y) * SharedPitch;

    for (ib = threadIdx.x; ib < BlockWidth + 2 * RADIUS; ib += blockDim.x) {
      LocalBlock[SharedIdx + 4 * ib + 0] = tex2D(tex,
         (float)(u + 4 * ib - RADIUS + 0), (float)(v + blockDim.y - RADIUS));
      LocalBlock[SharedIdx + 4 * ib + 1] = tex2D(tex,
         (float)(u + 4 * ib - RADIUS + 1), (float)(v + blockDim.y - RADIUS));
      LocalBlock[SharedIdx + 4 * ib + 2] = tex2D(tex,
         (float)(u + 4 * ib - RADIUS + 2), (float)(v + blockDim.y - RADIUS));
      LocalBlock[SharedIdx + 4 * ib + 3] = tex2D(tex,
         (float)(u + 4 * ib - RADIUS + 3), (float)(v + blockDim.y - RADIUS));
    }
  }

  __syncthreads();

  u >>= 2;  // index as uchar4 from here
  uchar4 *pSobel = (uchar4 *)(((char *) pSobelOriginal) + v * SobelPitch);
  SharedIdx = threadIdx.y * SharedPitch;

  for (ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x) {
    unsigned char pix00 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 0];
    unsigned char pix01 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 1];
    unsigned char pix02 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 2];
    unsigned char pix10 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 0];
    unsigned char pix11 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 1];
    unsigned char pix12 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 2];
    unsigned char pix20 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 0];
    unsigned char pix21 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 1];
    unsigned char pix22 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 2];

    uchar4 out;

    out.x = ComputeSobel(pix00, pix01, pix02,
       pix10, pix11, pix12,
       pix20, pix21, pix22, fScale);

    pix00 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 3];
    pix10 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 3];
    pix20 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 3];
    out.y = ComputeSobel(pix01, pix02, pix00,
       pix11, pix12, pix10,
       pix21, pix22, pix20, fScale);

    pix01 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 4];
    pix11 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 4];
    pix21 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 4];
    out.z = ComputeSobel(pix02, pix00, pix01,
       pix12, pix10, pix11,
       pix22, pix20, pix21, fScale);

    pix02 = LocalBlock[SharedIdx + 4 * ib + 0 * SharedPitch + 5];
    pix12 = LocalBlock[SharedIdx + 4 * ib + 1 * SharedPitch + 5];
    pix22 = LocalBlock[SharedIdx + 4 * ib + 2 * SharedPitch + 5];
    out.w = ComputeSobel(pix00, pix01, pix02,
       pix10, pix11, pix12,
       pix20, pix21, pix22, fScale);

    if (u + ib < w / 4 && v < h) {
      pSobel[u + ib] = out;
    }
  }

  __syncthreads();
}

__global__ void SobelCopyImage(Pixel *pSobelOriginal, unsigned int Pitch,
    int w, int h, float fscale) {
  unsigned char *pSobel = (unsigned char *) (((char *) pSobelOriginal) + blockIdx.x * Pitch);

  for (int i = threadIdx.x; i < w; i += blockDim.x) {
    pSobel[i] = min(max((tex2D(tex, (float) i, (float) blockIdx.x) * fscale), 0.f), 255.f);
  }
}

__global__ void SobelTex(Pixel *pSobelOriginal, unsigned int Pitch,
    int w, int h, float fScale) {
  unsigned char *pSobel = (unsigned char *)(((char *) pSobelOriginal) + blockIdx.x * Pitch);

  for (int i = threadIdx.x; i < w; i += blockDim.x) {
    unsigned char pix00 = tex2D(tex, (float) i - 1, (float) blockIdx.x - 1);
    unsigned char pix01 = tex2D(tex, (float) i + 0, (float) blockIdx.x - 1);
    unsigned char pix02 = tex2D(tex, (float) i + 1, (float) blockIdx.x - 1);
    unsigned char pix10 = tex2D(tex, (float) i - 1, (float) blockIdx.x + 0);
    unsigned char pix11 = tex2D(tex, (float) i + 0, (float) blockIdx.x + 0);
    unsigned char pix12 = tex2D(tex, (float) i + 1, (float) blockIdx.x + 0);
    unsigned char pix20 = tex2D(tex, (float) i - 1, (float) blockIdx.x + 1);
    unsigned char pix21 = tex2D(tex, (float) i + 0, (float) blockIdx.x + 1);
    unsigned char pix22 = tex2D(tex, (float) i + 1, (float) blockIdx.x + 1);
    pSobel[i] = ComputeSobel(pix00, pix01, pix02,
       pix10, pix11, pix12,
       pix20, pix21, pix22, fScale);
  }
}

// CPU code
void initializeData(char *file) {
  unsigned int w, h;
  size_t file_length = strlen(file);

  if (!strcmp(&file[file_length - 3], "pgm")) {
    if (loadPGM<unsigned char>(file, &pixels, &w, &h) != true) {
      printf("Failed to load PGM image file: %s\n", file);
      exit(EXIT_FAILURE);
    }

    g_Bpp = 1;
  }
  else if (!strcmp(&file[file_length - 3], "ppm")) {
    if (loadPPM4(file, &pixels, &w, &h) != true) {
      printf("Failed to load PPM image file: %s\n", file);
      exit(EXIT_FAILURE);
    }

    g_Bpp = 4;
  } else {
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }

  imWidth = (int)w;
  imHeight = (int)h;
}

void loadDefaultImage() {
  initializeData(image_path);
}

void setupTexture(int iw, int ih, Pixel *data, int Bpp) {
  cudaChannelFormatDesc desc;
  if (Bpp == 1) {
    desc = cudaCreateChannelDesc<unsigned char>();
  } else {
    desc = cudaCreateChannelDesc<uchar4>();
  }
  checkCudaErrors(cudaMallocArray(&array, &desc, iw, ih));
}

// GPUSYNC interface

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
  // Follow convention and initialize CUDA/GPU
  // used here to invoke initialization of GPU locking
  cudaFree(0);

  // Pin code
  if(!mlockall(MCL_CURRENT)) {
    fprintf(stderr, "Failed to lock code pages.\n");
    exit(EXIT_FAILURE);
  }
 
  // Set the device context 
  cudaSetDevice(0);

  // create a user defined stream
  cudaStreamCreate(&stream);

  sprintf(dump_file, "lena_shared.pgm");
}

extern "C" void mallocCPU(int num_elements) {
  loadDefaultImage(); 
  checkCudaErrors(cudaMallocHost(&h_result, imWidth * imHeight * sizeof(Pixel)));

  threads = dim3(16, 4);
#ifndef FIXED_BLOCKWIDTH
  BlockWidth = 80; // must be divisible by 16 for coalescing
#endif
  blocks = dim3(imWidth / (4 * BlockWidth) + (0 != imWidth % (4 * BlockWidth)),
      imHeight / threads.y + (0 != imHeight % threads.y));
  SharedPitch = ~0x3f & (4 * (BlockWidth + 2 * RADIUS) + 0x3f);
  sharedMem = SharedPitch * (threads.y + 2 * RADIUS);

  // for the shared kernel, width must be divisible by 4
  imWidth &= ~3;
}

extern "C" void mallocGPU(int num_elements) {
  setupTexture(imWidth, imHeight, pixels, g_Bpp);
  // may not be necessary
  // memset(pixels, 0x0, g_Bpp * sizeof(Pixel) * imWidth * imHeight);
  checkCudaErrors(cudaMalloc((void **)&d_result, imWidth * imHeight * sizeof(Pixel)));
}

extern "C" void copyin(int num_elements) {
  checkCudaErrors(cudaMemcpyToArrayAsync(array, 0, 0, pixels,
      g_Bpp * sizeof(Pixel) * imWidth * imHeight, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaBindTextureToArray(tex, array));
  cudaStreamSynchronize(stream);
}

extern "C" void exec(int num_elements) {
  SobelShared <<< blocks, threads, sharedMem, stream>>>((uchar4 *) d_result, imWidth,
#ifndef FIXED_BLOCKWIDTH
      BlockWidth, SharedPitch,
#endif
      imWidth, imHeight, imageScale);
  cudaStreamSynchronize(stream);
}

extern "C" void copyout() {
  checkCudaErrors(cudaMemcpyAsync(h_result, d_result, imWidth * imHeight * sizeof(Pixel), 
      cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
}

extern "C" void freeGPU() {
  checkCudaErrors(cudaUnbindTexture(tex));
  checkCudaErrors(cudaFree(d_result));
  checkCudaErrors(cudaFreeArray(array));
}

extern "C" void freeCPU() {
  cudaFreeHost(h_result);
}

extern "C" void finish() {
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  cudaError_t err = cudaDeviceReset();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to deinitialize the device! (error code %s).\n", cudaGetErrorString(err));
  }
  exit(EXIT_SUCCESS);
}

// int main(int argc, char **argv) {

//   printf("%s Starting...\n\n", sSDKsample);

//   printf("[%s] (automated testing w/ readback)\n", sSDKsample);
//   sf_init(0);
//   sf_mallocHost(argv[0]);
//   sf_cudaMalloc();
//   sf_copyin();
//   sf_exec();
//   sf_copyout();
//   printf("AutoTest %s done\n", argv[0]);
//   sdkSavePGM(dump_file, h_result, imWidth, imHeight);
//   sf_cudaFree();
//   sf_freeHost();
//   sf_finish(argv[0]);
// }


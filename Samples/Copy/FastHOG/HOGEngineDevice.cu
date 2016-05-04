#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cutil.h"
#include "HOGConvolution.h"
#include "HOGEngine.h"
#include "HOGHistogram.h"
#include "HOGPadding.h"
#include "HOGScale.h"
#include "HOGSVMSlider.h"
#include "HOGUtils.h"
#include "HOGEngineDevice.h"

int hWidth, hHeight;
int hWidthROI, hHeightROI;
int hPaddedWidth, hPaddedHeight;
int rPaddedWidth, rPaddedHeight;

int minX, minY, maxX, maxY;

int hNoHistogramBins, rNoHistogramBins;

int hPaddingSizeX, hPaddingSizeY;
int hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY, hWindowSizeX, hWindowSizeY;
int hNoOfCellsX, hNoOfCellsY, hNoOfBlocksX, hNoOfBlocksY;
int rNoOfCellsX, rNoOfCellsY, rNoOfBlocksX, rNoOfBlocksY;

int hNumberOfBlockPerWindowX, hNumberOfBlockPerWindowY;
int hNumberOfWindowsX, hNumberOfWindowsY;
int rNumberOfWindowsX, rNumberOfWindowsY;

float4 *paddedRegisteredImage;

float1 *resizedPaddedImageF1;
float4 *resizedPaddedImageF4;

float2 *colorGradientsF2;

float1 *blockHistograms;
float1 *cellHistograms;

float1 *svmScores;

bool hUseGrayscale;

uchar1* outputTest1;
uchar4* outputTest4;

float* hResult;

float scaleRatio;
float startScale;
float endScale;
int scaleCount;

int avSizeX, avSizeY, marginX, marginY;

extern uchar4* paddedRegisteredImageU4;

void DeviceAllocHOGEngineDeviceMemory(void) {
  DeviceAllocHOGConvolutionMemory();
  DeviceAllocHOGHistogramMemory();
  DeviceAllocHOGSVMMemory();
  DeviceAllocHOGPaddingMemory();
  DeviceAllocHOGScaleMemory();
  cutilSafeCall(cudaMalloc(&paddedRegisteredImage, sizeof(float4) *
    hPaddedWidth * hPaddedHeight));
  if (hUseGrayscale) {
    cutilSafeCall(cudaMalloc(&resizedPaddedImageF1, sizeof(float1) *
      hPaddedWidth * hPaddedHeight));
  } else {
    cutilSafeCall(cudaMalloc(&resizedPaddedImageF4, sizeof(float4) *
      hPaddedWidth * hPaddedHeight));
  }
  cutilSafeCall(cudaMalloc(&colorGradientsF2, sizeof(float2) * hPaddedWidth *
    hPaddedHeight));
  cutilSafeCall(cudaMalloc(&blockHistograms, sizeof(float1) * hNoOfBlocksX *
    hNoOfBlocksY * hCellSizeX * hCellSizeY * hNoHistogramBins));
  cutilSafeCall(cudaMalloc(&cellHistograms, sizeof(float1) * hNoOfCellsX *
    hNoOfCellsY * hNoHistogramBins));
  cutilSafeCall(cudaMalloc(&svmScores, sizeof(float1) * hNumberOfWindowsX *
    hNumberOfWindowsY * scaleCount));
  if (hUseGrayscale) {
    cutilSafeCall(cudaMalloc(&outputTest1, sizeof(uchar1) * hPaddedWidth *
      hPaddedHeight));
  } else {
    cutilSafeCall(cudaMalloc(&outputTest4, sizeof(uchar4) * hPaddedWidth *
      hPaddedHeight));
  }
}

void HostAllocHOGEngineDeviceMemory(void) {
  HostAllocHOGHistogramMemory();
  cutilSafeCall(cudaMallocHost(&hResult, sizeof(float) * hNumberOfWindowsX *
    hNumberOfWindowsY * scaleCount));
}

void CopyInHOGEngineDevice(void) {
  CopyInHOGConvolution();
  CopyInHOGHistogram();
  CopyInHOGSVM();
}

void HostFreeHOGEngineDeviceMemory(void) {
  cutilSafeCall(cudaFreeHost(hResult));
  hResult = NULL;
  HostFreeHOGHistogramMemory();
}

void DeviceFreeHOGEngineDeviceMemory(void) {
  cutilSafeCall(cudaFree(paddedRegisteredImage));
  if (hUseGrayscale) {
    cutilSafeCall(cudaFree(resizedPaddedImageF1));
  } else {
    cutilSafeCall(cudaFree(resizedPaddedImageF4));
  }
  cutilSafeCall(cudaFree(colorGradientsF2));
  cutilSafeCall(cudaFree(blockHistograms));
  cutilSafeCall(cudaFree(cellHistograms));
  cutilSafeCall(cudaFree(svmScores));
  DeviceFreeHOGConvolutionMemory();
  DeviceFreeHOGHistogramMemory();
  DeviceFreeHOGSVMMemory();
  DeviceFreeHOGPaddingMemory();
  DeviceFreeHOGScaleMemory();
  if (hUseGrayscale) {
    cutilSafeCall(cudaFree(outputTest1));
  } else {
    cutilSafeCall(cudaFree(outputTest4));
  }
}

void InitHOG(int width, int height) {
  cudaSetDevice(0);
  int i;
  int toaddxx = 0, toaddxy = 0, toaddyx = 0, toaddyy = 0;
  hWidth = width;
  hHeight = height;
  avSizeX = HOG.avSizeX;
  avSizeY = HOG.avSizeY;
  marginX = HOG.marginX;
  marginY = HOG.marginY;
  if (avSizeX != 0) {
    toaddxx = hWidth * marginX / avSizeX;
    toaddxy = hHeight * marginY / avSizeX;
  }
  if (avSizeY != 0) {
    toaddyx = hWidth * marginX / avSizeY;
    toaddyy = hHeight * marginY / avSizeY;
  }
  hPaddingSizeX = max(toaddxx, toaddyx);
  hPaddingSizeY = max(toaddxy, toaddyy);
  hPaddedWidth = hWidth + hPaddingSizeX * 2;
  hPaddedHeight = hHeight + hPaddingSizeY * 2;
  hUseGrayscale = HOG.useGrayscale;
  hNoHistogramBins = HOG.hNoOfHistogramBins;
  hCellSizeX = HOG.hCellSizeX;
  hCellSizeY = HOG.hCellSizeY;
  hBlockSizeX = HOG.hBlockSizeX;
  hBlockSizeY = HOG.hBlockSizeY;
  hWindowSizeX = HOG.hWindowSizeX;
  hWindowSizeY = HOG.hWindowSizeY;
  hNoOfCellsX = hPaddedWidth / hCellSizeX;
  hNoOfCellsY = hPaddedHeight / hCellSizeY;
  hNoOfBlocksX = hNoOfCellsX - hBlockSizeX + 1;
  hNoOfBlocksY = hNoOfCellsY - hBlockSizeY + 1;
  hNumberOfBlockPerWindowX = (hWindowSizeX - hCellSizeX * hBlockSizeX) /
    hCellSizeX + 1;
  hNumberOfBlockPerWindowY = (hWindowSizeY - hCellSizeY * hBlockSizeY) /
    hCellSizeY + 1;
  hNumberOfWindowsX = 0;
  for (i = 0; i < hNumberOfBlockPerWindowX; i++) {
    hNumberOfWindowsX += (hNoOfBlocksX - i) / hNumberOfBlockPerWindowX;
  }
  hNumberOfWindowsY = 0;
  for (i = 0; i < hNumberOfBlockPerWindowY; i++) {
    hNumberOfWindowsY += (hNoOfBlocksY - i) / hNumberOfBlockPerWindowY;
  }
  scaleRatio = 1.05f;
  startScale = 1.0f;
  endScale = min(hPaddedWidth / (float) hWindowSizeX, hPaddedHeight /
    (float) hWindowSizeY);
  scaleCount = (int)floor(logf(endScale / startScale) / logf(scaleRatio)) + 1;
  InitConvolution(hPaddedWidth, hPaddedHeight, hUseGrayscale);
  InitHistograms(hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY,
    hNoHistogramBins, HOG.wtScale);
  InitSVM();
  InitScale();
  InitPadding();
  rPaddedWidth = hPaddedWidth;
  rPaddedHeight = hPaddedHeight;
}

void CloseHOG() {
  CloseConvolution();
  CloseHistogram();
  CloseSVM();
  CloseScale();
  ClosePadding();
  cudaThreadExit();
}

void BeginHOGProcessing(unsigned char* hostImage, float minScale,
    float maxScale) {
  int i;
  minX = HOG.minX;
  minY = HOG.minY;
  maxX = HOG.maxX;
  maxY = HOG.maxY;
  PadHostImage((uchar4*)hostImage, paddedRegisteredImage, minX, minY, maxX,
    maxY);

  rPaddedWidth = hPaddedWidth; rPaddedHeight = hPaddedHeight;
  scaleRatio = 1.05f;
  startScale = (minScale < 0.0f) ? 1.0f : minScale;
  if (maxScale < 0.0f) {
    endScale = min(hPaddedWidth / (float) hWindowSizeX, hPaddedHeight /
      (float) hWindowSizeY);
  } else {
    endScale = maxScale;
  }
  scaleCount = (int) floor(logf(endScale / startScale) / logf(scaleRatio)) + 1;
  float currentScale = startScale;
  ResetSVMScores(svmScores);
  for (i = 0; i < scaleCount; i++) {
    DownscaleImage(0, scaleCount, i, currentScale, hUseGrayscale,
      paddedRegisteredImage, resizedPaddedImageF1, resizedPaddedImageF4);
    SetConvolutionSize(rPaddedWidth, rPaddedHeight);
    if (hUseGrayscale) {
      ComputeColorGradients1to2(resizedPaddedImageF1, colorGradientsF2);
    } else {
      ComputeColorGradients4to2(resizedPaddedImageF4, colorGradientsF2);
    }
    ComputeBlockHistogramsWithGauss(colorGradientsF2, blockHistograms,
      hNoHistogramBins, hCellSizeX, hCellSizeY, hBlockSizeX, hBlockSizeY,
      hWindowSizeX, hWindowSizeY,  rPaddedWidth, rPaddedHeight);
    NormalizeBlockHistograms(blockHistograms, hNoHistogramBins, hCellSizeX,
      hCellSizeY, hBlockSizeX, hBlockSizeY, rPaddedWidth, rPaddedHeight);
    LinearSVMEvaluation(svmScores, blockHistograms, hNoHistogramBins,
      hWindowSizeX, hWindowSizeY, hCellSizeX, hCellSizeY, hBlockSizeX,
      hBlockSizeY, rNoOfBlocksX, rNoOfBlocksY, i, rPaddedWidth, rPaddedHeight);
    currentScale *= scaleRatio;
  }
}

float* EndHOGProcessing() {
  cudaThreadSynchronize();
  cutilSafeCall(cudaMemcpyAsync(hResult, svmScores, sizeof(float) *
    scaleCount * hNumberOfWindowsX * hNumberOfWindowsY,
    cudaMemcpyDeviceToHost, stream));
  cutilSafeCall(cudaStreamSynchronize(stream));
  return hResult;
}

// NOTE (Nathan): I think this is unused.
void GetProcessedImage(unsigned char* hostImage, int imageType) {
  switch (imageType) {
  case 0:
    Float4ToUchar4(resizedPaddedImageF4, outputTest4, rPaddedWidth,
      rPaddedHeight);
    break;
  case 1:
    Float2ToUchar4(colorGradientsF2, outputTest4, rPaddedWidth, rPaddedHeight,
      0);
    break;
  case 2:
    Float2ToUchar4(colorGradientsF2, outputTest4, rPaddedWidth, rPaddedHeight,
      1);
    break;
  case 3:
    cutilSafeCall(cudaMemcpyAsync(hostImage, paddedRegisteredImageU4,
      sizeof(uchar4) * hPaddedWidth * hPaddedHeight, cudaMemcpyDeviceToHost,
      stream));
    cutilSafeCall(cudaStreamSynchronize(stream));
    return;
  case 4:
    cutilSafeCall(cudaMemcpy2DAsync(((uchar4*) hostImage) + minX + minY *
      hWidth, hWidth * sizeof(uchar4), paddedRegisteredImageU4 +
      hPaddingSizeX + hPaddingSizeY * hPaddedWidth, hPaddedWidth *
      sizeof(uchar4), hWidthROI * sizeof(uchar4), hHeightROI,
      cudaMemcpyDeviceToHost, stream));
    cutilSafeCall(cudaStreamSynchronize(stream));
    return;
  }
  cutilSafeCall(cudaMemcpy2DAsync(hostImage, hPaddedWidth * sizeof(uchar4),
    outputTest4, rPaddedWidth * sizeof(uchar4), rPaddedWidth * sizeof(uchar4),
    rPaddedHeight, cudaMemcpyDeviceToHost, stream));
  cutilSafeCall(cudaStreamSynchronize(stream));
}

void GetHOGParameters() {
  HOG.startScale = startScale;
  HOG.endScale = endScale;
  HOG.scaleRatio = scaleRatio;
  HOG.scaleCount = scaleCount;
  HOG.hPaddingSizeX = hPaddingSizeX;
  HOG.hPaddingSizeY = hPaddingSizeY;
  HOG.hPaddedWidth = hPaddedWidth;
  HOG.hPaddedHeight = hPaddedHeight;
  HOG.hNoOfCellsX = hNoOfCellsX;
  HOG.hNoOfCellsY = hNoOfCellsY;
  HOG.hNoOfBlocksX = hNoOfBlocksX;
  HOG.hNoOfBlocksY = hNoOfBlocksY;
  HOG.hNumberOfWindowsX = hNumberOfWindowsX;
  HOG.hNumberOfWindowsY = hNumberOfWindowsY;
  HOG.hNumberOfBlockPerWindowX = hNumberOfBlockPerWindowX;
  HOG.hNumberOfBlockPerWindowY = hNumberOfBlockPerWindowY;
}

cudaArray *imageArray2 = 0;
texture<float4, 2, cudaReadModeElementType> tex2;
cudaChannelFormatDesc channelDescDownscale2;

__global__ void resizeFastBicubic3(float4 *outputFloat, float4* paddedRegisteredImage, int width, int height, float scale)
{
  int x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  int i = __umul24(y, width) + x;

  float u = x*scale;
  float v = y*scale;

  if (x < width && y < height)
  {
    float4 cF;

    if (scale == 1.0f)
      cF = paddedRegisteredImage[x + y * width];
    else
      cF = tex2D(tex2, u, v);

    outputFloat[i] = cF;
  }
}

void DownscaleImage2(float scale, float4* paddedRegisteredImage,
    float4* resizedPaddedImageF4, int width, int height, int &rPaddedWidth,
    int &rPaddedHeight) {
  dim3 hThreadSize, hBlockSize;
  hThreadSize = dim3(THREAD_SIZE_W, THREAD_SIZE_H);
  rPaddedWidth = iDivUpF(width, scale);
  rPaddedHeight = iDivUpF(height, scale);
  hBlockSize = dim3(iDivUp(rPaddedWidth, hThreadSize.x), iDivUp(rPaddedHeight,
   hThreadSize.y));
  cutilSafeCall(cudaMemcpyToArrayAsync(imageArray2, 0, 0,
    paddedRegisteredImage, sizeof(float4) * width * height,
    cudaMemcpyDeviceToDevice, stream));
  cutilSafeCall(cudaStreamSynchronize(stream));
  cutilSafeCall(cudaBindTextureToArray(tex2, imageArray2,
    channelDescDownscale2));
  cutilSafeCall(cudaMemsetAsync(resizedPaddedImageF4, 0, width * height *
    sizeof(float4), stream));
  cutilSafeCall(cudaStreamSynchronize(stream));
  resizeFastBicubic3<<<hBlockSize, hThreadSize, stream>>>(
    (float4*)resizedPaddedImageF4, (float4*)paddedRegisteredImage,
    rPaddedWidth, rPaddedHeight, scale);
  cutilSafeCall(cudaStreamSynchronize(stream));
  cutilSafeCall(cudaUnbindTexture(tex2));
}

// NOTE (Nathan): I don't think this is ever used, so DownscaleImage2 is
// probably never used either.
float3* CUDAImageRescale(float3* src, int width, int height, int &rWidth,
    int &rHeight, float scale) {
  int i, j, offsetC, offsetL;
  float4* srcH;
  float4* srcD;
  float4* dstD;
  float4* dstH;
  float3 val3;
  float4 val4;
  channelDescDownscale2 = cudaCreateChannelDesc<float4>();
  tex2.filterMode = cudaFilterModeLinear;
  tex2.normalized = false;
  cudaMalloc((void**)&srcD, sizeof(float4) * width * height);
  cudaMalloc((void**)&dstD, sizeof(float4) * width * height);
  cudaMallocHost((void**)&srcH, sizeof(float4) * width * height);
  cudaMallocHost((void**)&dstH, sizeof(float4) * width * height);
  cutilSafeCall(cudaMallocArray(&imageArray2, &channelDescDownscale2, width,
    height));
  for (i = 0; i < width; i++) {
    for (j = 0; j < height; j++) {
      offsetC = j + i * height;
      offsetL = j * width + i;
      val3 = src[offsetC];
      srcH[offsetL].x = val3.x;
      srcH[offsetL].y = val3.y;
      srcH[offsetL].z = val3.z;
    }
  }
  cudaMemcpyAsync(srcD, srcH, sizeof(float4) * width * height,
    cudaMemcpyHostToDevice, stream);
  cutilSafeCall(cudaStreamSynchronize(stream));
  DownscaleImage2(scale, srcD, dstD, width, height, rWidth, rHeight);
  cudaMemcpyAsync(dstH, dstD, sizeof(float4) * rWidth * rHeight,
    cudaMemcpyDeviceToHost, stream);
  cutilSafeCall(cudaStreamSynchronize(stream));
  float3* dst = (float3*) malloc (rWidth * rHeight * sizeof(float3));
  for (i = 0; i < rWidth; i++) {
    for (j = 0; j < rHeight; j++) {
      offsetC = j + i * rHeight;
      offsetL = j * rWidth + i;
      val4 = dstH[offsetL];
      dst[offsetC].x = val4.x;
      dst[offsetC].y = val4.y;
      dst[offsetC].z = val4.z;
    }
  }
  cutilSafeCall(cudaFreeArray(imageArray2));
  cudaFree(srcD);
  cudaFree(dstD);
  cudaFreeHost(srcH);
  cudaFreeHost(dstH);
  return dst;
}

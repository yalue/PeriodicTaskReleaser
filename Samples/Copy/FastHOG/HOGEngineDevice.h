#ifndef __CUDA_HOG__
#define __CUDA_HOG__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "HOGDefines.h"

void InitHOG(int width, int height);
void CloseHOG();
void BeginHOGProcessing(unsigned char* hostImage, float minScale,
    float maxScale);
float* EndHOGProcessing();

void GetHOGParameters();

void GetProcessedImage(unsigned char* hostImage, int imageType);

extern float3* CUDAImageRescale(float3* src, int width, int height, int &rWidth, int &rHeight, float scale);

void InitCUDAHOG(int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY,
     int windowSizeX, int windowSizeY, int noOfHistogramBins, float wtscale,
     float svmBias, float* svmWeights, int svmWeightsCount, bool useGrayscale);
void CloseCUDAHOG();

#endif

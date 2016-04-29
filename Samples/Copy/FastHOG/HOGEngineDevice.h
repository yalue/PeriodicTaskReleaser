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
void BeginHOGProcessing(unsigned char* hostImage, int minx, int miny, int maxx, int maxy, 
                               float minScale, float maxScale);
float* EndHOGProcessing();

void GetHOGParameters(float *cStartScale, float *cEndScale, float *cScaleRatio, int *cScaleCount,
    int *cPaddingSizeX, int *cPaddingSizeY, int *cPaddedWidth, int *cPaddedHeight,
    int *cNoOfCellsX, int *cNoOfCellsY, int *cNoOfBlocksX, int *cNoOfBlocksY,
    int *cNumberOfWindowsX, int *cNumberOfWindowsY,
    int *cNumberOfBlockPerWindowX, int *cNumberOfBlockPerWindowY);

void GetProcessedImage(unsigned char* hostImage, int imageType);

extern float3* CUDAImageRescale(float3* src, int width, int height, int &rWidth, int &rHeight, float scale);

void InitCUDAHOG(int cellSizeX, int cellSizeY, int blockSizeX, int blockSizeY,
		 int windowSizeX, int windowSizeY, int noOfHistogramBins, float wtscale,
		 float svmBias, float* svmWeights, int svmWeightsCount, bool useGrayscale);
void CloseCUDAHOG();

#endif

/*
 * fastHog.cpp
 *
 *  Created on: May 14, 2009
 *      Author: viprad
 */
#include <stdio.h>
#include "HOGImage.h"
#include "Others/persondetectorwt.tcc"

extern void InitializeHOG(int iw, int ih, float svmBias, float* svmWeights,
    int svmWeightsCount);
extern void BeginProcess(HOGImage* hostImage, int _minx, int _miny, int _maxx,
    int _maxy, float minScale, float maxScale);
extern void EndProcess();
extern void GetImage(HOGImage *imageCUDA, ImageType imageType);
extern void FinalizeHOG();

HOGImage* image;
HOGImage* imageCUDA;

char file_name[] = "Files/Images/testImage.bmp";

int main(void) {
  image = HOGImageFile(file_name);
  imageCUDA = HOGImageCUDA(image->width,image->height);
  printf("Loaded Image\n");
  InitializeHOG(image->width, image->height, PERSON_LINEAR_BIAS,
    PERSON_WEIGHT_VEC, PERSON_WEIGHT_VEC_LENGTH);
  printf("Initialized HOG\n");
  BeginProcess(image, -1, -1, -1, -1, -1.0f, -1.0f);
  EndProcess();
  printf("Processed Image\n");
#ifdef FOOBAR
  FinalizeHOG();
#endif
  return 0;
}

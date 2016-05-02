/*
 * fastHog.cpp
 *
 *  Created on: May 14, 2009
 *      Author: viprad
 */
#include <stdio.h>
#include "HOGEngine.h"
#include "HOGImage.h"
#include "Others/persondetectorwt.tcc"

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

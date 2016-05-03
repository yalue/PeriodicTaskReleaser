/*
 * fastHog.cpp
 *
 *  Created on: May 14, 2009
 *      Author: viprad
 */
#include <stdio.h>
#include <stdlib.h>
#include "HOGEngine.h"
#include "HOGEngineDevice.h"
#include "HOGImage.h"
#include "Others/persondetectorwt.tcc"
extern "C" {
#include "../../gpusync.h"
}

HOGImage image;

char file_name[] = "Files/Images/testImage.bmp";

void init(int sync_level) {
  switch (sync_level) {
  case 0:
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    break;
  case 1:
    cudaSetDeviceFlags(cudaDeviceScheduleYield);
    break;
  default:
    printf("Unknown sync level: %d\n", sync_level);
    break;
  }
  if (!HOGImageFile(file_name, &image)) {
    printf("Unable to load image file.\n");
    exit(1);
  }
  // TODO (Nathan): Remove this if it's already further down the stack.
  cudaFree(0);
  InitializeHOG(image.width, image.height, PERSON_LINEAR_BIAS,
    PERSON_WEIGHT_VEC, PERSON_WEIGHT_VEC_LENGTH);
}

void mallocCPU(int numElements) {
  HostAllocHOGEngineDeviceMemory();
}

void mallocGPU(int numElements) {
  DeviceAllocHOGEngineDeviceMemory();
}

void copyin(int numElements) {
  CopyInHOGEngineDevice();
}

void exec(int numElements) {
  // TODO (Nathan): Get the remaining allocates/copies out of this. HOGScale
  // definitely has some, to start.
  BeginProcess(&image, -1, -1, -1, -1, -1.0f, -1.0f);
}

void copyout() {
  // TODO (Nathan): Split EndProcess into copyout() and finish(), remove disk
  // stuff.
  EndProcess();
}

void freeGPU() {
  DeviceFreeHOGEngineDeviceMemory();
}

void freeCPU() {
  HostFreeHOGEngineDeviceMemory();
}

void finish() {
  FinalizeHOG();
}

int main(void) {
  init(0);
  mallocCPU(0);
  mallocGPU(0);
  copyin(0);
  exec(0);
  copyout();
  freeGPU();
  freeCPU();
  finish();
}

/*
int main(void) {
  image = HOGImageFile(file_name);
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
*/

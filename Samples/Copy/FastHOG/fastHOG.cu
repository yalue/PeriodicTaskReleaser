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
cudaStream_t stream;

char file_name[] = "../Samples/Copy/FastHOG/Files/Images/testImage.bmp";


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
  if (cudaSetDevice(0) != cudaSuccess) {
    printf("Unable to set cuda device.\n");
    exit(1);
  }
  if (cudaFree(0) != cudaSuccess) {
    printf("Error running cudaFree(0).\n");
    exit(1);
  }
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    printf("Unable to create cuda stream.\n");
    exit(1);
  }
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
  // There are still memcpys to the device in HOGScale and HOGPadding--they
  // may require more work to get rid of because they seem to rely on variables
  // determined during the execution phase.
  BeginProcess(&image, -1, -1, -1, -1, -1.0f, -1.0f);
}

void copyout() {
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
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  if (cudaDeviceReset() != cudaSuccess) {
    printf("Failed to reset the device.\n");
    exit(1);
  }
}

/*
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
  return 0;
}
*/

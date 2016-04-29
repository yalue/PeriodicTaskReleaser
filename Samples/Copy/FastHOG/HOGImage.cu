/*
 * HOGImage.cpp
 *
 *  Created on: May 14, 2009
 *      Author: viprad
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <FreeImage.h>
#include "HOGImage.h"

HOGImage fromFile;
HOGImage fromCUDA;

HOGImage* HOGImageCUDA(int width, int height) {
  fromCUDA.width = width;
  fromCUDA.height = height;
  fromCUDA.isLoaded = false;
  fromCUDA.pixels = (unsigned char*) malloc(sizeof(unsigned char) * 4 * width * height);
  memset(fromCUDA.pixels, 0, sizeof(unsigned char) * 4 * width * height);
  return &fromCUDA;
}

HOGImage* HOGImageFile(const char* fileName) {
  bool bLoaded = false;
  int bpp;
  FIBITMAP *bmp = 0;
  FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
  fif = FreeImage_GetFileType(fileName);
  if (fif == FIF_UNKNOWN) {
    fif = FreeImage_GetFIFFromFilename(fileName);
  }
  if (fif != FIF_UNKNOWN && FreeImage_FIFSupportsReading(fif)) {
    bmp = FreeImage_Load(fif, fileName, 0);
    bLoaded = true;
    if (bmp == NULL) bLoaded = false;
  }
  if (!bLoaded) {
    fromFile.isLoaded = false;
    return &fromFile;
  }
  fromFile.width = FreeImage_GetWidth(bmp);
  fromFile.height = FreeImage_GetHeight(bmp);
  bpp = FreeImage_GetBPP(bmp);
  switch (bpp) {
  case 32:
    break;
  default:
    FIBITMAP *bmpTemp = FreeImage_ConvertTo32Bits(bmp);
    if (bmp != NULL) FreeImage_Unload(bmp);
    bmp = bmpTemp;
    bpp = FreeImage_GetBPP(bmp);
    break;
  }
  fromFile.pixels = (unsigned char*) malloc(sizeof(unsigned char) * 4 *
    fromFile.width * fromFile.height);
  FreeImage_ConvertToRawBits(fromFile.pixels, bmp, fromFile.width * 4, bpp,
    FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, true);
  fromFile.isLoaded = true;
  return &fromFile;
}


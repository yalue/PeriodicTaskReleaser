//-----------------------------------------------------------------------------
// These functions and macros were copied over from the old cutil header files,
// so that FastHOG could be compiled.
// The cutil files were picked from the GPU Computing SDK that shipped with
// the old CUDA 3.2 SDK.
//-----------------------------------------------------------------------------
#ifndef FASTHOG_CUTIL_H
#define FASTHOG_CUTIL_H
#include <stdio.h>

#define cutilSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

inline void __cudaSafeCall( cudaError err, const char *file, const int line ) {
  if (err == cudaSuccess) return;
  printf("Runtime error at %s, line %i: %s.\n", file, line,
    cudaGetErrorString(err));
  exit(-1);
}

#define MIN(a,b) ((a < b) ? a : b)
#define MAX(a,b) ((a > b) ? a : b)

inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
  typedef struct {
    // 0xMm, M = SM Major version, m = SM minor version
    int SM;
    int Cores;
  } sSMtoCores;
  sSMtoCores nGpuArchCoresPerSM[] = {
    {0x10, 8},
    {0x11, 8},
    {0x12, 8},
    {0x13, 8},
    {0x20, 32},
    {0x21, 48},
    {-1, -1},
  };
  int index = 0;
  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
      return nGpuArchCoresPerSM[index].Cores;
    }
    index++;
  }
  printf("MapSMtoCores: Undefined SMversion %d.%d!\n", major, minor);
  return -1;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int cutGetMaxGflopsDeviceId() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_compute_perf = 0, max_perf_device = 0;
  int device_count = 0, best_SM_arch = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceCount(&device_count);
  // Find the best major SM Architecture GPU device
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);
    if (deviceProp.major > 0 && deviceProp.major < 9999) {
      best_SM_arch = MAX(best_SM_arch, deviceProp.major);
    }
    current_device++;
  }
  // Find the best CUDA capable GPU device
  current_device = 0;
  while(current_device < device_count) {
    cudaGetDeviceProperties( &deviceProp, current_device );
    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
      sm_per_multiproc = 1;
    } else {
      sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major,
        deviceProp.minor);
    }
    int compute_perf = deviceProp.multiProcessorCount * sm_per_multiproc *
      deviceProp.clockRate;
    if (compute_perf > max_compute_perf ) {
      // If we find GPU with SM major > 2, search only these
      if ( best_SM_arch > 2 ) {
        // If our device == dest_SM_arch, choose this, or else pass
        if (deviceProp.major == best_SM_arch) {
          max_compute_perf = compute_perf;
          max_perf_device = current_device;
        }
      } else {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    }
    current_device++;
  }
  return max_perf_device;
}

#endif // FASTHOG_CUTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "physical_allocation.h"
#define PAGE_SIZE (4096)

// A macro wrapping the check function.
#define CheckError(val) check ( (val), #val, __FILE__, __LINE__ )

// Prints a message and exits if r != CUDA_SUCCESS.
static void check(cudaError_t r, const char *fn, const char *f, int line) {
  if (r == cudaSuccess) return;
  printf("CUDA error %d at %s:%d (%s)\n", (int) r, f, line, fn);
  cudaDeviceReset();
  exit(1);
}

void* AllocateOutsidePhysicalRegion(uint64_t size, uint64_t base_address,
    uint64_t max_address) {
  // TODO (next): Implement AllocateOutsidePhysicalRegion
  return NULL;
}

uint64_t GetPhysicalAddress(void *virtual_address) {
  uint64_t to_return;
  long virtual_pfn = ((long) virtual_address) / PAGE_SIZE;
  FILE *f = fopen("/proc/self/pagemap", "r");
  if (!f) return 0;
  if (fseek(f, virtual_pfn * 8, SEEK_SET) != 0) {
    fclose(f);
    return 0;
  }
  if (fread(&to_return, 1, sizeof(to_return), f) != sizeof(to_return)) {
    fclose(f);
    return 0;
  }
  fclose(f);
  return to_return * PAGE_SIZE;
}


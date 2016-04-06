#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include "../Samples/va.h"
#include "../Samples/mm.h"
#include "../util.h"

#define TOTAL_ITERATIONS 10000

int main(int argc, char** argv) {
  struct timespec start, end;
  int i, input_size;
  if (argc < 2) {
    error("Need size argument as second argument.");
    exit(EXIT_FAILURE);
  }
  input_size = atoi(argv[1]);
  if (argc == 3) {
    va_init(0);
    va_mallocHost(input_size);
  } else {
    mm_init(0);
    mm_mallocHost(input_size);
  }
  for (i = 0; i < TOTAL_ITERATIONS; ++i) {
    if (clock_gettime(CLOCK_REALTIME, &start)) {
      error("Error getting time");
      break;
    }
    if (argc == 3) {
      va_cudaMalloc(input_size);
      va_copyin(input_size);
      va_exec(input_size); 
      va_copyout();
      va_cudaFree();
    } else {
      mm_cudaMalloc(input_size);
      mm_copyin(input_size);
      mm_exec(input_size); 
      mm_copyout();
      mm_cudaFree();
    }
    if (clock_gettime(CLOCK_REALTIME, &end)) {
      error("Error getting time");
      break;
    }
    fprintf(stdout, "%3ld,\n", elapsed_ns(&start, &end));
  }
  if (argc == 3) {
    va_freeHost();
    va_finish();
  } else {
    mm_freeHost();
    mm_finish(); 
  }
  exit(EXIT_SUCCESS);
}

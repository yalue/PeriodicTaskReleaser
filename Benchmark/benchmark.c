#include <stdio.h>
#include <time.h>

#include "../Samples/va.h"
#include "../Samples/mm.h"

#define INPUT_SIZE 524288 // 2^19
#define TOTAL_ITERATIONS 1000

int main(int argc, char** argv) {
  struct timespec start, end;
  if (argc == 2)
    va_init_GPU_Op(0);
  else
    mm_init_GPU_Op(0);
  for (int i = 0; i < TOTAL_ITERATIONS; ++i) {
    if (clock_gettime(CLOCK_REALTIME, &start_time)) {
      error("Error getting time");
    }
    if (argc == 2)
      va_run_GPU_Op(INPUT_SIZE); 
    else 
      mm_run_GPU_Op(INPUT_SIZE);
    if (clock_gettime(CLOCK_REALTIME, &end_time)) {
      error("Error getting time");
    }
    fprintf(stdout, "%3ld,\n", elapsed_ms(&start_time, &end_time));
  }
  va_finish_GPU_Op();
  mm_init_GPU_Op(0);
  for (int i = 0; i < TOTAL_ITERATIONS; ++i) {
    va_run_GPU_Op(INPUT_SIZE);
  }
  mm_finish_GPU_Op();
}

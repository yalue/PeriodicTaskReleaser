#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include "../Samples/gpusync.h"
#include "../util.h"

#define TOTAL_ITERATIONS 10000

// Call program with two arguments: input_size and operation.
// input size = positive number
// operation = {0: all, 1: copy engine, 2: execution engine}

int main(int argc, char** argv) {
  struct timespec start, end;
  int i, input_size, operation;

  if (argc < 2) {
    error("First argument must be input size.");
    exit(EXIT_FAILURE);
  }
  input_size = atoi(argv[1]);
  if (argc < 3) {
    error("Second argument must specify operation. {0: all, 1: copy engine, 2: execution engine}.");
    exit(EXIT_FAILURE);
  }
  operation = atoi(argv[2]);

  init(0);
  mallocCPU(input_size);

  for (i = 0; i < TOTAL_ITERATIONS; ++i) {
    if (clock_gettime(CLOCK_REALTIME, &start)) {
      error("Error getting time.");
      break;
    }
    if (operation <= 1) { // (operation == 1 || operation == 0)
      mallocGPU(input_size);
      copyin(input_size);
    } else if (operation % 2 == 0) { // (operation == 2 || operation == 0)
      exec(input_size);
    }
    if (operation <= 1) {
      copyout();
      freeGPU();
    }
    if (clock_gettime(CLOCK_REALTIME, &end)) {
      error("Error getting time.");
      break;
    }
    fprintf(stdout, "%3ld,\n", elapsed_ns(&start, &end));
  }
  freeCPU();
  finish();
  exit(EXIT_SUCCESS);
}

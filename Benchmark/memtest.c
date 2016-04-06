#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include "../Samples/mem.h"
#include "../util.h"

#define TOTAL_ITERATIONS 10000

int main(int argc, char** argv) {
  struct timespec start, end;
  int i, input_size, operation;
  if (argc < 2) {
    error("Need size argument as second argument.");
    exit(EXIT_FAILURE);
  }
  input_size = atoi(argv[1]);
  if (argc < 3) {
    error("Need argument specifying operation: 1: allocate, 2: copyin, 3: copyout, 4: free.");
    exit(EXIT_FAILURE);
  }
  operation = atoi(argv[2]);
  if (operation < 1 || operation > 4) {
    error("Operation should be between 1 and 4.\n");
    exit(EXIT_FAILURE);
  }

  for (i = 0; i < TOTAL_ITERATIONS; ++i) {
    // set up
    memtest_init(0, input_size);
    if (operation > 1)
      memtest_alloc(input_size);
    if (operation > 2 && operation < 4)
      memtest_copyin(input_size);
    // start timer
    if (clock_gettime(CLOCK_REALTIME, &start)) {
      error("Error getting time");
      break;
    }
    // do operation
    if (operation == 1)
      memtest_alloc(input_size);
    else if (operation == 2) 
      memtest_copyin(input_size);
    else if (operation == 3)
      memtest_copyout(input_size);
    else if (operation == 4)
      memtest_cudafree();
    // stop timer
    if (clock_gettime(CLOCK_REALTIME, &end)) {
      error("Error getting time");
      break;
    }
    // clean up
    if (operation < 4)
      memtest_cudafree();
    memtest_cleanup();
    fprintf(stdout, "%3ld,\n", elapsed_ns(&start, &end));
  }
  exit(EXIT_SUCCESS);
}

#include <inttypes.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "runner.h"
#include "util.h"
#include "Samples/va.h"
#include "Samples/mm.h"

/**
 * Runs one job when the condition is notified.
 */
void * runner(void *runner_args) {
  int rc;
  int i;
  int datasize;
  int period_ms;
  int function;
  struct Runner_Args *args;
  pthread_mutex_t *mutex;
  pthread_barrier_t *barrier;
  FILE *ostream;
  struct timespec launch_time;
  struct timespec next_release;
  struct timespec start_time;
  struct timespec end_time;

  args = (struct Runner_Args*) runner_args;
  mutex = args->mutex;
  ostream = args->ostream;
  period_ms = args->ms;
  function = args->function;
  barrier = args->barrier;

  // Initialize GPU
  if (function == VECTOR_ADD) {
    va_init_GPU_Op(0);
  } else {
    mm_init_GPU_Op(0);
  }
  // Wait for work
  i = 1;
  clock_gettime(CLOCK_REALTIME, &launch_time);
  while (i <= MAX_ITERATIONS) {
    pthread_mutex_lock(mutex);
    if (!args->isActive) {
      break;
    }
    timespec_offset(&next_release, &launch_time, i * period_ms);
    if ((rc = clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &next_release, NULL))) {
      fprintf(stderr, "Error during sleep: %s. Args: %s.\n", strerror(rc), format_time(&next_release));
    }
    fprintf(ostream, "%s\tRELEASE:   %3d.\n", format_time(&next_release), i);
    datasize = args->datasize;
    if (clock_gettime(CLOCK_REALTIME, &start_time)) {
      error("Error getting current time");
    }
    fprintf(ostream, "%s\tACCEPT:    %3d.\n", format_time(&start_time), i);
 
    // Do periodic task work here
    if (function == VECTOR_ADD) {
      va_run_GPU_Op(datasize);
    } else {
      mm_run_GPU_Op(datasize);
    }

    if (clock_gettime(CLOCK_REALTIME, &end_time)) {
      error("Error getting current time");
    }
    fprintf(ostream, "%s\tFINISH:    %3d. (Execution %3ld ms) (Unused %3ld ms).\n", 
      format_time(&end_time), i, elapsed_ms(&start_time, &end_time),
      period_ms - elapsed_ms(&start_time, &end_time));
    i++;
    pthread_mutex_unlock(mutex);
  }
  pthread_barrier_wait(barrier);
  if (function == VECTOR_ADD) {
    va_finish_GPU_Op();
  } else {
    mm_finish_GPU_Op();
  }
  fprintf(stderr, "runner terminated.\n");
  pthread_exit(NULL);
}

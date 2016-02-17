#include <inttypes.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "releaser.h"
#include "runner.h"
#include "util.h"
#include "GPUOp.h"

/**
 * Runs one job when the condition is notified.
 */
void * runner(void *runner_args) {
  int rc;
  int i;
  int datasize;
  int period_ms;
  struct Runner_Args *args;
  pthread_mutex_t *mutex;
  FILE *ostream;
  struct timespec launch_time;
  struct timespec next_release;
  struct timespec start_time;
  struct timespec end_time;

  args = (struct Runner_Args*) runner_args;
  mutex = args->mutex;
  ostream = args->ostream;
  period_ms = args->ms;

  // Initialize GPU
  init_GPU_Op(0);
  // Wait for work
  i = 1;
  clock_gettime(CLOCK_REALTIME, &launch_time);
  while (i <= MAX_ITERATIONS) {
    pthread_mutex_lock(mutex);
    fprintf(ostream, "Locked\n");
    if (!args->isActive) {
      break;
    }
    next_release.tv_sec = launch_time.tv_sec + (i * period_ms) / MS_PER_SEC;
    next_release.tv_nsec = launch_time.tv_nsec + (i * period_ms % MS_PER_SEC) * NS_PER_MS;
    fprintf(ostream, "%lld.%.9ld\tRELEASE:   %3d.\n", (long long) next_release.tv_sec, next_release.tv_nsec, i);
    if ((rc = clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &next_release, NULL))) {
      fprintf(stderr, "Error during sleep %s\n", strerror(rc));
    }

    datasize = args->datasize;
    if (clock_gettime(CLOCK_REALTIME, &start_time)) {
      error("Error getting current time");
    }
    fprintf(ostream, "%lld.%.9ld\tACCEPT:    %3d.\n", (long long) start_time.tv_sec,
        start_time.tv_nsec, i % MAX_SIGNALS);
    fflush(ostream);

    // do work here instead of sleeping
    run_GPU_Op(datasize);

    if (clock_gettime(CLOCK_REALTIME, &end_time)) {
      error("Error getting current time");
    }
    fprintf(ostream, "%lld.%.9ld\tFINISH:    %3d. (Execution %3ld ms) (Unused %3ld ms).\n", 
      (long long) end_time.tv_sec, end_time.tv_nsec, i % MAX_SIGNALS,

      (long) ((end_time.tv_sec - start_time.tv_sec) * 1e3 +
      (end_time.tv_nsec - start_time.tv_nsec) * 1e-6),

      period_ms - (long) ((end_time.tv_sec - start_time.tv_sec) * 1e3 +
      (end_time.tv_nsec - start_time.tv_nsec) * 1e-6));
    fflush(ostream);
    i++;
    pthread_mutex_unlock(mutex);
    fprintf(ostream, "Unlocked\n");
  }
  finish_GPU_Op();
  pthread_exit(NULL);
}

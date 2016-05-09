#include <inttypes.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "runner.h"
#include "util.h"
#include "gpusync.h"

/**
 * Runs one job when the condition is notified.
 */
void * runner(void *runner_args) {
  int rc;
  int i;
  int datasize;
  long long period;
  int sync;
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
  period = args->period;
  barrier = args->barrier;
  datasize = args->datasize;
  sync = args->sync;

  // Initialize GPU
  init(sync);
  mallocCPU(datasize);
  // Wait for work
  i = 1;
  clock_gettime(CLOCK_REALTIME, &launch_time);
  timespec_offset(&next_release, &launch_time, i * period);
  while (i <= MAX_ITERATIONS) {
    pthread_mutex_lock(mutex);
    if (!args->isActive) {
      break;
    }
    if ((rc = clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &next_release, NULL))) {
      fprintf(stderr, "Error during sleep: %s. Args: %s.\n", strerror(rc), format_time(&next_release));
    }
    fprintf(ostream, "%s\tRELEASE:   %3d.\n", format_time(&next_release), i);
    if (clock_gettime(CLOCK_REALTIME, &start_time)) {
      error("Error getting current time");
    }
    fprintf(ostream, "%s\tACCEPT:    %3d.\n", format_time(&start_time), i);
 
    // Do periodic task work here
    mallocGPU(datasize);
    copyin(datasize);
    exec(datasize);
    copyout();
    freeGPU();
    // End periodic task work section

    if (clock_gettime(CLOCK_REALTIME, &end_time)) {
      error("Error getting current time");
    }
    fprintf(ostream, "%s\tFINISH:    %3d.\n",
      format_time(&end_time), i, elapsed_ns(&start_time, &end_time),
      period - elapsed_ns(&start_time, &end_time));
    pthread_mutex_unlock(mutex);

    // Compute the next release time
    i++;
    timespec_offset(&next_release, &launch_time, i * period);
    while (timespec_compare(&next_release, &end_time) <= 0 && i < MAX_ITERATIONS) {
      i++;
      timespec_offset(&next_release, &launch_time, i * period);
      fprintf(stderr, "Missed release %d.\n", i);
    }
  }
  pthread_barrier_wait(barrier);
  freeCPU();
  finish();
  fprintf(stderr, "runner %lld terminated.\n", period);
  pthread_exit(NULL);
}

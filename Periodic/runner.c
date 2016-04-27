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
  int period;
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
  while (i <= MAX_ITERATIONS) {
    pthread_mutex_lock(mutex);
    if (!args->isActive) {
      break;
    }
    timespec_offset(&next_release, &launch_time, i * period);
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

    if (clock_gettime(CLOCK_REALTIME, &end_time)) {
      error("Error getting current time");
    }
    fprintf(ostream, "%s\tFINISH:    %3d. (Execution %3ld ns) (Unused %3ld ns).\n", 
      format_time(&end_time), i, elapsed_ns(&start_time, &end_time),
      period - elapsed_ns(&start_time, &end_time));
    i++;
    pthread_mutex_unlock(mutex);
  }
  pthread_barrier_wait(barrier);
  freeCPU();
  finish();
  fprintf(stderr, "runner %d terminated.\n", period);
  pthread_exit(NULL);
}

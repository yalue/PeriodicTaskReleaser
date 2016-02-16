#include <inttypes.h>
#include <time.h>
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
  int i;
  int thread_id;
  int datasize;
  struct Runner_Args *args;
  pthread_cond_t *cond;
  pthread_mutex_t *mutex;
  FILE *ostream;
  struct timespec timeout;
  struct timespec start_time;
  struct timespec end_time;

  args = (struct Runner_Args*) runner_args;
  thread_id = args->thread_id;
  cond = args->cond;
  mutex = args->mutex;
  ostream = args->ostream;

  // Initialize GPU
  init_GPU_Op(0);
  // Wait for work
  i = 0;
  while (i <= MAX_ITERATIONS) {
    pthread_mutex_lock(mutex);
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += MAX_WAIT_TIME;
    pthread_cond_timedwait(cond, mutex, &timeout);
    if (!args->isActive) {
      break;
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
    fprintf(ostream, "%lld.%.9ld\tFINISH:    %3d. (Execution %3ld ms).\n", 
      (long long) end_time.tv_sec, end_time.tv_nsec, i % MAX_SIGNALS, 
      (long) ((end_time.tv_sec - start_time.tv_sec) * 1e3 +
      (end_time.tv_nsec - start_time.tv_nsec) * 1e-6));
    fflush(ostream);
    i++;
    pthread_mutex_unlock(mutex);
  }
  finish_GPU_Op();
  pthread_exit(NULL);
}

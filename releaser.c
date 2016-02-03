#include <inttypes.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "releaser.h"
#include "util.h"

/**
 * Releases one job on a periodic schedule. 
 * Release period is specified in args->ms.
 * Job id is specified in args->cond.
 */
void * releaser(void *releaser_args) {
  int i;
  int ms;
  int thread_id;
  struct timespec ts;
  struct Releaser_Args *args;
  pthread_cond_t *cond;
  FILE *ostream;
  struct timespec start_time;

  args = (struct Releaser_Args*) releaser_args;
  thread_id = args->thread_id;
  ms = args->ms;
  cond = args->cond;
  ostream = args->ostream;

  ts.tv_sec = ms / MS_PER_SEC;
  ts.tv_nsec = (ms % MS_PER_SEC) * NS_PER_MS;
  i = 0;
  while (i < MAX_SIGNALS) {
    if (clock_gettime(CLOCK_REALTIME, &start_time)) {
      error("Error getting current time");
    }
    fprintf(ostream, "%lld.%.9ld\tBROADCAST: %3d. (Period %5d ms).\n", (long long) start_time.tv_sec, start_time.tv_nsec, thread_id, ms);
    fflush(ostream);
    pthread_cond_broadcast(cond);
    nanosleep(&ts, NULL);
    i++;
  }
  pthread_exit(NULL);
}

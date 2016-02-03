#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "releaser.h"
#include "runner.h"

#define NUM_RUNNERS 3
#define WORK_MS 500
#define PERIOD_DURATION 3000

int main(int argc, char *argv[]) {
  int rc;
  pthread_t runners[NUM_RUNNERS];
  pthread_t releasers[NUM_RUNNERS];
  pthread_cond_t thread_conds[NUM_RUNNERS];
  pthread_mutex_t thread_mutexes[NUM_RUNNERS];
  struct Runner_Args thread_args[NUM_RUNNERS];
  struct Releaser_Args releaser_args[NUM_RUNNERS];
  FILE *output_files[NUM_RUNNERS];
  int i;


  for (i = 0; i < NUM_RUNNERS; ++i) {
    char file_name[32];

    // Create condition variables.
    pthread_cond_init(&thread_conds[i], NULL);

    // Create thread mutex.
    pthread_mutex_init(&thread_mutexes[i], NULL);
    // We may want to set mutex attributes to handle
    // priority ineversion differently than the defaults.

    // Create output files
    // Put "file" then k then ".txt" in to filename.
    snprintf(file_name, sizeof(char) * 32, "runner%i.txt", i);
    output_files[i] = (fopen(file_name,"w"));
    if (!output_files[i]) {
      fprintf(stderr, "Error opening output file %s\n.", file_name);
      exit(EXIT_FAILURE);
    }

    // Assign thread args.
    thread_args[i].thread_id = i;
    thread_args[i].ms = WORK_MS;
    thread_args[i].cond = &thread_conds[i];
    thread_args[i].mutex = &thread_mutexes[i];
    thread_args[i].ostream = output_files[i];

    // Assign releaser args
    releaser_args[i].thread_id = i;
    releaser_args[i].ms = PERIOD_DURATION / (i + 1); // To get varied periods in here...
    releaser_args[i].cond = &thread_conds[i];
    releaser_args[i].ostream = output_files[i];
  }

  // Delimit experimental runs with a series of '-';
  for (i = 0; i < NUM_RUNNERS; ++i) {
    fprintf(output_files[i], "------------------------------\n");
  }

  // Create threads
  for (i = 0; i < NUM_RUNNERS; ++i) {
    rc = pthread_create(&runners[i], NULL, runner, (void *) &thread_args[i]);
    if (rc) {
      fprintf(stderr, "Error creating runner: %d\n", rc);
      exit(-1);
    } else {
      fprintf(output_files[i], "Created thread: %d\n", i);
      fflush(output_files[i]);
    }
  }

  {
    // Sleep to allow all threads to initialize.
    struct timespec ts;
    int ms;
    ms = 1000;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000;
    nanosleep(&ts, NULL);
  }

  // Create the releasers
  for (i = 0; i < NUM_RUNNERS; ++i) {
    rc = pthread_create(&releasers[i], NULL, releaser, (void *) &releaser_args[i]);
    if (rc) {
      fprintf(stderr, "Error creating releaser: %d\n", rc);
      exit(-1);
    }
  }

  // Wait on releasers and runners
  for (i = 0; i < NUM_RUNNERS; ++i) {
    if ((rc = pthread_join(releasers[i], NULL))) {
      // error
      fprintf(stderr, "Error joining releaser: %d.\n", rc);
    }
    if ((rc = pthread_join(runners[i], NULL))) {
      // error
      fprintf(stderr, "Error joining runner: %d.\n", rc);
    }
  }
  pthread_exit(NULL);
}
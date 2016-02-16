#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "releaser.h"
#include "runner.h"

////////tmp
#include "GPUOp.h"

#define NUM_RUNNERS 3
#define PERIOD_DURATION 400
#define MAX_DATA_SIZE 8388608 // 2^23 ... 2^24 is too big
#define MAX_FILE_NAME 32

int main(int argc, char *argv[]) {
  int rc;
  pthread_t runners[NUM_RUNNERS];
  pthread_t releasers[NUM_RUNNERS];
  pthread_cond_t thread_conds[NUM_RUNNERS];
  pthread_mutex_t thread_mutexes[NUM_RUNNERS];
  struct Runner_Args runner_args[NUM_RUNNERS];
  struct Releaser_Args releaser_args[NUM_RUNNERS];
  FILE *output_files[NUM_RUNNERS];
  int i, j;

  // Initialize thread state
  for (i = 0; i < NUM_RUNNERS; ++i) {
    char file_name[MAX_FILE_NAME];

    // Create condition variables.
    pthread_cond_init(&thread_conds[i], NULL);

    // Create thread mutex.
    pthread_mutex_init(&thread_mutexes[i], NULL);
    // We may want to set mutex attributes to handle
    // priority ineversion differently than the defaults.

    // Create output files
    // Put "file" then k then ".txt" in to filename.
    snprintf(file_name, sizeof(char) * MAX_FILE_NAME, "runner%i.txt", i);
    output_files[i] = (fopen(file_name,"w"));
    if (!output_files[i]) {
      fprintf(stderr, "Error opening output file %s\n.", file_name);
      exit(EXIT_FAILURE);
    }

    // Assign thread args.
    runner_args[i].isActive = 1;
    runner_args[i].thread_id = i;
    runner_args[i].cond = &thread_conds[i];
    runner_args[i].mutex = &thread_mutexes[i];
    runner_args[i].ostream = output_files[i];

    // Assign releaser args
    releaser_args[i].thread_id = i;
    releaser_args[i].ms = PERIOD_DURATION;// All with equal periods / (i + 1); // To get varied periods in here...
    releaser_args[i].cond = &thread_conds[i];
    releaser_args[i].ostream = output_files[i];
  }
  // Create runners
  for (i = 0; i < NUM_RUNNERS; ++i) {
    rc = pthread_create(&runners[i], NULL, runner, (void *) &runner_args[i]);
    if (rc) {
      fprintf(stderr, "Error creating runner: %d\n", rc);
      exit(-1);
    } else {
      fprintf(output_files[i], "Created thread: %d\n", i);
      fflush(output_files[i]);
    }
  }
  {
    // Sleep launcher to allow all threads to initialize.
    fprintf(stderr, "Sleeping 1 second...\n");
    struct timespec ts;
    int ms;
    ms = 1000;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (ms % 1000) * 1000000;
    nanosleep(&ts, NULL);
    fprintf(stderr, "Here we go!\n");
  }

  // Create releasers for different input sizes
  for (j = 1024; j < MAX_DATA_SIZE; j*=2) {
    for (i = 0; i < NUM_RUNNERS; ++i) {
      runner_args[i].datasize = j;
    }

    // Delimit experimental runs with a series of '-';
    for (i = 0; i < NUM_RUNNERS; ++i) {
      fprintf(output_files[i], "------------------------------\n");
      fprintf(output_files[i], "Datasize: %d %d\n", j, MAX_SIGNALS);
    }

    // Create the releasers
    for (i = 0; i < NUM_RUNNERS; ++i) {
      rc = pthread_create(&releasers[i], NULL, releaser, (void *) &releaser_args[i]);
      if (rc) {
        fprintf(stderr, "Error creating releaser: %d\n", rc);
        exit(-1);
      }
    }
    // Wait on releasers
    for (i = 0; i < NUM_RUNNERS; ++i) {
      if ((rc = pthread_join(releasers[i], NULL))) {
        // error
        fprintf(stderr, "Error joining releaser: %d.\n", rc);
      }
    }
  }
  fprintf(stderr, "Finished runs.\n");
  // Signal termination to runners
  for (i = 0; i < NUM_RUNNERS; ++i) {
    runner_args[i].isActive = 0;
    if ((rc = pthread_join(runners[i], NULL))) {
      // error
      fprintf(stderr, "Error joining runner: %d.\n", rc);
    }
  }
  pthread_exit(NULL);
}
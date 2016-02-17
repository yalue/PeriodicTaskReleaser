#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "runner.h"

////////tmp
#include "GPUOp.h"

#define NUM_RUNNERS 3
#define PERIOD_DURATION 1000
#define MAX_DATA_SIZE 8388608 // 2^23 ... 2^24 is too big
#define MAX_FILE_NAME 32
#define EXPERIMENT_DURATION 10000 // 10 seconds

int main(int argc, char *argv[]) {
  int rc;
  pthread_t runners[NUM_RUNNERS];
  pthread_mutex_t thread_mutexes[NUM_RUNNERS];
  pthread_mutexattr_t mutex_attributes[NUM_RUNNERS];
  struct Runner_Args runner_args[NUM_RUNNERS];
  FILE *output_files[NUM_RUNNERS];
  int i, j;

  struct timespec start_time;
  struct timespec end_time;

  // Initialize thread state
  for (i = 0; i < NUM_RUNNERS; ++i) {
    char file_name[MAX_FILE_NAME];

    // Create mutex attributes
    pthread_mutexattr_init(&mutex_attributes[i]);
    // Priority inheritance
    pthread_mutexattr_setprotocol(&mutex_attributes[i], PTHREAD_PRIO_INHERIT);
    // Create thread mutex.
    pthread_mutex_init(&thread_mutexes[i], &mutex_attributes[i]);
    // We may want to set mutex attributes to handle
    // priority ineversion differently than the defaults.

    // Lock the mutex so we can release the threads at a specified time.
    pthread_mutex_lock(&thread_mutexes[i]);

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
    runner_args[i].mutex = &thread_mutexes[i];
    runner_args[i].ostream = output_files[i];
    runner_args[i].ms = PERIOD_DURATION;
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
    ms = PERIOD_DURATION;
    ts.tv_sec = ms / MS_PER_SEC;
    ts.tv_nsec = (ms % MS_PER_SEC) * NS_PER_MS;
    nanosleep(&ts, NULL);
    fprintf(stderr, "Here we go!\n");
  }

  {
    j = 1024;
    // Delimit experimental runs with a series of '-';
    for (i = 0; i < NUM_RUNNERS; ++i) {
      fprintf(output_files[i], "------------------------------\n");
      fprintf(output_files[i], "Datasize: %d\n", j);
    }

    // Record start time
    clock_gettime(CLOCK_REALTIME, &start_time);
    end_time.tv_sec = start_time.tv_sec + EXPERIMENT_DURATION / MS_PER_SEC;
    end_time.tv_nsec = start_time.tv_nsec + (EXPERIMENT_DURATION % MS_PER_SEC) * NS_PER_MS;
    // Specify datasize
    for (i = 0; i < NUM_RUNNERS; ++i) {
      runner_args[i].datasize = j;
    }
    // Release runners
    for (i = 0; i < NUM_RUNNERS; ++i) {
      pthread_mutex_unlock(&thread_mutexes[i]);
    }
    printf("Going to sleep.\n");
    if ((rc = clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &end_time, NULL))) {
      fprintf(stderr, "Error during sleep: %s. Args: %lld.%.9ld\n", strerror(rc),
          (long long) end_time.tv_sec, end_time.tv_nsec);
    }
    printf("Woke up.\n");
    for (i = 0; i < NUM_RUNNERS; ++i) {
      pthread_mutex_lock(&thread_mutexes[i]);
      printf("Locked thread %d\n", i);
    }
  }
  for (i = 0; i < NUM_RUNNERS; ++i) {
    runner_args[i].isActive = 0;
    pthread_mutex_unlock(&thread_mutexes[i]);
  }
  fprintf(stderr, "Finished runs.\n");
  for (i = 0; i < NUM_RUNNERS; ++i) {
    fprintf(output_files[i], "------------------------------\n");
    fprintf(output_files[i], "Start %lld.%.9ld.\nEnd:  %lld.%.9ld.\n", 
        (long long) start_time.tv_sec, start_time.tv_nsec,
        (long long) end_time.tv_sec, end_time.tv_nsec);

    clock_gettime(CLOCK_REALTIME, &end_time);
    fprintf(output_files[i], "Expected duration: %d. Actual duration: %lld\n", 
        EXPERIMENT_DURATION,
        (long long) ((end_time.tv_sec - start_time.tv_sec) * 1e3 +
        (end_time.tv_nsec - start_time.tv_nsec) * 1e-6));
    fclose(output_files[i]);
  }
  pthread_exit(NULL);
}
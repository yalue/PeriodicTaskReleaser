#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <argp.h>

#include "runner.h"
#include "util.h"

#define SET_UP_DELAY 100

#define DEFAULT_DATA_SIZE 524288 // 2^19
#define DEFAULT_PERIOD  33  // 33.3 ms
#define DEFAULT_N_RUNNERS 1 // 1 runner
#define DEFAULT_WORST_CASE 33 // 33.3 ms
#define DEFAULT_EXPERIMENT_DURATION 1000 // 1 seconds
#define MAX_FILE_NAME 32

/*
 * Program Args
 */
const char *argp_program_version = "v1";
const char *argp_program_bug_address = "<vbmiller@cs.unc.edu>";
static char doc[] = "Periodic task launcher.";
static char args_doc[] = "";
static struct argp_option options[] = {
  {"duration", 'd', "experiment_duration", 0, "Specifies the duration the experiment should run in milliseconds."},
  {"size", 's', "data_size", 0, "Specifies the size of input data to the task. Some tasks may disregard this value."},
  {"period", 'p', "period", 0, "Specifies the release period of the task in milliseconds."},
  {"copies", 'n', "n_runners", 0, "Specifies the number of copies of tasks to release simultaneously. Typically 1."},
  {"worstcase", 'w', "worst_case", 0, "Specifies the worst-case execution time of this task."},
  {0},
};

struct arguments {
  int data_size;
  int experiment_duration;
  int period;
  int n_runners;
  int worst_case;
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  struct arguments *arguments = state->input;
  switch (key) {
    case 'd':
      arguments->experiment_duration = atoi(arg);
      if (arguments->experiment_duration < 0) {
        return EINVAL;
      }
      break;
    case 'n':
      arguments->n_runners = atoi(arg);
      if (arguments->n_runners < 0) {
        return EINVAL;
      }
      break;
    case 'p':
      arguments->period = atoi(arg);
      if (arguments->period < 0) {
        return EINVAL;
      }
      break;
    case 's':
      arguments->data_size = atoi(arg);
      if (arguments->data_size < 0) {
        return EINVAL;
      }
      break;
    case 'w':
      arguments->worst_case = atoi(arg);
      break;
    default:
      return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};

void launchThreads(int data_size, int period, int n_runners, int worst_case, int experiment_duration) {
  int rc;
  pthread_t runners[n_runners];
  pthread_mutex_t thread_mutexes[n_runners];
  pthread_mutexattr_t mutex_attributes[n_runners];
  pthread_barrier_t barrier;
  struct Runner_Args runner_args[n_runners];
  FILE *output_files[n_runners];
  int i;

  struct timespec start_time;
  struct timespec end_time;

  // Initialize barrier
  pthread_barrier_init(&barrier, NULL, n_runners + 1);
  // Initialize thread state
  for (i = 0; i < n_runners; ++i) {
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

    // Create output files for each runner
    snprintf(file_name, sizeof(char) * MAX_FILE_NAME, "runner%i.txt", i);
    output_files[i] = (fopen(file_name,"w"));
    if (!output_files[i]) {
      fprintf(stderr, "Error opening output file %s\n.", file_name);
      exit(EXIT_FAILURE);
    }

    // Assign thread args.
    runner_args[i].isActive = 1;
    runner_args[i].mutex = &thread_mutexes[i];
    runner_args[i].barrier = &barrier;
    runner_args[i].ostream = output_files[i];

    runner_args[i].datasize = data_size;
    runner_args[i].period_ms = period;
    runner_args[i].worst_case = worst_case;

    printf("Thread %d period: %d\n", i, runner_args[i].period_ms);
  }
  // Create runners
  for (i = 0; i < n_runners; ++i) {
    rc = pthread_create(&runners[i], NULL, runner, (void *) &runner_args[i]);
    if (rc) {
      fprintf(stderr, "Error creating runner: %d\n", rc);
      exit(EXIT_FAILURE);
    } else {
      fprintf(output_files[i], "Created thread: %d\n", i);
    }
  }
  {
    // Sleep launcher to allow all threads to initialize.
    fprintf(stderr, "Initializing threads...\n");
    struct timespec delay;
    int ms;
    ms = SET_UP_DELAY;
    delay.tv_sec = ms / MS_PER_SEC;
    delay.tv_nsec = (ms % MS_PER_SEC) * NS_PER_MS;
    nanosleep(&delay, NULL);
    fprintf(stderr, "Here we go!\n");
  }
  {
    // Delimit experimental runs with a series of '-';
    for (i = 0; i < n_runners; ++i) {
      fprintf(output_files[i], "------------------------------\n");
      fprintf(output_files[i], "Datasize: %d\n", data_size);
      fflush(output_files[i]);
    }

    // Record start time
    clock_gettime(CLOCK_REALTIME, &start_time);
    timespec_offset(&end_time, &start_time, experiment_duration);
    // Release runners
    for (i = 0; i < n_runners; ++i) {
      pthread_mutex_unlock(&thread_mutexes[i]);
    }
    fprintf(stderr, "Waiting for threads to complete...\n");
    if ((rc = clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &end_time, NULL))) {
      fprintf(stderr, "Error during sleep: %s. Args: %s.\n", strerror(rc), format_time(&end_time));
    }
    for (i = 0; i < n_runners; ++i) {
      pthread_mutex_lock(&thread_mutexes[i]);
      fprintf(stderr, "Thread %d completed.\n", i);
    }
  }
  for (i = 0; i < n_runners; ++i) {
    runner_args[i].isActive = 0;
    pthread_mutex_unlock(&thread_mutexes[i]);
  }
  pthread_barrier_wait(&barrier);
  fprintf(stderr, "Finished runs.\n");
  for (i = 0; i < n_runners; ++i) {
    fflush(output_files[i]);
    fclose(output_files[i]);
  }
  pthread_barrier_destroy(&barrier);
  fprintf(stderr, "Closed output files.\n");
}

int main(int argc, char *argv[]) {
  struct arguments arguments;
  // Default value
  arguments.data_size = DEFAULT_DATA_SIZE;
  arguments.period = DEFAULT_PERIOD;
  arguments.n_runners = DEFAULT_N_RUNNERS;
  arguments.worst_case = DEFAULT_WORST_CASE;
  arguments.experiment_duration = DEFAULT_EXPERIMENT_DURATION;
  // Parse args
  argp_parse(&argp, argc, argv, 0, 0, &arguments);
  launchThreads(arguments.data_size, arguments.period, arguments.n_runners, arguments.worst_case, arguments.experiment_duration);
  pthread_exit(NULL);
}

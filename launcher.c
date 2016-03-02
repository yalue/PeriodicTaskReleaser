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

#define INITIAL_DATA_SIZE 524288 // 2^19
#define MAX_DATA_SIZE 2097152 // 2^21 ... 2^24 is too big
#define MAX_FILE_NAME 32
#define EXPERIMENT_DURATION 1000 // 1 seconds

// Valid for 2^19 data only sched_other with locks
#define VECTOR_ADD_AVG_CASE 13
#define VECTOR_ADD_WORST_CASE 114
#define MATRIX_MULTIPLY_AVG_CASE 22
#define MATRIX_MULTIPLY_WORST_CASE 50

/*
 * Program Args
 */
const char *argp_program_version = "v1";
const char *argp_program_bug_address = "<vbmiller@cs.unc.edu>";
static char doc[] = "Periodic task launcher.";
static char args_doc[] = "";
static struct argp_option options[] = {
  {"size", 's', "data_size", OPTION_ARG_OPTIONAL, "Operate on data of this size."},
  {"runners", 'r', "num_runners", OPTION_ARG_OPTIONAL, "Spawn this many runners."},
  {"utilization", 'u', "utilization (0-100)", OPTION_ARG_OPTIONAL, 
      "Percentate utilization of each runner. Total utilization = num_runners * utilization."},
  {"worstCase", 'w', 0, OPTION_ARG_OPTIONAL, "Uses the worst-case execution time instead of average case."}
};

struct arguments {
  int data_size;
  int num_runners;
  int utilization;
  int worst_case;
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  struct arguments *arguments = state->input;
  switch (key) {
    case 's':
      arguments->data_size = atoi(arg);
      if (arguments->data_size == 0) {
        return EINVAL;
      }
      break;
    case 'r':
      arguments->num_runners = atoi(arg);
      if (arguments->num_runners == 0) {
        return EINVAL;
      }
      break;
    case 'w':
      arguments->worst_case = 1;
      break;
    case 'u':
      arguments->utilization = atoi(arg);
      if (arguments->num_runners == 0) {
        return EINVAL;
      }
      break;
    default:
      return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};

void launchThreads(int data_size, int runner_fns[], int period[], int num_runners) {
  int rc;
  pthread_t runners[num_runners];
  pthread_mutex_t thread_mutexes[num_runners];
  pthread_mutexattr_t mutex_attributes[num_runners];
  pthread_barrier_t barrier;
  struct Runner_Args runner_args[num_runners];
  FILE *output_files[num_runners];
  int i;

  struct timespec start_time;
  struct timespec end_time;

  // Initialize barrier
  pthread_barrier_init(&barrier, NULL, num_runners + 1);
  // Initialize thread state
  for (i = 0; i < num_runners; ++i) {
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
    runner_args[i].barrier = &barrier;
    runner_args[i].ostream = output_files[i];
    runner_args[i].ms = period[i];
    runner_args[i].function = runner_fns[i];
    printf("Thread %d period: %d\n", i, runner_args[i].ms);
  }
  // Create runners
  for (i = 0; i < num_runners; ++i) {
    rc = pthread_create(&runners[i], NULL, runner, (void *) &runner_args[i]);
    if (rc) {
      fprintf(stderr, "Error creating runner: %d\n", rc);
      exit(-1);
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
    for (i = 0; i < num_runners; ++i) {
      fprintf(output_files[i], "------------------------------\n");
      fprintf(output_files[i], "Datasize: %d\n", data_size);
      fflush(output_files[i]);
    }

    // Record start time
    clock_gettime(CLOCK_REALTIME, &start_time);
    timespec_offset(&end_time, &start_time, EXPERIMENT_DURATION);
    // Specify datasize
    for (i = 0; i < num_runners; ++i) {
      runner_args[i].datasize = data_size;
    }
    // Release runners
    for (i = 0; i < num_runners; ++i) {
      pthread_mutex_unlock(&thread_mutexes[i]);
    }
    printf("Waiting for threads to complete.\n");
    if ((rc = clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &end_time, NULL))) {
      fprintf(stderr, "Error during sleep: %s. Args: %s.\n", strerror(rc), format_time(&end_time));
    }
    for (i = 0; i < num_runners; ++i) {
      pthread_mutex_lock(&thread_mutexes[i]);
      printf("Thread %d completed.\n", i);
    }
  }
  for (i = 0; i < num_runners; ++i) {
    runner_args[i].isActive = 0;
    pthread_mutex_unlock(&thread_mutexes[i]);
  }
  pthread_barrier_wait(&barrier);
  fprintf(stderr, "Finished runs.\n");
  for (i = 0; i < num_runners; ++i) {
    fflush(output_files[i]);
    fclose(output_files[i]);
  }
  fprintf(stderr, "Closed output files.\n");
}

int main(int argc, char *argv[]) {
  struct arguments arguments;
  // Default value
  arguments.data_size = INITIAL_DATA_SIZE;
  arguments.num_runners = 1;
  arguments.utilization = 100;
  arguments.worst_case = 0;
  // Parse args
  argp_parse(&argp, argc, argv, 0, 0, &arguments);
  {
    int runner_fns[arguments.num_runners];
    int period[arguments.num_runners];

    if (arguments.num_runners > 0) {
      period[0] = (arguments.worst_case ? VECTOR_ADD_WORST_CASE : VECTOR_ADD_AVG_CASE) *
          100.0 / (float) arguments.utilization;
      runner_fns[0] = VECTOR_ADD;
    }
    if (arguments.num_runners > 1) {
      period[1] = (arguments.worst_case ? MATRIX_MULTIPLY_WORST_CASE : MATRIX_MULTIPLY_AVG_CASE) * 
          100.0 / (float) arguments.utilization;
      runner_fns[1] = MATRIX_MUL;
    }

    launchThreads(arguments.data_size, runner_fns, period, arguments.num_runners);
  }
  pthread_exit(NULL);
}

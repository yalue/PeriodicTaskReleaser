#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <argp.h>
#include <stdint.h>
#include <unistd.h>

#include "gpusync.h"
#include "util.h"

#define DEFAULT_DATA_SIZE 1024 // 2^10
#define DEFAULT_EXPERIMENT_DURATION  1800 // 30 minutes
#define DEFAULT_ITERATION_COUNT 2147483647 // 2^31 - 1
#define DEFAULT_SYNC 2
#define DEFAULT_RAND_SLEEP 0

#define FIFTEEN_MS_IN_NS 15000000 // 15 million

/*
 * Program Args
 */
const char *argp_program_version = "v1";
const char *argp_program_bug_address = "<vbmiller+gpusync@cs.unc.edu>";
static char doc[] = "GPU Sample Program Benchmarking.";
static char args_doc[] = "";
static struct argp_option options[] = {
  {0, 0, 0, 0, "Experiment configuration parameters:"},
  {"size", 's', "data_size", 0, "Specifies the size of input data to the task. Some tasks may disregard this value."},
  {"sync", 'y', "{0|1}", 0, "Specifies how the CPU should synchronize with the GPU kernel. {0: spin, 1: yield, default: block}."},
  {"randsleep", 'r', 0, OPTION_ARG_OPTIONAL, "Specifies that the program should sleep for a random amount of time between 0-15ms after each iteration."},
  {0, 0, 0, 0, "Experiment duration specifiers. If both are used, whichever limit is reached first will terminate the experiment."},
  {"iterations", 'n', "iteration_count", 0, "Specifies the maximum number of iterations of the benchmark program. Defaults to infinity."},
  {"duration", 'd', "experiment_duration", 0, "Specifies the duration the experiment should run in seconds. Defaults to 30 minutes."},
  {0},
};

struct arguments {
  int data_size;
  uint64_t experiment_duration;
  uint32_t iteration_count;
  int operation;
  int sync;
  int randsleep;
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  struct arguments *arguments = state->input;
  switch (key) {
    case 'd':
      {
        int duration = atoi(arg);
        if (duration < 0) {
          return EINVAL;
        }
        arguments->experiment_duration = (uint64_t) duration;
      }
      break;
    case 'n':
      {
        int iterations = atoi(arg);
        if (iterations < 0) {
          return EINVAL;
        }
        arguments->iteration_count = iterations;
      }
      break;
    case 'r':
      arguments->randsleep = 1;
      break;
    case 's':
      arguments->data_size = atoi(arg);
      if (arguments->data_size < 0) {
        return EINVAL;
      }
      break;
    case 'y':
      arguments->sync= atoi(arg);
      if (arguments->sync < 0 || arguments->sync > 2) {
        return EINVAL;
      }
      break;
    default:
      return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};

int main(int argc, char** argv) {
  struct arguments arguments;
  // Default values
  arguments.data_size = DEFAULT_DATA_SIZE;
  arguments.experiment_duration = ((uint64_t) DEFAULT_EXPERIMENT_DURATION);
  arguments.iteration_count = (uint64_t) DEFAULT_ITERATION_COUNT;
  arguments.sync = DEFAULT_SYNC;
  arguments.randsleep = DEFAULT_RAND_SLEEP;
  // Parse args
  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  struct timespec start, end, experiment_start, tmp;
  int i;

  // print output header
  fprintf(stdout, "Timestamp, program, PID, function, call/ret, arg\n");
  CURRENT_TIME(&experiment_start);
  // initialize end time to experiment start time.
  // this copies the primitive fields tv_sec and tv_nsec.
  end = experiment_start;

  init(arguments.sync);
  mallocCPU(arguments.data_size);
  mallocGPU(arguments.data_size);

  for (i = 0; elapsed_sec(&experiment_start, &end) < arguments.experiment_duration && i < arguments.iteration_count; ++i) {
    CURRENT_TIME(&start);
    fprintf(stdout, "%s, %s, %d, start\n", format_time(&start),
        argv[0], getpid());

    CURRENT_TIME(&tmp);
    fprintf(stdout, "%s, %s, %d, cudaMemcpy, call, hostToDevice\n",
        format_time(&tmp), argv[0], getpid());
    copyin(arguments.data_size);
    CURRENT_TIME(&tmp);
    fprintf(stdout, "%s, %s, %d, cudaMemcpy, return, hostToDevice\n",
        format_time(&tmp), argv[0], getpid());

    CURRENT_TIME(&tmp);
    fprintf(stdout, "%s, %s, %d, cudaLaunch, call\n", format_time(&tmp),
        argv[0], getpid());
    exec(arguments.data_size);
    CURRENT_TIME(&tmp);
    fprintf(stdout, "%s, %s, %d, cudaLaunch, return\n", format_time(&tmp),
        argv[0], getpid());

    CURRENT_TIME(&tmp);
    fprintf(stdout, "%s, %s, %d, cudaMemcpy, call, deviceToHost\n",
        format_time(&tmp), argv[0], getpid());
    copyout();
    CURRENT_TIME(&tmp);
    fprintf(stdout, "%s, %s, %d, cudaMemcpy, return, deviceToHost\n",
        format_time(&tmp), argv[0], getpid());

    CURRENT_TIME(&end);
    fprintf(stdout, "%s, %s, %d, end\n", format_time(&end),
        argv[0], getpid());
    
    // Sleep for small amount of time to emulate periodicity.
    struct timespec delay;
    delay.tv_sec = 0;
    delay.tv_nsec = arguments.randsleep * (rand() % FIFTEEN_MS_IN_NS);
    nanosleep(&delay, NULL);
  }
  freeGPU();
  freeCPU();
  finish();
  exit(EXIT_SUCCESS);
}

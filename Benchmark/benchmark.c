#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <argp.h>
#include <stdint.h>

#include "gpusync.h"
#include "util.h"

#define ALL_OPERATIONS 0
#define COPY_OPERATIONS 1
#define EXEC_OPERATIONS 2
#define DEFAULT_OPERATION ALL_OPERATIONS
#define DEFAULT_DATA_SIZE 1024 // 2^10
#define DEFAULT_EXPERIMENT_DURATION  1800 // 30 minutes
#define DEFAULT_ITERATION_COUNT 2147483647 // 2^31 - 1
#define DEFAULT_SYNC 2

#define FIVE_MS_IN_NS 5000000 // 5 million

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

  {0, 0, 0, 0, "Mutual exclusive choice of GPU operations to benchmark:"},
  {"all", 'a', 0, OPTION_ARG_OPTIONAL, "Benchmarks the entire program, excluding setup and teardown. This is the default option."},
  {"copy", 'c', 0, OPTION_ARG_OPTIONAL, "Benchmarks the copy engine use only."},
  {"exec", 'e', 0, OPTION_ARG_OPTIONAL, "Benchmarks the execution engine use only."},
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
};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  struct arguments *arguments = state->input;
  switch (key) {
    case 'd':
      int duration = atoi(arg);
      if (duration < 0) {
        return EINVAL;
      }
      arguments->experiment_duration = (uint64_t) duration;
      break;
    case 'n':
      int iterations = atoi(arg);
      if (iterations < 0) {
        return EINVAL;
      }
      arguments->iteration_count = iterations;
      break;
    case 's':
      arguments->data_size = atoi(arg);
      if (arguments->data_size < 0) {
        return EINVAL;
      }
      break;
    case 'a':
      arguments->operation = ALL_OPERATIONS;
      break;
    case 'c':
      arguments->operation = COPY_OPERATIONS;
      break;
    case 'e':
      arguments->operation = EXEC_OPERATIONS;
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
  arguments.operation = DEFAULT_OPERATION;
  arguments.sync = DEFAULT_SYNC;
  // Parse args
  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  struct timespec start, end, experiment_start;
  int i;

  if (clock_gettime(CLOCK_REALTIME, &experiment_start)) {
    error("Error getting time.");
    exit(EXIT_FAILURE);
  }
  // initialize end time to experiment start time.
  // this copies the primitive fields tv_sec and tv_nsec.
  end = experiment_start;

  init(arguments.sync);
  mallocCPU(arguments.data_size);

  for (i = 0; elapsed_sec(&experiment_start, &end) < arguments.experiment_duration && i < arguments.iteration_count; ++i) {
    if (clock_gettime(CLOCK_REALTIME, &start)) {
      error("Error getting time.");
      exit(EXIT_FAILURE);
    }
    // operation = {0: all, 1: copy engine, 2: execution engine}
    if (arguments.operation <= 1) { // (operation == 1 || operation == 0)
      mallocGPU(arguments.data_size);
      copyin(arguments.data_size);
    } else if (arguments.operation % 2 == 0) { // (operation == 2 || operation == 0)
      exec(arguments.data_size);
    }
    if (arguments.operation <= 1) {
      copyout();
      freeGPU();
    }
    if (clock_gettime(CLOCK_REALTIME, &end)) {
      error("Error getting time.");
      exit(EXIT_FAILURE);
    }
    fprintf(stdout, "%3ld,\n", elapsed_ns(&start, &end));
    
    // Sleep for small amount of time to emulate periodicity.
    {
      struct timespec delay;
      delay.tv_sec = 0;
      delay.tv_nsec = FIVE_MS_IN_NS;
      nanosleep(&delay, NULL);
    }
  }
  freeCPU();
  finish();
  exit(EXIT_SUCCESS);
}

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <argp.h>

#include "gpusync.h"
#include "util.h"

#define ALL_OPERATIONS 0
#define COPY_OPERATIONS 1
#define EXEC_OPERATIONS 2
#define DEFAULT_DATA_SIZE 524288 // 2^19
#define DEFAULT_EXPERIMENT_DURATION 60000 // 1 minute 
#define DEFAULT_ITERATION_COUNT 10000

/*
 * Program Args
 */
const char *argp_program_version = "v1";
const char *argp_program_bug_address = "<vbmiller@cs.unc.edu>";
static char doc[] = "GPU Sample Program Benchmarking.";
static char args_doc[] = "";
static struct argp_option options[] = {
  {"duration", 'd', "experiment_duration", OPTION_HIDDEN, "Specifies the duration the experiment should run in milliseconds."},
  {"iterations", 'n', "iteration_count", 0, "Specifies the number of iterations to run before benchmarking is complete."},
  {"size", 's', "data_size", 0, "Specifies the size of input data to the task. Some tasks may disregard this value."},
  {"all", 'a', 0, OPTION_ARG_OPTIONAL, "Benchmarks the entire program, excluding setup and teardown."},
  {"copy", 'c', 0, OPTION_ARG_OPTIONAL, "Benchmarks the copy engine use only."},
  {"exec", 'e', 0, OPTION_ARG_OPTIONAL, "Benchmarks the execution engine use only."},
  {0},
};

struct arguments {
  int data_size;
  int experiment_duration;
  int iteration_count;
  int operation;
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
      arguments->iteration_count = atoi(arg);
      if (arguments->iteration_count < 0) {
        return EINVAL;
      }
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
  arguments.experiment_duration = DEFAULT_EXPERIMENT_DURATION;
  arguments.iteration_count = DEFAULT_ITERATION_COUNT;
  // Parse args
  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  struct timespec start, end;
  int i;

  init(0);
  mallocCPU(arguments.data_size);

  for (i = 0; i < arguments.iteration_count; ++i) {
    if (clock_gettime(CLOCK_REALTIME, &start)) {
      error("Error getting time.");
      break;
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
      break;
    }
    fprintf(stdout, "%3ld,\n", elapsed_ns(&start, &end));
  }
  freeCPU();
  finish();
  exit(EXIT_SUCCESS);
}

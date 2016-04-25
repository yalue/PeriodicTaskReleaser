#define MAX_ITERATIONS 10000

struct Runner_Args {
  int isActive;
  int datasize;
  int period;
  int worst_case;

  pthread_mutex_t *mutex;
  pthread_barrier_t *barrier;
  FILE *ostream;
};

void * runner(void *runner_args);

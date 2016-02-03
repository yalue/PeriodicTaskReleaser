#define MAX_ITERATIONS 10

struct Runner_Args {
  int thread_id;
  int ms;
  pthread_cond_t *cond;
  pthread_mutex_t *mutex;
  FILE *ostream;
};

void * runner(void *runner_args);